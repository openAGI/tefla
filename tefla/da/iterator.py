from __future__ import division, print_function, absolute_import

import Queue
import SharedArray
import multiprocessing
import os
import threading
from uuid import uuid4

import numpy as np

from . import data


class BatchIterator(object):

    def __init__(self, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, X, y=None):
        if self.shuffle:
            index_array = np.random.permutation(len(X))
            self.X = X[index_array]
            self.y = y[index_array] if y is not None else y
        else:
            self.X, self.y = X, y
        return self

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if attr in state:
                del state[attr]
        return state


class QueuedMixin(object):

    def __iter__(self):
        queue = Queue.Queue(maxsize=20)
        end_marker = object()

        def producer():
            for Xb, yb in super(QueuedMixin, self).__iter__():
                queue.put((np.array(Xb), np.array(yb)))
            queue.put(end_marker)

        thread = threading.Thread(target=producer)
        thread.daemon = True
        thread.start()

        item = queue.get()
        while item is not end_marker:
            yield item
            queue.task_done()
            item = queue.get()


class QueuedIterator(QueuedMixin, BatchIterator):
    pass


class DAIterator(BatchIterator):

    def __call__(self, X, y=None, crop_bbox=None, xform=None):
        self.crop_bbox = crop_bbox
        self.xform = xform
        return super(DAIterator, self).__call__(X, y)

    def __init__(self, batch_size, shuffle, preprocessor, crop_size, is_training,
                 aug_params=data.no_augmentation_params, fill_mode='constant', fill_mode_cval=0, standardizer=None,
                 save_to_dir=None):
        self.preprocessor = preprocessor if preprocessor else data.image_no_preprocessing
        self.w = crop_size[0]
        self.h = crop_size[1]
        self.is_training = is_training
        self.aug_params = aug_params
        self.fill_mode = fill_mode
        self.fill_mode_cval = fill_mode_cval
        self.standardizer = standardizer
        self.save_to_dir = save_to_dir
        if save_to_dir and not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        super(DAIterator, self).__init__(batch_size, shuffle)

    def da_args(self):
        kwargs = {'preprocessor': self.preprocessor, 'w': self.w, 'h': self.h, 'is_training': self.is_training,
                  'fill_mode': self.fill_mode, 'fill_mode_cval': self.fill_mode_cval, 'standardizer': self.standardizer,
                  'save_to_dir': self.save_to_dir}
        if self.crop_bbox is not None:
            assert not self.is_training, "crop bbox only in validation/prediction mode"
            kwargs['bbox'] = self.crop_bbox
        elif self.xform is not None:
            assert not self.is_training, "transform only in validation/prediction mode"
            kwargs['transform'] = self.xform
        else:
            kwargs['aug_params'] = self.aug_params
        return kwargs

    def transform(self, Xb, yb):
        fnames, labels = Xb, yb
        Xb = data.load_augmented_images(fnames, **self.da_args())
        return Xb, labels


class QueuedDAIterator(QueuedMixin, DAIterator):
    pass


pool_process_seed = None


def load_shared(args):
    import os
    i, array_name, fname, kwargs = args
    array = SharedArray.attach(array_name)
    global pool_process_seed
    if not pool_process_seed:
        pool_process_seed = os.getpid()
        # print("random seed: %d in pid %d" % (pool_process_seed, os.getpid()))
        np.random.seed(pool_process_seed)
    array[i] = data.load_augment(fname, **kwargs)


class ParallelDAIterator(QueuedDAIterator):

    def __init__(self, batch_size, shuffle, preprocessor, crop_size, is_training,
                 aug_params=data.no_augmentation_params, fill_mode='constant', fill_mode_cval=0, standardizer=None,
                 save_to_dir=None):
        self.pool = multiprocessing.Pool()
        super(ParallelDAIterator, self).__init__(batch_size, shuffle, preprocessor, crop_size, is_training, aug_params,
                                                 fill_mode, fill_mode_cval, standardizer, save_to_dir)

    def transform(self, Xb, yb):
        shared_array_name = str(uuid4())
        try:
            shared_array = SharedArray.create(
                shared_array_name, [len(Xb), self.w, self.h, 3], dtype=np.float32)

            fnames, labels = Xb, yb
            args = []
            da_args = self.da_args()
            for i, fname in enumerate(fnames):
                args.append((i, shared_array_name, fname, da_args))

            self.pool.map(load_shared, args)
            Xb = np.array(shared_array, dtype=np.float32)

        finally:
            SharedArray.delete(shared_array_name)

        # if labels is not None:
        #     labels = labels[:, np.newaxis]

        return Xb, labels


def balance_data(X, y, balance_ratio, count, balance_weights, final_balance_weights):
    alpha = balance_ratio ** count
    class_weights = balance_weights * alpha + \
        final_balance_weights * (1 - alpha)
    count += 1
    indices = data.balance_per_class_indices(y, weights=class_weights)
    X = X[indices]
    y = y[indices]
    return X, y, count


class BalancingDAIterator(ParallelDAIterator):

    def __init__(
            self, batch_size, shuffle, preprocessor, crop_size, is_training,
            balance_weights, final_balance_weights, balance_ratio, balance_epoch_count=0,
            aug_params=data.no_augmentation_params,
            fill_mode='constant', fill_mode_cval=0, standardizer=None, save_to_dir=None):
        self.count = balance_epoch_count
        self.balance_weights = balance_weights
        self.final_balance_weights = final_balance_weights
        self.balance_ratio = balance_ratio
        super(BalancingDAIterator, self).__init__(batch_size, shuffle, preprocessor, crop_size, is_training, aug_params,
                                                  fill_mode, fill_mode_cval, standardizer, save_to_dir)

    def __call__(self, X, y=None):
        if y is not None:
            X, y, self.count = balance_data(
                X, y, self.balance_ratio, self.count, self.balance_weights, self.final_balance_weights)
        return super(BalancingDAIterator, self).__call__(X, y)


# Todo remove code duplication with BalancingDAIterator (call method)
class BalancingQueuedDAIterator(QueuedDAIterator):

    def __init__(
            self, batch_size, shuffle, preprocessor, crop_size, is_training,
            balance_weights, final_balance_weights, balance_ratio, balance_epoch_count=0,
            aug_params=data.no_augmentation_params,
            fill_mode='constant', fill_mode_cval=0, standardizer=None, save_to_dir=None):
        self.count = balance_epoch_count
        self.balance_weights = balance_weights
        self.final_balance_weights = final_balance_weights
        self.balance_ratio = balance_ratio
        super(BalancingQueuedDAIterator, self).__init__(batch_size, shuffle, preprocessor, crop_size, is_training,
                                                        aug_params, fill_mode, fill_mode_cval, standardizer,
                                                        save_to_dir)

    def __call__(self, X, y=None):
        if y is not None:
            X, y, self.count = balance_data(
                X, y, self.balance_ratio, self.count, self.balance_weights, self.final_balance_weights)
        return super(BalancingQueuedDAIterator, self).__call__(X, y)
