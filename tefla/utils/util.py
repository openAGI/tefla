import importlib
import logging
import collections
import six
import os
import subprocess
import time
from datetime import datetime
from scipy.misc import imsave
import matplotlib.pyplot as plt
import numbers
import numpy as np
from progress.bar import Bar
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
import tensorflow as tf

from .quadratic_weighted_kappa import quadratic_weighted_kappa

OrderedDict = collections.OrderedDict


def roc(y_true, y_pred, classes=[0, 1, 2, 3, 4]):
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc


def float32(k):
    return np.cast['float32'](k)


def get_bbox_10crop(crop_size, im_size):
    im_center = im_size[:2] / 2.0
    h_indices = (0, (im_size[0] - crop_size[0]) / 2.0)
    w_indices = (0, (im_size[1] - crop_size[1]) / 2.0)
    bboxs = np.empty((5, 5), dtype=np.int32)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            bboxs[curr, :4] = (i, j, i + crop_size[0], j + crop_size[1])
            bboxs[curr, 4] = 1
            curr += 1
    bboxs[4, :4] = np.tile(im_center, (1, 2)) + \
        np.concatenate([-crop_size / 2.0, crop_size / 2.0])
    bboxs[4, 4] = 1
    bboxs = np.tile(bboxs, (2, 1))
    bboxs[5:, 4] = 0

    return bboxs


def get_predictions(feature_net, images, tf=None, bbox=None, color_vec=None):
    tic = time.time()
    feature_net.batch_iterator_test.xform = tf
    feature_net.batch_iterator_test.crop_bbox = bbox
    feature_net.batch_iterator_test.standardizer.color_vec = color_vec
    predictions = feature_net.predict_proba(images)
    print('took %6.1f seconds' % (time.time() - tic))

    return predictions


def kappa_from_proba(w, p, y_true):
    return kappa(y_true, p.dot(w))


def load_module(mod):
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def get_commit_sha():
    p = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                         stdout=subprocess.PIPE)
    output, _ = p.communicate()
    return output.strip().decode('utf-8')


def get_submission_filename():
    sha = get_commit_sha()
    return "data/sub_{}_{}.csv".format(sha,
                                       datetime.now().replace(microsecond=0))


def f1_score_wrapper(y_true, y_pred):
    # y_pred_2 = np.asarray([1 if b > a else 0 for [a, b] in y_pred])
    y_pred_2 = np.argmax(y_pred, axis=1)
    # print("F1 score inputs:")
    # print(y_true)
    # print(y_pred_2)
    # print("---")
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred_2)
    return accuracy_score(y_true, y_pred_2) if 0 in f1 else np.mean(f1)


def auroc_wrapper(y_true, y_pred):
    # print("Auroc score inputs:")
    # print(y_true.tolist())
    # y_pred_2 = np.argmax(y_pred, axis=1)
    # print(y_pred_2)
    # print(y_pred[:, 1].tolist())
    # print("---")
    try:
        return roc_auc_score(y_true, y_pred[:, 1])
    except ValueError as e:
        print(e.message)
        return accuracy_score(y_true, np.argmax(y_pred, axis=1))


def kappa_wrapper(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = y_true.dot(range(y_true.shape[1]))
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    try:
        return quadratic_weighted_kappa(y_true, y_pred)
    except IndexError:
        return np.nan


kappa = kappa_wrapper


def accuracy_wrapper(y_true, y_pred):
    try:
        return accuracy_score(y_true, y_pred.argmax(axis=1))
    except Exception:
        return accuracy_score(y_true, y_pred)


def dump_vars(sess):
    all_vars = set(tf.all_variables())
    trainable_vars = set(tf.trainable_variables())
    non_trainable_vars = all_vars.difference(trainable_vars)

    def _dump_set(var_set):
        names_vars = map(lambda v: (v.name, v), var_set)
        for n, v in sorted(names_vars, key=lambda nv: nv[0]):
            print("%s=%s" % (n, sess.run(v)))

    print("Variable values:")
    print("-----------")
    print("\n---Trainable vars:")
    _dump_set(trainable_vars)
    print("\n---Non Trainable vars:")
    _dump_set(non_trainable_vars)
    print("-----------")


def init_logging(file_name, file_log_level, console_log_level, clean=False):
    import sys
    logger = logging.getLogger('tefla')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    h1 = logging.StreamHandler(stream=sys.stdout)
    h1.setLevel(console_log_level)
    h1.setFormatter(formatter)
    logger.addHandler(h1)

    mode = 'w' if clean else 'a'
    h2 = logging.FileHandler(file_name, mode=mode)
    h2.setLevel(file_log_level)
    h2.setFormatter(formatter)
    logger.addHandler(h2)


def get_input_shape(x):
    "Return input shape"
    if isinstance(x, tf.Tensor):
        return x.get_shape().as_list()
    elif type(x) in [np.ndarray, list, tuple]:
        return np.shape(x)
    else:
        raise Exception("Invalid input layer")


def valid_dtypes():
    """Valid types for loss, variables and gradients.
        Subclasses should override to allow other float types.
    Returns:
        Valid types for loss, variables and gradients.
    """
    return set([dtypes.float16, dtypes.float32, dtypes.float64])


def assert_valid_dtypes(tensors):
    """Asserts tensors are all valid types (see `_valid_dtypes`).
    Args:
        tensors: Tensors to check.
    Raises:
        ValueError: If any tensor is not a valid type.
    """
    valid_dtype = valid_dtypes()
    for t in tensors:
        dtype = t.dtype.base_dtype
        if dtype not in valid_dtype:
            raise ValueError("Invalid type %r for %s, expected: %s." %
                             (dtype, t.name, [v for v in valid_dtype]))


def constant_value(value_or_tensor_or_var, dtype=None):
    """Returns value if value_or_tensor_or_var has a constant value.

    Args:
        value_or_tensor_or_var: A value, a `Tensor` or a `Variable`.
        dtype: Optional `tf.dtype`, if set it would check it has the right
          dtype.

    Returns:
        The constant value or None if it not constant.

    Raises:
        ValueError: if value_or_tensor_or_var is None or the tensor_variable has the
                    wrong dtype.
    """
    if value_or_tensor_or_var is None:
        raise ValueError('value_or_tensor_or_var cannot be None')
    value = value_or_tensor_or_var
    if isinstance(value_or_tensor_or_var, (ops.Tensor, variables.Variable)):
        if dtype and value_or_tensor_or_var.dtype != dtype:
            raise ValueError('It has the wrong type %s instead of %s' %
                             (value_or_tensor_or_var.dtype, dtype))
        if isinstance(value_or_tensor_or_var, variables.Variable):
            value = None
        else:
            value = tensor_util.constant_value(value_or_tensor_or_var)
    return value


def static_cond(pred, fn1, fn2):
    """Return either fn1() or fn2() based on the boolean value of `pred`.

    Same signature as `control_flow_ops.cond()` but requires pred to be a bool.

    Args:
        pred: A value determining whether to return the result of `fn1` or `fn2`.
        fn1: The callable to be performed if pred is true.
        fn2: The callable to be performed if pred is false.

    Returns:
        Tensors returned by the call to either `fn1` or `fn2`.

    Raises:
        TypeError: if `fn1` or `fn2` is not callable.
    """
    if not callable(fn1):
        raise TypeError('fn1 must be callable.')
    if not callable(fn2):
        raise TypeError('fn2 must be callable.')
    if pred:
        return fn1()
    else:
        return fn2()


def smart_cond(pred, fn1, fn2, name=None):
    """Return either fn1() or fn2() based on the boolean predicate/value `pred`.

    If `pred` is bool or has a constant value it would use `static_cond`,
     otherwise it would use `tf.cond`.

    Args:
        pred: A scalar determining whether to return the result of `fn1` or `fn2`.
        fn1: The callable to be performed if pred is true.
        fn2: The callable to be performed if pred is false.
        name: Optional name prefix when using tf.cond
    Returns:
        Tensors returned by the call to either `fn1` or `fn2`.
    """
    pred_value = constant_value(pred)
    if pred_value is not None:
        # Use static_cond if pred has a constant value.
        return static_cond(pred_value, fn1, fn2)
    else:
        # Use dynamic cond otherwise.
        return control_flow_ops.cond(pred, fn1, fn2, name)


def get_variable_collections(variables_collections, name):
    if isinstance(variables_collections, dict):
        variable_collections = variables_collections.get(name, None)
    else:
        variable_collections = variables_collections
    return variable_collections


def veryify_args(actual, allowed_keys, msg_prefix):
    actual_keys = set(actual.keys())
    extra = list(actual_keys - set(allowed_keys))
    if len(extra) > 0:
        raise ValueError("%s %s" % (msg_prefix, extra))


def rms(x, name=None):
    if name is None:
        name = x.op.name + '/rms'
    return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)


def weight_bias(W_shape, b_shape, w_init=tf.truncated_normal, b_init=0.0, w_regularizer=tf.nn.l2_loss, trainable=True, name='maingate'):
    W = tf.get_variable(name=name + 'W', shape=W_shape, initializer=w_init,
                        regularizer=w_regularizer, trainable=trainable)
    b = tf.get_variable(name=name + 'b', shape=b_shape, initializer=tf.constant_initializer(
        b_init), trainable=trainable)
    return W, b


def one_hot(labels, num_classes, name='one_hot'):
    """Transform numeric labels into onehot_labels.
    Args:
        labels: [batch_size] target labels.
        num_classes: total number of classes.
        scope: Optional scope for op_scope.
    Returns:
        one hot encoding of the labels.
    """
    with tf.op_scope(name):
        batch_size = labels.get_shape()[0]
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        labels = tf.cast(tf.expand_dims(labels, 1), indices.dtype)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(
            concated, tf.pack([batch_size, num_classes]), 1.0, 0.0)
        onehot_labels.set_shape([batch_size, num_classes])
        return onehot_labels


def is_sequence(seq):
    """Returns a true if its input is a collections.Sequence (except strings).

    Args:
        seq: an input sequence.

    Returns:
        True if the sequence is a not a string and is a collections.Sequence.
    """
    return (isinstance(seq, collections.Sequence) and not isinstance(seq, six.string_types))


def flatten_sq(nest_sq):
    """Returns a flat sequence from a given nested structure.
    If `nest` is not a sequence, this returns a single-element list: `[nest]`.

    Args:
        nest: an arbitrarily nested structure or a scalar object.
            Note, numpy arrays are considered scalars.

    Returns:
        A Python list, the flattened version of the input.
    """
    return list(_yield_flat_nest(nest_sq)) if is_sequence(nest_sq) else [nest_sq]


def _yield_flat_nest(nest_sq):
    for n in nest_sq:
        if is_sequence(n):
            for ni in _yield_flat_nest(n):
                yield ni
        else:
            yield n


class VariableDeviceChooser(object):
    """Device chooser for variables.
    When using a parameter server it will assign them in a round-robin fashion.
    When not using a parameter server it allows GPU:0 placement otherwise CPU:0.
    Initialize VariableDeviceChooser.

    Args:
        num_parameter_servers: number of parameter servers.
        ps_device: string representing the parameter server device.
        placement: string representing the placement of the variable either CPU:0
            or GPU:0. When using parameter servers forced to CPU:0.
    """

    def __init__(self, num_parameter_servers=0, ps_device='/job:ps', placement='CPU:0'):
        self._num_ps = num_parameter_servers
        self._ps_device = ps_device
        self._placement = placement if num_parameter_servers == 0 else 'CPU:0'
        self._next_task_id = 0

    def __call__(self, op):
        device_string = ''
        if self._num_ps > 0:
            task_id = self._next_task_id
            self._next_task_id = (self._next_task_id + 1) % self._num_ps
            device_string = '%s/task:%d' % (self._ps_device, task_id)
        device_string += '/%s' % self._placement
        return device_string


def last_dimension(shape, min_rank=1):
    """Returns the last dimension of shape while checking it has min_rank.

    Args:
        shape: A `TensorShape`.
        min_rank: Integer, minimum rank of shape.

    Returns:
        The value of the last dimension.

    Raises:
        ValueError: if inputs don't have at least min_rank dimensions, or if the
            last dimension value is not defined.
    """
    dims = shape.dims
    if dims is None:
        raise ValueError('dims of shape must be known but is None')
    if len(dims) < min_rank:
        raise ValueError(
            'rank of shape must be at least %d not: %d' % (min_rank, len(dims)))
    value = dims[-1].value
    if value is None:
        raise ValueError('last dimension shape must be known but is None')
    return value


def load_frozen_graph(frozen_graph):
    """Load Graph from frozen weights and model

    Args:
        frozen_graph: binary pb file

    Returns:
        loaded graph
    """
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    try:
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name='model',
                op_dict=None,
                producer_op_list=None
            )
        return graph
    except Exception as e:
        print(e.message)


def normalize(input_layer):
    """ Normalize a input layer

    Args:
        inmput_layer: input layer tp normalize

    Returns:
        normalized layer
    """
    return input_layer / 127.5 - 1.


def denormalize(input_layer):
    """ DeNormalize a input layer

    Args:
        input_layer: input layer to de normalize

    Returns:
        denormalized layer
    """
    return (input_layer + 1.) / 2.


class ProgressBar(Bar):
    """Display progress bar

    """
    message = 'Loading'
    fill = '...'
    suffix = '%(percent).1f%% | ETA: %(eta)ds'


def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v


def get_vars_maybe_avg(var_names, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, **kwargs))
    return vars


def save_images(fname, flat_img, width=28, height=28, sep=3, channels=3):
    N = flat_img.shape[0]
    pdim = int(np.ceil(np.sqrt(N)))
    image = np.zeros((pdim * (width + sep), pdim *
                      (height + sep), channels))
    for i in range(N):
        row = int(i / pdim) * (height + sep)
        col = (i % pdim) * (width + sep)
        image[row:row + width, col:col +
              height, :] = flat_img[i].reshape(width, height, channels)
    if channels == 1:
        image = image.reshape(pdim * (width + sep), pdim * (height + sep))
    imsave(fname, image)


def stride_2d(strides):
    if isinstance(strides, int):
        return [1, strides, strides, 1]
    elif isinstance(strides, (tuple, list)):
        if len(strides) == 2:
            return [1, strides[0], strides[1], 1]
        elif len(strides) == 4:
            return [strides[0], strides[1], strides[2], strides[3]]
        else:
            raise Exception("strides length error: " + str(len(strides))
                            + ", only a length of 2 or 4 is supported.")
    else:
        raise Exception("strides format error: " + str(type(strides)))


def kernel_2d(kernel):
    if isinstance(kernel, int):
        return [1, kernel, kernel, 1]
    elif isinstance(kernel, (tuple, list)):
        if len(kernel) == 2:
            return [1, kernel[0], kernel[1], 1]
        elif len(kernel) == 4:
            return [kernel[0], kernel[1], kernel[2], kernel[3]]
        else:
            raise Exception("kernel length error: " + str(len(kernel))
                            + ", only a length of 2 or 4 is supported.")
    else:
        raise Exception("kernel format error: " + str(type(kernel)))


def filter_2d(fsize, in_depth, out_depth):
    if isinstance(fsize, int):
        return [fsize, fsize, in_depth, out_depth]
    elif isinstance(fsize, (tuple, list)):
        if len(fsize) == 2:
            return [fsize[0], fsize[1], in_depth, out_depth]
        else:
            raise Exception("filter length error: " + str(len(fsize))
                            + ", only a length of 2 is supported.")
    else:
        raise Exception("filter format error: " + str(type(fsize)))


def kernel_padding(padding):
    if padding in ['same', 'SAME', 'valid', 'VALID']:
        return str.upper(padding)
    else:
        raise Exception("Unknown padding! Accepted values: 'same', 'valid'.")


def filter_3d(fsize, in_depth, out_depth):
    if isinstance(fsize, int):
        return [fsize, fsize, fsize, in_depth, out_depth]
    elif isinstance(fsize, (tuple, list)):
        if len(fsize) == 3:
            return [fsize[0], fsize[1], fsize[2], in_depth, out_depth]
        else:
            raise Exception("filter length error: " + str(len(fsize))
                            + ", only a length of 3 is supported.")
    else:
        raise Exception("filter format error: " + str(type(fsize)))


def stride_3d(strides):
    if isinstance(strides, int):
        return [1, strides, strides, strides, 1]
    elif isinstance(strides, (tuple, list)):
        if len(strides) == 3:
            return [1, strides[0], strides[1], strides[2], 1]
        elif len(strides) == 5:
            assert strides[0] == strides[
                4] == 1, "Must have strides[0] = strides[4] = 1"
            return [strides[0], strides[1], strides[2], strides[3], strides[4]]
        else:
            raise Exception("strides length error: " + str(len(strides))
                            + ", only a length of 3 or 5 is supported.")
    else:
        raise Exception("strides format error: " + str(type(strides)))


def kernel_3d(kernel):
    if isinstance(kernel, int):
        return [1, kernel, kernel, kernel, 1]
    elif isinstance(kernel, (tuple, list)):
        if len(kernel) == 3:
            return [1, kernel[0], kernel[1], kernel[2], 1]
        elif len(kernel) == 5:
            assert kernel[0] == kernel[
                4] == 1, "Must have kernel[0] = kernel[4] = 1"
            return [kernel[0], kernel[1], kernel[2], kernel[3], kernel[4]]
        else:
            raise Exception("kernels length error: " + str(len(kernel))
                            + ", only a length of 3 or 5 is supported.")
    else:
        raise Exception("kernel format error: " + str(type(kernel)))


def accuracy_tf(labels, predictions):
    return tf.contrib.metrics.accuracy(labels, predictions)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def get_hist(predictions, labels, num_classes, batch_size=1):
    hist = np.zeros((num_classes, num_classes))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(),
                          predictions[i].argmax(2).flatten(), num_classes)
    return hist


def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.

    Args:
        x: a tensor of shape [num_x_samples, num_features]
        y: a tensor of shape [num_y_samples, num_features]

    Returns:
        a distance matrix of dimensions [num_x_samples, num_y_samples].

    Raises:
        ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    def norm(x): return tf.reduce_sum(tf.square(x), 1)

    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
    """Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.

    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        sigmas: a tensor of floats which denote the widths of each of the
            gaussians in the kernel.

    Returns:
        A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def retrieve_seq_length(data):
    """ compute the length of a sequence. 0 are masked.

   Args:
       data: input sequence

   Returns:
      a `int`, length of the sequence
    """
    with tf.name_scope('GetLength'):
        used = tf.sign(tf.reduce_max(tf.abs(data), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.cast(length, tf.int32)
    return length


def advanced_indexing(inp, index):
    """ Advanced Indexing for Sequences
   Args:
       inp: input sequence
       index: input index for indexing

   Returns:
      a indexed sequence
    """
    batch_size = tf.shape(inp)[0]
    max_length = int(inp.get_shape()[1])
    dim_size = int(inp.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (index - 1)
    flat = tf.reshape(inp, [-1, dim_size])
    relevant = tf.gather(flat, index)
    return relevant


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences.
    Pad each sequence to the same length: the length of the longest sequence.
    If maxlen is provided, any sequence longer than maxlen is truncated to
    maxlen. Truncation happens off either the beginning or the end (default)
    of the sequence. Supports pre-padding and post-padding (default).

    Args:
        sequences: list of lists where each element is a sequence.
        maxlen: a `int`, maximum length.
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: `float`, value to pad the sequences to the desired value.

    Returns:
        x: `numpy array` with dimensions (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def chars_to_dictionary(string):
    """ Creates a dictionary char:integer for each unique character
    Args:
        string: a `string` input

    Returns:
        dictionary of chars
    """
    chars = set(string)
    char_idx = {c: i for i, c in enumerate(sorted(chars))}
    return char_idx


def string_to_semi_redundant_sequences(string, seq_maxlen=25, redun_step=3, char_idx=None):
    """ string_to_semi_redundant_sequences.
    Vectorize a string and returns parsed sequences and targets, along with
    the associated dictionary.

    Args:
        string: `str`. Lower-case text from input text file.
        seq_maxlen: `int`. Maximum length of a sequence. Default: 25.
        redun_step: `int`. Redundancy step. Default: 3.
        char_idx: 'dict'. A dictionary to convert chars to positions. Will be automatically generated if None

    Returns:
        A tuple: (inputs, targets, dictionary)
    """

    print("Vectorizing text...")

    if char_idx is None:
        char_idx = chars_to_dictionary(string)

    len_chars = len(char_idx)

    sequences = []
    next_chars = []
    for i in range(0, len(string) - seq_maxlen, redun_step):
        sequences.append(string[i: i + seq_maxlen])
        next_chars.append(string[i + seq_maxlen])

    X = np.zeros((len(sequences), seq_maxlen, len_chars), dtype=np.bool)
    Y = np.zeros((len(sequences), len_chars), dtype=np.bool)
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_idx[char]] = 1
        Y[i, char_idx[next_chars[i]]] = 1

    print("Text total length: {:,}".format(len(string)))
    print("Distinct chars   : {:,}".format(len_chars))
    print("Total sequences  : {:,}".format(len(sequences)))

    return X, Y, char_idx


def textfile_to_semi_redundant_sequences(path, seq_maxlen=25, redun_step=3,
                                         to_lower_case=False, pre_defined_char_idx=None):
    """ Vectorize Text file
    textfile_to_semi_redundant_sequences.
    Vectorize a string from a textfile and returns parsed sequences and targets, along with
    the associated dictionary.

    Args:
        path: `str`. path of the input text file.
        seq_maxlen: `int`. Maximum length of a sequence. Default: 25.
        redun_step: `int`. Redundancy step. Default: 3.
        to_lower_case: a `bool`, if true, convert to lowercase
        pre_defined_char_idx: 'dict'. A dictionary to convert chars to positions. Will be automatically generated if None

    Returns:
        A tuple: (inputs, targets, dictionary)
    """
    text = open(path).read()
    if to_lower_case:
        text = text.lower()
    return string_to_semi_redundant_sequences(text, seq_maxlen, redun_step, pre_defined_char_idx)


def logits_to_log_prob(logits):
    """Computes log probabilities using numerically stable trick.
    This uses two numerical stability tricks:
    1) softmax(x) = softmax(x - c) where c is a constant applied to all
    arguments. If we set c = max(x) then the softmax is more numerically
    stable.
    2) log softmax(x) is not numerically stable, but we can stabilize it
    by using the identity log softmax(x) = x - log sum exp(x)

    Args:
        logits: Tensor of arbitrary shape whose last dimension contains logits.

    Returns:
        A tensor of the same shape as the input, but with corresponding log
            probabilities.
    """

    with tf.variable_scope('log_probabilities'):
        axis = len(logits.get_shape().as_list()) - 1
        max_logits = tf.reduce_max(
            logits, axis=axis, keep_dims=True)
        safe_logits = tf.subtract(logits, max_logits)
        sum_exp = tf.reduce_sum(
            tf.exp(safe_logits),
            axis=axis,
            keep_dims=True)
        log_probs = tf.subtract(safe_logits, tf.log(sum_exp))
    return log_probs


def GetTensorOpName(x):
    """Get the name of the op that created a tensor.
    Useful for naming related tensors, as ':' in name field of op is not permitted

    Args:
      x: the input tensor.

    Returns:
      the name of the op.
    """

    t = x.name.rsplit(":", 1)
    if len(t) == 1:
        return x.name
    else:
        return t[0]


def ListUnion(list_1, list_2):
    """Returns the union of two lists.
    Python sets can have a non-deterministic iteration order. In some
    contexts, this could lead to TensorFlow producing two different
    programs when the same Python script is run twice. In these contexts
    we use lists instead of sets.
    This function is not designed to be especially fast and should only
    be used with small lists.

    Args:
      list_1: A list
      list_2: Another list

    Returns:
      A new list containing one copy of each unique element of list_1 and
      list_2. Uniqueness is determined by "x in union" logic; e.g. two
`      string of that value appearing in the union.

    Raises:
      TypeError: The arguments are not lists.
    """

    if not (isinstance(list_1, list) and isinstance(list_2, list)):
        raise TypeError("Arguments must be lists.")

    union = []
    for x in list_1 + list_2:
        if x not in union:
            union.append(x)

    return union


def Interface(ys, xs):
    """Maps xs to consumers.
      Returns a dict mapping each element of xs to any of its consumers that are
      indirectly consumed by ys.

    Args:
      ys: The outputs
      xs: The inputs

    Returns:
      out: Dict mapping each member x of `xs` to a list of all Tensors that are
           direct consumers of x and are eventually consumed by a member of
           `ys`.
    """

    if isinstance(ys, (list, tuple)):
        queue = list(ys)
    else:
        queue = [ys]

    out = OrderedDict()
    if isinstance(xs, (list, tuple)):
        for x in xs:
            out[x] = []
    else:
        out[xs] = []

    done = set()

    while queue:
        y = queue.pop()
        if y in done:
            continue
        done = done.union(set([y]))
        for x in y.op.inputs:
            if x in out:
                out[x].append(y)
            else:
                assert id(x) not in [id(foo) for foo in out]
        queue.extend(y.op.inputs)

    return out


def BatchClipByL2norm(t, upper_bound, name=None):
    """Clip an array of tensors by L2 norm.
    Shrink each dimension-0 slice of tensor (for matrix it is each row) such
    that the l2 norm is at most upper_bound. Here we clip each row as it
    corresponds to each example in the batch.

    Args:
      t: the input tensor.
      upper_bound: the upperbound of the L2 norm.
      name: optional name.

    Returns:
      the clipped tensor.
    """

    assert upper_bound > 0
    with tf.name_scope(values=[t, upper_bound], name=name,
                       default_name="batch_clip_by_l2norm") as name:
        saved_shape = tf.shape(t)
        batch_size = tf.slice(saved_shape, [0], [1])
        t2 = tf.reshape(t, tf.concat(axis=0, values=[batch_size, [-1]]))
        upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                                  tf.constant(1.0 / upper_bound))
        # Add a small number to avoid divide by 0
        l2norm_inv = tf.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)
        scale = tf.minimum(l2norm_inv, upper_bound_inv) * upper_bound
        clipped_t = tf.matmul(tf.diag(scale), t2)
        clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
        return clipped_t


def AddGaussianNoise(t, sigma, name=None):
    """Add i.i.d. Gaussian noise (0, sigma^2) to every entry of t.

    Args:
      t: the input tensor.
      sigma: the stddev of the Gaussian noise.
      name: optional name.

    Returns:
      the noisy tensor.
    """

    with tf.name_scope(values=[t, sigma], name=name,
                       default_name="add_gaussian_noise") as name:
        noisy_t = t + tf.random_normal(tf.shape(t), stddev=sigma)
        return noisy_t
