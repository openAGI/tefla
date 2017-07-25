# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf


class Decoder(object):
    """A Decoder class to decode examples

    Args:
        feature_keys: a dict, with features name and data types
        e.g.:
            features_keys = {
                'image/encoded/image': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
                'image/class/label': tf.FixedLenFeature([], tf.int64,
                    default_value=tf.zeros([], dtype=tf.int64)),
            }

    """

    def __init__(self, feature_keys):
        self._feature_keys = feature_keys
        self._feature_names = self._feature_keys.keys()

    @property
    def feature_names(self):
        return self._feature_names

    def decode(self, example_serialized, image_size, resize_size=None):
        """Parses an Example proto containing a training example of an image.
        Args:
            example_serialized: scalar Tensor tf.string containing a serialized
                Example protocol buffer.
            Returns:
                image_buffer: Tensor tf.string containing the contents of a JPEG file.
                label: Tensor tf.int32 containing the label.
                text: Tensor tf.string containing the human-readable label.
        """

        features = tf.parse_single_example(
            example_serialized, self._feature_keys)
        outputs = dict()
        for feature in self._feature_names:
            f_type = feature.split('/')[-1]
            if f_type == 'image':
                out = self._decode_feature(f_type, features[feature])
                out = self._process_raw_image(
                    out, image_size, resize_size=resize_size)
            elif f_type in ['format', 'text', 'colorspace', 'filename']:
                out = tf.convert_to_tensor(features[feature], dtype=tf.string)
            else:
                out = tf.convert_to_tensor(features[feature], dtype=tf.int64)
            outputs.update({f_type: out})
            # outputs.update({f_type: self._decode_feature(f_type, features[feature])})

        return outputs

    def _decode_feature(self, f_type, feature):
        return {
            'image': self._decode_jpeg(feature),
            'label': tf.cast(feature, dtype=tf.int64),
            'height': tf.cast(feature, dtype=tf.int64),
            'width': tf.cast(feature, dtype=tf.int64),
            'channels': tf.cast(feature, dtype=tf.int64),
            'text': tf.cast(feature, dtype=tf.string),
            'colorspace': tf.cast(feature, dtype=tf.string),
            'format': tf.cast(feature, dtype=tf.string),
            'filename': tf.cast(feature, dtype=tf.string),
        }[f_type]

    def _decode_jpeg(self, image_buffer, scope=None):
        """Decode a JPEG string into one 3-D float image Tensor.
        Args:
            image_buffer: scalar string Tensor.
            scope: Optional scope for name_scope.
        Returns:
            3-D float Tensor with values ranging from [0, 1).
        """
        with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
            image = tf.image.decode_jpeg(image_buffer, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image

    def distort_image(self, image, distort_op, height, width, thread_id=0, scope=None):
        """Distort one image for training a network.
        Args:
            image: 3-D float Tensor of image
            height: integer
            width: integer
            thread_id: integer indicating the preprocessing thread.
            scope: Optional scope for name_scope.
        Returns:
            3-D float Tensor of distorted image used for training.
        """
        with tf.name_scope(scope, 'distort_image', [image, height, width]):
            # Crop the image to the specified bounding box.
            # Resize image as per memroy constarints
            image = tf.image.resize_images(image, height, width, 3)
            distorted_image = distort_op(image)

            return distorted_image

    def eval_image(self, image, height, width, scope=None):
        """Prepare one image for evaluation.
        Args:
            image: 3-D float Tensor
            height: integer
            width: integer
            scope: Optional scope for name_scope.
        Returns:
            3-D float Tensor of prepared image.
        """
        with tf.name_scope(scope, 'eval_image', [image, height, width]):
            # Crop the central region of the image with an area containing 87.5% of
            # the original image.
            image = tf.image.central_crop(image, central_fraction=0.875)

            # Resize the image to the original height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(
                image, [height, width], align_corners=False)
            image = tf.squeeze(image, [0])
            return image

    def _process_raw_image(self, image, im_size, resize_size=None):
        with tf.name_scope('process_raw_image'):
            image = tf.reshape(image, shape=im_size)
            if resize_size is not None:
                image = tf.image.resize_bilinear(
                    image, resize_size, align_corners=False)
            return image

    # TODO Mainly useful for ImageNet Dataset
    def parse_example_proto(self, example_serialized, is_bbox=False):
        """Parses an Example proto containing a training example of an image.
        The output of the build_image_data.py image preprocessing script is a dataset
        containing serialized Example protocol buffers. Each Example proto contains
        the following fields:
            image/height: 462
            image/width: 581
            image/colorspace: 'RGB'
            image/channels: 3
            image/class/label: 615
            image/class/synset: 'n03623198'
            image/class/text: 'knee pad'
            image/object/bbox/xmin: 0.1
            image/object/bbox/xmax: 0.9
            image/object/bbox/ymin: 0.2
            image/object/bbox/ymax: 0.6
            image/object/bbox/label: 615
            image/format: 'JPEG'
            image/filename: 'ILSVRC2012_val_00041207.JPEG'
            image/encoded: <JPEG encoded string>

        Args:
            example_serialized: scalar Tensor tf.string containing a serialized
            Example protocol buffer.

        Returns:
            image_buffer: Tensor tf.string containing the contents of a JPEG file.
            label: Tensor tf.int32 containing the label.
            bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged as
            [ymin, xmin, ymax, xmax].
            text: Tensor tf.string containing the human-readable label.
        """
        # Dense features in Example proto.
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
            'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                    default_value=-1),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                                   default_value=''),
        }
        sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
        # Sparse features in Example proto.
        if is_bbox:
            feature_map.update(
                {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                             'image/object/bbox/ymin',
                                             'image/object/bbox/xmax',
                                             'image/object/bbox/ymax']})

        features = tf.parse_single_example(example_serialized, feature_map)
        label = tf.cast(features['image/class/label'], dtype=tf.int32)
        if is_bbox:
            xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
            ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
            xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
            ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

            bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

            # Force the variable number of bounding boxes into the shape
            # [1, num_boxes, coords].
            bbox = tf.expand_dims(bbox, 0)
            bbox = tf.transpose(bbox, [0, 2, 1])

            return features['image/encoded'], label, bbox, features['image/class/text']
        else:
            return features['image/encoded'], label, features['image/class/text']


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        """Convert png image to jpeg images

        Args:
            image_data: image is a 3-D uint8 Tensor of shape [height, width, channels].

        Returns:
             jpeg formated image
        """
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        """Convert cmyk image to rgg images

        Args:
            image_data: image is a 3-D uint8 Tensor of shape [height, width, channels].

        Returns:
            rgb formated image
        """
        return self._sess.run(self._cmyk_to_rgb, feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        """ Decode jpeg images

        Args:
            image_data: image is a 3-D uint8 Tensor of shape [height, width, channels].

        Returns:
            decoded image
        """
        image = self._sess.run(self._decode_jpeg, feed_dict={
                               self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
