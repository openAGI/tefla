# -------------------------------------------------------------------#
# Written by Mrinal Haloi
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf


class Decoder(object):
    def __init__(self, feature_keys):
        self._feature_keys = feature_keys
        self.feature_names = self._feature_keys.keys()

    @property
    def feature_names(self):
        return self.feature_names

    def decode(self, example_serialized):
        """Parses an Example proto containing a training example of an image.
        Args:
            example_serialized: scalar Tensor tf.string containing a serialized
            Example protocol buffer.
        Returns:
            image_buffer: Tensor tf.string containing the contents of a JPEG file.
            label: Tensor tf.int32 containing the label.
            text: Tensor tf.string containing the human-readable label.
        """

        features = tf.parse_single_example(example_serialized, self._feature_keys)
        outputs = dict()
        for feature in self.feature_names:
            f_type = feature.split('/')[-1]
            outputs.update({f_type: (self._decode_feature(f_type, features[feature]))})

        return outputs

    def _decode_feature(self, f_type, feature):
        return {
            'label': tf.cast(feature, dtype=tf.int64),
            'height': tf.cast(feature, dtype=tf.int64),
            'width': tf.cast(feature, dtype=tf.int64),
            'channels': tf.cast(feature, dtype=tf.int64),
            'text': tf.cast(feature, dtype=tf.string),
            'colorspace': tf.cast(feature, dtype=tf.string),
            'format': tf.cast(feature, dtype=tf.string),
            'filename': tf.cast(feature, dtype=tf.string),
            'image': self._decode_jpeg(feature),
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


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""
    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

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
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
