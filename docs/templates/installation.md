# Installation

## Tensorflow Installation

Tefla requires Tensorflow (version >= 1.8.0) to be installed.

Select the correct binary to install, according to your system:
```python
# Ubuntu/Linux/macOS 64-bit, CPU only, Python 3.6
pip install tensorflow

# Ubuntu/Linux/macOS 64-bit, GPU enabled, Python 3.6
# Requires CUDA toolkit 8.0/9.0 and CuDNN v5/v7. For other versions, see "Installing from sources" below.
pip install tensorflow-gpu
```

- For more details: [Tensorflow installation instructions](https://www.tensorflow.org/install).

## Tefla Installation

To install Tefla, the easiest way is to run

For the bleeding edge version:
```python
pip install git+https://github.com/n3011/tefla.git
```
Otherwise, you can also install from pypi
```python
pip install tefla
```

## Upgrade Tensorflow

If you version for Tensorflow is too old (under 0.12.0), you may upgrade Tensorflow to avoid some incompatibilities with Tefla.
To upgrade Tensorflow, you first need to uninstall Tensorflow and Protobuf:

```python
pip uninstall protobuf
pip uninstall tensorflow
```

Then you can re-install Tensorflow:
## Using Latest Tensorflow

Tefla is compatible with [master version](https://github.com/tensorflow/tensorflow) of Tensorflow, but some warnings may appear.
