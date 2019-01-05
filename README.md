[![Build Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://openagi.github.io/tefla/)
[![Build Status](https://travis-ci.org/openAGI/tefla.svg?branch=master)](https://travis-ci.org/openAGI/tefla)
[![PyPI version](https://badge.fury.io/py/tefla.svg)](https://badge.fury.io/py/tefla)
[![Build Status](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/openagi/tefla/blob/master/LICENSE)
# Tefla

Tefla is built on top of Tensorflow for fast prototyping of deep learning algorithms. It provides high level access to the features of tensorflow. An interface to easily build complex models. 

Tefla features:

        - Support for custom optimizers

        . Support for data-sets, data-augmentation
       
        . Support for text datasets

        . Easy to define complex deep models

        . Single and multi GPU training

        . Various prediction functions including ensembling of models

        . Different metrics for performance measurement

        . Custom losses

        . Learning rate schedules, polynomial, step, validation_loss based

        . Semantic segmentation learning

        . Semi-supervised learning
         



## Prerequisite to install Tefla

Before you install Tefla you need to install Tensorflow version r1.8.0 or later.
```Shell
pip install tensorflow-gpu
or 
pip install tensorflow
```

### Install Tefla 
The latest release of Tefla is version 1.9.0.
- To install the latest stable version: </p>
```python
pip install tefla
```

- To install the current version:
```python
pip install git+https://github.com/openagi/tefla.git
```

- To develop or work with source and modifying source code:
```Shell
git clone https://github.com/openagi/tefla.git
cd tefla
pip install -r requirements.txt
export PYTHONPATH=.
```
## Documentation

For more information about installing, configuring, and managing see, [Tefla Docs](https://openAGI.github.io/tefla/)


## Tefla Models
Recent deep convolutional models are easy to implement using Tefla. For more information about the latest state-of-the-art models that are implemented using tefla, see [Recent Models](https://github.com/openagi/models)

## Getting Started with just three easy steps

1. Import the layers
```python
>>>from tefla.core.layers import conv2d
>>>convolved = conv2d(input, 48, False, None)

```
2. Create the data directory and download the TFRecords
- Data Directory structure for using normal images
```Shell
|-- Data_Dir
|   |-- training_image_size (eg. training_256, for 256 image size)
|   |-- validation_image_size (eg. validation_256, for 256 image size)
|   |-- training_labels.csv
|   |-- validation_labels.csv
```
- Download the TFRecords support available using tefla/dataset class
       [Train v2](https://github.com/openagi/tefla/blob/master/tefla/trainv2.py)

3. Run the training
  ```Shell
python tefla/train.py --model models/alexnet.py --training_cnf models/multiclass_cnf.py --data_dir /path/to/data/dir (as per instructions 2.a)
  ```
## Mnist example that gives a overview about how to use Tefla
 
```python
image_size =(32, 32)
crop_size = (28, 28)
def model(is_training, reuse):
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=prelu, **common_args)
    fc_args = make_args(activation=prelu, **common_args)
    logit_args = make_args(activation=None, **common_args)

    x = input((None, height, width, 1), **common_args)
    x = conv2d(x, 32, name='conv1_1', **conv_args)
    x = conv2d(x, 32, name='conv1_2', **conv_args)
    x = max_pool(x, name='pool1', **common_args)
    x = dropout(x, drop_p=0.25, name='dropout1', **common_args)
    x = fully_connected(x, n_output=128, name='fc1', **fc_args)
    x = dropout(x, drop_p=0.5, name='dropout2', **common_args)
    logits = fully_connected(x, n_output=10, name="logits", **logit_args)
    predictions = softmax(logits, name='predictions', **common_args)

    return end_points(is_training)

training_cnf = {
    'classification': True,
    'validation_scores': [('validation accuracy', util.accuracy_wrapper), ('validation kappa', util.kappa_wrapper)],
    'num_epochs': 50,
    'lr_policy': StepDecayPolicy(
        schedule={
            0: 0.01,
            30: 0.001,
        }
    )
}
util.init_logging('train.log', file_log_level=logging.INFO, console_log_level=logging.INFO)

trainer = SupervisedTrainer(model, training_cnf, classification=training_cnf['classification'])
trainer.fit(data_set, weights_from=None, start_epoch=1, verbose=1, summary_every=10)
```



## Contributions

Welcome to the fourth release of Tefla, if you find any bug, please report the issue on GitHub. Improvements and requests for new features are more than welcome. Do not hesitate to twist and tweak Tefla, and send pull-requests.


## License

[MIT License](https://openAGI.github.io/tefla/license/)
