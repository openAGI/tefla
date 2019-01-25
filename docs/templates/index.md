# Tefla: Deep Learning library, a Higher level API for TensorFlow

Tefla is built on top of Tensorflow for fast prototyping of deep learning algorithms. It provides high level access to the features of tensorflow. An interface to easily build complex models. 

Tefla features:

        . Supports custom optimizers

        . Supports data-sets, data-augmentation, and others
       
        . Supports text datasets

        . Easy to define complex deep models

        . Single and multi GPU training

        . Various prediction functions including ensembling of models

        . Different metrics for performance measurement

        . Custom losses

        . Learning rate schedules, polynomial, step, validation_loss based

        . Semantic segmentation learning

        . Semi-supervised learning
         



## Installation

### Prerequisite to install Tefla

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

## Example

## MNIST example that gives a overview about how to use Tefla
 
```python
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
