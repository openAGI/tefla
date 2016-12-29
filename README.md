# Tefla: Deep Learning library, a Higher level API for TensorFlow

Tefla is built on top of Tensorflow. It provides higher level access to tensorflow's features.  

Tefla features:

        . Support for data-sets, data-augmentation

        . easy to define complex deep models

        . single and multi GPU training

        . various prediction fnctions including ensembling of models

        . different metrics for performance measurement\

        . custom losses

        . learning rate schedules, polynomial, step, validation_loss based



**TensorFlow Installation**

Tefla requires Tensorflow(version >=r0.12)

**Tefla Installation**

for current version installation:
```python
pip install git+https://github.com/n3011/tefla.git
```

## Examples

Mnist example gives a overview about Tefla usages
 
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

#Note: This project was originally forked from: www.github.com/litan/tefla. Both projects are evolving independently, with a cross-pollination of ideas.
