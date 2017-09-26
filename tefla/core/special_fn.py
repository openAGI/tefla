import random
import tensorflow as tf
from tensorflow.python.framework import function


def fn_with_custom_grad(grad_fn, use_global_vars=False):
    """Decorator to create a subgraph with a custom gradient function.

    The subgraph created by the decorated function is NOT put in a Defun and so
    does not suffer from the limitations of the Defun (all subgraph ops on the
    same device, no summaries).

    Args:
        grad_fn: function with signature
          (inputs, variables, outputs, output_grads) -> (grad_inputs, grad_vars),
           all of which are lists of Tensors.
        use_global_vars: if True, variables will be the global variables created.
            If False, will be the trainable variables.

    Returns:
        Decorator for function such that the gradient is defined by grad_fn.
    """

    def dec(fn):

        def wrapped(*args):
            return _fn_with_custom_grad(fn, args, grad_fn, use_global_vars=use_global_vars)

        return wrapped

    return dec


def _fn_with_custom_grad(fn, inputs, grad_fn, use_global_vars=False):
    """Create a subgraph with a custom gradient.

    Args:
        fn: function that takes inputs as arguments and produces 1 or more Tensors.
        inputs: list<Tensor>, will be passed as fn(*inputs).
        grad_fn: function with signature
            (inputs, vars, outputs, output_grads) -> (grad_inputs, grad_vars),
            all of which are lists of Tensors.
        use_global_vars: if True, variables will be the global variables created.
           If False, will be the trainable variables.

    Returns:
        fn(*inputs)
    """
    with tf.variable_scope(None, default_name="fn_with_custom_grad") as vs:
        inputs = list(inputs)
        outputs = fn(*inputs)
        if use_global_vars:
            train_vars = list(vs.global_variables())
        else:
            train_vars = list(vs.trainable_variables())

    if grad_fn is None:
        return outputs
    else:
        if not (isinstance(outputs, tuple) or isinstance(outputs, list)):
            outputs = [outputs]
        outputs = list(outputs)

        in_types = [t.dtype for t in inputs]
        out_types = [t.dtype for t in outputs]
        var_types = [t.dtype for t in train_vars]

        def custom_grad_fn(op, *dys):
            """Custom grad fn applying grad_fn for identity Defun."""
            dys = list(dys)
            fn_inputs = op.inputs[:len(inputs)]
            fn_vars = op.inputs[len(inputs):len(inputs) + len(train_vars)]
            fn_outputs = op.inputs[len(inputs) + len(train_vars):]
            assert len(fn_outputs) == len(outputs)
            assert len(fn_outputs) == len(dys)

            grad_inputs, grad_vars = grad_fn(
                fn_inputs, fn_vars, fn_outputs, dys)
            grad_outputs = [None] * len(fn_outputs)
            return tuple(grad_inputs + grad_vars + grad_outputs)

        # The Defun takes as input the original inputs, the trainable variables
        # created in fn, and the outputs. In the forward it passes through the
        # outputs. In the backwards, it produces gradients for the original inputs
        # and the trainable variables.
        @function.Defun(
            *(in_types + var_types + out_types),
            func_name="identity_custom_grad%d" % random.randint(1, 10**9),
            python_grad_func=custom_grad_fn,
            shape_func=lambda _: [t.get_shape() for t in outputs])
        def identity(*args):
            outs = args[len(inputs) + len(train_vars):]
            return tuple([tf.identity(t) for t in outs])

        id_out = identity(*(inputs + train_vars + outputs))
        return id_out
