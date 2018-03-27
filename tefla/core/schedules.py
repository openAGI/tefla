"""Schedule functions for controlling hparams over time."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import abc
import math
import ast
import itertools
from six.moves import xrange


@six.add_metaclass(abc.ABCMeta)
class Schedule(object):
  """Schedule is a function which sets a hyperparameter's value over time.

  For example, a schedule can be used to decay an hparams, or oscillate
  it over time. This object is constructed with an instance of Config
  (will be specific to each class implementation). For example if this
  is a decay schedule, the config may specify the rate of decay and
  decay start time. Then the object instance is called like a function,
  mapping global step (an integer counting how many calls to the train
  op have been made) to the hparam value. Properties of a schedule
  function f(t): 0) Domain of t is the non-negative integers (t may be
  0). 1) Range of f is the reals. 2) Schedule functions can assume that
  they will be called in time order. This    allows schedules to be
  stateful. 3) Schedule functions should be deterministic. Two schedule
  instances with the    same config must always give the same value for
  each t, and regardless of    what t's it was previously called on.
  Users may call f(t) on arbitrary    (positive) time jumps.
  Essentially, multiple schedule instances used in    replica training
  will behave the same. 4) Duplicate successive calls on the same time
  are allowed.
  """

  @abc.abstractmethod
  def __init__(self, config):
    """Construct this schedule with a config specific to each class impl.

    Args:
      config: An instance of config_lib.Config.
    """
    pass

  @abc.abstractmethod
  def __call__(self, global_step):
    """Map `global_step` to a value. `global_step` is an integer counting how
    many calls to the train op have been made across all replicas (hence why it
    is global). Implementations may assume calls to be made in time order, i.e.
    `global_step` now >= previous `global_step` values.

    Args:
      global_step: Non-negative integer.

    Returns:
      Hparam value at this step. A number.
    """
    pass


class ConstSchedule(Schedule):
  """Constant function.

  Args
     config:
         const: Constant value at every step.
         f(t) = const.
  """

  def __init__(self, config):
    super(ConstSchedule, self).__init__(config)
    self.const = config.const

  def __call__(self, global_step):
    return self.const


class LinearDecaySchedule(Schedule):
  """Linear decay function.

  Args:
      config:
        initial: Decay starts from this value.
        final: Decay ends at this value.
        start_time: Step when decay starts. Constant before it.
        end_time: When decay ends. Constant after it.
        f(t) is a linear function when start_time <= t <= end_time, with slope of
        (final - initial) / (end_time - start_time). f(t) = initial
        when t <= start_time. f(t) = final when t >= end_time.
        If start_time == end_time, this becomes a step function.
  """

  def __init__(self, config):
    super(LinearDecaySchedule, self).__init__(config)
    self.initial = config.initial
    self.final = config.final
    self.start_time = config.start_time
    self.end_time = config.end_time

    if self.end_time < self.start_time:
      raise ValueError('start_time must be before end_time.')

    # Linear interpolation.
    self._time_diff = float(self.end_time - self.start_time)
    self._diff = float(self.final - self.initial)
    self._slope = (self._diff / self._time_diff if self._time_diff > 0 else float('inf'))

  def __call__(self, global_step):
    if global_step <= self.start_time:
      return self.initial
    if global_step > self.end_time:
      return self.final
    return self.initial + (global_step - self.start_time) * self._slope


class ExponentialDecaySchedule(Schedule):
  """Exponential decay function. See
  https://en.wikipedia.org/wiki/Exponential_decay. Use this decay function to
  decay over orders of magnitude. For example, to decay learning rate from 1e-2
  to 1e-6. Exponential decay will decay the exponent linearly.

  Args:
      config:
        initial: Decay starts from this value.
        final: Decay ends at this value.
        start_time: Step when decay starts. Constant before it.
        end_time: When decay ends. Constant after it.
      f(t) is an exponential decay function when start_time <= t <= end_time. The
      decay rate and amplitude are chosen so that f(t) = initial when
      t = start_time, and f(t) = final when t = end_time. f(t) is constant for
      t < start_time or t > end_time. initial and final must be positive values.
      If start_time == end_time, this becomes a step function.
  """

  def __init__(self, config):
    super(ExponentialDecaySchedule, self).__init__(config)
    self.initial = config.initial
    self.final = config.final
    self.start_time = config.start_time
    self.end_time = config.end_time

    if self.initial <= 0 or self.final <= 0:
      raise ValueError('initial and final must be positive numbers.')

    # Linear interpolation in log space.
    self._linear_fn = LinearDecaySchedule(
        Config(
            initial=math.log(self.initial),
            final=math.log(self.final),
            start_time=self.start_time,
            end_time=self.end_time))

  def __call__(self, global_step):
    return math.exp(self._linear_fn(global_step))


class SmootherstepDecaySchedule(Schedule):
  """Smootherstep decay function. A sigmoidal like transition from initial to
  final values. A smoother transition than linear and exponential decays, hence
  the name. See https://en.wikipedia.org/wiki/Smoothstep.

  Args:
      config:
        initial: Decay starts from this value.
        final: Decay ends at this value.
        start_time: Step when decay starts. Constant before it.
        end_time: When decay ends. Constant after it.
      f(t) is fully defined here:
      https://en.wikipedia.org/wiki/Smoothstep#Variations.
      f(t) is smooth, as in its first-derivative exists everywhere.
  """

  def __init__(self, config):
    super(SmootherstepDecaySchedule, self).__init__(config)
    self.initial = config.initial
    self.final = config.final
    self.start_time = config.start_time
    self.end_time = config.end_time

    if self.end_time < self.start_time:
      raise ValueError('start_time must be before end_time.')

    self._time_diff = float(self.end_time - self.start_time)
    self._diff = float(self.final - self.initial)

  def __call__(self, global_step):
    if global_step <= self.start_time:
      return self.initial
    if global_step > self.end_time:
      return self.final
    x = (global_step - self.start_time) / self._time_diff

    # Smootherstep
    return self.initial + x * x * x * (x * (x * 6 - 15) + 10) * self._diff


class HardOscillatorSchedule(Schedule):
  """Hard oscillator function.

  Args:
      config:
        high: Max value of the oscillator. Value at constant plateaus.
        low: Min value of the oscillator. Value at constant valleys.
        start_time: Global step when oscillation starts. Constant before this.
        period: Width of one oscillation, i.e. number of steps over which the
            oscillation takes place.
        transition_fraction: Fraction of the period spent transitioning between high
            and low values. 50% of this time is spent rising, and 50% of this time
            is spent falling. 50% of the remaining time is spent constant at the
            high value, and 50% of the remaining time is spent constant at the low
            value. transition_fraction = 1.0 means the entire period is spent
            rising and falling. transition_fraction = 0.0 means no time is spent
            rising and falling, i.e. the function jumps instantaneously between
            high and low.
      f(t) = high when t < start_time.
      f(t) is periodic when t >= start_time, with f(t + period) = f(t).
      f(t) is linear with positive slope when rising, and negative slope when
      falling. At the start of the period t0, f(t0) = high and begins to descend.
      At the middle of the period f is low and is constant until the ascension
      begins. f then rises from low to high and is constant again until the period
      repeats.
      Note: when transition_fraction is 0, f starts the period low and ends high.
  """

  def __init__(self, config):
    super(HardOscillatorSchedule, self).__init__(config)
    self.high = config.high
    self.low = config.low
    self.start_time = config.start_time
    self.period = float(config.period)
    self.transition_fraction = config.transition_fraction
    self.half_transition_fraction = config.transition_fraction / 2.0

    if self.transition_fraction < 0 or self.transition_fraction > 1.0:
      raise ValueError('transition_fraction must be between 0 and 1.0')
    if self.period <= 0:
      raise ValueError('period must be positive')

    self._slope = (float(self.high - self.low) / self.half_transition_fraction
                   if self.half_transition_fraction > 0 else float('inf'))

  def __call__(self, global_step):
    if global_step < self.start_time:
      return self.high
    period_pos = ((global_step - self.start_time) / self.period) % 1.0
    if period_pos >= 0.5:
      # ascending
      period_pos -= 0.5
      if period_pos < self.half_transition_fraction:
        return self.low + period_pos * self._slope
      else:
        return self.high
    else:
      # descending
      if period_pos < self.half_transition_fraction:
        return self.high - period_pos * self._slope
      else:
        return self.low


_NAME_TO_CONFIG = {
    'const': ConstSchedule,
    'linear_decay': LinearDecaySchedule,
    'exp_decay': ExponentialDecaySchedule,
    'smooth_decay': SmootherstepDecaySchedule,
    'hard_osc': HardOscillatorSchedule,
}


def make_schedule(config):
  """Schedule factory. Given `config` containing a `fn` property, a Schedule
  implementation is instantiated with `config`. See `_NAME_TO_CONFIG` for `fn`
  options.

  Args:
    config: Config with a `fn` option that specifies which Schedule
        implementation to use. `config` is passed into the constructor.

  Returns:
    A Schedule impl instance.
  """
  schedule_class = _NAME_TO_CONFIG[config.fn]
  return schedule_class(config)


class Config(dict):
  """Stores model configuration, hyperparameters, or dataset parameters."""

  def __getattr__(self, attr):
    return self[attr]

  def __setattr__(self, attr, value):
    self[attr] = value

  def pretty_str(self, new_lines=True, indent=2, final_indent=0):
    prefix = (' ' * indent) if new_lines else ''
    final_prefix = (' ' * final_indent) if new_lines else ''
    kv = [
        '%s%s=%s' % (prefix, k, (repr(v) if not isinstance(v, Config) else v.pretty_str(
            new_lines=new_lines, indent=indent + 2, final_indent=indent))) for k, v in self.items()
    ]
    if new_lines:
      return 'Config(\n%s\n%s)' % (',\n'.join(kv), final_prefix)
    else:
      return 'Config(%s)' % ', '.join(kv)

  def _update_iterator(self, *args, **kwargs):
    """Convert mixed input into an iterator over (key, value) tuples. Follows
    the dict.update call signature.

    Args:
      *args: (Optional) Pass a dict or iterable of (key, value) 2-tuples as
          an unnamed argument. Only one unnamed argument allowed.
      **kwargs: (Optional) Pass (key, value) pairs as named arguments, where the
          argument name is the key and the argument value is the value.

    Returns:
      An iterator over (key, value) tuples given in the input.

    Raises:
      TypeError: If more than one unnamed argument is given.
    """
    if len(args) > 1:
      raise TypeError('Expected at most 1 unnamed arguments, got %d' % len(args))
    obj = args[0] if args else dict()
    if isinstance(obj, dict):
      return itertools.chain(obj.items(), kwargs.items())
    # Assume obj is an iterable of 2-tuples.
    return itertools.chain(obj, kwargs.items())

  def make_default(self, keys=None):
    """Convert OneOf objects into their default configs. Recursively calls into
    Config objects.

    Args:
      keys: Iterable of key names to check. If None, all keys in self will be
          used.
    """
    if keys is None:
      keys = self.keys()
    for k in keys:
      # Replace OneOf with its default value.
      if isinstance(self[k], OneOf):
        self[k] = self[k].default()
      # Recursively call into all Config objects, even those that came from
      # OneOf objects in the previous code line (for nested OneOf objects).
      if isinstance(self[k], Config):
        self[k].make_default()

  def update(self, *args, **kwargs):
    """Same as dict.update except nested Config objects are updated.

    Args:
      *args: (Optional) Pass a dict or list of (key, value) 2-tuples as unnamed
          argument.
      **kwargs: (Optional) Pass (key, value) pairs as named arguments, where the
          argument name is the key and the argument value is the value.
    """
    key_set = set(self.keys())
    for k, v in self._update_iterator(*args, **kwargs):
      if k in key_set:
        # This key is updated so exclude from make_default.
        key_set.remove(k)
      if k in self and isinstance(self[k], Config) and isinstance(v, dict):
        self[k].update(v)
      elif k in self and isinstance(self[k], OneOf) and isinstance(v, dict):
        # Replace OneOf with the chosen config.
        self[k] = self[k].update(v)
      else:
        self[k] = v
    self.make_default(key_set)

  def strict_update(self, *args, **kwargs):
    """Same as Config.update except keys and types are not allowed to change.
    If a given key is not already in this instance, an exception is raised. If
    a given value does not have the same type as the existing value for the
    same key, an exception is raised. Use this method to catch config mistakes.

    Args:
      *args: (Optional) Pass a dict or list of (key, value) 2-tuples as unnamed
          argument.
      **kwargs: (Optional) Pass (key, value) pairs as named arguments, where the
          argument name is the key and the argument value is the value.

    Raises:
      TypeError: If more than one unnamed argument is given.
      TypeError: If new value type does not match existing type.
      KeyError: If a given key is not already defined in this instance.
    """
    key_set = set(self.keys())
    for k, v in self._update_iterator(*args, **kwargs):
      if k in self:
        # This key is updated so exclude from make_default.
        key_set.remove(k)
        if isinstance(self[k], Config):
          if not isinstance(v, dict):
            raise TypeError('dict required for Config value, got %s' % type(v))
          self[k].strict_update(v)
        elif isinstance(self[k], OneOf):
          if not isinstance(v, dict):
            raise TypeError('dict required for OneOf value, got %s' % type(v))
          # Replace OneOf with the chosen config.
          self[k] = self[k].strict_update(v)
        else:
          if not isinstance(v, type(self[k])):
            raise TypeError('Expecting type %s for key %s, got type %s' % (type(self[k]), k,
                                                                           type(v)))
          self[k] = v
      else:
        raise KeyError('Key %s does not exist. New key creation not allowed in '
                       'strict_update.' % k)
    self.make_default(key_set)

  @staticmethod
  def from_str(config_str):
    """Inverse of Config.__str__."""
    parsed = ast.literal_eval(config_str)
    assert isinstance(parsed, dict)

    def _make_config(dictionary):
      for k, v in dictionary.items():
        if isinstance(v, dict):
          dictionary[k] = _make_config(v)
      return Config(**dictionary)

    return _make_config(parsed)

  @staticmethod
  def parse(key_val_string):
    """Parse hyperparameter string into Config object. Format is
    'key=val,key=val,...' Values can be any python literal, or another Config
    object encoded as 'c(key=val,key=val,...)'. c(...) expressions can be
    arbitrarily nested. Example: 'a=1,b=3e-5,c=[1,2,3],d="hello
    world",e={"a":1,"b":2},f=c(x=1,y=[10,20])'.

    Args:
      key_val_string: The hyperparameter string.

    Returns:
      Config object parsed from the input string.
    """
    if not key_val_string.strip():
      return Config()

    def _pair_to_kv(pair):
      split_index = pair.find('=')
      key, val = pair[:split_index].strip(), pair[split_index + 1:].strip()
      if val.startswith('c(') and val.endswith(')'):
        val = Config.parse(val[2:-1])
      else:
        val = ast.literal_eval(val)
      return key, val

    return Config(**dict([_pair_to_kv(pair) for pair in _comma_iterator(key_val_string)]))


def _next_comma(string, start_index):
  """Finds the position of the next comma not used in a literal collection."""
  paren_count = 0
  for i in xrange(start_index, len(string)):
    c = string[i]
    if c == '(' or c == '[' or c == '{':
      paren_count += 1
    elif c == ')' or c == ']' or c == '}':
      paren_count -= 1
    if paren_count == 0 and c == ',':
      return i
  return -1


def _comma_iterator(string):
  index = 0
  while 1:
    next_index = _next_comma(string, index)
    if next_index == -1:
      yield string[index:]
      return
    yield string[index:next_index]
    index = next_index + 1
