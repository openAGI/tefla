from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Error(Exception):
  """Base class for all errors from tefla.

  This is thrown to indicate a Neural Network specific problem, e.g.
  wrong module arity, module is not connected to the graph when it
  should be, tried to wire together incompatible modules, etc.
  """


class NotConnectedError(Error):
  """Error raised when operating on a module that has not yet been connected.

  Some module properties / methods are valid to access before the module
  has been connected into the graph, but some are not. This Error is
  raised when the user attempts to do anything not valid before
  connection.
  """


class ParentNotBuiltError(Error):
  """Error raised when the parent of a module has not been built yet.

  the parent has to be connected to the graph before the child transpose
  to ensure that shape inference has already occurred.
  """


class IncompatibleShapeError(Error):
  """Error raised when the shape of the input at build time is incompatible."""


class UnderspecifiedError(Error):
  """Error raised when too little information is available.

  This does not typically mean the user is trying to do something that
  doesn't work (in which case `IncompatibleShapeError` should be used),
  just that some more information needs to be provided in order to build
  the Graph.
  """


class NotSupportedError(Error):
  """Error raised when something that cannot be supported is requested.

  For example a Dilated Convolution module cannot be transposed.
  """


class NotInitializedError(Error):
  """Error raised when connecting an uninitialized module.

  Before they can be connected, all modules must call
  `AbstractModule.__init__` (e.g. via a `super` call).
  """


class DifferentGraphError(Error):
  """Error raised when trying to connect a module to multiple Graphs."""
