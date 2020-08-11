from tensorflow.python.layers import base as base_layer
import tensorflow.compat.v1 as tf
rnn = tf.nn.rnn_cell

class MultiRNNHacked(rnn.RNNCell):
  """RNN cell composed sequentially of multiple simple cells.
  Example:
  ```python
  num_units = [128, 64]
  cells = [BasicLSTMCell(num_units=n) for n in num_units]
  stacked_rnn_cell = MultiRNNCell(cells)
  ```
  """

  @deprecated(None, "This class is equivalent as "
              "tf.keras.layers.StackedRNNCells, and will be replaced by "
              "that in Tensorflow 2.0.")
  def __init__(self, cells, state_is_tuple=True):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all concatenated along the
        column axis.  This latter behavior will soon be deprecated.
    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    super(MultiRNNHacked, self).__init__()
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    if not nest.is_sequence(cells):
      raise TypeError("cells must be a list or tuple, but saw: %s." % cells)

    if len(set(id(cell) for cell in cells)) < len(cells):
      logging.log_first_n(
          logging.WARN, "At least two cells provided to MultiRNNCell "
          "are the same object and will share weights.", 1)

    self._cells = cells
    for cell_number, cell in enumerate(self._cells):
      # Add Trackable dependencies on these cells so their variables get
      # saved with this object when using object-based saving.
      if isinstance(cell, trackable.Trackable):
        # TODO(allenl): Track down non-Trackable callers.
        self._track_trackable(cell, name="cell-%d" % (cell_number,))
    self._state_is_tuple = state_is_tuple
    if not state_is_tuple:
      if any(nest.is_sequence(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s" %
                         str([c.state_size for c in self._cells]))

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum(cell.state_size for cell in self._cells)

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._state_is_tuple:
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
      else:
        # We know here that state_size of each cell is not a tuple and
        # presumably does not contain TensorArrays or anything else fancy
        return super(MultiRNNCell, self).zero_state(batch_size, dtype)

  @property
  def trainable_weights(self):
    if not self.trainable:
      return []
    weights = []
    for cell in self._cells:
      if isinstance(cell, base_layer.Layer):
        weights += cell.trainable_weights
    return weights

  @property
  def non_trainable_weights(self):
    weights = []
    for cell in self._cells:
      if isinstance(cell, base_layer.Layer):
        weights += cell.non_trainable_weights
    if not self.trainable:
      trainable_weights = []
      for cell in self._cells:
        if isinstance(cell, base_layer.Layer):
          trainable_weights += cell.trainable_weights
      return trainable_weights + weights
    return weights

  def call(self, inputs, state):
    """Run this multi-layer cell on inputs, starting from state."""
    cur_state_pos = 0
    cur_inp = inputs
    new_states = []
    for i, cell in enumerate(self._cells):
      with vs.variable_scope("cell_%d" % i):
        if self._state_is_tuple:
          if not nest.is_sequence(state):
            raise ValueError(
                "Expected state to be a tuple of length %d, but received: %s" %
                (len(self.state_size), state))
          cur_state = state[i]
        else:
          cur_state = array_ops.slice(state, [0, cur_state_pos],
                                      [-1, cell.state_size])
          cur_state_pos += cell.state_size
        cur_inp, new_state = cell(cur_inp, cur_state)
        new_states.append(new_state)

    new_states = (
        tuple(new_states) if self._state_is_tuple else array_ops.concat(
            new_states, 1))

    return cur_inp, new_states