import tensorflow.compat.v1 as tf

rnn = tf.nn.rnn_cell

class MultiRNNCellWithAutoencoder(rnn.MultiRNNCell):
  def call(self, inputs, state):
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