import tensorflow as tf
from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

class BahdanauCoverageAttention(tf.contrib.seq2seq.BahdanauAttention):
  """Implements Bahdanau-style (additive) attention.
  This attention has two forms.  The first is Bahdanau attention,
  as described in:
  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473
  The second is the normalized form.  This form is inspired by the
  weight normalization article:
  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868
  To enable the second form, construct the object with parameter
  `normalize=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               name="BahdanauCoverageAttention"):
    """Construct the Attention mechanism.
    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is @{tf.nn.softmax}. Other options include
        @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      dtype: The data type for the query and memory layers of the attention
        mechanism.
      name: Name to use when creating ops.
    """
    super(BahdanauCoverageAttention, self).__init__(num_units,
               memory,
               memory_sequence_length=memory_sequence_length,
               normalize=normalize,
               probability_fn=probability_fn,
               score_mask_value=score_mask_value,
               dtype=dtype,
               name=name)

  def initial_coverage(self, batch_size, dtype):
    """Creates the initial alignment values for the `AttentionWrapper` class.
    This is important for AttentionMechanisms that use the previous alignment
    to calculate the alignment at the next time step (e.g. monotonic attention).
    The default behavior is to return a tensor of all zeros.
    Args:
      batch_size: `int32` scalar, the batch_size.
      dtype: The `dtype`.
    Returns:
      A `dtype` tensor shaped `[batch_size, alignments_size]`
      (`alignments_size` is the values' `max_time`).
    """
    max_time = self._alignments_size
    return tf.getVariable("coverage", [batch_size, max_time])

  def __call__(self, query, state):
    """Score the query based on the keys and values.
    Args:
      query: Tensor of dtype matching `self.values` and shape
        `[batch_size, query_depth]`.
      state: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]`
        (`alignments_size` is memory's `max_time`).
    Returns:
      alignments: Tensor of dtype matching `self.values` and shape
        `[batch_size, alignments_size]` (`alignments_size` is memory's
        `max_time`).

        We use the state vector as the coverage mechanism
    """
    with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      score = _bahdanau_coverage_score(processed_query, self._keys, state, self._normalize)
    rubish = 0
    alignments = self._probability_fn(score, rubish)
    next_state = state + alignments
    return alignments, next_state

def _bahdanau_coverage_score(processed_query, keys, coverage, normalize):
  """Implements Bahdanau-style (additive) scoring function.
  This attention has two forms.  The first is Bhandanau attention,
  as described in:
  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473
  The second is the normalized form.  This form is inspired by the
  weight normalization article:
  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868
  To enable the second form, set `normalize=True`.
  Args:
    processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
    keys: Processed memory, shape `[batch_size, max_time, num_units]`.
    coverage : Added attention until this point [batch_size, max_time]
    normalize: Whether to normalize the score function.
  Returns:
    A `[batch_size, max_time]` tensor of unnormalized score values.
  """
  dtype = processed_query.dtype
  # Get the number of hidden units from the trailing dimension of keys
  num_units = keys.shape[2].value or array_ops.shape(keys)[2]
  # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
  processed_query = array_ops.expand_dims(processed_query, 1)
  v = variable_scope.get_variable(
      "attention_v", [num_units], dtype=dtype)
  print("plop")
  #Coverage vector
  w = variable_scope.get_variable("coverage_w", [num_units], dtype=dtype)

  coverage_expanded = tf.expand_dims(coverage, 2)
  if normalize:
    #Not implemented
    pass
  else:
    return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query + coverage_expanded*w), [2])