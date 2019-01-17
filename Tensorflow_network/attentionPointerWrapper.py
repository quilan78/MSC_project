import tensorflow as tf
from AttentionPointerWrapperState import *


import collections
import functools
import math

import numpy as np

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

_zero_state_tensors = rnn_cell_impl._zero_state_tensors

class AttentionPointerWrapper(tf.contrib.seq2seq.AttentionWrapper):
	"""Wraps another `RNNCell` with attention.
	"""

	def __init__(self,
							 cell,
							 attention_mechanism,
							 attention_layer_size=None,
							 alignment_history=False,
							 cell_input_fn=None,
							 output_attention=True,
							 initial_cell_state=None,
							 pointer=False,
							 name=None):
		super(AttentionPointerWrapper, self).__init__(
							 cell,
							 attention_mechanism,
							 attention_layer_size=attention_layer_size,
							 alignment_history=alignment_history,
							 cell_input_fn=cell_input_fn,
							 output_attention=output_attention,
							 initial_cell_state=initial_cell_state,
							 name=name)
		self.pointer = pointer

	def call(self, inputs, state):
		"""Perform a step of attention-wrapped RNN.
		- Step 1: Mix the `inputs` and previous step's `attention` output via
			`cell_input_fn`.
		- Step 2: Call the wrapped `cell` with this input and its previous state.
		- Step 3: Score the cell's output with `attention_mechanism`.
		- Step 4: Calculate the alignments by passing the score through the
			`normalizer`.
		- Step 5: Calculate the context vector as the inner product between the
			alignments and the attention_mechanism's values (memory).
		- Step 6: Calculate the attention output by concatenating the cell output
			and context through the attention layer (a linear layer with
			`attention_layer_size` outputs).
		Args:
			inputs: (Possibly nested tuple of) Tensor, the input at this time step.
			state: An instance of `AttentionWrapperState` containing
				tensors from the previous time step.
		Returns:
			A tuple `(attention_or_cell_output, next_state)`, where:
			- `attention_or_cell_output` depending on `output_attention`.
			- `next_state` is an instance of `AttentionWrapperState`
				 containing the state calculated at this time step.
		Raises:
			TypeError: If `state` is not an instance of `AttentionWrapperState`.
		"""
		if not isinstance(state, AttentionPointerWrapperState):
			raise TypeError("Expected state to be instance of AttentionWrapperState. "
											"Received type %s instead."  % type(state))

		# Step 1: Calculate the true inputs to the cell based on the
		# previous attention value.
		cell_inputs = self._cell_input_fn(inputs, state.attention)
		cell_state = state.cell_state
		cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

		cell_batch_size = (
				cell_output.shape[0].value or array_ops.shape(cell_output)[0])
		error_message = (
				"When applying AttentionWrapper %s: " % self.name +
				"Non-matching batch sizes between the memory "
				"(encoder output) and the query (decoder output).  Are you using "
				"the BeamSearchDecoder?  You may need to tile your memory input via "
				"the tf.contrib.seq2seq.tile_batch function with argument "
				"multiple=beam_width.")
		with ops.control_dependencies(
				self._batch_size_checks(cell_batch_size, error_message)):
			cell_output = array_ops.identity(
					cell_output, name="checked_cell_output")

		if self._is_multi:
			previous_attention_state = state.attention_state
			previous_alignment_history = state.alignment_history
		else:
			previous_attention_state = [state.attention_state]
			previous_alignment_history = [state.alignment_history]

		all_alignments = []
		all_attentions = []
		all_attention_states = []
		maybe_all_histories = []
		for i, attention_mechanism in enumerate(self._attention_mechanisms):
			attention, alignments, next_attention_state = _compute_attention(
					attention_mechanism, cell_output, previous_attention_state[i],
					self._attention_layers[i] if self._attention_layers else None)
			alignment_history = previous_alignment_history[i].write(
					state.time, alignments) if self._alignment_history else ()



			all_attention_states.append(next_attention_state)
			all_alignments.append(alignments)
			all_attentions.append(attention)
			maybe_all_histories.append(alignment_history)

		if self.pointer:
			num_units = self._cell._num_units
			Wh = tf.get_variable("Wh", [num_units])
			Ws = tf.get_variable("Ws", [num_units])
			Wx = tf.get_variable("Wx", shape=inputs.shape)
			bptr = tf.get_variable("Bptr", shape=())
			pgen = tf.sigmoid(tf.reduce_sum(Wh * state.attention, [1]) + tf.reduce_sum(Ws * cell_output, [1]) + tf.reduce_sum(inputs * Wx, [1]) + bptr)
			pgen_all = state.pgen.write(state.time, pgen)
		else:
			pgen_all = ()
			

		attention = array_ops.concat(all_attentions, 1)
		next_state = AttentionPointerWrapperState(
				time=state.time + 1,
				cell_state=next_cell_state,
				attention=attention,
				attention_state=self._item_or_tuple(all_attention_states),
				alignments=self._item_or_tuple(all_alignments),
				alignment_history=self._item_or_tuple(maybe_all_histories),
				pgen =pgen_all
				)

		if self._output_attention:
			return attention, next_state
		else:
			return cell_output, next_state

	def zero_state(self, batch_size, dtype):
		"""Return an initial (zero) state tuple for this `AttentionWrapper`.
		**NOTE** Please see the initializer documentation for details of how
		to call `zero_state` if using an `AttentionWrapper` with a
		`BeamSearchDecoder`.
		Args:
			batch_size: `0D` integer tensor: the batch size.
			dtype: The internal state data type.
		Returns:
			An `AttentionWrapperState` tuple containing zeroed out tensors and,
			possibly, empty `TensorArray` objects.
		Raises:
			ValueError: (or, possibly at runtime, InvalidArgument), if
				`batch_size` does not match the output size of the encoder passed
				to the wrapper object at initialization time.
		"""
		with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
			if self._initial_cell_state is not None:
				cell_state = self._initial_cell_state
			else:
				cell_state = self._cell.zero_state(batch_size, dtype)
			error_message = (
					"When calling zero_state of AttentionWrapper %s: " % self._base_name +
					"Non-matching batch sizes between the memory "
					"(encoder output) and the requested batch size.  Are you using "
					"the BeamSearchDecoder?  If so, make sure your encoder output has "
					"been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
					"the batch_size= argument passed to zero_state is "
					"batch_size * beam_width.")
			with ops.control_dependencies(
					self._batch_size_checks(batch_size, error_message)):
				cell_state = nest.map_structure(
						lambda s: array_ops.identity(s, name="checked_cell_state"),
						cell_state)
			initial_alignments = [
					attention_mechanism.initial_alignments(batch_size, dtype)
					for attention_mechanism in self._attention_mechanisms]
			return AttentionPointerWrapperState(
					cell_state=cell_state,
					time=array_ops.zeros([], dtype=dtypes.int32),
					attention=_zero_state_tensors(self._attention_layer_size, batch_size,
																				dtype),
					alignments=self._item_or_tuple(initial_alignments),
					attention_state=self._item_or_tuple(
							attention_mechanism.initial_state(batch_size, dtype)
							for attention_mechanism in self._attention_mechanisms),
					alignment_history=self._item_or_tuple(
							tensor_array_ops.TensorArray(
									dtype,
									size=0,
									dynamic_size=True,
									element_shape=alignment.shape,
									clear_after_read=False)
							if self._alignment_history else () for alignment in initial_alignments),
					pgen=tensor_array_ops.TensorArray(
									dtype,
									size=0,
									dynamic_size=True,
									clear_after_read=False) if self.pointer else ()
					)

def _compute_attention(attention_mechanism, cell_output, attention_state,
											 attention_layer):
	"""Computes the attention and alignments for a given attention_mechanism."""
	alignments, next_attention_state = attention_mechanism(
			cell_output, state=attention_state)

	# Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
	expanded_alignments = array_ops.expand_dims(alignments, 1)
	# Context is the inner product of alignments and values along the
	# memory time dimension.
	# alignments shape is
	#   [batch_size, 1, memory_time]
	# attention_mechanism.values shape is
	#   [batch_size, memory_time, memory_size]
	# the batched matmul is over memory_time, so the output shape is
	#   [batch_size, 1, memory_size].
	# we then squeeze out the singleton dim.
	context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
	context = array_ops.squeeze(context, [1])

	if attention_layer is not None:
		attention = attention_layer(array_ops.concat([cell_output, context], 1))
	else:
		attention = context

	return attention, alignments, next_attention_state