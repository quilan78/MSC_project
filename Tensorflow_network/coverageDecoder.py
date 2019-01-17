import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

class CoverageDecoder(tf.contrib.seq2seq.BasicDecoder):
	"""Basic sampling decoder."""

	def __init__(self, cell, helper, initial_state, pgen, pointer, embedding_size, output_layer=None):
		"""Initialize BasicDecoder.
		Args:
			cell: An `RNNCell` instance.
			helper: A `Helper` instance.
			initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
				The initial state of the RNNCell.
			output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
				`tf.layers.Dense`. Optional layer to apply to the RNN output prior
				to storing the result or sampling.
		Raises:
			TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
		"""
		super(CoverageDecoder, self).__init__(cell, helper, initial_state, output_layer=output_layer)
		print(self.batch_size.get_shape())
		self.pgen = pgen
		self.pointer = pointer
		self.counter=  0
		self.embedding_size = embedding_size

	@property
	def batch_size(self):
		return self._helper.batch_size

	def _rnn_output_size(self):
		size = self._cell.output_size
		if self._output_layer is None:
			return size
		else:
			# To use layer's compute_output_shape, we need to convert the
			# RNNCell's output_size entries into shapes with an unknown
			# batch size.  We then pass this through the layer's
			# compute_output_shape and read off all but the first (batch)
			# dimensions to get the output size of the rnn with the layer
			# applied to the top.
			output_shape_with_unknown_batch = nest.map_structure(
					lambda s: tensor_shape.TensorShape([None]).concatenate(s),
					size)
			layer_output_shape = self._output_layer.compute_output_shape(
					output_shape_with_unknown_batch)
			return nest.map_structure(lambda s: s[1:], layer_output_shape)

	def step(self, time, inputs, state, name=None):
		"""Perform a decoding step.
		Args:
			time: scalar `int32` tensor.
			inputs: A (structure of) input tensors.
			state: A (structure of) state tensors and TensorArrays.
			name: Name scope for any created operations.
		Returns:
			`(outputs, next_state, next_inputs, finished)`.
		"""
		with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
			cell_outputs, cell_state = self._cell(inputs, state)

			if self.pointer:
				num_units = self._cell._cell._num_units

				Wh = tf.get_variable("Wh", [num_units], initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
				Ws = tf.get_variable("Ws", [num_units], initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
				Wx = tf.get_variable("Wx", [self.embedding_size], initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
				bptr = tf.get_variable("Bptr", shape=(), initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

				pgen = tf.sigmoid(tf.reduce_sum(Wh * state.attention, [1]) + tf.reduce_sum(Ws * cell_outputs, [1]) + tf.reduce_sum(inputs * Wx, [1]) + bptr)
				tf.assign(self.pgen[self.counter],pgen)
				# self.pgen[self.counter] = pgen
				self.counter += 1

			if self._output_layer is not None:
				cell_outputs = self._output_layer(cell_outputs)
			sample_ids = self._helper.sample(
					time=time, outputs=cell_outputs, state=cell_state)
			(finished, next_inputs, next_state) = self._helper.next_inputs(
					time=time,
					outputs=cell_outputs,
					state=cell_state,
					sample_ids=sample_ids)
		outputs = tf.contrib.seq2seq.BasicDecoderOutput(cell_outputs, sample_ids)
		return (outputs, next_state, next_inputs, finished)