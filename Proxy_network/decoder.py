import numpy as np
from utils import *
import tensorflow as tf
from denseLayer import *
class Decoder:
	"""
		Class that represents the decoder part of the architecture, including the attention mechanism (Bahdanau's)

		Variable:
			lstm_cell is the LSTM cell of the decoder
			cellSize is the dimension of the hidden layers in the LSTM cell
			attentionSize is the dimension of the attention vector (in my implementation, it is the same as cellSize)
			max_decoding_length is the size of the output sequence
			memoryLayer is the memory layer matrix 
			queryLayer is the query layer matrix
			attentionV is the attention v vector
			attentonLayer is the attention Layer 

	"""
	def __init__(self, lstm_cell=None, cellSize=128, attentionSize=128, max_decoding_length=50, memoryLayer=[], queryLayer=[], attentionV = [], attentonLayer=[], library="nps"):
		if lstm_cell == None:
			self.lstm_cell = Lstm()
		else:
			self.lstm_cell = lstm_cell


		self.cellSize = cellSize
		self.attentionSize = attentionSize
		self.max_decoding_length = max_decoding_length


		self.memoryLayer = np.array(memoryLayer).transpose()
		self.queryLayer = np.array(queryLayer).transpose()
		self.attentionV = np.array(attentionV).reshape((len(attentionV), 1))
		self.attentonLayer = np.array(attentonLayer).transpose()

	def compute_attention(self, decoder_state, forward_encoder_states, backward_encoder_states):
		"""
			Computes the attention vector from the encoder outputs and the decoder state
			returns the attention vetor
		"""

		assert len(forward_encoder_states) == len(backward_encoder_states)
		max_encoding_len = len(forward_encoder_states)
		output_states=[]
		for i in range(max_encoding_len):
			state = np.concatenate([forward_encoder_states[i]['h'], backward_encoder_states[i]['h']])
			output_states.append(state)

		tiled_outputs = np.concatenate(output_states, 1)
		tiled_decoder = np.tile(decoder_state['h'], (1, max_encoding_len))

		alpha = np.zeros((len(output_states), 1))
		for i in range(max_encoding_len):
			alpha[i] = np.matmul(self.attentionV.transpose(), np.tanh(np.matmul(self.memoryLayer, output_states[i]) + np.matmul(self.queryLayer, decoder_state['h'])))
		alpha = softmax(alpha)

		u = np.zeros(output_states[0].shape)
		for i in range(max_encoding_len):
			u += alpha[i, 0] * output_states[i]
		a = np.matmul(self.attentonLayer, np.concatenate([decoder_state['h'], u]))

		return a, alpha

	def null_attention_vectors(self, forward_encoder_states):
		max_encoding_len = len(forward_encoder_states)
		alpha = np.zeros((max_encoding_len, 1))
		a = np.zeros((self.attentonLayer.shape[0], 1))
		return a, alpha


	def compute_decoding(self, dec_inp, initial_h, initial_c, forward_encoder_states, backward_encoder_states, denseLayer, embedding):
		"""
			Computes the forward propagation of the attention + LSTM
			We store at one time step the attention vector and the alpha used to compute the input of the LSTM, so they have been computed at the previous timestep
			The first time step is the initial state of the LSTM
			return a list of vectors
		"""
		outputs = []
		decoder_state={}
		decoder_state["lstm"] = self.lstm_cell.zero_state()

		attention_vector, alpha = self.null_attention_vectors(forward_encoder_states)
		decoder_state['a'] = attention_vector
		decoder_state['alpha'] = alpha
		decoder_state["lstm"]['h'] = initial_h
		decoder_state["lstm"]['c'] = initial_c

		for i in range(self.max_decoding_length):

			#To keep track of the input for LRP when we transmit the relevance from the input
			decoder_state["input_int"] = dec_inp[i].reshape((len(dec_inp[i]),1))

			#Concatenating the input vector with the attention vector
			input_ = np.concatenate([dec_inp[i].reshape((len(dec_inp[i]),1)), decoder_state['a']])

			#We compute the LSTM step
			decoder_state["lstm"] = self.lstm_cell.forward(decoder_state["lstm"], input_)

			#We compute the output after the output layer
			decoder_state["output"] =  denseLayer.compute_forward(decoder_state["lstm"])

			# The decoder state stores the attention vector is had as an input
			outputs.append(decoder_state.copy())

			#We compute the attention vector used for the next step
			attention_vector, alpha = self.compute_attention(decoder_state["lstm"], forward_encoder_states, backward_encoder_states)
			decoder_state['a'] = attention_vector
			decoder_state['alpha'] = alpha


		self.outputs = outputs
		return outputs

	def countParams(self):
		return self.lstm_cell.countParams() + self.memoryLayer.shape[0] * self.memoryLayer.shape[1] + self.queryLayer.shape[0] * self.queryLayer.shape[1] + self.attentonLayer.shape[0] * self.attentonLayer.shape[1] + self.attentionV.shape[0] 

	def	computeLRP(self, output_relevance, denseLayer, forward_encoder_states, backward_encoder_states, transmit_input=False, start_decode=None):
		"""
			Computes LRP

			Input :
			output_relevance : list of length max_decoding_length representing the output relevance. shape (max_decoding_length, vocab_size)
			denseLayer : The dense layer of the network
			forward_encoder_states : states of the forward encoder
			backward_encoder_states : states of the backward encoder
			transmit_input : if True, the relevance of the input at time step T will be assigned to the corresponding word to the output layer of time step t-1

			Return :
			h_relevance : Relevance of the initial hidden state
			c_relevance : Relevance of the initial context vector
			forward_encoder_relevance : relevance attributed to each of the encoder forward outputs
			backward_decoder_relevance : relevance attributed to each of the encoder backward outputs
		"""
		if start_decode is None:
			start_decode = self.max_decoding_length

		h_relevance = np.zeros((self.cellSize, 1))
		c_relevance = np.zeros((self.cellSize, 1))

		#Initialising the encoder relevance
		forward_encoder_relevance = [np.zeros((self.cellSize, 1)) for i in forward_encoder_states]
		backward_encoder_relevance = [np.zeros((self.cellSize, 1)) for i in forward_encoder_states]

		#Relevance for the input of the decoder
		inputs_relevance = []

		
		max_encoding_len = len(forward_encoder_states)
		output_states=[]
		for i in range(max_encoding_len):
			state = np.concatenate([forward_encoder_states[i]['h'], backward_encoder_states[i]['h']])
			output_states.append(state)


		#Last step LRP step : 
		for i in reversed(range(self.max_decoding_length)):
			if i > start_decode:
				inputs_relevance.append(np.zeros((self.cellSize,1)).tolist())
			else:

				#Compute LRP for the output layer
				h_relevance += denseLayer.computeLRP(output_relevance[i], self.outputs[i])
				#print("Dense Layer")
				#print(np.sum(output_relevance[i]))
				#print(np.sum(h_relevance))
				#Compute LRP for this step of the LSTM
				x_rel, h_rel, c_rel = self.lstm_cell.step_LRP(self.outputs[i]["lstm"], c_relevance.copy(), h_relevance.copy())

				#print("LSTM")
				#print(np.sum(c_relevance) + np.sum(h_relevance))
				#print(np.sum(x_rel) + np.sum(h_rel) + np.sum(c_rel))
				h_relevance = h_rel
				c_relevance = c_rel
				x_relevance = x_rel[:len(x_rel)-self.attentionSize]
				a_relevance = x_rel[len(x_rel)-self.attentionSize:]

				inputs_relevance.append(x_relevance.tolist())

				#The attention mechanism is initialised independently from the encoder
				if i > 0:
					#Computing LRP for the attention mechanism. Adds to the previous hidden state and to the encoder outputs
					#ha_rel, forward_encoder_rel, backward_decoder_rel = self.compute_AttentionLRP(self.outputs[i], a_relevance, output_states)
					
					ha_rel=np.zeros((self.cellSize, 1))
					forward_encoder_rel = [np.zeros((self.cellSize, 1)) for i in forward_encoder_states]
					backward_decoder_rel = [np.zeros((self.cellSize, 1)) for i in forward_encoder_states]

					h_relevance += ha_rel
					forward_encoder_relevance = addElementwiseListArray(forward_encoder_relevance, forward_encoder_rel)
					backward_encoder_relevance = addElementwiseListArray(backward_encoder_relevance, backward_decoder_rel)

					#if transmit_input:
						#We add the relevance attributed to the input word to the corresponding neuron on the output layer of the previous time step
						#output_relevance[i-1][self.outputs[i-1]["input_int"]] += np.sum(x_relevance)

		return h_relevance, c_relevance, forward_encoder_relevance, backward_encoder_relevance, inputs_relevance[::-1], a_relevance


	def compute_AttentionLRP(self, cell_state, relevance_a, output_states):
		"""
			Computes LRP for the attention part
			
			Input : 
			cell_state : the cell state at this time step
			relevance_a : the relevance of the attention vector
			forward_encoder_states : states of the forward encoder
			backward_encoder_states : states of the backward encoder

			Return :
			h_relevance : relevance attributed to the hidden state of the previous time step
			forward_encoder_relevance : relevance attributed to each of the encoder forward outputs
			backward_decoder_relevance : relevance attributed to each of the encoder backward outputs
		"""


		#Reconstructing the concatenated encoder states
		max_encoding_len = len(output_states)
		u = np.zeros(output_states[0].shape)
		for i in range(max_encoding_len):
			u += cell_state["alpha"][i, 0] * output_states[i]
		a = np.matmul(self.attentonLayer, np.concatenate([cell_state["lstm"]['last_h'], u]))

		# LRP for the attention layer
		inp_a_rel = layerLRP(np.concatenate([cell_state["lstm"]['last_h'], u]), self.attentonLayer, np.zeros((self.attentionSize, 1)), a, relevance_a)

		h_relevance= inp_a_rel[:self.cellSize]
		u_relevance = inp_a_rel[self.cellSize:]

		forward_encoder_relevance = []
		backward_decoder_relevance = []


		input_lrp_vector = np.concatenate([cell_state["alpha"][i, 0] * output_states[i] for i in range(max_encoding_len)])
		input_lrp_matrix = np.concatenate([np.identity(2*self.cellSize) for i in range(max_encoding_len)], 1)
		#for i in range(max_encoding_len):
			#inp_c_rel = layerLRP(cell_state["alpha"][i, 0] * output_states[i], np.identity(2*self.cellSize), np.zeros((2*self.cellSize, 1)), u, u_relevance, epsilon = 0.001, delta=1.0)
			#forward_encoder_relevance.append(inp_c_rel[:self.cellSize])
			#backward_decoder_relevance.append(inp_c_rel[self.cellSize:])
		inp_c_rel = layerLRP(input_lrp_vector, input_lrp_matrix, np.zeros((2*self.cellSize, 1)), u, u_relevance)
		for i in range(max_encoding_len):
			forward_encoder_relevance.append(inp_c_rel[2*i*self.cellSize:(2*i+1)*self.cellSize])
			backward_decoder_relevance.append(inp_c_rel[(2*i+1)*self.cellSize:(2*(i+1))*self.cellSize])

		return h_relevance, forward_encoder_relevance, backward_decoder_relevance


