import numpy as np
from lstm import *

class Encoder():
	"""
		Implementation of the Encoder of the network

		Loads 2 LSTM cells (forward and backward)

		Computes the states of the encoder for each time step

		Variables :
			lstm_forward is the LSTM cell reading forward
			lstm_backward is the LSTM cell reading backward
			cellSize is the dimension of the hidden layers in the LSTM cell
			embeddingSize is the dimension of the input
			max_encoding_length is the size of the input sequence


	"""
	def __init__(self, lstm_forward=None, lstm_backward=None, cellSize=128, embeddingSize = 64, max_encoding_length=200):
		if lstm_forward == None:
			self.lstm_forward = Lstm()
		else:
			self.lstm_forward = lstm_forward
		if lstm_backward == None:
			self.lstm_backward = Lstm()
		else:
			self.lstm_backward = lstm_backward

		self.cellSize = cellSize
		self.embeddingSize = embeddingSize
		self.max_encoding_length = max_encoding_length

	def compute_encoding(self, input_seq):
		"""
			Computes the state of the encoder for every time step

			input_seq is a list of vectors representing the input words 
			

			Returns the list of states of the encoder in the right order
		"""
		assert len(input_seq) == self.max_encoding_length

		#Reverse the list
		rev_input_seq = input_seq[::-1]

		forward_encoder_states = []
		backward_encoder_states = []
		state_forward = self.lstm_forward.zero_state()
		state_backward = self.lstm_backward.zero_state()
		for i in range(self.max_encoding_length):
			state_forward = self.lstm_forward.forward(state_forward, input_seq[i])
			forward_encoder_states.append(state_forward.copy())

			state_backward = self.lstm_backward.forward(state_backward, rev_input_seq[i])
			backward_encoder_states.append(state_backward.copy())

		self.forward_encoder_states = forward_encoder_states
		self.backward_encoder_states = backward_encoder_states[::-1]
		return forward_encoder_states, backward_encoder_states[::-1]

	def countParams(self):
		return self.lstm_forward.countParams() + self.lstm_backward.countParams()

	def computeLRP(self, last_forward_h_relevance, last_backward_h_relevance, last_forward_c_relevance, last_backward_c_relevance, forward_encoder_relevance, backward_encoder_relevance):
		"""
			Computes LRP for the encoder

			Inputs :
			last_forward_h_relevance : Relevance attributed to the forward encoder last hidden state
			last_backward_h_relevance : Relevance attributed to the backward encoder last hidden state
			last_forward_c_relevance : Relevance attributed to the forward encoder last context vector
			last_backward_c_relevance : Relevance attributed to the backward encoder last context vector
			forward_encoder_relevance : Relevance for the hidden states of the forward encoder
			backward_encoder_relevance : Relevance for the hidden states of the backward encoder. WARNING : they are in the order of the input text and not of computation


			Return :
			input_relevance_forward : Relevance attributed to the inputs by the forward encoder
			input_relevance_backward : Relevance attributed to the inputs by the backward encoder
		"""
		#The encoder states, we reverse the backward states so that the last state is really the last computed one
		forward_encoder_states = self.forward_encoder_states
		backward_encoder_states = self.backward_encoder_states[::-1]

		#We reverse the backward hidden states relevance so that they correspond to the order of computation
		backward_encoder_relevance = backward_encoder_relevance[::-1]

		#Are going to take relevance of the input. The forward one is going to be reversed originaly compared to the real text, the backward in the right order
		input_relevance_forward = []
		input_relevance_backward = []

		#Initialisin the relevance of the output of the encoder
		h_relevance_forward = last_forward_h_relevance
		c_relevance_forward = last_forward_c_relevance

		h_relevance_backward = last_backward_h_relevance
		c_relevance_backward = last_backward_c_relevance

		for i in reversed(range(self.max_encoding_length)):
			#Computation for the forward encoder
			h_relevance_forward += forward_encoder_relevance[i]
			x_rel_forward, h_rel_forward, c_rel_forward = self.lstm_forward.step_LRP(forward_encoder_states[i], c_relevance_forward, h_relevance_forward)

			h_relevance_forward = h_rel_forward
			c_relevance_forward = c_rel_forward
			input_relevance_forward.append(x_rel_forward.copy().tolist())

			#Computation for the backward encoder 
			h_relevance_backward += backward_encoder_relevance[i]
			x_rel_backward, h_rel_backward, c_rel_backward = self.lstm_backward.step_LRP(backward_encoder_states[i], c_relevance_backward, h_relevance_backward)

			h_relevance_backward = h_rel_backward
			c_relevance_backward = c_rel_backward
			input_relevance_backward.append(x_rel_backward.copy().tolist())

		#We have to reverse the forward relevance to get in the order of the text
		return input_relevance_forward[::-1], input_relevance_backward, h_relevance_forward, c_relevance_forward, h_relevance_backward, c_relevance_backward