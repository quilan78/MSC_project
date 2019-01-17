import numpy as np
from encoder import *
from embedding import *
from decoder import *
from lstm import *
from utils import *
import sys
sys.path.append('/home/dwt17/MSc_project/neural_sum_1/code/Commons/')
sys.path.append('../Commons/')
from vocab import *
from stateReducer import *
from denseLayer import *
class Seq2seqSummariser:
	"""
		Class that represent the entire architecture end to end
		
		Variable :
			cellSize is the dimension of the hidden layers in the LSTM cell
			embeddingSize is the dimension of the input
			attentionSize is the dimension of the attention vector
			vocabSize is the number of words
			max_encoding_length is the size of the input sequence
			max_decoding_length is the size of the output sequence
			
			
	"""
	def __init__(self, cellSize = 128, embeddingSize=64, attentionSize=128, vocabSize=150000, max_encoding_length=200, max_decoding_length=50, vocab_path="../data/useful files/"):
		self.cellSize = cellSize
		self.embeddingSize = embeddingSize
		self.attentionSize = attentionSize
		self.vocabSize = vocabSize
		self.max_encoding_length = max_encoding_length
		self.max_decoding_length = max_decoding_length

		self.embedding = None
		self.encoder = None
		self.decoder = None
		self.stateReducer = None
		self.denseLayer = None

		self.vocab = Vocab(path=vocab_path)
		self.vocabSize =self. vocab.LoadVocab(max_size=vocabSize)

	def _initialise_embedding(self, embedding_matrix):
		self.embedding = Embedding(embedding_matrix)
		assert self.vocabSize == self.embedding.vocabSize
		assert self.embeddingSize == self.embedding.embeddingSize

	def _initialise_encoder(self, kernel_forward, bias_forward, kernel_backward, bias_backward):
		"""
			Initialise the encoder
			The kernel are as given by Tensorflow, so of size (embedding + cellSize) x 4*cellSize
			Bias are of shape (4*cellSize, )
		"""


		lstm_forward = Lstm(kernel_forward, bias_forward, self.cellSize)
		lstm_backward = Lstm(kernel_backward, bias_backward, self.cellSize)
		assert lstm_forward.embeddingSize == self.embeddingSize
		assert lstm_backward.embeddingSize == self.embeddingSize

		self.encoder = Encoder( lstm_forward=lstm_forward, lstm_backward=lstm_backward, cellSize=self.cellSize, embeddingSize = self.embeddingSize, max_encoding_length=self.max_encoding_length)

	def _initialise_stateReducer(self, W_h, W_c, b_h, b_c):
		"""
			Initialise the state Reducer from the encoder and the decoder
			The matrices as given by tensorflow are transposed

		"""
		self.stateReducer = StateReducer(W_h = W_h,
										W_c = W_c,
										b_h = b_h,
										b_c = b_c)

	def _initialise_decoder(self, kernel, bias, memoryLayer, queryLayer, attentionV, attentonLayer):
		cell = Lstm(kernel, bias, self.cellSize	)
		self.decoder = Decoder(lstm_cell=cell, 
								cellSize=self.cellSize, 
								attentionSize=self.attentionSize, 
								max_decoding_length=self.max_decoding_length,
								memoryLayer=memoryLayer, 
								queryLayer=queryLayer, 
								attentionV = attentionV, 
								attentonLayer=attentonLayer)

	def _initialise_DenseLayer(self, kernel, bias):
		self.denseLayer = DenseLayer(kernel = kernel,
									bias = bias)

	def compute_forward(self, input_seq_int, dec_seq_int):
		"""
		Compute all the states of the network for the specific input
		
		input : sequence of words

		"""
		#integer_seq_inp = self.vocab.TransalteSentence(input_seq)
		#print(dec_seq_int)
		embedded_inp_seq = self.embedding.transform_input(input_seq_int)
		forward_encoder_states, backward_encoder_states = self.encoder.compute_encoding(embedded_inp_seq)
		#print("Comparing the transmitted states C")
		#print(np.linalg.norm(forward_encoder_states[-1]["c"]))
		#print(np.linalg.norm(backward_encoder_states[0]["c"]))
		h, c = self.stateReducer.compute_reduction(forward_encoder_states[-1], backward_encoder_states[0])

		#integer_seq_out = self.vocab.TransalteSentence(dec_seq)
		embedded_dec_seq = self.embedding.transform_input([self.vocab.start_decode_id] + dec_seq_int)

		decoder_states = self.decoder.compute_decoding(embedded_dec_seq, h, c, forward_encoder_states, backward_encoder_states, self.denseLayer, self.embedding)
		#print("compare context vectors")
		#print(decoder_states[0]["lstm"]["last_c"][:10])
		#print(np.max(decoder_states[0]["lstm"]["f"]))
		#print(decoder_states[0]["lstm"]["f"][:10])
		#print(c[:10])

		return forward_encoder_states, backward_encoder_states, h, c, decoder_states

	def countParams(self):
		count = 0
		count += self.embedding.countParams()
		count += self.encoder.countParams()
		count += self.decoder.countParams()
		count += self.stateReducer.countParams()
		count += self.denseLayer.countParams()
		return count

	def computeLRP(self, targets, start_decode=False):
		"""
			Compute the LRP for the entire network
		
			Input :
			targets : of shape (max_decoding_length, vocab_size). Table of True of False. If the position (i,j) is true, then we want to compute the relevance for the prediction we had of word number j at the decoding step i
		
		"""

		decoder_states = self.decoder.outputs
		forward_encoder_states  = self.encoder.forward_encoder_states
		backward_encoder_states = self.encoder.backward_encoder_states

		output_relevance = []
		for i in range(self.max_decoding_length):
			output_relevance.append(np.zeros((self.vocabSize, 1)))
			for j in range(self.vocabSize):
				if targets[i][j]:
					output_relevance[i][j] = softmax(decoder_states[i]["output"])[j]

		print(np.sum(output_relevance))
		h_relevance, c_relevance, forward_encoder_relevance, backward_encoder_relevance, decoder_input_relevance, rest_relevance = self.decoder.computeLRP(output_relevance, self.denseLayer, forward_encoder_states, backward_encoder_states, transmit_input=False, start_decode=start_decode)

		#h_relevance = np.random.random_sample((self.cellSize, 1)) *10 -5
		#c_relevance = np.random.random_sample((self.cellSize, 1)) *10 -5

		print("Check conservation After decoding")
		print(np.sum(output_relevance))
		print(np.sum(h_relevance)+ np.sum(c_relevance)+ np.sum(forward_encoder_relevance)+ np.sum(backward_encoder_relevance)  + np.sum(np.sum(decoder_input_relevance)) + np.sum(rest_relevance))
		#print(rest_relevance)

		#print("Length of the attention shit")
		#print(len(forward_encoder_relevance))
		last_forward_h_relevance, last_backward_h_relevance, last_forward_c_relevance, last_backward_c_relevance = self.stateReducer.computeLRP(h_relevance, c_relevance)
		#print(np.linalg.norm(last_forward_h_relevance))
		#print(np.linalg.norm(last_backward_h_relevance))
		#print(np.linalg.norm(last_forward_c_relevance))
		#print(np.linalg.norm(last_backward_c_relevance))
		
		print("Check conservation State reducer")
		print(np.sum(h_relevance) + np.sum(c_relevance))
		print(np.sum(last_forward_h_relevance)+ np.sum(last_backward_h_relevance)+ np.sum(last_forward_c_relevance)+ np.sum(last_backward_c_relevance))

		input_relevance_forward, input_relevance_backward, h_relevance_rest_forward, c_relevance_rest_forward, h_relevance_rest_backward, c_relevance_rest_backward = self.encoder.computeLRP(last_forward_h_relevance, last_backward_h_relevance, last_forward_c_relevance, last_backward_c_relevance, forward_encoder_relevance, backward_encoder_relevance)


		print("Check conservation State reducer")
		print(np.sum(last_forward_h_relevance)+ np.sum(last_backward_h_relevance)+ np.sum(last_forward_c_relevance)+ np.sum(last_backward_c_relevance)+ np.sum(forward_encoder_relevance)+ np.sum(backward_encoder_relevance))
		print(np.sum(input_relevance_forward)+ np.sum(input_relevance_backward) + np.sum(h_relevance_rest_forward)+ np.sum(c_relevance_rest_forward) + np.sum(h_relevance_rest_backward)+ np.sum(c_relevance_rest_backward))

		print("Check conservation LRP")
		print(np.sum(output_relevance))
		print(np.sum(input_relevance_forward)+ np.sum(input_relevance_backward)+ np.sum(np.sum(decoder_input_relevance)) + np.sum(rest_relevance)+ np.sum(h_relevance_rest_forward)+ np.sum(c_relevance_rest_forward) + np.sum(h_relevance_rest_backward)+ np.sum(c_relevance_rest_backward))


		print("Difference between start and attributed LRP")
		print(np.sum(output_relevance))
		print(np.sum(input_relevance_forward)+ np.sum(input_relevance_backward)+ np.sum(np.sum(decoder_input_relevance)))

		print("Rest relevance in attention")
		print(np.sum(rest_relevance))
		print("Rest relevance in forward (h, c)")
		print(np.sum(h_relevance_rest_forward))
		print(np.sum(c_relevance_rest_forward))
		print("Rest relevance in backward (h, c)")
		print(np.sum(h_relevance_rest_backward))
		print(np.sum(c_relevance_rest_backward))
		# + np.sum(input_relevance_backward) + np.sum(decoder_input_relevance)
		return input_relevance_forward, input_relevance_backward, decoder_input_relevance