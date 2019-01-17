import numpy as np


class Embedding:
	"""
		Class aimed at representing the embedding part of the architecture

		Stores the embeddingMatrix and translates the input integers into vectors

		Variables :
			cellSize is the dimension of the hidden layers in the LSTM cell
			embeddingSize is the dimension of the input
			vocabSize is the number of words
	"""
	def __init__(self, matrix = []):
		self.embeddingMat = np.array(matrix)
		self.vocabSize = self.embeddingMat.shape[0]
		self.embeddingSize = self.embeddingMat.shape[1]

	def transform_input(self, input_seq):
		"""
			Translate the integer sequence into a sequence of vectors
			
			input_seq is a list of intergers
			outputs a list of numpy arrays
		"""
		embeddingMat = self.embeddingMat

		assert(max(input_seq) < self.vocabSize)
		assert(min(input_seq) >= 0)
		embedded_seq = []
		for i in range(len(input_seq)):
			vector = embeddingMat[input_seq[i], :].transpose()
			embedded_seq.append(vector)

		return embedded_seq

	def transform_word(self, id_word):
		"""
			Translate the integer sequence into a sequence of vectors
			
			input_seq is a list of intergers
			outputs a list of numpy arrays
		"""
		embeddingMat = self.embeddingMat

		assert(id_word < self.vocabSize)
		assert(id_word >= 0)
		embedded =  embeddingMat[id_word, :].transpose()
		return embedded

	def countParams(self):
		return self.vocabSize * self.embeddingSize