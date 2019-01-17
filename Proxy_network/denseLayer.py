import numpy as np
from utils import *

class DenseLayer:
	"""
		Represents the dense Layer that maps the output of the decoder LSTM to a probability distribution over 
		Applies the computation Ax + b
	"""
	def __init__(self, kernel = [], bias = []):
		self.kernel = np.array(kernel).transpose()
		self.bias = np.array(bias).reshape((len(bias), 1))

	def compute_forward(self, state):
		return np.matmul(self.kernel, state['h']) + self.bias

	def countParams(self):
		return self.kernel.shape[0] * self.kernel.shape[1] + self.bias.shape[0]

	def computeLRP(self, output_relevance, cell_state):
		input_relevance = layerLRP(cell_state["lstm"]["h"], self.kernel, self.bias, cell_state["output"], output_relevance)
		return input_relevance
