import numpy as np
from utils import *

class StateReducer:
	"""
		Class that generates the initial state of the decoder from the last states of the encoder

	"""
	def __init__(self, W_c = [], W_h= [], b_c = [], b_h = []):
		self.W_c = np.array(W_c).transpose()
		self.W_h = np.array(W_h).transpose()
		self.b_c = np.array(b_c).reshape((len(b_c), 1))
		self.b_h = np.array(b_h).reshape((len(b_h), 1))
		self.cellSize = self.b_c.shape[0]
	
	def compute_reduction(self, forward_state, backward_state):
		"""
			Computes the forward pass for the state reducer

			Applies the transformation Ax+b to the concatenated hidden state and context vector

		"""
		old_h = np.concatenate([forward_state["h"], backward_state["h"]])
		old_c = np.concatenate([forward_state["c"], backward_state["c"]])


		new_h = np.matmul(self.W_h, old_h) + self.b_h
		new_c = np.matmul(self.W_c, old_c) + self.b_c


		#print(np.linalg.norm(np.matmul(self.W_c[:,:self.cellSize], forward_state["c"])))
		#print(np.linalg.norm(np.matmul(self.W_c[:,self.cellSize:], backward_state["c"])))
		self.old_h = old_h
		self.old_c = old_c

		self.new_h = new_h
		self.new_c = new_c

		return new_h,new_c

	def countParams(self):
		return self.W_c.shape[0] * self.W_c.shape[1] + self.W_h.shape[0] * self.W_h.shape[1] + self.b_c.shape[0] + self.b_h.shape[0]

	def computeLRP(self, h_relevance, c_relevance):
		"""
			Compute LRP for the state reducer

			Input :
			h_relevance : relevance of the initial decoder hidden state
			c_relevance : c_relevance of the initial decoder context vector

			Return :
			forward_h_relevance : Relevance attributed to the forward encoder last hidden state
			backward_h_relevance : Relevance attributed to the backward encoder last hidden state
			forward_c_relevance : Relevance attributed to the forward encoder last context vector
			backward_c_relevance : Relevance attributed to the backward encoder last context vector
		"""
		h_rel = layerLRP(self.old_h, self.W_h, self.b_h, self.new_h, h_relevance)
		c_rel = layerLRP(self.old_c, self.W_c, self.b_c, self.new_c, c_relevance)

		forward_h_relevance = h_rel[:self.cellSize]
		backward_h_relevance = h_rel[self.cellSize:]

		forward_c_relevance = c_rel[:self.cellSize]
		backward_c_relevance = c_rel[self.cellSize:]

		return forward_h_relevance, backward_h_relevance, forward_c_relevance, backward_c_relevance