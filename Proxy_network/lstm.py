import numpy as np
from utils import *
import tensorflow as tf

class Lstm:
	"""
		Implementation of a LSTM cell

		Variables :
			Wx are the matrices applied for each gate on the input
			Rx are the matrices applied for each gate on the previous hidden state
			bx are the bias vector
			cellSize is the dimension of the hidden layers in the LSTM cell
			embeddingSize is the dimension of the input



	"""

	def __init__(self, kernel = [], bias=[], cellSize=128, forget_added_bias=1.0, activation="np"):

		self.raw_kernel = np.array(kernel).transpose()
		self.raw_bias = np.array(bias)

		self.cellSize = cellSize
		self.embeddingSize = self.raw_kernel.shape[1] - cellSize

		self.activation = activation

		self.forget_added_bias = forget_added_bias * np.ones((cellSize,1))

		#Extracting the matrices from the raw matrix given by tensorflow
		self.Wi = self.raw_kernel[:self.cellSize,:self.embeddingSize]
		self.Ri = self.raw_kernel[:self.cellSize,self.embeddingSize:]
		self.Wz = self.raw_kernel[self.cellSize:2*self.cellSize,:self.embeddingSize]
		self.Rz = self.raw_kernel[self.cellSize:2*self.cellSize,self.embeddingSize:]
		self.Wf = self.raw_kernel[2*self.cellSize:3*self.cellSize,:self.embeddingSize]
		self.Rf = self.raw_kernel[2*self.cellSize:3*self.cellSize,self.embeddingSize:]
		self.Wo = self.raw_kernel[3*self.cellSize:,:self.embeddingSize]
		self.Ro = self.raw_kernel[3*self.cellSize:,self.embeddingSize:]

		self.bi = self.raw_bias[:self.cellSize].reshape((self.cellSize, 1))
		self.bz = self.raw_bias[self.cellSize:2*self.cellSize].reshape((self.cellSize, 1))
		self.bf = self.raw_bias[2*self.cellSize:3*self.cellSize].reshape((self.cellSize, 1))
		self.bo = self.raw_bias[3*self.cellSize:].reshape((self.cellSize, 1))

		if activation == "np":
			self.tanh = np.tanh
			self.sigmoid = sigmoid
		elif activation == "tf":
			self.tanh = lambda x : tf.tanh(x).eval()
			self.sigmoid = lambda x : tf.sigmoid(x).eval()

	def forward(self, last_state, input):
		"""
			Compute the forward computation of the LSTM
			last_state : cell state (dictionnary) with elements t, x, z,i, f, c, o, h from last time step
			input : vector representation of the word
			

			output : cell state after computation
		"""
		cell_state = {}
		cell_state["t"] = last_state["t"] +1
		cell_state["last_h"] = last_state["h"]
		cell_state["last_c"] = last_state["c"]
		cell_state["x"] = input.reshape((len(input), 1))
		cell_state["raw_i"] = np.matmul(self.Wi, cell_state["x"]) + np.matmul(self.Ri, cell_state["last_h"]) + self.bi
		cell_state["i"] = self.sigmoid(cell_state["raw_i"])
		cell_state["raw_z"] = np.matmul(self.Wz, cell_state["x"]) + np.matmul(self.Rz, cell_state["last_h"]) + self.bz
		cell_state["z"] = self.tanh(cell_state["raw_z"])
		cell_state["raw_f"] = np.matmul(self.Wf, cell_state["x"]) + np.matmul(self.Rf, cell_state["last_h"]) + self.bf + self.forget_added_bias #Forget bias
		cell_state["f"] = self.sigmoid(cell_state["raw_f"]) 
		cell_state["c"] = np.multiply(cell_state["z"], cell_state["i"]) + np.multiply(cell_state["last_c"], cell_state["f"])
		cell_state["raw_o"] = np.matmul(self.Wo, cell_state["x"]) + np.matmul(self.Ro, cell_state["last_h"]) + self.bo
		cell_state["o"] = self.sigmoid(cell_state["raw_o"])
		cell_state["h"] = np.multiply(self.tanh(cell_state["c"]), cell_state["o"])

		return cell_state

	def zero_state(self):
		"""
			Return the zero cell state of this LSTM cell

		"""
		cell_state = {}
		cell_state["t"] = np.zeros((self.cellSize,1))
		cell_state["x"] = np.zeros((self.embeddingSize,1))
		cell_state["z"] = np.zeros((self.cellSize,1))
		cell_state["i"] = np.zeros((self.cellSize,1))
		cell_state["f"] = np.zeros((self.cellSize,1))
		cell_state["c"] = np.zeros((self.cellSize,1))
		cell_state["o"] = np.zeros((self.cellSize,1))
		cell_state["h"] = np.zeros((self.cellSize,1))
		cell_state["last_h"] = np.zeros((self.cellSize,1))
		cell_state["last_c"] = np.zeros((self.cellSize,1))
		cell_state["raw_z"] = np.zeros((self.cellSize,1))
		cell_state["raw_f"] = np.zeros((self.cellSize,1))
		cell_state["raw_o"] = np.zeros((self.cellSize,1))
		return cell_state

	def countParams(self):
		return self.raw_kernel.shape[0] * self.raw_kernel.shape[1] + self.raw_bias.shape[0]

	def step_LRP(self, cell_state, c_relevance, h_relevance):
		"""
			Compute one step of LRP

			cell_state : cell_state after one time step
			c_relevance : relevance of the output context vector
			h_relevance : relevance of the output hidden state vector


			Returns :
			inp_x_rel : Relevance of the input at that time step
			imp_h_rel : relevance of the last time step hidden state
			imp_c_rel : relevance of the last time step context vector
		"""
		c_relevance += h_relevance
		#inp_c_rel = layerLRP(np.multiply(cell_state["last_c"], cell_state["f"]), np.identity(self.cellSize), np.zeros((self.cellSize, 1)), cell_state["c"], c_relevance, epsilon = 0.001, delta=1.0)
		#temp_rel = layerLRP(np.multiply(cell_state["z"], cell_state["i"]), np.identity(self.cellSize), np.zeros((self.cellSize, 1)), cell_state["c"], c_relevance, epsilon = 0.001, delta=1.0)
		# We have to do it together or modify the LRP function to notify the real number of input neuron
		
		full_lrp_c = layerLRP(np.concatenate([np.multiply(cell_state["last_c"], cell_state["f"]),np.multiply(cell_state["z"], cell_state["i"])]), np.concatenate([np.identity(self.cellSize),np.identity(self.cellSize)],1), np.zeros((self.cellSize, 1)), cell_state["c"], c_relevance)
		inp_c_rel = full_lrp_c[:self.cellSize]
		temp_rel = full_lrp_c[self.cellSize:]
		#print("Check computaiton of C")
		#print(np.multiply(cell_state["last_c"], cell_state["f"]) + np.multiply(cell_state["z"], cell_state["i"]) - cell_state["c"])
		
		#print("Check transfer c_relevance")
		#print(np.sum(c_relevance))
		#print(np.sum(full_lrp_c))
		#print(np.sum(c_relevance) - np.sum(inp_c_rel + temp_rel))

		full_lrp_input = layerLRP(np.concatenate([cell_state["x"], cell_state["last_h"]]), np.concatenate([self.Wz,self.Rz],1), self.bz, cell_state["raw_z"], temp_rel)
		#inp_x_rel = layerLRP(cell_state["x"], self.Wi, self.bi, cell_state["raw_z"], temp_rel, epsilon = 0.001, delta=1.0)
		#inp_h_rel = layerLRP(cell_state["last_h"], self.Ri, self.bi, cell_state["raw_z"], temp_rel, epsilon = 0.001, delta=1.0)	
		inp_x_rel = full_lrp_input[:self.embeddingSize]
		inp_h_rel = full_lrp_input[self.embeddingSize:]	

		#print("Check computaiton of z")
		#print(np.matmul(np.concatenate([self.Wz,self.Rz],1), np.concatenate([cell_state["x"], cell_state["last_h"]])) - cell_state["raw_z"] + self.bz)
		
		#print("Check transfer input relevance")
		#print(np.sum(temp_rel))
		#print(np.sum(full_lrp_input))
		return inp_x_rel, inp_h_rel, inp_c_rel
		
		
		
		

