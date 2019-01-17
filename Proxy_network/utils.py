import numpy as np
import math
import scipy.special as sp
from numpy import newaxis as na
#def sigmoid(x):
#	return 1 / (1 + np.exp(-x))
#sigmoid = np.vectorize(sigmoid)	
sigmoid = sp.expit

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def no_activation(x):
	return x

def layerLRP(input_, weights, bias, outputs, output_relevance, epsilon = 0.00001, delta=0.0):
	"""
		Applies LRP to a one layer fully connected layer
		Noting C the size of the lower layer and D the size of the higher layer
		input : activation of the lower layer. shape (C,1)
		weights : weights of the layer, shape (D,C)
		bias : bias of the layer, shape (D,1)
		outputs : Output activation (D,1)
		output_relevance : Relevance of the higher layer, shape (D,1)
		epsilon : stabilizer
		delta : factor that sets the consevativeness of LRP
	"""

	N = float(len(input_))
	signs = np.tile(np.where(outputs>=0, 1., -1.), (1, weights.shape[1]))

	# note : not matrix multiplication
	numerator = (weights * input_.transpose() + 1./N * (epsilon * signs + delta*np.tile(bias, (1, weights.shape[1]))) )

	denominator = np.tile(outputs, (1, weights.shape[1])) + epsilon * signs
	#print(np.min(np.abs(denominator)))
	

	relevance_matrix = (numerator / denominator) * output_relevance

	input_relevance = np.sum(relevance_matrix, axis=0).reshape((weights.shape[1], 1))
	return input_relevance
	#Rin = lrp_linear(input_.reshape(input_.shape[0],), weights.transpose(), bias.reshape((bias.shape[0],)), outputs.reshape((bias.shape[0],)), output_relevance.reshape((bias.shape[0],)), len(input_), epsilon, delta, debug=False)
	#return Rin.reshape((weights.shape[1], 1))

def addElementwiseListArray(list1, list2):
	"""
		Add together the arrays of two lists
		(takes the length of the list1)
	"""
	l3 = []
	for i in range(len(list1)):
		l3.append(list1[i] + list2[i])
	return l3

def lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor, debug=False):
	"""
    LRP for a linear layer with input dim D and output dim M.
    Args:
    - hin:            forward pass input, of shape (D,)
    - w:              connection weights, of shape (D, M)
    - b:              biases, of shape (M,)
    - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
    - Rout:           relevance at layer output, of shape (M,)
    - bias_nb_units:  number of lower-layer units onto which the bias/stabilizer contribution is redistributed
    - eps:            stabilizer (small positive number)
    - bias_factor:    for global relevance conservation set to 1.0, otherwise 0.0 to ignore bias redistribution
    Returns:
    - Rin:            relevance at layer input, of shape (D,)
    """
	sign_out = np.where(hout[na,:]>=0, 1., -1.) # shape (1, M)

	numer    = (w * hin[:,na]) + ( (bias_factor*b[na,:]*1. + eps*sign_out*1.) * 1./bias_nb_units ) # shape (D, M)
	denom    = hout[na,:] + (eps*sign_out*1.)   # shape (1, M)

	message  = (numer/denom) * Rout[na,:]       # shape (D, M)
	Rin      = message.sum(axis=1)              # shape (D,)

	# Note: local  layer   relevance conservation if bias_factor==1.0 and bias_nb_units==D
	#       global network relevance conservation if bias_factor==1.0 (can be used for sanity check)
	if debug:
		print("local diff: ", Rout.sum() - Rin.sum())

	return Rin


if __name__ == "__main__":
	print(softmax([0.0, -1.0, 2.0, 3.0]))
	print(softmax([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]))


