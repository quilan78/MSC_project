import numpy as np
from seq2seqSummariser import *
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
import time
import tensorflow as tf
import sys
sys.path.append('/home/dwt17/MSc_project/neural_sum_1/code/Commons/')
sys.path.append('../Commons/')
from read_data import *
from treatedData import *	
from utils import *
from LRP_output import *
class ModelManager:

	def __init__(self):
		pass


	def loadModel(self, model_path, max_encoding_length=200, max_decoding_length=50, vocab_path="../../Data/finished_files/"):
		"""
			Loads the proxy model from the checkpoint


		"""
		sess = tf.InteractiveSession()
		saver = tf.train.import_meta_graph(model_path+'.meta')
		saver.restore(sess, model_path)
		#for i in tf.trainable_variables():
		#	print(i)
		# get the graph
		g = tf.get_default_graph()

		#First get all the size constants
		embeddingMatrix = g.get_tensor_by_name('embedding/embedding_encoder:0')
		encoderBias = g.get_tensor_by_name('encoder/bidirectional_rnn/fw/forward_cell/bias:0')
		attentionLayer = g.get_tensor_by_name('decoder/decoder_training/decoder/attention_wrapper/attention_layer/kernel:0')
		vocabSize = embeddingMatrix.eval().shape[0]
		embeddingSize = embeddingMatrix.eval().shape[1]
		cellSize = int(encoderBias.eval().shape[0] / 4)
		attentionSize = attentionLayer.eval().shape[1]


		#Adding info to object
		self.vocabSize = vocabSize
		self.embeddingSize = embeddingSize
		self.cellSize = cellSize
		self.attentionSize = attentionSize
		self.max_encoding_length = max_encoding_length
		self.max_decoding_length = max_decoding_length

		self.seq2seqSummariser = Seq2seqSummariser(cellSize = cellSize, embeddingSize=embeddingSize, attentionSize=attentionSize, vocabSize=vocabSize, max_encoding_length=max_encoding_length, max_decoding_length=max_decoding_length, vocab_path=vocab_path)
		

		#Initialising embedding
		embeddingMatrix = g.get_tensor_by_name('embedding/embedding_encoder:0')
		self.seq2seqSummariser._initialise_embedding(embeddingMatrix.eval())

		kernel_forward = g.get_tensor_by_name('encoder/bidirectional_rnn/fw/forward_cell/kernel:0')
		bias_forward = g.get_tensor_by_name('encoder/bidirectional_rnn/fw/forward_cell/bias:0')
		kernel_backward = g.get_tensor_by_name('encoder/bidirectional_rnn/bw/backward_cell/kernel:0')
		bias_backward = g.get_tensor_by_name('encoder/bidirectional_rnn/bw/backward_cell/bias:0')
		self.seq2seqSummariser._initialise_encoder(kernel_forward.eval(), bias_forward.eval(), kernel_backward.eval(), bias_backward.eval())

		W_h = g.get_tensor_by_name('reduce_transfered_states/W_h:0')
		W_c = g.get_tensor_by_name('reduce_transfered_states/W_c:0')
		b_h = g.get_tensor_by_name('reduce_transfered_states/b_h:0')
		b_c = g.get_tensor_by_name('reduce_transfered_states/b_c:0')
		self.seq2seqSummariser._initialise_stateReducer(W_h.eval(), W_c.eval(), b_h.eval(), b_c.eval())

		kernel1 = g.get_tensor_by_name('decoder/decoder_training/decoder/attention_wrapper/cell/kernel:0')
		bias1 = g.get_tensor_by_name('decoder/decoder_training/decoder/attention_wrapper/cell/bias:0')
		memoryLayer = g.get_tensor_by_name('decoder/memory_layer/kernel:0')
		queryLayer = g.get_tensor_by_name('decoder/decoder_training/decoder/attention_wrapper/bahdanau_attention/query_layer/kernel:0')
		attentionV = g.get_tensor_by_name('decoder/decoder_training/decoder/attention_wrapper/bahdanau_attention/attention_v:0')
		attentonLayer = g.get_tensor_by_name('decoder/decoder_training/decoder/attention_wrapper/attention_layer/kernel:0')
		self.seq2seqSummariser._initialise_decoder(kernel1.eval(), bias1.eval(), memoryLayer.eval(), queryLayer.eval(), attentionV.eval(), attentonLayer.eval())

		kernel2 = g.get_tensor_by_name('decoder/decoder_training/decoder/dense/kernel:0')
		bias2 = g.get_tensor_by_name('decoder/decoder_training/decoder/dense/bias:0')
		self.seq2seqSummariser._initialise_DenseLayer(kernel2.eval(), bias2.eval())

	def prepare_data(self, path="../../Data/finished_files/", filename="test"):
		"""
			Loads the data for further use
		"""
		data = Data(path=path)
		examples, number = data.read_preprocessed(filename, maxi=10)
		self.data= examples
		self.nbre_data = number


	def compute_forward_from_source(self, id_, path="../../../Data/finished_files/"):
		data = Data(path=path)

		example = self.data[id_]
		abstract_input, abstract_target, length = data.generate_abstract_data(example.abstract, self.max_decoding_length)
		text, length = data.generate_text_data(example.text, self.max_encoding_length)

		return self.seq2seqSummariser.compute_forward(text, [abstract_input[0]])

	def compute_forward_from_prediction(self, text, summary):
		return self.seq2seqSummariser.compute_forward([int(x) for x in text], [int(x) for x in summary])

	def compare_results(self, proxy_outputs, prediction):
		pass

	def load_both_prediction(self, id_, path="../../Training/Exp1/Output/Obj/"):
		data = TreatedData()
		prediction = data.load_object(id_, filepath=path)
		logits = prediction.logits[0]

		forward_encoder_states, backward_encoder_states, h, c, decoder_states = self.compute_forward_from_prediction(prediction.input_seq, prediction.greed_seq_num)
		#print(prediction.input_seq)
		self.compare_transmitted(prediction, h, c)
		self.compare_encoder_output(399, prediction, forward_encoder_states, backward_encoder_states)
		self.compare_alignment(15, prediction, decoder_states)
		self.compare_logits(49, prediction, decoder_states)


	def compare_transmitted(self, prediction, h, c):
		print("Transmitted state proxy")
		print(h[:10])
		print("Transmitted state tensorflow")
		print(prediction.enc_state.h[0][:10])
		print("Transmitted context proxy")
		print(c[:10])
		print("Transmitted context tensorflow")
		print(prediction.enc_state.c[0][:10])

	def compare_encoder_output(self, id_, prediction, forward_encoder_states, backward_encoder_states):
		print("Forward output")
		print(prediction.encoder_outputs[0][id_][:10])
		print(forward_encoder_states[id_]["h"][:10])
		print("Backward output")
		print(prediction.encoder_outputs[0][id_][-10:])
		print(backward_encoder_states[id_]["h"][-10:])

	def compare_alignment(self, id_, prediction, decoder_states):
		print("Tensorflow's alignment")
		print(prediction.alignment_history[id_][0][:10])
		print(sum(prediction.alignment_history[id_][0]))
		print("Proxy's alignment")
		print(decoder_states[id_+1]["alpha"][:10])
		print(sum(decoder_states[id_+1]["alpha"]))

	def compare_logits(self, id_, prediction, decoder_states):
		print("Logits for tensorflow")
		print(prediction.logits[0][id_][:10])
		print("Logits for the proxy")
		print(decoder_states[id_]["output"][:10])


	def countParams(self):
		return self.seq2seqSummariser.countParams()

	def requestLRPOnPredictedWord(self,decoding_step, target_seq, decoder_states):
		targets = []
		for i in range(self.max_decoding_length):
			targets.append([False for i in range(self.vocabSize)])
			if i == decoding_step:
				targets[i][target_seq[decoding_step]] = True
		
		input_relevance_forward, input_relevance_backward, decoder_input_relevance = self.seq2seqSummariser.computeLRP(targets, start_decode=decoding_step)
		return input_relevance_forward, input_relevance_backward, decoder_input_relevance
		
	def generateLRP_output(self, id_, path="../../Training/Exp1/Output/Obj/", json_path="../Visualiser/JSON/"):
		data = TreatedData()
		prediction = data.load_object(id_, filepath=path)

		lrp_output = LRP_output()
		lrp_output.original_text = prediction.original_text.copy()
		lrp_output.input_text = prediction.input_text.copy()
		lrp_output.original_summary = prediction.original_summary.copy()
		lrp_output.input_summary = prediction.input_summary.copy()
		#print(type([self.seq2seqSummariser.vocab.start_decode_token]))
		#print(type(prediction.greed_seq))
		lrp_output.greedy_summary = [self.seq2seqSummariser.vocab.start_decode_token] + prediction.greed_seq.tolist().copy()
		lrp_output.beam_summary = [self.seq2seqSummariser.vocab.stop_decode_token] +prediction.beam_seq.tolist().copy()

		#For the greedy decoding
		#Compute the forward pass

		time_start = time.time()
		print("greedy forward pass started")
		forward_encoder_states, backward_encoder_states, h, c, decoder_states = self.compute_forward_from_prediction(prediction.input_seq, prediction.greed_seq_num)
		print("greedy forward pass ended")
		time_stop = time.time()
		with open(json_path+str(id_)+".txt", 'a') as outfile:
			outfile.write("Just finished greedy search,  took {} h \n".format((time_stop-time_start)/3600))	

		
		logits = []
		attention = []
		lrp_forward = []
		lrp_backward = []
		lrp_dec = []

		for i in range(len(decoder_states)):
			logits.append(softmax(decoder_states[i]["output"]).tolist())
			attention.append(decoder_states[i]["alpha"].tolist())

			time_start = time.time()
			#LRP
			print("greedy LRP pass started for word {}".format(i))
			input_relevance_forward, input_relevance_backward, decoder_input_relevance = self.requestLRPOnPredictedWord(i, prediction.greed_seq_num, decoder_states)
			print("greedy LRP pass ended for word {}".format(i))
			
			lrp_forward.append([np.sum(x) for x in input_relevance_forward])
			lrp_backward.append([np.sum(x) for x in input_relevance_backward])
			lrp_dec.append([np.sum(x) for x in decoder_input_relevance])


			time_stop = time.time()
			with open(json_path+str(id_)+".txt", 'a') as outfile:
				outfile.write("Just treated greedy word {}, took {} h \n".format(i,(time_stop-time_start)/3600))	

		lrp_output.greedy_attention = attention.copy()
		lrp_output.greedy_logits = logits.copy()
		lrp_output.greedy_LRP_encoder_forward = lrp_forward.copy()
		lrp_output.greedy_LRP_encoder_backward = lrp_backward.copy()
		lrp_output.greedy_LRP_decoder = lrp_dec.copy()
		#print(lrp_dec)
		time_start = time.time()
		print("checking match")
		print(lrp_forward[0][0])
		print(lrp_backward[0][0])
		#For the beam decoding
		#Compute the forward pass
		"""
		print("Beam forward pass started")
		forward_encoder_states, backward_encoder_states, h, c, decoder_states = self.compute_forward_from_prediction(prediction.input_seq, prediction.beam_seq_number)
		print("Beam forward pass ended")
		
		
		time_stop = time.time()
		with open(json_path+str(id_)+".txt", 'a') as outfile:
			outfile.write("Just finished beam search,  took {} h \n".format((time_stop-time_start)/3600))	
		
		logits = []
		attention = []
		lrp_forward = []
		lrp_backward = []
		lrp_dec = []

		for i in range(len(decoder_states)):
			logits.append(softmax(decoder_states[i]["output"]).tolist())
			attention.append(decoder_states[i]["alpha"].tolist())

			time_start = time.time()
			#LRP
			print("Beam LRP pass started {}".format(i))
			input_relevance_forward, input_relevance_backward, decoder_input_relevance = self.requestLRPOnPredictedWord(i, prediction.beam_seq_number, decoder_states)
			print("Beam LRP pass ended {}".format(i))

			lrp_forward.append([np.sum(x) for x in input_relevance_forward])
			lrp_backward.append([np.sum(x) for x in input_relevance_backward])
			lrp_dec.append([np.sum(x) for x in decoder_input_relevance])


			time_stop = time.time()
			with open(json_path+str(id_)+".txt", 'a') as outfile:
				outfile.write("Just treated beam word {}, took {} h \n".format(i, (time_stop-time_start)/3600))	

		lrp_output.beam_attention = attention.copy()
		lrp_output.beam_logits = logits.copy()
		lrp_output.beam_LRP_encoder_forward = lrp_forward.copy()
		lrp_output.beam_LRP_encoder_backward = lrp_backward.copy()
		lrp_output.beam_LRP_decoder = lrp_dec.copy()
		print("checking match")
		print(lrp_forward[0][0])
		print(lrp_backward[0][0])
		"""


		with open(json_path+str(id_)+".json", 'w') as outfile:
			json.dump(lrp_output.__dict__, outfile)
		
		




if __name__ == "__main__":

	manager = ModelManager()
	manager.loadModel("../../Experiment/Model/model1.ckpt", max_encoding_length=400, max_decoding_length=50)
	#manager.load_both_prediction(1, path="../../Experiment/Test/Obj/")
	manager.generateLRP_output(0, path="../../Experiment/Results/Important/2/Obj/", json_path="../../Experiment/attention_lrp_test/")
	#manager.requestLRPOnPredictedWord(1, 3, path="../../Training/Exp1/Output/Obj/")
	print(manager.countParams())	
	#manager.prepare_data(path="../../Data/finished_files/", filename="test")
	#manager.compute_forward(0)