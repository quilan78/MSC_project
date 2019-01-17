import tensorflow as tf
import numpy as np
import time
import sys
sys.path.append('/home/dwt17/MSc_project/neural_sum_1/code/Commons/')
sys.path.append('../Commons/')
from read_data import *
from vocab import *
from treatedData import *
from LRP_output import *
from batch import *
from model import *
import os

class Seq2SeqSummarisation:

	def __init__(self, cellSize = 128, batch_size = 15, max_encoding_length = 200, max_decoding_length = 50, vocab_size = 2000, embedding_size = 64, learning_rate = 0.0001, learning_decay = 0.8, minimum_rate = 0.000001, nbre_epoch = 50, display_batch_freq = 2, gradient_clip = 5, beam_width = 10, save_frequency = 1, coverage=False, pointer=False) :

		self.beam_width = beam_width
		self.model = Model(cellSize = cellSize, batch_size = batch_size, max_encoding_length = max_encoding_length, max_decoding_length = max_decoding_length, vocab_size = vocab_size, embedding_size = embedding_size, learning_rate = learning_rate, learning_decay = learning_decay, minimum_rate =minimum_rate, nbre_epoch = nbre_epoch, display_batch_freq = display_batch_freq, gradient_clip = gradient_clip, beam_width = beam_width, save_frequency = save_frequency, coverage=coverage, pointer=pointer)

	def train(self, nb_data = 100, create_batches=True, load_from_checkpoint=False, checkpoint_path = "../../Train/Model/model1.ckpt", tensorboard_path="../../Train/tensorboard/", writting_path_batches="../../Data/Batches", data_path="../../Data/finished_files/"):
		
		#Initialising the Model
		self.model.init_graph_and_data(task="train", nb_data = nb_data, create_batches=create_batches, writting_path_batches=writting_path_batches, data_path=data_path)
		 
		#Training variables
		display_batch_freq = self.model.display_batch_freq
		save_frequency = self.model.save_frequency
		nbre_epoch = self.model.nbre_epoch
		nb_batches = self.model.nb_batches
		learning_rate = self.model.learning_rate
		learning_decay = self.model.learning_decay
		minimum_rate = self.model.minimum_rate
		save_frequency = self.model.save_frequency

		#Data loader
		batch_loader = Batch()

		#For testing on CPU
		#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
		full_start = time.time()
		# Blocks the execution to only one CPU
		session_conf = tf.ConfigProto(device_count = {'CPU': 1, 'GPU' : 1},inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)
		with tf.Session(config=session_conf) as sess:
		#with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			#print(tf.trainable_variables())
			if load_from_checkpoint:
				print("Restoring saved network")
				self.model.saver.restore(sess, checkpoint_path)
				print("Last version of the network loaded")
			#For tensorboard
			train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
			#batch_loader.load_object(4)
			step = 0
			for epoch in range(nbre_epoch):
				average_loss = 0
				epoch_start = time.time()
				summed_loss = 0
				print("START OF EPOCH {}/{}".format(epoch+1, nbre_epoch))
				for batch in range(nb_batches):
					batch_loader.load_object(batch, filepath=writting_path_batches)

					start_time = time.time()
					summary, loss, _ = sess.run([self.model.merged, self.model.tf_loss, self.model.tf_update_step],{
						self.model.tf_input_enc_batch : batch_loader.input_enc,
						self.model.tf_input_dec_batch : batch_loader.input_dec,
						self.model.tf_target_dec_batch : batch_loader.target_dec,
						self.model.tf_input_enc_seq_lengths : batch_loader.input_enc_seq,
						self.model.tf_input_dec_seq_lengths : batch_loader.input_dec_seq,
						self.model.tf_learning_rate : learning_rate,
						self.model.tf_batch_max_oov : batch_loader.max_oovs

						})
					end_time = time.time()
					time_spent = end_time - start_time
					average_loss += loss
					if batch % display_batch_freq == 0:
						print("EPOCH {}/{}, BATCH {}/{}, Loss {}, Time {}, rate {}".format(epoch+1,
																				nbre_epoch,
																				batch,
																				nb_batches,
																				loss,
																				time_spent,
																				learning_rate))
						train_writer.add_summary(summary, step)
					step += 1
				print("Average epoch loss : {}".format(average_loss/nb_batches))
				if learning_rate * learning_decay > minimum_rate:
					learning_rate *= learning_decay
				if epoch % save_frequency == 0:
					print("Saving the model")
					save_path = self.model.saver.save(sess, checkpoint_path)
					print("Model saved")

				epoch_end = time.time()
				print("EPOCH TIME : {} h".format((epoch_end-epoch_start)/3600))
			print("Training finished")
			print("Saving the model")
			save_path = self.model.saver.save(sess, checkpoint_path)
			print("Model saved")
			full_stop = time.time()
			print("FULL TIME : {} h".format((full_stop-full_start)/3600))


	def infer(self, nb_data = 10, checkpoint_path = "../../Train/Model/model1.ckpt", save_path ="../../Output/", data_path="../../Data/finished_files/"):
		
		self.model.init_graph_and_data(task="test", nb_data=nb_data, data_path=data_path)



		#Training variables
		nb_batches = self.model.nb_batches
		batch_size = self.model.batch_size

		# Blocks the execution to only one CPU
		session_conf = tf.ConfigProto(device_count = {'CPU': 1, 'GPU' : 1},inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)
		with tf.Session(config=session_conf) as sess:
			sess.run(tf.global_variables_initializer())
			print(tf.trainable_variables())


			train_writer = tf.summary.FileWriter("tensorboard/", sess.graph)
			print("Restoring saved network")
			self.model.saver.restore(sess, checkpoint_path)
			print("Last version of the network loaded")
			#For tensorboard
			
			print("START OF INFERENCE")
			for batch in range(nb_batches):
				for elem in range(batch_size): #We infer on one element at the time

					id_ = batch * batch_size + elem + 1
					print("START OF INFERENCE FOR WORD {}/{}".format(id_, nb_batches*batch_size))
					start_time = time.time()

					#We tile the input to match the batch size
					input_inf = [[word for word in self.model.input_enc_batches[batch][elem]] for i in range(batch_size)]
					input_length_inf = [self.model.input_enc_seq_lengths[batch][elem] for i in range(batch_size)]

					#Without beam
					encoder_outputs, prediction_greedy, logits_greedy, enc_state, alignment_history, keys = sess.run([self.model.enc_outputs, self.model.output_prediction_greedy, self.model.logits_prediction_greedy, self.model.encoder_state, self.model.dec_states_greedy.alignment_history.stack(), self.model.attention_mechanism.values,],{
						self.model.tf_input_enc_batch : input_inf,
						self.model.tf_input_enc_seq_lengths : input_length_inf,
						self.model.tf_batch_max_oov : self.model.max_oovs[batch]
						})

					#With beam
					"""
					encoder_outputs, prediction_greedy, logits_greedy, prediction_beam, enc_state, alignment_history, keys = sess.run([self.model.enc_outputs, self.model.output_prediction_greedy, self.model.logits_prediction_greedy, self.model.output_prediction_beam, self.model.encoder_state, self.model.dec_states_greedy.alignment_history.stack(), self.model.attention_mechanism.values,],{
						self.model.tf_input_enc_batch : input_inf,
						self.model.tf_input_enc_seq_lengths : input_length_inf,
						self.model.tf_batch_max_oov : self.model.max_oovs[batch]
						})
					"""
					#print("hello")
					#print(alignment_history)
					#print(encoder_outputs)
					#print(keys)
					end_time = time.time()
					time_spent = end_time - start_time
					print("Element {}/{}, Time {}".format(id_, nb_batches*batch_size, time_spent))
					#Getting the first summary(they are all identical)
					greedy_seq_num = prediction_greedy[0]
					#Getting the best summary from the beam search
					#beam_seq_num = prediction_beam[0,:,0]

					beam_seq_num = []

					#print(beam_seq_num)
					greedy_seq = self.model.vocab.TransalteAnswer(greedy_seq_num)
					beam_seq = self.model.vocab.TransalteAnswer(beam_seq_num)
					original_text = self.model.save_enc_input[batch][elem]
					input_text = self.model.vocab.TransalteAnswer(self.model.input_enc_batches[batch][elem])
					original_summary = self.model.save_dec_output[batch][elem]
					input_summary = self.model.vocab.TransalteAnswer(self.model.target_dec_batches[batch][elem])
					print("Starting saving to file element {}".format(id_))
					treated = TreatedData(original_text=original_text, input_text=input_text, original_summary=original_summary, input_summary=input_summary,  keys=keys, encoder_outputs=encoder_outputs, alignment_history=alignment_history, input_seq=self.model.input_enc_batches[batch][elem], enc_state=enc_state, greed_seq_num=greedy_seq_num, greed_seq=greedy_seq, beam_seq_number=beam_seq_num, beam_seq=beam_seq, logits = logits_greedy, id_ = id_)
					treated.saveToFileText(filepath=save_path+"Text/")
					treated.save_object(filepath= save_path+"Obj/")
					print("Finished saving to file element {}".format(id_))

	def experiment(self, nb_data = 10, checkpoint_path = "../../Experiment/Model/model1.ckpt", save_path ="../../Experiment/Results/1/",  data_path="../../Data/finished_files/",  exp_data_path="../../Experiment/ModifiedTexts/1/"):
		
		self.model.init_graph_and_data(task="test", nb_data=nb_data, data_path=data_path)

		# Blocks the execution to only one CPU
		session_conf = tf.ConfigProto(device_count = {'CPU': 1, 'GPU' : 1},inter_op_parallelism_threads=1,intra_op_parallelism_threads=1)
		with tf.Session(config=session_conf) as sess:
			sess.run(tf.global_variables_initializer())
			#print(tf.trainable_variables())

			print("Restoring saved network")
			self.model.saver.restore(sess, checkpoint_path)
			print("Last version of the network loaded")
			#For tensorboard
			

			batch_size = self.model.batch_size
			data = LRP_output()

			print("START OF INFERENCE")
			for batch in range(10):
				#for elem in range(batch_size): #We infer on one element at the time
					id_ = batch
					print("START OF INFERENCE FOR WORD {}/{}".format(id_, 10))
					start_time = time.time()



					data.load_json(batch, filepath=exp_data_path)

					#We tile the input to match the batch size
					input_inf = self.model.vocab.TranslateBatches(np.array([np.array([data.input_text for i in range(batch_size)])]))[0]
					input_length_inf = [400 for i in range(batch_size)]

					#Without beam
					encoder_outputs, prediction_greedy, logits_greedy, enc_state, alignment_history, keys = sess.run([self.model.enc_outputs, self.model.output_prediction_greedy, self.model.logits_prediction_greedy, self.model.encoder_state, self.model.dec_states_greedy.alignment_history.stack(), self.model.attention_mechanism.values,],{
						self.model.tf_input_enc_batch : input_inf,
						self.model.tf_input_enc_seq_lengths : input_length_inf,
						self.model.tf_batch_max_oov : 0
						})

					#With beam
					"""
					encoder_outputs, prediction_greedy, logits_greedy, prediction_beam, enc_state, alignment_history, keys = sess.run([self.model.enc_outputs, self.model.output_prediction_greedy, self.model.logits_prediction_greedy, self.model.output_prediction_beam, self.model.encoder_state, self.model.dec_states_greedy.alignment_history.stack(), self.model.attention_mechanism.values,],{
						self.model.tf_input_enc_batch : input_inf,
						self.model.tf_input_enc_seq_lengths : input_length_inf,
						self.model.tf_batch_max_oov : self.model.max_oovs[batch]
						})
					"""
					#print("hello")
					#print(alignment_history)
					#print(encoder_outputs)
					#print(keys)
					end_time = time.time()
					time_spent = end_time - start_time
					print("Element {}/{}, Time {}".format(id_, 10, time_spent))
					#Getting the first summary(they are all identical)
					greedy_seq_num = prediction_greedy[0]
					#Getting the best summary from the beam search
					#beam_seq_num = prediction_beam[0,:,0]

					beam_seq_num = []

					#print(beam_seq_num)
					greedy_seq = self.model.vocab.TransalteAnswer(greedy_seq_num)
					beam_seq = self.model.vocab.TransalteAnswer(beam_seq_num)
					original_text = data.original_text
					input_text = data.input_text
					original_summary = data.original_summary
					input_summary = data.input_summary
					print("Starting saving to file element {}".format(id_))
					treated = TreatedData(input_text=input_text, greed_seq=greedy_seq, id_ = id_)
					treated.saveToFileText(filepath=save_path+"Text/")
					treated.save_object(filepath= save_path+"Obj/")
					print("Finished saving to file element {}".format(id_))

	

if __name__ == "__main__":
	net = Seq2SeqSummarisation()
	net.train(nb_data = 20000, create_batches=False, load_from_checkpoint=True)
	#net.infer()