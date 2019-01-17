import tensorflow as tf
import numpy as np
import time
import sys
#sys.path.append('/home/dwt17/MSc_project/neural_sum_1/code/Proxy_network/Commons/')
sys.path.append('../Commons/')
from read_data import *
from vocab import *
from treatedData import *	
from batch import *
from bahdanauCoverageAttention import *
from attentionPointerWrapper import *
import os


class Model:

	def __init__(self, cellSize = 128, batch_size = 15, max_encoding_length = 200, max_decoding_length = 50, vocab_size = 2000, embedding_size = 64, learning_rate = 0.0001, learning_decay = 0.8, minimum_rate = 0.000001, nbre_epoch = 50, display_batch_freq = 2, gradient_clip = 5, beam_width = 10, save_frequency = 1, coverage=False, pointer=False) :
		self.cellSize = cellSize # 256
		self.batch_size = batch_size
		self.max_encoding_length = max_encoding_length # 400
		self.max_decoding_length = max_decoding_length # 200
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size # 128
		self.learning_rate = learning_rate
		self.learning_decay = learning_decay
		self.minimum_rate = minimum_rate
		self.nbre_epoch = nbre_epoch
		self.display_batch_freq = display_batch_freq
		self.gradient_clip = gradient_clip
		self.beam_width = beam_width
		self.save_frequency = save_frequency
		self.coverage = coverage
		self.pointer = pointer

	def init_graph_and_data(self, task="train", nb_data=100, data_path="../Data/finished_files/", writting_path_batches="../Data/Batches",create_batches=True):
		learning_rate = self.learning_rate
		learning_decay = self.learning_decay
		minimum_rate = self.minimum_rate
		nbre_epoch = self.nbre_epoch
		batch_size = self.batch_size
		beam_width = self.beam_width

		cellSize = self.cellSize
		batch_size = self.batch_size
		max_encoding_length= self.max_encoding_length
		max_decoding_length = self.max_decoding_length
		vocab_size = self.vocab_size
		embedding_size = self.embedding_size
		gradient_clip = self.gradient_clip



		print("Loading vocabulary")
		vocab = Vocab(path=data_path)
		vocab_size = vocab.LoadVocab(max_size=vocab_size)
		self.vocab = vocab
		self.vocab_size = vocab_size # Making sure the vocabulary sizes matches
		# Tokens for the 
		self.start_token = vocab.start_decode_id
		self.stop_token = vocab.stop_decode_id
		print("vocabulary loaded, size : {}".format(vocab_size))

		print("Loading Data")
		data = Data(path=data_path)

		if task == "train":
			if create_batches == True:
				nb_batches = data.GenerateBatchesOnDisk(batch_size, vocab, max_text_length=max_encoding_length, max_abstract_length=max_decoding_length, max_data =nb_data, reading_file="train", writting_path=writting_path_batches, pointer=self.pointer)
				max_summary_length = max_decoding_length
			else:
				max_summary_length = max_decoding_length
				nb_batches = nb_data // batch_size#
			self.max_summary_length=max_summary_length
			self.nb_batches = nb_batches
		elif task =="test":
			filename = task
			input_enc_batches, input_dec_batches, target_dec_batches, input_enc_seq_lengths, input_dec_seq_lengths, nb_batches = data.data_pipeline(batch_size,nb_example=nb_data,max_text_length = max_encoding_length, max_abstract_length = max_decoding_length, filename = filename)
			self.max_summary_length = data.max_abstract_length
			print("Transforming words into id")
			self.save_enc_input = input_enc_batches
			self.save_dec_output = target_dec_batches
			if self.pointer:
				translated_batches, oov_words, max_oovs = vocab.TranslateTextBatchesWithOOV(input_enc_batches)
				self.input_enc_oov = translated_batches
				self.max_oovs = max_oovs
				self.oov_words = oov_words
				self.input_dec_batches = vocab.TranslateBatches(input_dec_batches)
				self.input_enc_batches = vocab.TranslateBatches(input_enc_batches)
				self.target_dec_batches = vocab.TranslateSummaryBatchesWithOOV(target_dec_batches, oov_words)
			else:
				self.input_enc_batches = vocab.TranslateBatches(input_enc_batches)
				self.input_dec_batches = vocab.TranslateBatches(input_dec_batches)
				self.target_dec_batches = vocab.TranslateBatches(target_dec_batches)
				self.max_oovs = [0 for i in self.input_enc_batches]
			self.nb_batches = nb_batches
			self.input_enc_seq_lengths = input_enc_seq_lengths
			self.input_dec_seq_lengths = input_dec_seq_lengths
			print("Inputs of rnn prepared")
		print("Data loaded")
		#print(input_dec_seq_lengths)


		print("Creating Graph")
		self.create_Graph(task)
		print("Graph created")
		#For tensorboard
		#tf.summary.scalar('Loss', tf_loss)
		self.merged = tf.summary.merge_all()

		#Save the trained model
		self.saver = tf.train.Saver()
	
	def create_Graph(self, task="train"):

		self._create_placeholders()

		self._generate_Embeddings()
		self._generate_Encoder()
		
		self.reduce_transfered_states()

		self._generate_Decoder(task=task)
		if task == "train":
			self._generate_Optimisation()

	def _create_placeholders(self):


		batch_size = self.batch_size
		max_encoding_length = self.max_encoding_length
		max_decoding_length = self.max_decoding_length


		#Batch Major
		input_enc_batch = tf.placeholder(tf.int32, [batch_size, max_encoding_length], name="input_enc_batch")
		input_dec_batch = tf.placeholder(tf.int32, [batch_size, max_decoding_length], name="input_dec_batch")
		target_dec_batch = tf.placeholder(tf.int32, [batch_size, max_decoding_length], name="target_dec_batch")

		#Length of text/summary
		input_enc_seq_lengths = tf.placeholder(tf.int32, [batch_size], name="input_enc_seq_lengths")
		input_dec_seq_lengths = tf.placeholder(tf.int32, [batch_size], name="input_dec_seq_lengths")
		max_summary_length = tf.constant(max_decoding_length, dtype=tf.int32, name="max_summary_length")
		max_text_length = tf.constant(max_encoding_length, dtype=tf.int32, name="text_length")
		fake_summary_length = tf.constant(max_decoding_length, dtype=tf.int32, shape=[batch_size])
		#Hyperparameters
		learning_rate = tf.placeholder(tf.float32, name="learning_rate")
		coverage_multiplier = tf.constant(1, tf.float32, name="coverage_multiplier")

		#For pointer network
		batch_max_oov = tf.placeholder(tf.int32, shape=(), name="batch_max_oov")

		self.tf_input_enc_batch = input_enc_batch
		self.tf_input_dec_batch = input_dec_batch
		self.tf_target_dec_batch = target_dec_batch
		self.tf_input_enc_seq_lengths = input_enc_seq_lengths
		self.tf_input_dec_seq_lengths = input_dec_seq_lengths
		self.tf_max_summary_length = max_summary_length
		self.tf_max_text_length = max_text_length
		self.tf_fake_summary_length = fake_summary_length
		self.tf_learning_rate = learning_rate
		self.coverage_multiplier = coverage_multiplier
		self.tf_batch_max_oov = batch_max_oov


	def _generate_Embeddings(self):

		vocab_size = self.vocab_size
		embedding_size = self.embedding_size
		input_enc_batch = self.tf_input_enc_batch
		input_dec_batch = self.tf_input_dec_batch


		with tf.variable_scope("embedding"):
			embedding = tf.get_variable("embedding_encoder", [vocab_size, embedding_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
			tf.summary.histogram('embedding', embedding)
			emb_enc_batch = tf.nn.embedding_lookup(embedding, input_enc_batch)
			emb_dec_batch = tf.nn.embedding_lookup(embedding, input_dec_batch)

		self.emb_enc_batch = emb_enc_batch
		self.emb_dec_batch = emb_dec_batch
		self.embedding_matrix = embedding

	def _generate_Encoder(self):
		cellSize = self.cellSize
		input_batch = self.emb_enc_batch
		seq_length_batch = self.tf_input_enc_seq_lengths

		with tf.variable_scope("encoder"):
			forward_LSTM = tf.contrib.rnn.LSTMCell(cellSize,
											   initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2), forget_bias=1.0, name="forward_cell")
			backward_LSTM = tf.contrib.rnn.LSTMCell(cellSize,
											   initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2), forget_bias=1.0, name="backward_cell")
			enc_outputs, enc_states = tf.nn.bidirectional_dynamic_rnn(
																forward_LSTM,
																backward_LSTM,
																input_batch,
																sequence_length= seq_length_batch,
																dtype=tf.float32)
			enc_outputs = tf.concat(enc_outputs, 2)
			forward_state, backward_state = enc_states
		self.enc_outputs = enc_outputs
		self.forward_state = forward_state
		self.backward_state = backward_state

	def reduce_transfered_states(self):
		cellSize = self.cellSize
		forward_state = self.forward_state
		backward_state = self.backward_state

		#In the article, he build a 1 layer neural net
		with tf.variable_scope('reduce_transfered_states'):


			W_c = tf.get_variable('W_c', [2*cellSize, cellSize], dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
			W_h = tf.get_variable('W_h', [2*cellSize, cellSize], dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
			b_c = tf.get_variable('b_c', [cellSize], dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
			b_h = tf.get_variable('b_h', [cellSize], dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))


			concat_c = tf.concat(axis=1, values=[forward_state.c, backward_state.c])
			concat_h = tf.concat(axis=1, values=[forward_state.h, backward_state.h])

			new_c = tf.matmul(concat_c, W_c) + b_c
			new_h = tf.matmul(concat_h, W_h) + b_h

		self.encoder_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state


	def _generate_Decoder(self, task="train"):
		
		cellSize = self.cellSize
		enc_input_length = self.tf_input_enc_seq_lengths
		enc_outputs = self.enc_outputs
		enc_state = self.encoder_state
		vocab_size = self.vocab_size
		batch_size = self.batch_size
		beam_width = self.beam_width

		if self.coverage:
			Attention_mech_chosen = BahdanauCoverageAttention
		else:
			Attention_mech_chosen = tf.contrib.seq2seq.BahdanauAttention

		print(Attention_mech_chosen)
		
		if task=="train":
			with tf.variable_scope("decoder",  reuse=tf.AUTO_REUSE):
				enc_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs, multiplier=1)
				enc_state = tf.contrib.seq2seq.tile_batch(enc_state, multiplier=1)
				enc_input_length = tf.contrib.seq2seq.tile_batch(enc_input_length, multiplier=1)
				self.decoder_cell = tf.contrib.rnn.LSTMCell(cellSize, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2), forget_bias=1.0, name="cell")

				self.attention_mechanism = Attention_mech_chosen(cellSize,
		                                                  enc_outputs,
		                                                  enc_input_length,
		                                                  normalize=False,
		                                                  name="attention_model")

				self.decoder_cell = AttentionPointerWrapper(self.decoder_cell,
		                                                          self.attention_mechanism,
		                                                          cellSize,
		                                                          alignment_history=True,
		                                                          initial_cell_state=enc_state,
		                                                          output_attention=False,
		                                                          pointer=self.pointer,
		                                                          name="attention_wrapper")

				self.initial_state_normal = self.decoder_cell.zero_state(batch_size, dtype=tf.float32).clone(cell_state=enc_state)

				self.projection_layer = tf.layers.Dense(vocab_size)
				#kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1),activation=tf.nn.softmax

				self._generate_Decoder_training()

		elif task=="test":
			with tf.variable_scope("decoder",  reuse=tf.AUTO_REUSE):
				tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs, multiplier=1)
				tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(enc_state, multiplier=1)
				tiled_sequence_length = tf.contrib.seq2seq.tile_batch(enc_input_length, multiplier=1)
				self.decoder_cell_original = tf.contrib.rnn.LSTMCell(cellSize, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2), name="cell")

				self.attention_mechanism = Attention_mech_chosen(cellSize,
		                                                  enc_outputs,
		                                                  enc_input_length,
		                                                  normalize=False,
		                                                  name="attention_model")

				self.decoder_cell = AttentionPointerWrapper(self.decoder_cell_original,
		                                                          self.attention_mechanism,
		                                                          cellSize,
		                                                          alignment_history=True,
		                                                          initial_cell_state=enc_state,
		                                                          output_attention=False,
		                                                          pointer=self.pointer,
		                                                          name="attention_wrapper")

				self.initial_state_normal = self.decoder_cell.zero_state(batch_size, dtype=tf.float32).clone(cell_state=enc_state)

				self.projection_layer = tf.layers.Dense(vocab_size)
				self._generate_Decoder_prediction()
				#kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1),activation=tf.nn.softmax
			with tf.variable_scope("decoder",  reuse=tf.AUTO_REUSE):
				tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(enc_outputs, multiplier=beam_width)
				tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(enc_state, multiplier=beam_width)
				tiled_sequence_length = tf.contrib.seq2seq.tile_batch(enc_input_length, multiplier=beam_width)

				self.attention_mechanism_beam = Attention_mech_chosen(cellSize,
	                                              tiled_encoder_outputs,
	                                              tiled_sequence_length,
	                                              normalize=False,
	                                              name="attention_model")

				self.decoder_cell = AttentionPointerWrapper(self.decoder_cell_original,
		                                                          self.attention_mechanism_beam,
		                                                          cellSize,
		                                                          initial_cell_state=tiled_encoder_final_state,
		                                                          alignment_history=True,
		                                                          output_attention=False,
		                                                          pointer=self.pointer,
		                                                          name="attention_wrapper")

				# Replicate encoder infos beam_width times
				decoder_initial_state = self.decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size * beam_width)
				self.decoder_initial_state_beam = decoder_initial_state.clone(cell_state=tiled_encoder_final_state)

				#self._generate_Decoder_prediction_beam()


	def _generate_Decoder_prediction(self):
		decoder_cell = self.decoder_cell
		projection_layer = self.projection_layer
		max_summary_length = self.tf_max_summary_length
		enc_state = self.initial_state_normal
		embedding = self.embedding_matrix
		start_token = self.start_token
		stop_token = self.stop_token
		batch_size = self.batch_size

		with tf.variable_scope("decoder_training",  reuse=tf.AUTO_REUSE):
			inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding,
	                                                                tf.fill([batch_size], start_token),
	                                                                stop_token)

			decoder_greedy = tf.contrib.seq2seq.BasicDecoder(decoder_cell, 
														 inference_helper,
														 enc_state,
														 output_layer=projection_layer)

			outputs_greedy, state_greedy, _= tf.contrib.seq2seq.dynamic_decode(decoder_greedy,
																		impute_finished=True,
																		maximum_iterations=max_summary_length)
		
		self.output_prediction_greedy = outputs_greedy.sample_id
		self.logits_prediction_greedy = outputs_greedy.rnn_output
		self.dec_states_greedy = state_greedy

	def _generate_Decoder_training(self):

		decoder_cell = self.decoder_cell
		projection_layer = self.projection_layer
		dec_input = self.emb_dec_batch
		dec_input_length = self.tf_fake_summary_length
		max_summary_length = self.tf_max_summary_length
		enc_state = self.initial_state_normal
		batch_size = self.batch_size

		with tf.variable_scope("decoder_training"):

			helper= tf.contrib.seq2seq.TrainingHelper(dec_input, dec_input_length)

			decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, 
													 helper,
													 enc_state,
													 output_layer=projection_layer)
			outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,impute_finished=True,maximum_iterations=max_summary_length)

			if self.pointer:
				final_dist = self.compute_pointer_distribution(outputs.rnn_output, final_state.alignment_history.stack(), final_state.pgen.stack())
			else:
				final_dist = outputs.rnn_output
		self.outputs_training = final_dist
		self.final_state_training = final_state

	def _generate_Decoder_prediction_beam(self):
		cellSize = self.cellSize
		decoder_cell = self.decoder_cell
		projection_layer = self.projection_layer
		max_summary_length = self.tf_max_summary_length
		enc_outputs = self.enc_outputs
		enc_state = self.encoder_state
		enc_input_length = self.tf_input_enc_seq_lengths
		embedding = self.embedding_matrix
		start_token = self.start_token
		stop_token = self.stop_token
		batch_size = self.batch_size
		beam_width = self.beam_width

		#Attention mechanism for beam search
		with tf.variable_scope("decoder_training", reuse=tf.AUTO_REUSE):
			#Generating specific sizes for all tensors for attention


			#print(decoder_initial_state_beam.c.get_shape())
			# Define a beam-search decoder
			decoder_beam = tf.contrib.seq2seq.BeamSearchDecoder(
				        cell=self.decoder_cell,
				        embedding=embedding,
				        start_tokens=tf.fill([batch_size], start_token),
				        end_token=stop_token,
				        initial_state=self.decoder_initial_state_beam,
				        beam_width=beam_width,
				        output_layer=projection_layer,
				        length_penalty_weight=0.0)
			#Beams are ordered from best to worse
			output_beam, state_beam, _ = tf.contrib.seq2seq.dynamic_decode(decoder_beam,
																		impute_finished=False,
																		maximum_iterations=max_summary_length)


			self.output_prediction_beam = output_beam.predicted_ids
			self.dec_states_beam = state_beam


	def compute_pointer_distribution(self, logits, attention, pgen):
		print(self.tf_input_enc_batch.get_shape())

		pgen_extended = tf.expand_dims(pgen, [2])

		# Is the attention * (1-pgen) for each TT
		attention_dist = tf.transpose(attention * (1-pgen_extended), perm=[1,0,2])
		#Is logits * pgen for each TT
		vocab_dist = logits * tf.transpose(pgen_extended, perm=[1,0,2])

		#Adding the extra words to the vocabulary
		new_zeros = tf.zeros((self.batch_size,tf.shape(vocab_dist)[1], self.tf_batch_max_oov))
		vocab_dist = tf.concat([vocab_dist, new_zeros],2)

		#Adding the attention distributions
		shape = tf.shape(vocab_dist)
		extented_enc_batch = tf.tile(tf.expand_dims(self.tf_input_enc_batch, [1]), [1, tf.shape(vocab_dist)[1], 1])
		i1, i2 = tf.meshgrid(tf.range(self.batch_size),
                     tf.range(tf.shape(vocab_dist)[1]), indexing="ij")
		i1 = tf.tile(i1[:, :, tf.newaxis], [1, 1, tf.shape(extented_enc_batch)[2]])
		i2 = tf.tile(i2[:, :, tf.newaxis], [1, 1, tf.shape(extented_enc_batch)[2]])
		# Create final indices
		idx = tf.stack([i1, i2, extented_enc_batch], axis=-1)


		attention_vocab = tf.scatter_nd(idx, attention_dist, shape)

		final_dist = attention_vocab + vocab_dist
		return final_dist


	def _generate_Optimisation(self):
		batch_size = self.batch_size
		output_batch = self.outputs_training
		target_batch = self.tf_target_dec_batch
		summaries_length = self.tf_input_dec_seq_lengths
		max_summary_length = self.tf_max_summary_length
		max_encoding_length = self.tf_max_text_length
		learning_rate = self.learning_rate
		gradient_clip = self.gradient_clip
		coverage_multiplier = self.coverage_multiplier

		with tf.variable_scope("optimisation"):
			#mask is of shape [batch_size, max_decoding_length]
			mask = tf.sequence_mask(summaries_length, maxlen = max_summary_length, dtype=tf.float32)
			logit = tf.identity(output_batch)
			#Loss of the sequence
			cost = tf.contrib.seq2seq.sequence_loss(logit, target_batch, mask)


			#Loss of the coverage part
			if self.coverage:
				#alignment history of size [Dec_length, batch_size, encoder_length]
				alignment_history = self.final_state_training.alignment_history.stack()
				coverage = tf.cumsum(alignment_history, axis=0, exclusive=True)
				#print(coverage.get_shape())
				#cov loss of shape [dec_length, batch_size]
				coverage_loss = tf.reduce_sum(tf.minimum(alignment_history, coverage), [2])
				masked_coverage_loss = tf.reduce_sum(tf.transpose(coverage_loss) * mask) /  tf.to_float(max_encoding_length, name='ToFloat')
				#print(cost.get_shape())
				loss = (cost + coverage_multiplier * masked_coverage_loss)/ batch_size
			else:
				loss = (cost)/ batch_size

			optimizer = tf.train.AdamOptimizer(learning_rate)



			params = tf.trainable_variables()
			gradients = tf.gradients(loss, params)
			clipped_gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)


			update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
		self.tf_loss = loss
		self.tf_update_step = update_step

if __name__ == "__main__":
	net = Seq2SeqSummarisation()
	net.train(nb_data = 20000, create_batches=False, load_from_checkpoint=True)
	#net.infer()