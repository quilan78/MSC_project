import tensorflow as tf
from seq2SeqSummarisation import *
class RNN_wrapper:


	def exp1_rnn(self):
		net = Seq2SeqSummarisation(cellSize = 256, 
									batch_size = 16, 
									max_encoding_length = 400,
									max_decoding_length = 200, 
									vocab_size = 50000, 
									embedding_size = 128, 
									learning_rate = 0.00015, 
									learning_decay = 1, 
									minimum_rate = 0.01, 
									nbre_epoch = 6, 
									display_batch_freq = 300, 
									gradient_clip = 2, 
									beam_width = 10, 
									save_frequency = 1)
		return net
	def exp1_train(self, create_batches = False, load_from_checkpoint=True, writting_path_batches="../../Training/Exp1/Data/", checkpoint_path = "../../Training/Exp1/Model/model1.ckpt", tensorboard_path="../../Training/Exp1/tensorboard/",  data_path="../../Data/finished_files/"):
		net = self.exp1_rnn()
		net.train(nb_data = 250000, create_batches=create_batches, writting_path_batches=writting_path_batches, load_from_checkpoint=load_from_checkpoint, checkpoint_path=checkpoint_path, tensorboard_path=tensorboard_path,  data_path=data_path)

	def exp1_test(self, data_path="../../Data/finished_files/", checkpoint_path = "../../Training/Exp1/model/model1.ckpt", save_path ="../../Training/Exp1/Output/"):
		net = self.exp1_rnn()
		net.infer(nb_data = 20,checkpoint_path =checkpoint_path, save_path =save_path, data_path=data_path)

	def exp1_exp(self, important, id_, checkpoint_path = "../../Experiment/Model/model1.ckpt", save_path ="../../Experiment/Results/", data_path="../../Data/finished_files/",  exp_data_path="../../Experiment/ModifiedTexts/"):
		net = self.exp1_rnn()
		if important:
			save_path += "Important/" + str(id_) + "/"
			exp_data_path += "Important/" + str(id_) + "/"
		else:
			save_path += "Unimportant/" + str(id_) + "/"
			exp_data_path += "Unimportant/" + str(id_) + "/"
		net.experiment(nb_data = 20,checkpoint_path =checkpoint_path, save_path =save_path, data_path=data_path, exp_data_path=exp_data_path)

	def exp1_exp_rand(self, id_, counter, checkpoint_path = "../../Experiment/Model/model1.ckpt", save_path ="../../Experiment/Results/", data_path="../../Data/finished_files/",  exp_data_path="../../Experiment/ModifiedTexts/"):
		net = self.exp1_rnn()
		save_path += "Random/" + str(counter) + "/" + str(id_) + "/"
		exp_data_path += "Random/" + str(counter) + "/" + str(id_) + "/"
		net.experiment(nb_data = 20,checkpoint_path =checkpoint_path, save_path =save_path, data_path=data_path, exp_data_path=exp_data_path)


	def exp2_rnn(self):
		net = Seq2SeqSummarisation(cellSize = 128, 
									batch_size = 10, 
									max_encoding_length = 200,
									max_decoding_length = 50, 
									vocab_size = 50000, 
									embedding_size = 64, 
									learning_rate = 0.04, 
									learning_decay = 1,
									minimum_rate = 0.000001, 
									nbre_epoch = 1, 
									display_batch_freq = 2, 
									gradient_clip = 5, 
									beam_width = 10, 
									save_frequency = 1,
									coverage=True,
									pointer=True)
		return net

	def exp2_train(self, create_batches = False, load_from_checkpoint=True, writting_path_batches="../../Training/Exp2/Data/", checkpoint_path = "../../Training/Exp2/Model/model1.ckpt", tensorboard_path="../../Training/Exp2/tensorboard/", data_path="../../Data/finished_files/"):
		net = self.exp2_rnn()
		net.train(nb_data = 100, create_batches=create_batches, writting_path_batches=writting_path_batches, load_from_checkpoint=load_from_checkpoint, checkpoint_path=checkpoint_path, tensorboard_path=tensorboard_path, data_path=data_path)

	def exp2_test(self, data_path="../../Data/finished_files/", checkpoint_path = "../../Training/Exp2/Model/model1.ckpt", save_path ="../../Training/Exp2/Output/"):
		net = self.exp2_rnn()
		net.infer(nb_data = 20, checkpoint_path =checkpoint_path, save_path =save_path, data_path=data_path)

	def exp3_rnn(self):
		net = Seq2SeqSummarisation(cellSize = 256, 
									batch_size = 16, 
									max_encoding_length = 400,
									max_decoding_length = 200, 
									vocab_size = 50000, 
									embedding_size = 128, 
									learning_rate = 0.00015, 
									learning_decay = 1, 
									minimum_rate = 0.01, 
									nbre_epoch = 6, 
									display_batch_freq = 300, 
									gradient_clip = 2, 
									beam_width = 10, 
									save_frequency = 1,
									coverage=True,
									pointer=True)
		return net

	def exp3_train(self, create_batches = False, load_from_checkpoint=True, writting_path_batches="../../Training/Exp3/Data/", checkpoint_path = "../../Training/Exp3/Model/model1.ckpt", tensorboard_path="../../Training/Exp3/tensorboard/", data_path="../../Data/finished_files/"):
		net = self.exp3_rnn()
		net.train(nb_data = 250000, create_batches=create_batches, writting_path_batches=writting_path_batches, load_from_checkpoint=load_from_checkpoint, checkpoint_path=checkpoint_path, tensorboard_path=tensorboard_path, data_path=data_path)

	def exp3_test(self, data_path="../../Data/finished_files/", checkpoint_path = "../../Training/Exp3/Model/model1.ckpt", save_path ="../../Training/Exp3/Output/"):
		net = self.exp3_rnn()
		net.infer(nb_data = 20, checkpoint_path =checkpoint_path, save_path =save_path, data_path=data_path)

	def see_variable(self, link):
		net = self.exp2_rnn()
		nbre_epoch, learning_rate, learning_decay, minimum_rate, nb_batches, tf_input_enc_batch, tf_input_dec_batch, tf_target_dec_batch, tf_input_enc_seq_lengths, tf_input_dec_seq_lengths, tf_max_summary_length, tf_learning_rate, tf_loss, tf_update_step, merged, saver = net.init_graph_and_data(task="train", nb_data = 10000, create_batches=False, writting_path_batches="")
		#Save the trained model
		with tf.Session() as sess:
			saver.restore(sess, link)
			variables = tf.trainable_variables()
			print(variables)

if __name__=="__main__":
	rnn_wrapper = RNN_wrapper()
	#rnn_wrapper.exp1_train(create_batches = False, load_from_checkpoint=False)
	#rnn_wrapper.exp1_test(data_path="../../Data/finished_files/", checkpoint_path = "../../Experiment/Model/model1.ckpt", save_path ="../../Experiment/Test/")
	#for counter in range(1,11):
	#	for i in range(1, 13):
	#		tf.reset_default_graph()
	#		rnn_wrapper.exp1_exp_rand(i, counter)
	for i in range(1, 13):
		print("Text number {}/12".format(i))
		tf.reset_default_graph()
		rnn_wrapper.exp1_exp(False, i)
		tf.reset_default_graph()
		rnn_wrapper.exp1_exp(True, i)
	#rnn_wrapper.see_variable("../../Training/Exp2/Model/model1.ckpt")