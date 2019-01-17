import numpy as np
import pickle

class Batch:

	def __init__(self, _id=0, batch_size = 0, input_enc = [], input_dec= [], target_dec= [], input_enc_seq= [], input_dec_seq= [], oov_words=[], input_enc_oov=[],max_oovs=0):
		self.id=_id
		self.batch_size = batch_size
		self.input_enc = input_enc
		self.input_dec = input_dec
		self.target_dec = target_dec
		self.input_enc_seq = input_enc_seq
		self.input_dec_seq = input_dec_seq
		self.oov_words = oov_words
		self.max_oovs = max_oovs
		self.input_enc_oov = input_enc_oov
		
	def save_object(self, filepath="../Data/Batches/"):
		filename = filepath + str(self.id)
		with open(filename, 'wb') as output:  # Overwrites any existing file.
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	def load_object(self, id_, filepath="../Data/Batches/"):
		filename = filepath + str(id_)
		with open(filename, 'rb') as input:
			obj = pickle.load(input)
			self.id = id_
			self.batch_size = obj.batch_size
			self.input_enc = obj.input_enc
			self.input_dec = obj.input_dec
			self.target_dec = obj.target_dec
			self.input_enc_seq = obj.input_enc_seq
			self.input_dec_seq = obj.input_dec_seq
			self.oov_words = obj.oov_words
			self.max_oovs = obj.max_oovs
			self.input_enc_oov = obj.input_enc_oov


if __name__ == "__main__":
	batch = Batch()
	batch.load_object(1, filepath="../../Training/Exp1/Data/")
	print(batch.input_dec_seq)