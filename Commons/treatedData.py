import numpy as np
import pickle
class TreatedData:

	def __init__(self, original_text=[], input_text = [], input_summary =[], original_summary = [], input_seq = [], greed_seq_num=[], greed_seq=[], beam_seq_number=[], beam_seq=[], logits=[], enc_state=[], encoder_outputs=[], keys=[], alignment_history=[], id_ = 0):
		self.original_text = original_text
		self.input_text = input_text
		self.input_seq = input_seq
		self.original_summary = original_summary
		self.input_summary = input_summary
		self.greed_seq_num = greed_seq_num
		self.greed_seq = greed_seq
		self.beam_seq_number = beam_seq_number
		self.enc_state = enc_state
		self.beam_seq = beam_seq
		self.logits = logits
		self.id = id_
		self.alignment_history = alignment_history
		self.encoder_outputs = encoder_outputs
		self.keys = keys

	def saveToFileText(self, filepath = "../Output/Text/"):
		filename = filepath + str(self.id) + ".txt"
		with open(filename, 'w') as writer:
			writer.write("Original Text : \n")
			writer.write(str(self.original_text) + "\n")
			writer.write("Input Text : \n")
			writer.write(str(self.input_text) + "\n")
			writer.write("Original summary : \n")
			writer.write(str(self.original_summary) + "\n")
			writer.write("Input summary : \n")
			writer.write(str(self.input_summary) + "\n")
			writer.write("Greedy summary : \n")
			writer.write(str(self.greed_seq) + "\n")
			writer.write("Beam summary : \n")
			writer.write(str(self.beam_seq) + "\n")

	def save_object(self, filepath="../Output/Obj/"):
		filename = filepath + str(self.id)
		with open(filename, 'wb') as output:  # Overwrites any existing file.
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	def load_object(self, id_, filepath="../Output/Obj/"):
		filename = filepath + str(id_)
		with open(filename, 'rb') as input:
			return pickle.load(input)