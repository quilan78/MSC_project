import json
import pickle

class LRP_output:

	def __init__(self, original_text=[], input_text=[], input_summary=[], original_summary=[], greedy_summary=[], beam_summary=[], greedy_LRP_encoder_forward= [], greedy_LRP_encoder_backward= [], greedy_LRP_decoder = [], greedy_attention=[], beam_LRP=[], beam_attention=[], greedy_logits=[], beam_logits=[], beam_LRP_encoder_forward=[], beam_LRP_encoder_backward=[], beam_LRP_decoder=[]):

		#Data info & prediction
		self.original_text = original_text
		self.input_text = input_text
		self.original_summary = original_summary
		self.input_summary = input_summary
		self.greedy_summary = greedy_summary
		self.beam_summary = beam_summary

		#logits
		self.greedy_logits = greedy_logits
		self.beam_logits = beam_logits

		#LRP
		self.greedy_LRP_encoder_forward = greedy_LRP_encoder_forward
		self.greedy_LRP_encoder_backward = greedy_LRP_encoder_backward
		self.greedy_LRP_decoder = greedy_LRP_decoder
		self.beam_LRP_encoder_forward = beam_LRP_encoder_forward
		self.beam_LRP_encoder_backward = beam_LRP_encoder_backward
		self.beam_LRP_decoder = beam_LRP_decoder

		#Attention
		self.greedy_attention = greedy_attention
		self.beam_attention = beam_attention



	def load_json(self, id_, filepath="../Output/Obj/"):
		filename = filepath + str(id_) + ".json"
		with open(filename, "rb") as f:
			data = json.load(f)
			self.original_text = data["original_text"]
			self.input_text = data["input_text"]
			self.original_summary = data["original_summary"]
			self.input_summary = data["input_summary"]
			self.greedy_summary = data["greedy_summary"]
			self.beam_summary = data["beam_summary"]

			#logits
			self.greedy_logits = data["greedy_logits"]
			self.beam_logits = data["beam_logits"]

			#LRP
			self.greedy_LRP_encoder_forward = data["greedy_LRP_encoder_forward"]
			self.greedy_LRP_encoder_backward = data["greedy_LRP_encoder_backward"]
			self.greedy_LRP_decoder = data["greedy_LRP_decoder"]
			self.beam_LRP_encoder_forward = data["beam_LRP_encoder_forward"]
			self.beam_LRP_encoder_backward = data["beam_LRP_encoder_backward"]
			self.beam_LRP_decoder = data["beam_LRP_decoder"]

			#Attention
			self.greedy_attention = data["greedy_attention"]
			self.beam_attention = data["beam_attention"]

if __name__ == "__main__":
	lrp_output = LRP_output()
	print(lrp_output)
	with open('../Visualiser/JSON/'+str(1)+".json", 'w') as outfile:
		json.dump(lrp_output.__dict__, outfile)