import tensorflow as tf
import numpy as np
import struct
from tensorflow.core.example import example_pb2
from batch import *
from treatedData import *

ABST_READ_BEGINNING = '<s>'
ABST_READ_END = "</s>"

START_DECODE = "<s>"
STOP_DECODE = "</s>"
FILL = "<FILL>"

#Structure to hold our examples
class Example_raw:
	def __init__(self, text=[], abstract=[]):
		self.text = np.array(text)
		self.abstract = np.array(abstract)

class Data:

	def GenerateBatchesOnDisk(self, batch_size, vocab, max_text_length=400, max_abstract_length=100, max_data =250000, reading_file="train", writting_path="../Data/Batches", pointer=False):
		nb_batches = max_data // batch_size
		fullpath = self.path + reading_file + ".bin"

		#input_enc_batches = vocab.TranslateBatches(input_enc_batches)
		#input_dec_batches = vocab.TranslateBatches(input_dec_batches)
		#target_dec_batches = vocab.TranslateBatches(target_dec_batches)
		id_ = 0
		with open(fullpath, 'rb') as reader:
			b_length = reader.read(8)
			for i in range(nb_batches):
				print("Generating batch {}/{}".format(id_, nb_batches))
				examples = []
				count = 0
				while b_length and count < batch_size:
					
					length = struct.unpack('q', b_length)[0]
					str_example = struct.unpack('%ds' % length, reader.read(length))[0]
					e = example_pb2.Example.FromString(str_example)

					article = e.features.feature['article'].bytes_list.value[0].decode("utf-8").split()
					abstract = e.features.feature['abstract'].bytes_list.value[0].decode("utf-8").split()
					
					splitted_abstract = self.split_abstract(abstract)
					if article != [] and abstract != []:
						examples.append(Example_raw(article, splitted_abstract))
						count += 1
					
					b_length = reader.read(8)


				input_enc_batches, input_dec_batches, target_dec_batches, input_enc_seq_lengths, input_dec_seq_lengths = self.generate_data_batches([examples], batch_size, max_text_length, max_abstract_length)
				
				if pointer:
					translated_batches, oov_words, max_oovs = vocab.TranslateTextBatchesWithOOV(input_enc_batches)
					input_enc_batches = vocab.TranslateBatches(input_enc_batches)
					input_dec_batches = vocab.TranslateBatches(input_dec_batches)
					input_enc_oov = translated_batches
					target_dec_batches = vocab.TranslateSummaryBatchesWithOOV(target_dec_batches, oov_words)
				else:
					input_enc_batches = vocab.TranslateBatches(input_enc_batches)
					input_dec_batches = vocab.TranslateBatches(input_dec_batches)
					target_dec_batches = vocab.TranslateBatches(target_dec_batches)
					input_enc_oov = [[]]
					max_oovs = [0]
					oov_words=[0]
				batch = Batch(_id=id_, batch_size = batch_size, input_enc = input_enc_batches[0], input_dec= input_dec_batches[0], target_dec= target_dec_batches[0], input_enc_seq= input_enc_seq_lengths[0], input_dec_seq= input_dec_seq_lengths[0], oov_words=oov_words[0], max_oovs=max_oovs[0])
				batch.save_object(filepath=writting_path)


				id_ += 1
		return nb_batches

	def __init__(self, path = "../Data/finished_files/", data=[]):
		self.path = path
		self.data = []

	def read_preprocessed(self, filename, maxi=0):
		examples = []
		fullpath = self.path + filename + ".bin"
		count = -1
		with open(fullpath, 'rb') as reader:
			b_length = reader.read(8)
			while b_length and count < maxi:
				length = struct.unpack('q', b_length)[0]
				str_example = struct.unpack('%ds' % length, reader.read(length))[0]
				e = example_pb2.Example.FromString(str_example)

				article = e.features.feature['article'].bytes_list.value[0].decode("utf-8").split()
				abstract = e.features.feature['abstract'].bytes_list.value[0].decode("utf-8").split()
				
				splitted_abstract = self.split_abstract(abstract)
				if article != [] and abstract != []:
					examples.append(Example_raw(article, splitted_abstract))
					if maxi > 0 : count+=1

				del article
				del abstract
				del splitted_abstract
				
				b_length = reader.read(8)

		number = len(examples)
		#print(number)
		return examples, number

	def data_pipeline(self, batch_size, max_text_length = 400, max_abstract_length = 100, nb_example=0, filename="train"):
		examples, number = self.read_preprocessed(filename, maxi = nb_example)
		batches, nb_batches = self.createBatches(examples, batch_size)

		self.max_abstract_length = max_abstract_length
		input_enc_batches, input_dec_batches, target_dec_batches, input_enc_seq_lengths, input_dec_seq_lengths = self.generate_data_batches(batches, batch_size, max_text_length, max_abstract_length)
		return input_enc_batches, input_dec_batches, target_dec_batches, input_enc_seq_lengths, input_dec_seq_lengths, nb_batches

	def split_abstract(self, abstract):
		output = []

		while True:
			try:
				start = abstract.index(ABST_READ_BEGINNING)
				end = abstract.index(ABST_READ_END)
				output += ["."] + abstract[start+1:end]
				abstract = abstract[end+1:]
			except:
				break
		return output

	def generate_abstract_data(self, abstract, max_abstract_length):
		
		if len(abstract) > max_abstract_length:
			abstract_input = abstract[1:max_abstract_length-1]
			abstract_target = abstract[1:max_abstract_length]
		else:
			abstract_input = abstract[1:len(abstract)-1]
			abstract_target = abstract[1:len(abstract)]

		abstract_input = [START_DECODE] +  [elem for elem in abstract_input] + [STOP_DECODE]
		abstract_target = [elem for elem in abstract_target] + [STOP_DECODE]
		#print(abstract_target)
		length = len(abstract_input)
		#Padding Operation
		abstract_input = [elem for elem in abstract_input] + [FILL] * (max_abstract_length - length)
		abstract_target = [elem for elem in abstract_target] +  [FILL] * (max_abstract_length - length)

		#print(abstract_target)
		return abstract_input, abstract_target, length

	def generate_text_data(self, text, max_text_length):

		if len(text) > max_text_length:
			text = text[:max_text_length]
		else:
			text = text[:]

		length = len(text)
		#Padding Operation
		text = [elem for elem in text] + [FILL] * (max_text_length - length)

		return text, length
	
	def createBatches(self, examples, size):
		number = len(examples) // size
		batches = []
		#print(number)
		for i in range(0,number):
			batches.append(examples[i*size:(i+1)*size])
		#batches.append(examples[number*size:])
		
		return batches, number

	def generate_data_batches(self, batches, size, max_text_length, max_abstract_length):
		input_enc_batches = []
		input_dec_batches = []
		target_dec_batches  = []
		input_enc_seq_lengths = []
		input_dec_seq_lengths = []
		#print(batches)

		for batch in batches:
			temp_input_enc_batches = []
			temp_input_dec_batches = []
			temp_target_dec_batches  = []
			temp_input_enc_seq_lengths = []
			temp_input_dec_seq_lengths = []
			for example in batch:
				abstract_input, abstract_target, length = self.generate_abstract_data(example.abstract, max_abstract_length)
				temp_input_dec_batches.append(abstract_input)
				temp_target_dec_batches.append(abstract_target)
				temp_input_dec_seq_lengths.append(length)

				text, length = self.generate_text_data(example.text, max_text_length)
				temp_input_enc_batches.append(text)
				temp_input_enc_seq_lengths.append(length)

			input_enc_batches.append(temp_input_enc_batches)
			input_dec_batches.append(temp_input_dec_batches)
			target_dec_batches.append(temp_target_dec_batches)
			input_enc_seq_lengths.append(temp_input_enc_seq_lengths)
			input_dec_seq_lengths.append(temp_input_dec_seq_lengths)
		#print(target_dec_batches)
		#print(np.array(target_dec_batches))

		return np.array(input_enc_batches), np.array(input_dec_batches), np.array(target_dec_batches), np.array(input_enc_seq_lengths), np.array(input_dec_seq_lengths)

	def getMaxSummaryLength(self, examples):

		maxi = 0
		for elem in examples:
			if len(elem.abstract) > maxi:
				maxi = len(elem.abstract)
		return maxi

	def getMaxTextLength(self, examples):

		maxi = 0
		for elem in examples:
			if len(elem.text) > maxi:
				maxi = len(elem.text)
		return maxi

	def getAvgSummaryLength(self, examples):

		avg = 0
		for elem in examples:
				avg +=len(elem.abstract)
		avg /= len(examples)
		return avg

	def getAvgTextLength(self, examples):

		avg = 0
		for elem in examples:
				avg += len(elem.text)
		avg /= len(examples)
		return avg

if __name__ == "__main__":
	data = Data("../Data/finished_files/")
	#data.GenerateBatchesOnDisk(10, , max_text_length=400, max_abstract_length=100, max_data =100, reading_file="train", writting_path="../Data/Batches")
	#batch = Batch()
	#batch.load_object(119)
	#print(batch.input_enc)
	#print(batch.input_dec)
	#print(batch.target_dec)
	#print(batch.input_enc_seq)
	#print(batch.input_dec_seq)
	#_ = data.data_pipeline(	10, number=100, filename="train")
	data.read_preprocessed(filename="train", maxi=2)