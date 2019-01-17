import tensorflow as tf
import numpy as np
import json
from read_data import *
UNKNOWN = "<UNKNOWN>"

class Vocab:
	def __init__(self, path="../../Data/finished_files/"):
			self.path = path

	def LoadVocab(self, filename="vocab", max_size=-1):
			vocab_word_to_id = {}
			vocab_id_to_words = {}
			count = 0


			unknown_word = UNKNOWN
			vocab_id_to_words[count] = unknown_word
			vocab_word_to_id[unknown_word] = count
			count += 1


			fill = FILL
			vocab_id_to_words[count] = fill
			vocab_word_to_id[fill] = count
			count += 1

			start_decode = START_DECODE
			vocab_id_to_words[count] = start_decode
			vocab_word_to_id[start_decode] = count
			count += 1

			stop_decode = STOP_DECODE
			vocab_id_to_words[count] = stop_decode
			vocab_word_to_id[stop_decode] = count
			count += 1

			self.start_decode_token = start_decode
			self.stop_decode_token = stop_decode
			self.start_decode_id = vocab_word_to_id[start_decode]
			self.stop_decode_id = vocab_word_to_id[stop_decode]

			with open(self.path+filename, 'r',  encoding="utf8") as file:
				for line in file:
					word = line.split()[0]
					vocab_word_to_id[word] = count
					vocab_id_to_words[count] = word

					count += 1

					if count == max_size: 
						break
			
			self.vocab_size = count
			self.vocab_word_to_id = vocab_word_to_id
			self.vocab_id_to_words = vocab_id_to_words

			return count

	def TranslateTextBatchesWithOOV(self, batches):
		vocab_word_to_id = self.vocab_word_to_id
		oov_words_batches = []
		translated = []
		max_oov_batches = []

		for i in range(len(batches)):
			articles, oov_words, max_oov = self.TranslateBatchArticleWithOOV(batches[i])
			translated.append(articles)
			oov_words_batches.append(oov_words)
			max_oov_batches.append(max_oov)
		return np.array(translated, dtype=np.int32), oov_words_batches, max_oov_batches


	def TranslateSummaryBatchesWithOOV(self, batches, oov_words_batches):
		vocab_word_to_id = self.vocab_word_to_id
		translated = []


		for i in range(len(batches)):
			summaries = self.TranslateBatchSummaryWithOOV(batches[i], oov_words_batches[i])
			translated.append(summaries)

		return np.array(translated, dtype=np.int32)

	def TranslateBatchArticleWithOOV(self, articles):
			oov_words = []
			vocab_word_to_id = self.vocab_word_to_id
			translated = np.zeros(articles.shape)
			for i in range(len(articles)):
				oov_words.append([])
				for j in range(len(articles[0])):
					if articles[i][j] in vocab_word_to_id:
						translated[i,j] = vocab_word_to_id[articles[i][j]] 
					else:
						if articles[i][j] not in oov_words[i]:
							oov_words[i].append(articles[i][j])
						id_ = oov_words[i].index(articles[i][j])
						translated[i,j] =  self.vocab_size + id_
			return translated, oov_words, np.max([len(x) for x in oov_words])


	def TranslateBatchSummaryWithOOV(self, summary, oov_words):
			vocab_word_to_id = self.vocab_word_to_id
			translated = np.zeros(summary.shape)
			for i in range(len(summary)):
				for j in range(len(summary[0])):
					if summary[i][j] in vocab_word_to_id:
						translated[i,j] = vocab_word_to_id[summary[i][j]] 
					else:
						if summary[i][j] in oov_words[i]:
							id_ = oov_words[i].index(summary[i][j]) + self.vocab_size
						else:
							id_ = vocab_word_to_id[UNKNOWN]
						translated[i,j] = id_
			return translated


	def TranslateBatches(self, batches):
			vocab_word_to_id = self.vocab_word_to_id
			translated = np.zeros(batches.shape)
			for i in range(len(batches)):
				for j in range(len(batches[0])):
					for k in range(len(batches[0][0])):
						if batches[i][j][k] in vocab_word_to_id:
							translated[i][j][k] = vocab_word_to_id[batches[i][j][k]]
						else:
							translated[i][j][k] = vocab_word_to_id[UNKNOWN]
			return translated

	def TransalteAnswer(self, sentence):
			vocab_id_to_words = self.vocab_id_to_words
			translated = []
			for i in range(len(sentence)):
					if sentence[i] in vocab_id_to_words:
						translated.append(vocab_id_to_words[sentence[i]])
					else:
						translated.append([UNKNOWN])
			return np.array(translated)
			
	def TransalteSentence(self, sentence):
			vocab_word_to_id = self.vocab_word_to_id
			translated = []
			for i in range(len(sentence)):
					if sentence[i] in vocab_word_to_id:
						translated.append(vocab_word_to_id[sentence[i]])
					else:
						translated.append(vocab_word_to_id[UNKNOWN])
			return np.array(translated)

if __name__ == "__main__":
	vocab = Vocab()
	vocab.LoadVocab()
	with open("vocab.json", 'w') as outfile:
		json.dump(vocab.vocab_word_to_id, outfile)