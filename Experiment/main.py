import numpy as np
import sys
sys.path.append('../Commons/')
from LRP_output import *
from random import randint

def main(id_, path, important,write_path):
	data = LRP_output()
	data.load_json(id_, filepath=path)

	if important:
			write_path += "Important/"
	else:
			write_path += "Unimportant/"
	for i in range(10):
		newData = LRP_output()

		full_lrp = np.array(data.greedy_LRP_encoder_forward) + np.array(data.greedy_LRP_encoder_backward)

		#avg = averageLRP(full_lrp.copy())
		avg = average_pos_LRP(full_lrp.copy(), 0.5)
		#avg = average_weighted_absolute_lrp(full_lrp.copy(), 0.5)
		if important:
			newText = deleteImportantWords(i, data.input_text.copy(), avg.copy())
		else:
			newText = deleteUnImportantWords(i, data.input_text.copy(), avg.copy())
		print(newText.count("<UNKNOWN>"))
		newData.original_text = data.original_text
		newData.input_text = newText
		newData.original_summary = data.original_summary
		newData.input_summary = data.input_summary
		newData.greedy_summary = data.greedy_summary
		newData.beam_summary = data.beam_summary

		

		with open(write_path+str(id_)+"/"+str(i)+".json", 'w') as outfile:
			json.dump(newData.__dict__, outfile)
		

def main_random(id_, counter, path, write_path):
	data = LRP_output()
	data.load_json(id_, filepath=path)

	for i in range(10):
		newData = LRP_output()

		full_lrp = np.array(data.greedy_LRP_encoder_forward) + np.array(data.greedy_LRP_encoder_backward)

		avg = averageLRP(full_lrp)
		newText = deleteRandomWords(i, data.input_text.copy(), avg.copy())
		print(newText.count("<UNKNOWN>"))
		newData.original_text = data.original_text
		newData.input_text = newText
		newData.original_summary = data.original_summary
		newData.input_summary = data.input_summary
		newData.greedy_summary = data.greedy_summary
		newData.beam_summary = data.beam_summary

		

		with open(write_path+str(counter)+"/"+str(id_)+"/"+str(i)+".json", 'w') as outfile:
			json.dump(newData.__dict__, outfile)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def average_weighted_absolute_lrp(lrp, alpha):
	avg_lrp = np.zeros(len(lrp[0]))
	for word_lrp in lrp:
		word_lrp = normalize(word_lrp)
		for i in range(len(word_lrp)):
			if word_lrp[i] < 0 :
				avg_lrp[i] -= alpha * word_lrp[i]
			else:
				avg_lrp[i] += word_lrp[i]
		

	avg_lrp /= len(lrp)
	return avg_lrp

def deleteRandomWords(percent, text,lrp):
	nbre = int(0.01*percent * len(text))
	rand = []
	for i in range(nbre):

		index = randint(0,len(lrp)-1)
		while index in rand:
			index = randint(0,len(lrp)-1)
		rand.append(index)
		text[index] = "<UNKNOWN>"
		lrp[index] = 0
	return text


def average_pos_LRP(lrp, alpha):
	avg_lrp = np.zeros(len(lrp[0]))
	for word_lrp in lrp:
		#word_lrp = np.abs(word_lrp)
		for i in range(len(word_lrp)):
			if word_lrp[i] < 0 :
				word_lrp[i] = - alpha * word_lrp[i]
			else:
				word_lrp[i] = word_lrp[i]
		current_order = np.zeros(len(lrp[0]))
		for i in range(len(word_lrp)):
			index = np.argmax(word_lrp)
			current_order[index] = len(word_lrp)-i
			word_lrp[index] = np.min(lrp)-1
		avg_lrp += current_order

	avg_lrp /= len(lrp)
	return avg_lrp

def averageLRP(lrp): 
	avg_lrp = np.zeros(len(lrp[0]))
	for word_lrp in lrp:
		avg_lrp += np.abs(np.array(word_lrp))

	avg_lrp /= len(lrp)
	return avg_lrp

def deleteImportantWords(percent, text,lrp):
	nbre = int(0.01*percent * len(text))
	for i in range(nbre):
		index = np.argmax(lrp)
		text[index] = "<UNKNOWN>"
		lrp[index] = 0
	return text

def deleteUnImportantWords(percent, text,lrp):
	nbre = int(0.01*percent * len(text))
	for i in range(nbre):
		index = np.argmin(lrp)
		text[index] = "<UNKNOWN>"
		lrp[index] = np.max(lrp)
	return text

if __name__ == "__main__":
	for i in range(1,13):
		main(i, "../../Experiment/JSON/", True, "../../Experiment/ModifiedTexts/")
		main(i, "../../Experiment/JSON/", False, "../../Experiment/ModifiedTexts/")

