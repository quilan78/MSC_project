import sys
sys.path.append('../Commons/')
from LRP_output import *
from treatedData import *
import numpy as np

def main():
	data = TreatedData()
	for i in range(1, 13):
		print(i)
		text_results = {}
		important = {}
		for j in range(10):
			data = data.load_object(j, filepath="../../Experiment/Results/Important/"+str(i)+"/Obj/")
			important["text_"+str(j)] = data.input_text
			important["summary_"+str(j)] = data.greed_seq.tolist()
		unimportant = {}
		for j in range(10):
			data = data.load_object(j, filepath="../../Experiment/Results/Unimportant/"+str(i)+"/Obj/")
			unimportant["text_"+str(j)] = data.input_text
			unimportant["summary_"+str(j)] = data.greed_seq.tolist()

		"""random_list = []
		for j in range(1,8):
			random = {}
			for k in range(10):
				data = data.load_object(k, filepath="../../Experiment/Results/Random/"+str(j)+"/"+str(i)+"/Obj/")
				
				random["text_"+str(k)] = data.input_text
				random["summary_"+str(k)] = data.greed_seq.tolist()
			random_list.append(random)
		text_results["random"] = random"""
		text_results["important"] = important
		text_results["unimportant"] = unimportant


		with open("../../Experiment/Results/JSON/exp_"+str(i)+".json", 'w') as outfile:
			json.dump(text_results, outfile)

if __name__ == "__main__":
	main()