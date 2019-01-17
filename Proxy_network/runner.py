from modelManager import *



import sys




def run_LRP(model_path, vocab_path, obj_path):
	
	manager = ModelManager()
	manager.loadModel(model_path, max_encoding_length=400, max_decoding_length=50, vocab_path=vocab_path)
	#manager.loadModel("../../Training/Exp1/Model/model1.ckpt", max_encoding_length=400, max_decoding_length=200)
	for i in range(1,17):
		manager.generateLRP_output(i, path=obj_path,json_path=json_path)

vocab_path = sys.argv[1]
model_path = sys.argv[2] + "model/model1.ckpt"
obj_path = sys.argv[2] + "Output/Obj/"
json_path = sys.argv[3] + "JSON/"
run_LRP(model_path, vocab_path, obj_path)

