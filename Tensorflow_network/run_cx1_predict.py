from rnn_wraper import *
import sys



rnn_wrapper = RNN_wrapper()

data_path = sys.argv[1]
checkpoint_path = sys.argv[2] + "model/model1.ckpt"
save_path = sys.argv[2] + "Output/"
rnn_wrapper.exp1_test(data_path=data_path, checkpoint_path = checkpoint_path, save_path =save_path)

