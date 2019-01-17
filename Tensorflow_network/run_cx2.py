from rnn_wraper import *
import sys



rnn_wrapper = RNN_wrapper()

data_path = sys.argv[1]
writting_path_batches=sys.argv[2]
checkpoint_path = sys.argv[3] + "model/model1.ckpt"
tensorboard_path = sys.argv[3] + "tensorboard/"
rnn_wrapper.exp3_train(create_batches = True, load_from_checkpoint=False, data_path=data_path, writting_path_batches=writting_path_batches,checkpoint_path=checkpoint_path, tensorboard_path=tensorboard_path)

