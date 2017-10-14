from RNN_models import *
import sys
import numpy as np
import time
from utils import load_pMNIST_data, load_synthetic_data, load_PTB_data

# Global variables
args = sys.argv
data_path = '/data/mha001/IJCNN_2017_paper/data/'
seed = int(args[1])
rng = np.random.RandomState(seed)
task = args[2]
sequence_length = int(args[3])
model_type = args[4]
modules = int(args[5])
hidden_units_per_module = int(args[6])
n_r= int(args[7])
learning_rate = np.float32(args[8])
batch_size=int(args[9])
activation=args[10]
n_epoch = 100000 
test_frequency = 200
f_out = 'linear'

if task in ['MNIST', 'pMNIST']:
    n_in = 1
    n_out = 10
    n_epoch = 100
    if batch_size == 1:
        test_frequency = 5000
    else:  
        test_frequency = 20
elif task in ['copying', 'copyingVariable']:
    n_in = n_out = 10
    f_out = 'softmax'
elif task in ['PTB', 'PTB_5']:
    n_in = n_out = 49
    if batch_size == 1:
        test_frequency = 4000
    else:   
        test_frequency = 400
    n_epoch = 20
else:
    n_in = 2
    n_out = 1

if task != 'MNIST':
    string = 'seed-%i__task-%s__T-%i__m-%i__nh-%i__nr-%i__model-%s__lr-%.6f__bs-%i__f-%s' % \
             (seed, task, sequence_length, modules, hidden_units_per_module, n_r, model_type, learning_rate, batch_size, activation)
else:
    string = 'seed-%i__task-%s__T-%i__m-%i__nh-%i__nr-%i__model-%s__lr-%.6f__bs-%i' % \
             (seed, task, sequence_length, modules, hidden_units_per_module, n_r, model_type, learning_rate, batch_size)

print('This is for testing')
load_progress = data_path + 'saved_models/FEB17_' + string
save_progress = data_path + 'saved_models/FEB17_' + string  
print(string)
sys.stdout.flush()

# Loading the data
if task == 'MNIST':
    train_set, valid_set, test_data = load_pMNIST_data(rng, data_path, perm=False)
elif task == 'pMNIST':
    train_set, valid_set, test_data = load_pMNIST_data(rng, data_path, perm=True)
elif task == 'PTB':
    train_set, valid_set, test_data = load_PTB_data(data_path)
elif task == 'PTB_5':
    train_set, valid_set, test_data = load_PTB_data(data_path, 5)
else:
    train_set, valid_set = load_synthetic_data(data_path, task, sequence_length)

# Defining the model
start = time.time()
rnn = eval(model_type)(rng, n_in=n_in, n_out=n_out, m=modules, n_r=n_r, hupm=hidden_units_per_module, task_type=task, f_out=eval(f_out), f_act=eval(activation))
end = time.time() - start
print('It took %.6f seconds to built the model' % end)
# Training the model
rnn.optimiser.test(task, sequence_length, valid_set, load_progress)
# test_cost, test_accuracy = np.mean(results, axis=0)

