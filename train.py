# Author: Zakaria Mhammedi
# The University of Melbourne and Data61 (2016 - 2017)

from RNN_models import *
import sys
import numpy as np
import time
from utils import load_pMNIST_data, load_synthetic_data, load_PTB_data

# Global variables
args = sys.argv
seed = int(args[1])
rng = np.random.RandomState(seed)
task = args[2]
sequence_length = int(args[3])
model_type = args[4]
hidden_units = int(args[5])
n_reflections = int(args[6])
learning_rate = np.float32(args[7])
batch_size=int(args[8])
activation=args[9]
data_path = args[10] #'/data/mha001/IJCNN_2017_paper/data/'

n_epoch = 100000 
valid_frequency = 200
f_out = 'linear'

if task in ['MNIST', 'pMNIST']:
    n_in = 1
    n_out = 10
    n_epoch = 100
    if batch_size == 1:
        valid_frequency = 5000
    else:  
        valid_frequency = 20
elif task in ['copying', 'copyingVariable']:
    n_in = n_out = 10
    f_out = 'softmax'
elif task in ['PTB', 'PTB_5']:
    n_in = n_out = 49
    if batch_size == 1:
        valid_frequency = 4000
    else:   
        valid_frequency = 400
    if task == 'PTB_5':
        n_epoch = 20
    else:
        n_epoch = 20
else:
    n_in = 2
    n_out = 1

if task != 'MNIST':
    string = 'seed-%i__task-%s__T-%i__nh-%i__m-%i__model-%s__lr-%.6f__bs-%i__f-%s' % \
             (seed, task, sequence_length, hidden_units, n_reflections, model_type, learning_rate, batch_size, activation)
else:
    string = 'seed-%i__task-%s__T-%i__m-%i__nh-%i__model-%s__lr-%.6f__bs-%i' % \
             (seed, task, sequence_length, hidden_units, n_reflections, model_type, learning_rate, batch_size)

load_progress = '' # data_path + 'saved_models/' + string
save_progress = data_path + 'saved_models/' + string  
print(string)
sys.stdout.flush()

# Loading the data
if task == 'MNIST':
    train_set, valid_set, _ = load_pMNIST_data(rng, data_path, perm=False)
elif task == 'pMNIST':
    train_set, valid_set, _ = load_pMNIST_data(rng, data_path, perm=True)
elif task == 'PTB':
    train_set, valid_set, _ = load_PTB_data(data_path)
elif task == 'PTB_5':
    train_set, valid_set, _ = load_PTB_data(data_path, 5)
else:
    train_set, valid_set = load_synthetic_data(rng, data_path, task, sequence_length)

# Defining the model
start = time.time()
rnn = eval(model_type)(rng, n_in=n_in, n_out=n_out, n_h=hidden_units, m=n_reflections, task_type=task, f_out=eval(f_out), f_act=eval(activation))
end = time.time() - start
print('It took %.6f seconds to built the model' % end)

# Training the model
best_params, valid_cost = rnn.optimiser.train(rng, task, sequence_length, train_set, valid_set, instances_per_batch=batch_size, n_epoch=n_epoch, 
                                              lambda_=learning_rate, valid_freq=valid_frequency,
                                              load_progress=load_progress, save_progress=save_progress)

