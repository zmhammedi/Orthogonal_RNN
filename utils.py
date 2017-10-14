from toy_problems import *
import six.moves.cPickle as cPickle
import sys
args = sys.argv
import os
from mnist import MNIST

D = {'o': 7, '6': 43, 'p': 24, 'N': 28, 'k': 6, 's': 16, 'c': 12, 'n': 5, '9': 34, '/': 47, '\\': 46, 'z': 11,
     'j': 29, 'b': 4, '7': 44, 'w': 13, 'x': 25, '*': 48, '#': 45, '<': 26, 'u': 15, 'i': 10, 'd': 21, '.': 31,
     '4': 41, 't': 8, 'e': 2, '2': 40, 'y': 14, ' ': 0, '$': 38, "'": 32, 'l': 9, '0': 36, 'q': 23, 'f': 17,
     'g': 19, '>': 27, 'v': 30, '5': 35, '1': 33, 'r': 3, 'a': 1, '8': 42, 'm': 18, '&': 37, 'h': 20, '-': 22, '3': 39}


def load_lines(lines, filename, steps_ahead):
    arr = [[],[],[]]
    sys.stdin = open(filename, 'r')
    for i in range(lines):
        tmp = []
        for c in sys.stdin.readline()[:-2]:
            tmp.append(D[c])
        seq_len = len(tmp) - steps_ahead
        if seq_len <= 1:
            continue
        arr[0].append(tmp[:-steps_ahead])
        arr[1].append(tmp[steps_ahead:])
        arr[2].append(seq_len)
    return arr


def load_PTB_data(data_path, steps_ahead=1):
    train_set = load_lines(42068, data_path + '/PTB/train.txt', steps_ahead)
    valid_set = load_lines(3370, data_path + '/PTB/valid.txt', steps_ahead)
    test_set = load_lines(3761, data_path + '/PTB/test.txt', steps_ahead)
    return train_set, valid_set, test_set



def load_pMNIST_data(rng, data_path, perm=False): 
    data_path += '/MNIST/python-mnist/data/'
    mndata = MNIST(data_path)
    tmp_train =  mndata.load_training()
    tmp_test =  mndata.load_testing()
    len_data = len(tmp_train[0])
    ind_train = rng.permutation(len_data)[:55000]
    ind_valid = rng.permutation(len_data)[55000:]
    train_set = [1. / 256 * np.asarray(tmp_train[0])[ind_train], np.asarray(tmp_train[1])[ind_train],
                 784 * np.ones(55000)]
    valid_set = [1. / 256 * np.asarray(tmp_train[0])[ind_valid], np.asarray(tmp_train[1])[ind_valid],
                 784 * np.ones(5000)]
    test_set = [1. / 256 * np.asarray(tmp_test[0]), np.asarray(tmp_test[1]),
                784 * np.ones(len(tmp_test[0])).astype('int32')]
    if os.path.isfile('prm'):
        with open('prm', 'rb') as file:
            prm = cPickle.load(file)
        file.close()
    else:
        prm = rng.permutation(784)
        with open('prm', 'wb') as file:
            cPickle.dump(prm, file, cPickle.HIGHEST_PROTOCOL)
        file.close()
    if perm:
        train_set[0] = train_set[0][:, prm]
        valid_set[0] = valid_set[0][:, prm]
        test_set[0] = test_set[0][:, prm]
    print('Data loaded.')
    return train_set, valid_set, test_set


def load_synthetic_data(rng, data_path, task, sequence_length):
    filename_train = '%s/synthetic_data/%s/T%i_train' % (data_path, task, sequence_length)
    filename_test = '%s/synthetic_data/%s/T%i_test' % (data_path, task, sequence_length)
    if os.path.isfile(filename_train) and os.path.isfile(filename_test):
        # Load the training set
        with open(filename_train, 'rb') as file:
            train_set = cPickle.load(file)
        file.close()
        # Load the test set
        with open(filename_test, 'rb') as file:
            test_set = cPickle.load(file)
        file.close()
        print('Data loaded.')
        return train_set, test_set
    else:
        train_set, test_set = generate_data(rng, task, sequence_length, filename_train, filename_test)
        print('Data generated.')
        return train_set, test_set


def generate_data(rng, task, sequence_length, filename_train, filename_test, train_size=100, test_size=10000):
    train_set = eval(task)(rng, sequence_length, train_size)
    test_set = eval(task)(rng, sequence_length, test_size)
    with open(filename_train, 'wb') as file:
        cPickle.dump(train_set, file, cPickle.HIGHEST_PROTOCOL)
    file.close()
    with open(filename_test, 'wb') as file:
        cPickle.dump(test_set, file, cPickle.HIGHEST_PROTOCOL)
    file.close()
    return train_set, test_set

