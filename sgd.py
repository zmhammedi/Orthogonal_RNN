# Author: Zakaria Mhammedi
# The University of Melbourne, 2016-2017
# Code adapted from https://github.com/boulanni/theano-hf

import theano
import theano.tensor as T
import six.moves.cPickle as cPickle
import os
import sys
import numpy as np


class sgd_optimizer:
    def __init__(self, model, model_type):
        '''Constructs and compiles the necessary Theano functions.'''
        self.model_type = model_type

        self.p = model.p
        self.shapes = [i.get_value().shape for i in model.p]
        self.sizes = [np.prod(s) for s in self.shapes]
        self.positions = np.cumsum([0] + self.sizes)[:-1]

        g = T.grad(model.cost, model.p)
        flat_g = T.as_tensor_variable(self.list_to_flat(g))

        self.f_gc = theano.function(inputs=[model.x, model.y, model.seq_len],
                                    outputs=[flat_g, model.cost],
                                    on_unused_input='ignore'
                                    )

        self.f_cost = theano.function(inputs=[model.x, model.y, model.seq_len],
                                      outputs=[model.cost, model.accuracy],
                                      on_unused_input='ignore'
                                      )

        self.m = theano.shared(name='m', value=np.zeros(sum(self.sizes)).astype(theano.config.floatX))
        self.v = theano.shared(name='v', value=np.zeros(sum(self.sizes)).astype(theano.config.floatX))

        beta1 = T.scalar()
        beta2 = T.scalar()
        time_step = T.scalar()
        lambda_ = T.scalar()
        grad = T.vector()

        # adam method for SGD - https://arxiv.org/pdf/1412.6980v8.pdf
        m = beta1 * self.m + (1 - beta1) * grad
        v = beta2 * self.v + (1 - beta2) * grad ** 2
        delta_adam = - lambda_ * m / (1 - beta1 ** time_step) / (T.sqrt(v / (1 - beta2 ** time_step)) + 1e-8)

        self.sgd_adam = theano.function(
            [grad, lambda_, time_step, theano.In(beta1, value=0.9), theano.In(beta2, value=0.999)],
            delta_adam,
            updates=[(self.m, m),
                     (self.v, v)]
        )

    def flat_to_list(self, vector):
        return [vector[position:position + size].reshape(shape) for shape, size, position in
                zip(self.shapes, self.sizes, self.positions)]

    def list_to_flat(self, l):
        return T.concatenate([i.flatten() for i in l])

    def update_param(self, flat_delta):
        if self.model_type == 'oRNN':
            delta = self.flat_to_list(flat_delta)
            U = np.tril(self.p[0].get_value() + delta[0])
            norms = np.linalg.norm(U, axis=0)
            U = 1. / norms * U
            self.p[0].set_value(U)
            for i in range(1, len(self.p)):
                self.p[i].set_value(self.p[i].get_value() + delta[i])
        else:
            delta = self.flat_to_list(flat_delta)
            for i, d in zip(self.p, delta):
                i.set_value(i.get_value() + d)

    def set_params(self, params):
        for i, j in zip(self.p, params):
            i.set_value(j.astype(theano.config.floatX))

    def load_model(self, load_progress):
        first_iteration = 1
        best = [0, np.inf, None]
        if isinstance(load_progress, str) and os.path.isfile(load_progress):
            with open(load_progress, 'rb') as file:
                save = cPickle.load(file)
            file.close()
            best, m, v, first_iteration, init_p = save
            first_iteration += 1
            self.m.set_value(m.astype(theano.config.floatX))
            self.v.set_value(v.astype(theano.config.floatX))
            for i, j in zip(self.p, init_p): i.set_value(j.astype(theano.config.floatX))
            print('* recovered saved model')
            sys.stdout.flush()
        return best, first_iteration

    def save_model(self, save_progress, best, u):
        if isinstance(save_progress, str):
            save = best, self.m.get_value().copy(), self.v.get_value().copy(), u, \
                   [i.get_value().copy() for i in self.p]
            with open(save_progress, 'wb') as file:
                cPickle.dump(save, file, cPickle.HIGHEST_PROTOCOL)
            file.close()

    def train(self, rng, train_data, valid_data=None, instances_per_batch=None, lambda_=0.001, n_epoch=2000,
              valid_freq=10, patience=np.inf, load_progress=None, save_progress=None):
        '''
        :param rng: Numpy RandomState
        :param train_data: Training data
        :param valid_data: Validation data
        :param instances_per_batch: batch size of the training data
        :param lambda_: learning rate
        :param n_epoch: number of epochs
        :param valid_freq: Validation frequency
        :param patience: Patience
        :param load_progress: File name to load the model
        :param save_progress: File name to save the model
        :return:
        '''
        self.indices_train = IndexDataset(rng, train_data[0].shape[0], instances_per_batch=instances_per_batch)
        if valid_data is not None:
            self.indices_valid = IndexDataset(rng, valid_data[0].shape[0], instances_per_batch=valid_data[0].shape[0])

        self.n_epoch = n_epoch

        valid_cost = None
        valid_accuracy = None
        best, first_iteration = self.load_model(load_progress)
        try:
            u = first_iteration
            num_iter = n_epoch * self.indices_train.max_index // self.indices_train.instances_per_batch
            self.indices_train.epoch = (self.indices_train.instances_per_batch * u) // self.indices_train.max_index
            while self.indices_train.epoch <= n_epoch:
                indices = self.indices_train.get_indices()
                results = [self.f_gc(train_data[0][i], train_data[1][i], train_data[2][i]) for i in indices]
                gradient, cost = np.mean(results, axis=0)
                flat_delta = self.sgd_adam(gradient, lambda_, u, 0.9, 0.999)
                self.update_param(flat_delta)

                if u % valid_freq == 0:
                    if valid_data is not None:
                        # Compute the validation cost
                        indices = self.indices_valid.get_indices()
                        results = [self.f_cost(valid_data[0][i], valid_data[1][i], valid_data[2][i])
                                       for i in indices]
                        valid_cost, valid_accuracy = np.mean(results, axis=0)
                        if valid_cost < best[1]:
                            best = u, valid_cost, [i.get_value().copy() for i in self.p]

                    print_progress(u, num_iter, self.indices_train.epoch, cost, valid_cost, valid_accuracy)
                    self.save_model(save_progress, best, u)
                    if u - best[0] > patience:
                        print('REACHED STOPPING CONDITION.')
                        break
                u += 1
                sys.stdout.flush()

        except KeyboardInterrupt:
            print('Interrupted by user.')

        if best[2] is None:
            best[2] = [i.get_value().copy() for i in self.p]
        return best[2], best[1]


class IndexDataset:
    def __init__(self, rng, data_size, instances_per_batch=None):
        self.current_instance = 0
        if instances_per_batch is None:
            self.instances_per_batch = data_size
        else:
            self.instances_per_batch = instances_per_batch
        self.indices = np.arange(data_size, dtype='int64')
        self.epoch = 1
        self.max_index = data_size
        self.rng = rng
        self.shuffle()

    def shuffle(self):
        self.rng.shuffle(self.indices)

    def get_indices(self):
        start_index = self.current_instance
        end_index = start_index + self.instances_per_batch
        if end_index >= self.max_index:
            end_index = self.max_index
        result = self.indices[start_index:end_index]
        if start_index + self.instances_per_batch >= self.max_index:
            self.update()
        else:
            self.current_instance += self.instances_per_batch
        return result

    def update(self):
        self.epoch += 1
        self.shuffle()
        self.current_instance = 0


def print_progress(u, num_iter, epoch, train_cost, valid_cost=None, valid_accuracy=None):
    if valid_accuracy is not None:
        print('update %i/%i, epoch=%i, train_cost=%.5f, valid_cost=%.5f, valid_accuracy=%.5f'
              % (u, num_iter, epoch, train_cost, valid_cost, valid_accuracy))
    elif valid_cost is not None:
        print('update %i/%i, epoch=%i, train_cost=%.5f, valid_cost=%.5f,' % (u, num_iter, epoch, train_cost,
                                                                             valid_cost))
    else:
        print('update %i/%i, epoch=%i, train_cost=%.5f' % (u, num_iter, epoch, train_cost))
    sys.stdout.flush()

