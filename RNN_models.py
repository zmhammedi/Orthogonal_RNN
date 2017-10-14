# Author: Zakaria Mhammedi
# The University of Melbourne and Data61 (2016 - 2017)

import numpy as np
import theano as theano
import theano.tensor as T
import theano.tensor.shared_randomstreams
import theano.tensor.nlinalg
from sgd import sgd_optimizer
import sys
from tensor_ops import H_wy
theano.config.optimizer = 'fast_run'  # 'fast_compile'

def leaky_relu(z, b):
    return T.switch(z + b < 0, .1 * (z + b), z + b)

def tanh(z, b):
    return T.tanh(z + b)

def modReLU(h, b):
    m = h.shape[0] // 2
    a = T.reshape(h, (2, m)).T
    abs_z = T.sum(T.sqr(a), axis=1)
    tmp = T.concatenate([(abs_z + b[:m]) / abs_z, (abs_z + b[:m]) / abs_z])
    return T.switch(tmp > 0, tmp * h, 0)

def OPLU(h, b):
    tmp = T.reshape(h + b, (2, h.shape[0] // 2)).T
    tmpl = T.switch(tmp[:,0] < tmp[:,1], tmp[:,1], tmp[:,0])
    tmpr = T.switch(tmp[:,0] < tmp[:,1], tmp[:,0], tmp[:,1])
    return T.concatenate([tmpl, tmpr])

def linear(z):
    return z

def softmax(z):
    return T.nnet.softmax(z)[0]



'''-----------------------------------------------'''
'''------------------- oRNN ----------------------'''
'''-----------------------------------------------'''
class oRNN():
    def __init__(self, rng, n_in, n_out, n_h, m, task_type, f_act=leaky_relu, f_out=linear):
        U_ = np.tril(rng.normal(0, 0.01, (n_h, m)))
        norms = np.linalg.norm(U_, axis=0)
        U_ = 1. / norms * U_

        W_ = rng.uniform(-np.sqrt(6. / (n_in + n_h)), np.sqrt(6. / (n_in + n_h)), (n_h, n_in))
        bh_ = np.zeros( n_h)
        Y_ = rng.uniform(-np.sqrt(6. / (n_out + n_h)), np.sqrt(6. / (n_h + n_out)), (n_out, n_h))
        bo_ = np.zeros(n_out)
        h0_ = rng.uniform(-np.sqrt(3. / (2. * n_h)), np.sqrt(3. / (2. * n_h)), n_h)

        W_ = rng.uniform(-np.sqrt(6. / (n_in + n_h)), np.sqrt(6. / (n_in + n_h)), (n_h, n_in))
        bh_ = np.zeros( n_h)
        Y_ = rng.uniform(-np.sqrt(6. / (n_out + n_h)), np.sqrt(6. / (n_h + n_out)), (n_out, n_h))
        bo_ = np.zeros(n_out)
        h0_ = rng.uniform(-np.sqrt(3. / (2. * n_h)), np.sqrt(3. / (2. * n_h)), n_h)

        # Theano: Created shared variables
        W = theano.shared(name='W', value=W_.astype(theano.config.floatX))
        U = theano.shared(name='U', value=U_.astype(theano.config.floatX))
        bh = theano.shared(name='bh', value=bh_.astype(theano.config.floatX))
        Y = theano.shared(name='Y', value=Y_.astype(theano.config.floatX))
        bo = theano.shared(name='bo', value=bo_.astype(theano.config.floatX))
        h0 = theano.shared(name='h0', value=h0_.astype(theano.config.floatX))
        I = theano.shared(name='I', value=np.ones(n_h).astype(theano.config.floatX))
        n_eq_m = theano.shared(name='n_eq_h', value=(n_h == m))

        self.p = [U, W, Y, bh, bo, h0]

        seq_len = T.iscalar('seq_len')
        self.seq_len = seq_len

        if task_type in ['MNIST', 'pMNIST']:
            self.x = T.vector()
            x_scan = T.shape_padright(self.x)
        elif task_type in ['PTB', 'PTB_5']:
            self.x = T.ivector()
            x_scan = self.x
        else:
            self.x = T.matrix()
            x_scan = self.x

        if task_type in ['PTB', 'PTB_5']:
            def forward_prop_step(x_t, h_t_prev):
                X_t = T.eye(49)[x_t]
                h_t_prev = T.set_subtensor(h_t_prev[-1], h_t_prev[-1] * (U[-1, -1] * n_eq_m + (1 - n_eq_m)))
                h_t = f_act(W.dot(X_t) + H_wy(U[:, :m - n_eq_m], h_t_prev), bh)
                o_t = Y.dot(h_t) + bo
                return [o_t, h_t]
        else:
            def forward_prop_step(x_t, h_t_prev):
                h_t_prev = T.set_subtensor(h_t_prev[-1], h_t_prev[-1] * (U[-1, -1] * n_eq_m + (1 - n_eq_m)))
                h_t = f_act(W.dot(x_t) + H_wy(U[:, :m - n_eq_m], h_t_prev), bh)
                o_t = Y.dot(h_t) + bo
                return [o_t, h_t]

        # if task_type in ['PTB', 'PTB_5']:
        #     def forward_prop_step(x_t, h_t_prev):
        #         X_t = T.eye(49)[x_t]
        #         h_t = f_act(W.dot(X_t) + H_wy(U, h_t_prev), bh)
        #         o_t = Y.dot(h_t) + bo
        #         return [o_t, h_t]
        # else:
        #     def forward_prop_step(x_t, h_t_prev):
        #         h_t = f_act(W.dot(x_t) + H_wy(U, h_t_prev), bh)
        #         o_t = Y.dot(h_t) + bo
        #         return [o_t, h_t]

        [o_scan, _], _ = theano.scan(
            forward_prop_step,
            sequences=[x_scan],
            outputs_info=[None, h0],
            n_steps=seq_len
        )

        if task_type in ['add', 'multiply']:
            self.y = T.scalar('y')
            self.o = o_scan[-1]
            self.cost = T.sqr(self.o - self.y)[0]
            self.accuracy = T.switch(T.abs_(self.o - self.y) < 0.04, 1., 0)
        elif task_type in ['copying', 'copyingVariable']:
            self.y = T.matrix('y')
            self.o = T.nnet.softmax(o_scan)
            self.cost = T.nnet.categorical_crossentropy(self.o, self.y).mean()
            self.accuracy = T.switch(T.eq(T.argmax(self.o, axis=1), T.argmax(self.y, axis=1)), 1., 0.).mean()
        elif task_type in ['PTB', 'PTB_5']:
            self.y = T.ivector('y')
            self.o = T.nnet.softmax(o_scan)
            self.cost = T.nnet.categorical_crossentropy(self.o, T.eye(49)[self.y]).mean() / T.log(2)
            self.accuracy = T.switch(T.eq(T.argmax(self.o, axis=1), self.y), 1., 0.).mean()
        elif task_type in ['MNIST', 'pMNIST']:
            self.y = T.bscalar('y')
            self.o = T.nnet.softmax(o_scan[-1])[0]
            self.cost = T.nnet.categorical_crossentropy(self.o, T.eye(10)[self.y])
            self.accuracy = T.switch(T.eq(T.argmax(self.o), self.y), 1., 0.)
            self.prediction = np.argmax(self.o)

        self.optimiser = sgd_optimizer(self, 'oRNN')


