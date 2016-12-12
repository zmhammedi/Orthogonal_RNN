import numpy as np
import theano as theano
import theano.tensor as T
from tensor_ops import H_wy
from sgd import sgd_optimizer


def leaky_relu(z):
    return T.switch(z < 0, .1 * z, z)


def linear(z):
    return z


def softmax(z):
    return T.nnet.softmax(z)[0]


class oRNN():
    def __init__(self, rng, n_in, n_out, n_h, n_r, f_act=leaky_relu, f_out=softmax):
        '''
        :param rng: Numpy RandomState
        :param n_in: Input dimension (int)
        :param n_out: Output dimension (int)
        :param n_h: Hidden dimension (int)
        :param n_r: Number of reflection vectors (int)
        :param f_act: Hidden-to-hidden activation function
        :param f_out: Output activation function
        '''
        U_ = np.tril(rng.normal(0, 0.01, (n_h, n_r)))
        norms = np.linalg.norm(U_, axis=0)
        U_ = 1. / norms * U_

        Whi_ = rng.uniform(-np.sqrt(6. / (n_in + n_h)), np.sqrt(6. / (n_in + n_h)), (n_h, n_in))
        bh_ = np.zeros( n_h)
        Woh_ = rng.uniform(-np.sqrt(6. / (n_out + n_h)), np.sqrt(6. / (n_h + n_out)), (n_out, n_h))
        bo_ = np.zeros(n_out)
        h0_ = rng.uniform(-np.sqrt(3. / (2. * n_h)), np.sqrt(3. / (2. * n_h)), n_h)

        # Theano: Created shared variables
        Whi = theano.shared(name='Whi', value=Whi_.astype(theano.config.floatX))
        U = theano.shared(name='U', value=U_.astype(theano.config.floatX))
        bh = theano.shared(name='bh', value=bh_.astype(theano.config.floatX))
        Woh = theano.shared(name='Woh', value=Woh_.astype(theano.config.floatX))
        bo = theano.shared(name='bo', value=bo_.astype(theano.config.floatX))
        h0 = theano.shared(name='h0', value=h0_.astype(theano.config.floatX))

        self.p = [U, Whi, Woh, bh, bo, h0]

        seq_len = T.iscalar('seq_len')
        self.seq_len = seq_len

        self.x = T.vector()
        x_scan = T.shape_padright(self.x)

        if n_h != n_r:  # Number of reflection vectors is less than the hidden dimension
            def forward_prop_step(x_t, h_t_prev):
                h_t = f_act(Whi.dot(x_t) + H_wy(U, h_t_prev) + bh)
                o_t = Woh.dot(h_t) + bo
                return [o_t, h_t]
        else:
            def forward_prop_step(x_t, h_t_prev):
                h_t_prev = T.set_subtensor(h_t_prev[-1], h_t_prev[-1] * U[-1, -1])
                h_t = f_act(Whi.dot(x_t) + H_wy(U[:,:-1], h_t_prev) + bh)
                o_t = Woh.dot(h_t) + bo
                return [o_t, h_t]

        ## For loop version below (when n_r < n_h)
        # def forward_prop_step(x_t, h_t_prev):
        #     Wh = h_t_prev
        #     for i in range(n_r):
        #         Wh -= 2. * U[:, n_r - i - 1] * T.dot(U[:, n_r - i - 1], Wh)
        #     h_t = f_act(Whi.dot(x_t) + Wh + bh)
        #     o_t = Woh.dot(h_t) + bo
        #     return [o_t, h_t]

        [o_scan, _], _ = theano.scan(
            forward_prop_step,
            sequences=[x_scan],
            outputs_info=[None, h0],
            n_steps=seq_len
        )

        self.y = T.bscalar('y')
        self.o = f_out(o_scan[-1])
        self.cost = T.nnet.categorical_crossentropy(self.o, T.eye(10)[self.y])
        self.accuracy = T.switch(T.eq(T.argmax(self.o), self.y), 1., 0.)
        self.prediction = np.argmax(self.o)

        self.optimiser = sgd_optimizer(self, 'oRNN')
