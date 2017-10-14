import numpy
import C_fun
import theano.tensor
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply

class H_WY(Op):
    """
    Calculating  h - U T^{-1} U h, where T = triu(U^TU, 1) + 1/2 diag(U^TU).
    """
    __props__ = ()

    def make_node(self, U, h):
        U = as_tensor_variable(U)
        h = as_tensor_variable(h)
        assert U.ndim == 2
        assert h.ndim == 1
        out_dtype = theano.scalar.upcast(U.dtype, h.dtype)
        c = theano.tensor.vector(dtype=out_dtype)
        return Apply(self, [U, h], [c])

    def perform(self, node, inputs, outputs):
        U, h = inputs
        n, m = U.shape
        c = numpy.zeros_like(h)
        c_tilde = numpy.zeros(m)
        
        C_fun.F(U, h, c, c_tilde)

        outputs[0][0] = c

    def infer_shape(self, node, shapes):
        return [shapes[1]]

    def grad(self, inputs, output_gradients):
        U, h = inputs
        BPg = output_gradients[0]
        return H_WYGrad()(U, h, BPg)


H_wy = H_WY()

class H_WYGrad(Op):
    """
    Calculating backpropagated gradients (G and g) of loss with respect to U and h, respectively.
    """
    __props__ = ()

    def make_node(self, U, h, BPg):
        U = as_tensor_variable(U)
        h = as_tensor_variable(h)
        assert U.ndim == 2
        assert h.ndim == 1
        assert BPg.ndim == 1
        out_dtype = theano.scalar.upcast(U.dtype, h.dtype, BPg.dtype)
        G = theano.tensor.matrix(dtype=out_dtype)
        g = theano.tensor.vector(dtype=out_dtype)
        return Apply(self, [U, h, BPg], [G, g])

    def perform(self, node, inputs, outputs):
        U, h, BPg = inputs
        n, m = U.shape
        G = numpy.zeros_like(U)
        g = numpy.zeros_like(h)
        h_tilde = numpy.zeros(m)
        c_tilde = numpy.zeros(m)

        C_fun.GradF(U, h, BPg, G, g, h_tilde, c_tilde)

        outputs[0][0] = numpy.asarray(G)
        outputs[1][0] = g

    def infer_shape(self, node, shapes):
        return [shapes[0], shapes[1]]
