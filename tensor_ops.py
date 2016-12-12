import numpy
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
        n = U.shape[1]
        tmp = numpy.zeros_like(h)
        c_tilde = numpy.zeros(n)

        h_hat = numpy.dot(U.T, h)
        for k in range(1, n + 1):
            c_tilde[n - k] = 2. * (h_hat[n - k] - U[n - k:, n - k].dot(tmp[n - k:]))
            tmp[n - k:] = c_tilde[n - k] * U[n - k:, n - k] + tmp[n - k:]
        c = h - numpy.dot(U, c_tilde)
        outputs[0][0] = c

    def infer_shape(self, node, shapes):
        return [shapes[1]]

    def grad(self, inputs, output_gradients):
        U, h = inputs
        c_bar = output_gradients[0]
        return H_WYGrad()(U, h, c_bar)


H_wy = H_WY()

class H_WYGrad(Op):
    """
    Calculating U_bar and h_bar.
    """
    __props__ = ()

    def make_node(self, U, h, c_bar):
        U = as_tensor_variable(U)
        h = as_tensor_variable(h)
        assert U.ndim == 2
        assert h.ndim == 1
        assert c_bar.ndim == 1
        out_dtype = theano.scalar.upcast(U.dtype, h.dtype, c_bar.dtype)
        U_bar = theano.tensor.matrix(dtype=out_dtype)
        h_bar = theano.tensor.vector(dtype=out_dtype)
        return Apply(self, [U, h, c_bar], [U_bar, h_bar])

    def perform(self, node, inputs, outputs):
        U, h, c_bar = inputs
        n = U.shape[1]
        U_bar = numpy.zeros_like(U)
        h_tilde = numpy.zeros(n)
        c_tilde = numpy.zeros(n)
        tmp1 = numpy.zeros_like(h)
        tmp2 = numpy.zeros_like(h)

        h_hat = numpy.dot(U.T, h)
        c_hat = numpy.dot(U.T, c_bar)
        for k in range(n):
            # Compute c_tilde;  h_tilde = T^{-1} U^T h
            h_tilde[n-k-1] = 2. * (h_hat[n-k-1] - U[n-k-1:, n-k-1].dot(tmp1[n-k-1:]))
            tmp1[n-k-1:] += h_tilde[n-k-1] * U[n-k-1:, n-k-1]
            # Compute h_bar;  c_tilde = T^{-T} U^T c_bar
            c_tilde[k] = 2. * (c_hat[k] - U[k:, k].dot(tmp2[k:]))
            tmp2[k:] += c_tilde[k] * U[k:, k]

        # Compute U_bar
        tmp1 *= 0
        tmp2 *= 0
        for k in range(n - 1):
            tmp1[k:] += U[k:, k] * c_tilde[k]
            U_bar[:, k+1] += tmp1 * h_tilde[k+1]
            tmp2[n-k-1:] += U[n-k-1:, n-k-1] * h_tilde[n-k-1]
            U_bar[:, n-k-1] += tmp2 * c_tilde[n-k-1]
        tmp2 += U[:, 0] * h_tilde[0]
        U_bar[:, 0] += tmp2 * c_tilde[0]

        U_bar -= numpy.outer(c_bar, h_tilde) + numpy.outer(h, c_tilde)
        h_bar = c_bar - numpy.dot(U, c_tilde)
        outputs[0][0] = U_bar
        outputs[1][0] = h_bar

    def infer_shape(self, node, shapes):
        return [shapes[0], shapes[1]]