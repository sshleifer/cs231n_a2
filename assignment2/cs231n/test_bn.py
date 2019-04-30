import numpy as np
import unittest

from cs231n.layers import batchnorm_forward, batchnorm_backward, rel_error, batchnorm_backward_alt
from cs231n.gradient_check import eval_numerical_gradient_array
import time

class TestBN(unittest.TestCase):
    def test_batchnorm(self):

        # Gradient check batchnorm backward pass
        np.random.seed(231)
        N, D = 4, 5
        x = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)

        bn_param = {'mode': 'train'}
        fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
        fg = lambda a: batchnorm_forward(x, a, beta, bn_param)[0]
        fb = lambda b: batchnorm_forward(x, gamma, b, bn_param)[0]

        dx_num = eval_numerical_gradient_array(fx, x, dout)
        da_num = eval_numerical_gradient_array(fg, gamma.copy(), dout)
        db_num = eval_numerical_gradient_array(fb, beta.copy(), dout)

        _, cache = batchnorm_forward(x, gamma, beta, bn_param)
        dx, dgamma, dbeta = batchnorm_backward(dout, cache)
        # You should expect to see relative errors between 1e-13 and 1e-8
        print('dx error: ', rel_error(dx_num, dx))
        print('dgamma error: ', rel_error(da_num, dgamma))
        print('dbeta error: ', rel_error(db_num, dbeta))
        self.assertGreater(1e-8, rel_error(dx_num, dx),)

    def test_bn_alt(self):
        np.random.seed(231)
        N, D = 100, 500
        x = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)

        bn_param = {'mode': 'train'}
        out, cache = batchnorm_forward(x, gamma, beta, bn_param)

        t1 = time.time()
        dx1, dgamma1, dbeta1 = batchnorm_backward(dout, cache)
        t2 = time.time()
        dx2, dgamma2, dbeta2 = batchnorm_backward_alt(dout, cache)
        t3 = time.time()


        print('dx difference: ', rel_error(dx1, dx2))
        print('dgamma difference: ', rel_error(dgamma1, dgamma2))
        print('dbeta difference: ', rel_error(dbeta1, dbeta2))
        print('speedup: %.2fx' % ((t2 - t1) / (t3 - t2)))
        self.assertGreater(1e-8, rel_error(dx1, dx2), )
