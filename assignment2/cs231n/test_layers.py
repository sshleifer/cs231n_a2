import numpy as np
import unittest

from cs231n.layers import *
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.classifiers.fc_net import FullyConnectedNet
from cs231n.solver import Solver
import time


import pickle
def read_pickle(path):
    """pickle.load(path)"""
    with open(path, 'rb') as f:
        return pickle.load(f)


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

    def test_in_net(self):
        return
        np.random.seed(231)
        # Try training a very deep net with batchnorm
        hidden_dims = [100, 100]
        small_data = read_pickle('small_data.pkl')

        weight_scale = 2e-2
        bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale,
                                     normalization='batchnorm')
        model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, normalization=None)

        print('Solver with batch norm:')
        bn_solver = Solver(bn_model, small_data,
                           num_epochs=10, batch_size=50,
                           update_rule='adam',
                           optim_config={
                               'learning_rate': 1e-3,
                           },
                           verbose=True, print_every=20)
        bn_solver.train()

        print('\nSolver without batch norm:')
        solver = Solver(model, small_data,
                        num_epochs=10, batch_size=50,
                        update_rule='adam',
                        optim_config={
                            'learning_rate': 1e-3,
                        },
                        verbose=True, print_every=20)
        solver.train()

class TestDropout(unittest.TestCase):
    def test_dropout_fwd(self):
        np.random.seed(231)
        x = np.random.randn(500, 500) + 10

        for p in [0.25, 0.4, 0.7]:
            out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
            out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})
            self.assertAlmostEqual(out.mean(),  out_test.mean(), 1)
            # self.assertAlmostEqual((out == 0).mean(), (out_test == 0).mean(), 1)
            print('Running tests with p = ', p)
            print('Mean of input: ', x.mean())
            print('Mean of train-time output: ', out.mean())
            print('Mean of test-time output: ', out_test.mean())
            print('Fraction of train-time output set to zero: ', (out == 0).mean())
            print('Fraction of test-time output set to zero: ', (out_test == 0).mean())
            print()

    def test_dropout_backward(self):
        np.random.seed(231)
        x = np.random.randn(10, 10) + 10
        dout = np.random.randn(*x.shape)

        dropout_param = {'mode': 'train', 'p': 0.2, 'seed': 123}
        out, cache = dropout_forward(x, dropout_param)
        dx = dropout_backward(dout, cache)
        dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x,
                                               dout)

        # Error should be around e-10 or less
        self.assertGreater(1e-10, rel_error(dx, dx_num))
        print('dx relative error: ', rel_error(dx, dx_num))

    def test_in_net(self):
        np.random.seed(231)
        N, D, H1, H2, C = 2, 15, 20, 30, 10
        X = np.random.randn(N, D)
        y = np.random.randint(C, size=(N,))

        for dropout in [1, 0.75, 0.5]:
            print('Running check with dropout = ', dropout)
            model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                                      weight_scale=5e-2, dtype=np.float64,
                                      dropout=dropout, seed=123)

            loss, grads = model.loss(X, y)
            print('Initial loss: ', loss)

            # Relative errors should be around e-6 or less; Note that it's fine
            # if for dropout=1 you have W2 error be on the order of e-5.
            for name in sorted(grads):
                f = lambda _: model.loss(X, y)[0]
                grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
                err = rel_error(grad_num, grads[name])
                print('%s relative error: %.2e' % (name, err))
                self.assertGreaterEqual(1e-6, err)

            print()
