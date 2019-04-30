import numpy as np
import os
import unittest

from cs231n.layers import *
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.classifiers.fc_net import FullyConnectedNet
from cs231n.classifiers.cnn import ThreeLayerConvNet
from cs231n.solver import Solver
import time

IS_MAC = os.path.exists('/Users/shleifer/')

import pickle


def read_pickle(path):
    """pickle.load(path)"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def print_mean_std(x, axis=0):
    print('  means: ', x.mean(axis=axis))
    print('  stds:  ', x.std(axis=axis))
    print()


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
        self.assertGreater(1e-8, rel_error(dx_num, dx), )

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
            self.assertAlmostEqual(out.mean(), out_test.mean(), 1)
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
        return
        np.random.seed(231)
        N, D, H1, H2, C = 2, 15, 20, 30, 10
        X = np.random.randn(N, D)
        y = np.random.randint(C, size=(N,))

        for dropout in [0.5, .75, 1.]:
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
                self.assertGreaterEqual(1e-5, err)

            print()

    def test_layernorm_fwd(self):
        np.random.seed(231)
        N, D1, D2, D3 = 4, 50, 60, 3
        X = np.random.randn(N, D1)
        W1 = np.random.randn(D1, D2)
        W2 = np.random.randn(D2, D3)
        a = np.maximum(0, X.dot(W1)).dot(W2)

        print('Before layer normalization:')
        print_mean_std(a, axis=1)

        gamma = np.ones(D3)
        beta = np.zeros(D3)
        # Means should be close to zero and stds close to one
        print('After layer normalization (gamma=1, beta=0)')
        a_norm, _ = layernorm_forward(a, gamma, beta, {'mode': 'train'})
        print_mean_std(a_norm, axis=1)

        gamma = np.asarray([3.0, 3.0, 3.0])
        beta = np.asarray([5.0, 5.0, 5.0])
        # Now means should be close to beta and stds close to gamma
        print('After layer normalization (gamma=', gamma, ', beta=', beta, ')')
        a_norm, _ = layernorm_forward(a, gamma, beta, {'mode': 'train'})
        print_mean_std(a_norm, axis=1)
        self.assertEqual(a.shape, a_norm.shape)

    def test_layernorm_backward(self):
        np.random.seed(231)
        N, D = 4, 5
        x = 5 * np.random.randn(N, D) + 12
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)

        ln_param = {}
        fx = lambda x: layernorm_forward(x, gamma, beta, ln_param)[0]
        fg = lambda a: layernorm_forward(x, a, beta, ln_param)[0]
        fb = lambda b: layernorm_forward(x, gamma, b, ln_param)[0]

        dx_num = eval_numerical_gradient_array(fx, x, dout)
        da_num = eval_numerical_gradient_array(fg, gamma.copy(), dout)
        db_num = eval_numerical_gradient_array(fb, beta.copy(), dout)

        _, cache = layernorm_forward(x, gamma, beta, ln_param)
        dx, dgamma, dbeta = layernorm_backward(dout, cache)

        # You should expect to see relative errors between 1e-12 and 1e-8
        self.assertGreaterEqual(1e-8, rel_error(dx_num, dx))
        self.assertGreaterEqual(1e-8, rel_error(da_num, dgamma))
        self.assertGreaterEqual(1e-8, rel_error(db_num, dbeta))

        print('dx error: ', rel_error(dx_num, dx))
        print('dgamma error: ', rel_error(da_num, dgamma))
        print('dbeta error: ', rel_error(db_num, dbeta))


class TestConv(unittest.TestCase):
    def test_conv_forward_naive(self):
        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)

        conv_param = {'stride': 2, 'pad': 1}
        out, _ = conv_forward_naive(x, w, b, conv_param)
        correct_out = np.array([[[[-0.08759809, -0.10987781],
                                  [-0.18387192, -0.2109216]],
                                 [[0.21027089, 0.21661097],
                                  [0.22847626, 0.23004637]],
                                 [[0.50813986, 0.54309974],
                                  [0.64082444, 0.67101435]]],
                                [[[-0.98053589, -1.03143541],
                                  [-1.19128892, -1.24695841]],
                                 [[0.69108355, 0.66880383],
                                  [0.59480972, 0.56776003]],
                                 [[2.36270298, 2.36904306],
                                  [2.38090835, 2.38247847]]]])
        print(out)
        # Compare your output to ours; difference should be around e-8
        self.assertGreaterEqual(1e-7, rel_error(out, correct_out))
        print('Testing conv_forward_naive')
        print('difference: ', rel_error(out, correct_out))

    def test_conv_backward_naive(self):
        np.random.seed(231)
        x = np.random.randn(4, 3, 5, 5)
        w = np.random.randn(2, 3, 3, 3)
        b = np.random.randn(2, )
        dout = np.random.randn(4, 2, 5, 5)
        conv_param = {'stride': 1, 'pad': 1}

        dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0],
                                               x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0],
                                               w, dout)
        db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0],
                                               b, dout)

        out, cache = conv_forward_naive(x, w, b, conv_param)
        dx, dw, db = conv_backward_naive(dout, cache)

        # Your errors should be around e-8 or less.
        print('Testing conv_backward_naive function')
        print('dx error: ', rel_error(dx, dx_num))
        print('dw error: ', rel_error(dw, dw_num))
        print('db error: ', rel_error(db, db_num))

        self.assertGreaterEqual(1e-7, rel_error(dx, dx_num))
        self.assertGreaterEqual(1e-7, rel_error(dw, dw_num))
        self.assertGreaterEqual(1e-7, rel_error(db, db_num))
        print('dx error: ', rel_error(dx, dx_num))
        print('dw error: ', rel_error(dw, dw_num))
        print('db error: ', rel_error(db, db_num))

    def test_maxpool_forward_naive(self):
        x_shape = (2, 3, 4, 4)
        x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
        pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}
        out, _ = max_pool_forward_naive(x, pool_param)
        correct_out = np.array([[[[-0.26315789, -0.24842105],
                                  [-0.20421053, -0.18947368]],
                                 [[-0.14526316, -0.13052632],
                                  [-0.08631579, -0.07157895]],
                                 [[-0.02736842, -0.01263158],
                                  [0.03157895, 0.04631579]]],
                                [[[0.09052632, 0.10526316],
                                  [0.14947368, 0.16421053]],
                                 [[0.20842105, 0.22315789],
                                  [0.26736842, 0.28210526]],
                                 [[0.32631579, 0.34105263],
                                  [0.38526316, 0.4]]]])

        # Compare your output with ours. Difference should be on the order of e-8.
        self.assertGreaterEqual(1e-7, rel_error(out, correct_out))
        print('Testing max_pool_forward_naive function:')
        print('difference: ', rel_error(out, correct_out))

    def test_maxpool_backward_naive(self):
        np.random.seed(231)
        x = np.random.randn(3, 2, 8, 8)
        dout = np.random.randn(3, 2, 4, 4)
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0],
                                               x, dout)

        out, cache = max_pool_forward_naive(x, pool_param)
        dx = max_pool_backward_naive(dout, cache)

        # Your error should be on the order of e-12
        self.assertGreaterEqual(1e-11, rel_error(dx, dx_num))
        print('Testing max_pool_backward_naive function:')
        print('dx error: ', rel_error(dx, dx_num))

    def test_three_layer_conv(self):
        model = ThreeLayerConvNet()

        N = 50
        X = np.random.randn(N, 3, 32, 32)
        y = np.random.randint(10, size=N)

        loss, grads = model.loss(X, y)
        self.assertAlmostEqual(loss, np.log(10), 3)

    def test_three_layer_grads(self):
        num_inputs = 2
        input_dim = (3, 16, 16)
        reg = 0.0
        num_classes = 10
        np.random.seed(231)
        X = np.random.randn(num_inputs, *input_dim)
        y = np.random.randint(num_classes, size=num_inputs)

        model = ThreeLayerConvNet(num_filters=3, filter_size=3,
                                  input_dim=input_dim, hidden_dim=7,
                                  dtype=np.float64)
        loss, grads = model.loss(X, y)
        # Errors should be small, but correct implementations may have
        # relative errors up to the order of e-2
        for param_name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False,
                                                     h=1e-6)
            e = rel_error(param_grad_num, grads[param_name])
            self.assertGreaterEqual(1e-1, e)

    def test_overfit(self):
        small_data = read_pickle('small_data.pkl')
        model = ThreeLayerConvNet(weight_scale=1e-2)

        solver = Solver(model, small_data,
                        num_epochs=15, batch_size=50,
                        update_rule='adam',
                        optim_config={
                            'learning_rate': 1e-3,
                        },
                        verbose=True, print_every=1)
        solver.train()
        self.assertGreaterEqual(solver.train_acc_history[-1], .6)

    def test_spatial_batchnorm_fwd(self):
        np.random.seed(231)
        # Check the training-time forward pass by checking means and variances
        # of features both before and after spatial batch normalization

        N, C, H, W = 2, 3, 4, 5
        x = 4 * np.random.randn(N, C, H, W) + 10

        print('Before spatial batch normalization:')
        print('  Shape: ', x.shape)
        print('  Means: ', x.mean(axis=(0, 2, 3)))
        print('  Stds: ', x.std(axis=(0, 2, 3)))

        # Means should be close to zero and stds close to one
        gamma, beta = np.ones(C), np.zeros(C)
        bn_param = {'mode': 'train'}
        out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
        print('After spatial batch normalization:')
        print('  Shape: ', out.shape)
        print('  Means close to zero: ', out.mean(axis=(0, 2, 3)))
        print('  Stds close to 1: ', out.std(axis=(0, 2, 3)))

        # Means should be close to beta and stds close to gamma
        gamma, beta = np.asarray([3, 4, 5]), np.asarray([6, 7, 8])
        out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)
        print('After spatial batch normalization (nontrivial gamma, beta):')
        print('  Shape: ', out.shape)
        mns = out.mean(axis=(0, 2, 3))
        print('  Means close to [6,7,8]', out.mean(axis=(0, 2, 3)))
        print('  Stds close to [3,4,5] ', out.std(axis=(0, 2, 3)))
        self.assertEqual(x.shape, out.shape)
        self.assertTrue((np.round(mns, 1) == beta).all())
        self.assertTrue((np.round(out.std(axis=(0, 2, 3)), 1) == gamma).all())

    def test_spatial_batchnorm_bwd(self):
        np.random.seed(231)
        N, C, H, W = 2, 3, 4, 5
        x = 5 * np.random.randn(N, C, H, W) + 12
        gamma = np.random.randn(C)
        beta = np.random.randn(C)
        dout = np.random.randn(N, C, H, W)

        bn_param = {'mode': 'train'}
        fx = lambda x: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
        fg = lambda a: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]
        fb = lambda b: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]

        dx_num = eval_numerical_gradient_array(fx, x, dout)
        da_num = eval_numerical_gradient_array(fg, gamma, dout)
        db_num = eval_numerical_gradient_array(fb, beta, dout)

        # You should expect errors of magnitudes between 1e-12~1e-06
        _, cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
        dx, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)
        ACCEPTABLE = 1e-6
        self.assertGreaterEqual(ACCEPTABLE, rel_error(dx_num, dx))
        self.assertGreaterEqual(ACCEPTABLE, rel_error(da_num, dgamma))
        self.assertGreaterEqual(ACCEPTABLE, rel_error(db_num, dbeta))

    def test_spatial_groupnorm_fwd(self):
        np.random.seed(231)
        # Check the training-time forward pass by checking means and variances
        # of features both before and after spatial batch normalization

        N, C, H, W = 2, 6, 4, 5
        G = 2
        x = 4 * np.random.randn(N, C, H, W) + 10
        x_g = x.reshape((N * G, -1))
        print('Before spatial group normalization:')
        print('  Shape: ', x.shape)
        print('  Means: ', x_g.mean(axis=1))
        print('  Stds: ', x_g.std(axis=1))

        # Means should be close to zero and stds close to one
        gamma, beta = np.ones((1, C, 1, 1)), np.zeros((1, C, 1, 1))
        gn_param = {'mode': 'train'}

        out, _ = spatial_groupnorm_forward(x, gamma, beta, G, gn_param)
        out_g = out.reshape((N * G, -1))
        print('After spatial group normalization:')

        print('  Shape: ', out.shape)

        print('  Means: ', out_g.mean(axis=1))
        print('  Stds: ', out_g.std(axis=1))
        self.assertEqual(x.shape, out.shape)
        np.testing.assert_allclose(out_g.mean(axis=1),  0, atol=1e-4)
        np.testing.assert_allclose(out_g.std(axis=1), 1, atol=1e-4)

    def test_spatial_groupnorm_fwd_v2(self):
        np.random.seed(231)
        # Check the training-time forward pass by checking means and variances
        # of features both before and after spatial batch normalization

        N, C, H, W = 2, 12, 4, 5
        G = 3
        x = 4 * np.random.randn(N, C, H, W) + 10
        x_g = x.reshape((N * G, -1))
        print('Before spatial group normalization:')
        print('  Shape: ', x.shape)
        print('  Means: ', x_g.mean(axis=1))
        print('  Stds: ', x_g.std(axis=1))

        # Means should be close to zero and stds close to one
        gamma, beta = np.ones((1, C, 1, 1)), np.zeros((1, C, 1, 1))
        gn_param = {'mode': 'train'}

        out, _ = spatial_groupnorm_forward(x, gamma, beta, G, gn_param)
        out_g = out.reshape((N * G, -1))
        print('After spatial group normalization:')

        print('  Shape: ', out.shape)

        print('  Means: ', out_g.mean(axis=1))
        print('  Stds: ', out_g.std(axis=1))
        self.assertEqual(x.shape, out.shape)
        np.testing.assert_allclose(out_g.mean(axis=1), 0, atol=1e-4)
        np.testing.assert_allclose(out_g.std(axis=1), 1, atol=1e-4)

    def test_spatial_groupnorm_bwd(self):
        np.random.seed(231)
        N, C, H, W = 2, 6, 4, 5
        G = 2
        x = 5 * np.random.randn(N, C, H, W) + 12
        gamma = np.random.randn(1, C, 1, 1)
        beta = np.random.randn(1, C, 1, 1)
        dout = np.random.randn(N, C, H, W)

        gn_param = {}
        fx = lambda x: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]
        fg = lambda a: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]
        fb = lambda b: spatial_groupnorm_forward(x, gamma, beta, G, gn_param)[0]

        dx_num = eval_numerical_gradient_array(fx, x, dout)
        da_num = eval_numerical_gradient_array(fg, gamma, dout)
        db_num = eval_numerical_gradient_array(fb, beta, dout)

        _, cache = spatial_groupnorm_forward(x, gamma, beta, G, gn_param)
        dx, dgamma, dbeta = spatial_groupnorm_backward(dout, cache)

        ACCEPTABLE = 1e-7
        self.assertGreaterEqual(ACCEPTABLE, rel_error(dx_num, dx))
        self.assertGreaterEqual(ACCEPTABLE, rel_error(da_num, dgamma))
        self.assertGreaterEqual(ACCEPTABLE, rel_error(db_num, dbeta))
