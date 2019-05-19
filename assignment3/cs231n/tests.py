import unittest
import numpy as np
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.rnn_layers import *
from cs231n.captioning_solver import CaptioningSolver
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

class TestNB1(unittest.TestCase):
    def test_rnn_step_forward(self):
        N, D, H = 3, 10, 4

        x = np.linspace(-0.4, 0.7, num=N * D).reshape(N, D)
        prev_h = np.linspace(-0.2, 0.5, num=N * H).reshape(N, H)
        Wx = np.linspace(-0.1, 0.9, num=D * H).reshape(D, H)
        Wh = np.linspace(-0.3, 0.7, num=H * H).reshape(H, H)
        b = np.linspace(-0.2, 0.4, num=H)

        next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
        expected_next_h = np.asarray([
            [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
            [0.66854692, 0.79562378, 0.87755553, 0.92795967],
            [0.97934501, 0.99144213, 0.99646691, 0.99854353]])

        err = rel_error(expected_next_h, next_h)
        self.assertGreaterEqual(1e-8, err)


    def test_rnn_step_backward(self):
        from cs231n.rnn_layers import rnn_step_forward, rnn_step_backward
        np.random.seed(231)
        N, D, H = 4, 5, 6
        x = np.random.randn(N, D)
        h = np.random.randn(N, H)
        Wx = np.random.randn(D, H)
        Wh = np.random.randn(H, H)
        b = np.random.randn(H)

        out, cache = rnn_step_forward(x, h, Wx, Wh, b)

        dnext_h = np.random.randn(*out.shape)

        fx = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]
        fh = lambda prev_h: rnn_step_forward(x, h, Wx, Wh, b)[0]
        fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]
        fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]
        fb = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]

        dx_num = eval_numerical_gradient_array(fx, x, dnext_h)
        dprev_h_num = eval_numerical_gradient_array(fh, h, dnext_h)
        dWx_num = eval_numerical_gradient_array(fWx, Wx, dnext_h)
        dWh_num = eval_numerical_gradient_array(fWh, Wh, dnext_h)
        db_num = eval_numerical_gradient_array(fb, b, dnext_h)

        dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)

        print('dx error: ', rel_error(dx_num, dx))
        print('dprev_h error: ', rel_error(dprev_h_num, dprev_h))
        print('dWx error: ', rel_error(dWx_num, dWx))
        print('dWh error: ', rel_error(dWh_num, dWh))
        print('db error: ', rel_error(db_num, db))


    def test_rnn_forward_backward(self):
        np.random.seed(231)

        N, D, T, H = 2, 3, 10, 5

        x = np.random.randn(N, T, D)
        h0 = np.random.randn(N, H)
        Wx = np.random.randn(D, H)
        Wh = np.random.randn(H, H)
        b = np.random.randn(H)

        out, cache = rnn_forward(x, h0, Wx, Wh, b)

        dout = np.random.randn(*out.shape)

        dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)

        fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
        fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
        fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
        fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
        fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]

        dx_num = eval_numerical_gradient_array(fx, x, dout)
        dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
        dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
        dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
        db_num = eval_numerical_gradient_array(fb, b, dout)

        print('dx error: ', rel_error(dx_num, dx))
        print('dh0 error: ', rel_error(dh0_num, dh0))
        print('dWx error: ', rel_error(dWx_num, dWx))
        print('dWh error: ', rel_error(dWh_num, dWh))
        print('db error: ', rel_error(db_num, db))


    def test_word_embedding_forward(self):
        N, T, V, D = 2, 4, 5, 3

        x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
        W = np.linspace(0, 1, num=V * D).reshape(V, D)

        out, _ = word_embedding_forward(x, W)
        expected_out = np.asarray([
            [[0., 0.07142857, 0.14285714],
             [0.64285714, 0.71428571, 0.78571429],
             [0.21428571, 0.28571429, 0.35714286],
             [0.42857143, 0.5, 0.57142857]],
            [[0.42857143, 0.5, 0.57142857],
             [0.21428571, 0.28571429, 0.35714286],
             [0., 0.07142857, 0.14285714],
             [0.64285714, 0.71428571, 0.78571429]]])

        print('out error: ', rel_error(expected_out, out))


    def test_word_embedding_backward(self):
        np.random.seed(231)

        N, T, V, D = 50, 3, 5, 6
        x = np.random.randint(V, size=(N, T))
        W = np.random.randn(V, D)

        out, cache = word_embedding_forward(x, W)
        dout = np.random.randn(*out.shape)
        dW = word_embedding_backward(dout, cache)

        f = lambda W: word_embedding_forward(x, W)[0]
        dW_num = eval_numerical_gradient_array(f, W, dout)

        print('dW error: ', rel_error(dW, dW_num))



    def test_temporal_affine_forward(self):
        np.random.seed(231)

        # Gradient check for temporal affine layer
        N, T, D, M = 2, 3, 4, 5
        x = np.random.randn(N, T, D)
        w = np.random.randn(D, M)
        b = np.random.randn(M)

        out, cache = temporal_affine_forward(x, w, b)

        dout = np.random.randn(*out.shape)

        fx = lambda x: temporal_affine_forward(x, w, b)[0]
        fw = lambda w: temporal_affine_forward(x, w, b)[0]
        fb = lambda b: temporal_affine_forward(x, w, b)[0]

        dx_num = eval_numerical_gradient_array(fx, x, dout)
        dw_num = eval_numerical_gradient_array(fw, w, dout)
        db_num = eval_numerical_gradient_array(fb, b, dout)

        dx, dw, db = temporal_affine_backward(dout, cache)

        print('dx error: ', rel_error(dx_num, dx))
        print('dw error: ', rel_error(dw_num, dw))
        print('db error: ', rel_error(db_num, db))
    def test_temporal_softmax_loss(self):
        N, T, V = 100, 1, 10

        def check_loss(N, T, V, p):
            x = 0.001 * np.random.randn(N, T, V)
            y = np.random.randint(V, size=(N, T))
            mask = np.random.rand(N, T) <= p
            print(temporal_softmax_loss(x, y, mask)[0])

        check_loss(100, 1, 10, 1.0)  # Should be about 2.3
        check_loss(100, 10, 10, 1.0)  # Should be about 23
        check_loss(5000, 10, 10, 0.1)  # Should be about 2.3

        # Gradient check for temporal softmax loss
        N, T, V = 7, 8, 9

        x = np.random.randn(N, T, V)
        y = np.random.randint(V, size=(N, T))
        mask = (np.random.rand(N, T) > 0.5)

        loss, dx = temporal_softmax_loss(x, y, mask, verbose=False)

        dx_num = eval_numerical_gradient(lambda x: temporal_softmax_loss(x, y, mask)[0], x,
                                         verbose=False)

        print('dx error: ', rel_error(dx, dx_num))



    def test_captioning_rnn(self):
        N, D, W, H = 10, 20, 30, 40
        word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
        V = len(word_to_idx)
        T = 13

        model = CaptioningRNN(word_to_idx,
                              input_dim=D,
                              wordvec_dim=W,
                              hidden_dim=H,
                              cell_type='rnn',
                              dtype=np.float64)

        # Set all model parameters to fixed values
        for k, v in model.params.items():
            model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)

        features = np.linspace(-1.5, 0.3, num=(N * D)).reshape(N, D)
        captions = (np.arange(N * T) % V).reshape(N, T)

        loss, grads = model.loss(features, captions)
        expected_loss = 9.83235591003

        print('loss: ', loss)
        print('expected loss: ', expected_loss)
        print('difference: ', abs(loss - expected_loss))


    def test_captioning_rnn_grads(self):
        np.random.seed(231)

        batch_size = 2
        timesteps = 3
        input_dim = 4
        wordvec_dim = 5
        hidden_dim = 6
        word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
        vocab_size = len(word_to_idx)

        captions = np.random.randint(vocab_size, size=(batch_size, timesteps))
        features = np.random.randn(batch_size, input_dim)

        model = CaptioningRNN(word_to_idx,
                              input_dim=input_dim,
                              wordvec_dim=wordvec_dim,
                              hidden_dim=hidden_dim,
                              cell_type='rnn',
                              dtype=np.float64,
                              )

        loss, grads = model.loss(features, captions)

        for param_name in sorted(grads):
            f = lambda _: model.loss(features, captions)[0]
            param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False,
                                                     h=1e-6)
            e = rel_error(param_grad_num, grads[param_name])
            print('%s relative error: %e' % (param_name, e))

import sys
print(sys.executable)

class TestNB2(unittest.TestCase):

    def test_lstm_step_forward(self):
        N, D, H = 3, 4, 5
        x = np.linspace(-0.4, 1.2, num=N * D).reshape(N, D)
        prev_h = np.linspace(-0.3, 0.7, num=N * H).reshape(N, H)
        prev_c = np.linspace(-0.4, 0.9, num=N * H).reshape(N, H)
        Wx = np.linspace(-2.1, 1.3, num=4 * D * H).reshape(D, 4 * H)
        Wh = np.linspace(-0.7, 2.2, num=4 * H * H).reshape(H, 4 * H)
        b = np.linspace(0.3, 0.7, num=4 * H)

        next_h, next_c, cache = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)

        expected_next_h = np.asarray([
            [0.24635157, 0.28610883, 0.32240467, 0.35525807, 0.38474904],
            [0.49223563, 0.55611431, 0.61507696, 0.66844003, 0.7159181],
            [0.56735664, 0.66310127, 0.74419266, 0.80889665, 0.858299]])
        expected_next_c = np.asarray([
            [0.32986176, 0.39145139, 0.451556, 0.51014116, 0.56717407],
            [0.66382255, 0.76674007, 0.87195994, 0.97902709, 1.08751345],
            [0.74192008, 0.90592151, 1.07717006, 1.25120233, 1.42395676]])

        print('next_h error: ', rel_error(expected_next_h, next_h))
        print('next_c error: ', rel_error(expected_next_c, next_c))


    def test_lstm_step_backward(self):
        np.random.seed(231)

        N, D, H = 4, 5, 6
        x = np.random.randn(N, D)
        prev_h = np.random.randn(N, H)
        prev_c = np.random.randn(N, H)
        Wx = np.random.randn(D, 4 * H)
        Wh = np.random.randn(H, 4 * H)
        b = np.random.randn(4 * H)

        next_h, next_c, cache = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)

        dnext_h = np.random.randn(*next_h.shape)
        dnext_c = np.random.randn(*next_c.shape)

        fx_h = lambda x: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]
        fh_h = lambda h: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]
        fc_h = lambda c: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]
        fWx_h = lambda Wx: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]
        fWh_h = lambda Wh: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]
        fb_h = lambda b: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]

        fx_c = lambda x: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]
        fh_c = lambda h: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]
        fc_c = lambda c: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]
        fWx_c = lambda Wx: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]
        fWh_c = lambda Wh: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]
        fb_c = lambda b: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]

        num_grad = eval_numerical_gradient_array

        dx_num = num_grad(fx_h, x, dnext_h) + num_grad(fx_c, x, dnext_c)
        dh_num = num_grad(fh_h, prev_h, dnext_h) + num_grad(fh_c, prev_h, dnext_c)
        dc_num = num_grad(fc_h, prev_c, dnext_h) + num_grad(fc_c, prev_c, dnext_c)
        dWx_num = num_grad(fWx_h, Wx, dnext_h) + num_grad(fWx_c, Wx, dnext_c)
        dWh_num = num_grad(fWh_h, Wh, dnext_h) + num_grad(fWh_c, Wh, dnext_c)
        db_num = num_grad(fb_h, b, dnext_h) + num_grad(fb_c, b, dnext_c)

        dx, dh, dc, dWx, dWh, db = lstm_step_backward(dnext_h, dnext_c, cache)

        print('dx error: ', rel_error(dx_num, dx))
        print('dh error: ', rel_error(dh_num, dh))
        print('dc error: ', rel_error(dc_num, dc))
        print('dWx error: ', rel_error(dWx_num, dWx))
        print('dWh error: ', rel_error(dWh_num, dWh))
        print('db error: ', rel_error(db_num, db))


    def test_lstm_forward(self):
        N, D, H, T = 2, 5, 4, 3
        x = np.linspace(-0.4, 0.6, num=N * T * D).reshape(N, T, D)
        h0 = np.linspace(-0.4, 0.8, num=N * H).reshape(N, H)
        Wx = np.linspace(-0.2, 0.9, num=4 * D * H).reshape(D, 4 * H)
        Wh = np.linspace(-0.3, 0.6, num=4 * H * H).reshape(H, 4 * H)
        b = np.linspace(0.2, 0.7, num=4 * H)

        h, cache = lstm_forward(x, h0, Wx, Wh, b)

        expected_h = np.asarray([
            [[0.01764008, 0.01823233, 0.01882671, 0.0194232],
             [0.11287491, 0.12146228, 0.13018446, 0.13902939],
             [0.31358768, 0.33338627, 0.35304453, 0.37250975]],
            [[0.45767879, 0.4761092, 0.4936887, 0.51041945],
             [0.6704845, 0.69350089, 0.71486014, 0.7346449],
             [0.81733511, 0.83677871, 0.85403753, 0.86935314]]])

        print('h error: ', rel_error(expected_h, h))

    def test_lstm_backward(self):
        from cs231n.rnn_layers import lstm_forward, lstm_backward
        np.random.seed(231)

        N, D, T, H = 2, 3, 10, 6

        x = np.random.randn(N, T, D)
        h0 = np.random.randn(N, H)
        Wx = np.random.randn(D, 4 * H)
        Wh = np.random.randn(H, 4 * H)
        b = np.random.randn(4 * H)

        out, cache = lstm_forward(x, h0, Wx, Wh, b)

        dout = np.random.randn(*out.shape)

        dx, dh0, dWx, dWh, db = lstm_backward(dout, cache)

        fx = lambda x: lstm_forward(x, h0, Wx, Wh, b)[0]
        fh0 = lambda h0: lstm_forward(x, h0, Wx, Wh, b)[0]
        fWx = lambda Wx: lstm_forward(x, h0, Wx, Wh, b)[0]
        fWh = lambda Wh: lstm_forward(x, h0, Wx, Wh, b)[0]
        fb = lambda b: lstm_forward(x, h0, Wx, Wh, b)[0]

        dx_num = eval_numerical_gradient_array(fx, x, dout)
        dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
        dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
        dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
        db_num = eval_numerical_gradient_array(fb, b, dout)

        print('dx error: ', rel_error(dx_num, dx))
        print('dh0 error: ', rel_error(dh0_num, dh0))
        print('dWx error: ', rel_error(dWx_num, dWx))
        print('dWh error: ', rel_error(dWh_num, dWh))
        print('db error: ', rel_error(db_num, db))


    def test_lstm_captioning(self):
        N, D, W, H = 10, 20, 30, 40
        word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
        V = len(word_to_idx)
        T = 13

        model = CaptioningRNN(word_to_idx,
                              input_dim=D,
                              wordvec_dim=W,
                              hidden_dim=H,
                              cell_type='lstm',
                              dtype=np.float64)

        # Set all model parameters to fixed values
        for k, v in model.params.items():
            model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)

        features = np.linspace(-0.5, 1.7, num=N * D).reshape(N, D)
        captions = (np.arange(N * T) % V).reshape(N, T)

        loss, grads = model.loss(features, captions)
        expected_loss = 9.82445935443

        print('loss: ', loss)
        print('expected loss: ', expected_loss)
        print('difference: ', abs(loss - expected_loss))
