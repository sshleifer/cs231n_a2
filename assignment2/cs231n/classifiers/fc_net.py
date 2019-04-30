from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params = {}
        self.reg = reg
        self.W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.b2 = np.zeros(num_classes)
        self.params = {'W1': self.W1, 'W2': self.W2, 'b1': self.b1, 'b2': self.b2}
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(x, self.params['W2'], self.params['b2'])

        if y is None:  # test mode
            return scores
        loss, dout = softmax_loss(scores, y)
        reg_losses = [np.sum(v * v) for k, v in self.params.items()
                      if k in ['W1', 'W2']]
        reg_loss = np.sum(reg_losses) * self.reg * .5
        loss += reg_loss
        dx, dw2, db2 = affine_backward(dout, cache2)
        grads = dict(W2=dw2 + self.reg * self.params['W2'], b2=db2)
        _, dw1, db1 = affine_relu_backward(dx, cache1)
        grads.update(dict(W1=dw1 + self.reg * self.params['W1'], b1=db1))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return loss, grads

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_batchnorm = self.normalization == 'batchnorm'
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.cache = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        def random_init_w(in_dim, out_dim): return np.random.randn(in_dim, out_dim) * weight_scale

        #self.params['W1'] = random_init_w(input_dim, hidden_dims[0])
        #self.params['b1'] = np.zeros(hidden_dims[0])

        #self.params[f'W{self.num_layers}'] = random_init_w(hidden_dims[-1], num_classes)
        #self.params[f'b{self.num_layers}'] = np.zeros(num_classes)
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            if f'W{i+1}' not in self.params:
                self.params[f'W{i+1}'] = random_init_w(dims[i], dims[i+1])
                self.params[f'b{i+1}'] = np.zeros(dims[i+1])

            if self.use_batchnorm and (i+1) < self.num_layers:
                print(f'Adding bn for {i}')
                out_shape = self.params[f'W{i + 1}'].shape[-1]
                self.params[f'gamma{i+1}'] = np.ones(out_shape)
                self.params[f'beta{i+1}'] = np.zeros(out_shape)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.


        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    @property
    def _funcs_to_use(self):
        funcs = {}
        for i in range(self.num_layers):
            # g, b = self.params[f'gamma{i+1}'], self.params[f'beta{i+1}']
            if (i == self.num_layers): funcs[i] = (affine_forward,affine_backward)
            else: funcs[i] = (affine_relu_forward, affine_relu_backward)
        return funcs

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        inputs = {-1: X}
        self.cache = {}
        bn_cache = {}
        dropout_cache = {}
        funcs = self._funcs_to_use
        for i in range(self.num_layers):
            forward_func = funcs[i][0]
            W, b = self.params[f'W{i+1}'], self.params[f'b{i+1}']
            X, self.cache[i] = forward_func(inputs[i-1], W, b)
            if self.use_dropout:
                 X, do_cache = dropout_forward(X, self.dropout_param)
                 dropout_cache[i] = do_cache
            if self.use_batchnorm and i < len(self.bn_params):
                 g, b = self.params[f'gamma{i + 1}'], self.params[f'beta{i + 1}']
                 X, bn_cache[i] = batchnorm_forward(X, g, b, self.bn_params[i])
            inputs[i] = X
        scores = inputs[i]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # If test mode return early
        if mode == 'test':
            return scores

        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout = softmax_loss(scores, y)
        reg_losses = [np.sum(v * v) for k, v in self.params.items()
                      if k.startswith('W')]
        reg_loss = np.sum(reg_losses) * self.reg * .5
        loss += reg_loss
        grads = {}
        dx = dout
        for i in reversed(range(self.num_layers)):
            if self.use_batchnorm and i < len(self.bn_params):
                gk,betak = f'gamma{i+1}', f'beta{i+1}'
                dx, grads[gk], grads[betak] = batchnorm_backward_alt(dx, bn_cache[i])

            backward_func = funcs[i][1]
            wk, bk = f'W{i+1}', f'b{i+1}'

            dx, dw, db = backward_func(dx, self.cache[i])
            grads[wk] = dw + self.reg * self.params[wk]
            grads[bk] = db

            # grads[gk], grads[betak] =
        def assert_set_equal(a,b):
            a,b = set(a), set(b)
            msg = f'left difference{a.difference(b)}, right_difference: {b.difference(a)}'
            assert a == b, msg

        assert_set_equal(grads.keys(), self.params.keys())
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
