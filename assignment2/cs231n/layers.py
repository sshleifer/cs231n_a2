import numpy as np
from cs231n.optim import lin_combo

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    n_rows = x.shape[0]
    x2 = x.reshape(n_rows, w.shape[0])
    out = (x2 @ w) + b
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = (dout @ w.T).reshape(x.shape)
    x2 = x.reshape(x.shape[0], w.shape[0])
    dw = x2.T @ dout
    db = dout.sum(axis=0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x.clip(0, None)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dx = dout.copy()
    dx[cache < 0] = 0
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    batch_mn, batch_var = x.mean(axis=0), x.var(axis=0)
    bn_param['running_mean'] = lin_combo(running_mean, batch_mn, momentum)
    bn_param['running_var'] = lin_combo(running_var, batch_var, momentum)
    # Store the updated running means back into bn_param
    if mode == 'train':
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        batch_norm_x = (x - batch_mn) / np.sqrt(batch_var + eps)
        out = gamma * batch_norm_x + beta
        cache = (x - batch_mn, batch_norm_x, batch_mn, batch_var, gamma, beta, bn_param, eps)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    elif mode == 'test':
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = ((x - bn_param['running_mean']) / (np.sqrt(bn_param['running_var']) + eps)) * gamma + beta
        cache = ()
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)****
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = dout.shape
    xmu, xhat, batch_mn, batch_var, gamma, beta, bn_param, eps = cache
    dbeta = dout.sum(axis=0)
    dgamma = (xhat * dout).sum(axis=0)

    bve = batch_var + eps
    sqrtvar = np.sqrt(bve)
    dxhat = dout * gamma
    divar = (dxhat * xmu).sum(axis=0)

    dsqrtvar = -1. / bve * divar
    dvar = 0.5 / sqrtvar * dsqrtvar
    dxmu1 = dxhat / sqrtvar
    dxmu2 = dvar / N * 2 * xmu

    dx1 = dxmu1 + dxmu2
    dx2 = -dx1.sum(axis=0) / N
    dx = dx1 + dx2
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return dx, dgamma, dbeta

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    xmu, xhat, batch_mn, batch_var, gamma, beta, bn_param, eps = cache
    dbeta, dgamma = dout.sum(axis=0), (xhat * dout).sum(axis=0)
    N, D = dout.shape
    dxhat = dout * gamma
    sqrtvar = np.sqrt(batch_var + eps)
    dx = 1. / N / sqrtvar * (N * dxhat - dxhat.sum(axis=0)
                             - xhat * (dxhat * xhat).sum(axis=0))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return dx, dgamma, dbeta


def reshape_to_bn(X, N, C, H, W):
    return np.swapaxes(X, 0, 1).reshape(C, -1).T


def reshape_from_bn(out, N, C, H, W):
    return np.swapaxes(out.T.reshape(C, N, H, W), 0, 1)


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X = x.T
    batch_mn, batch_var = X.mean(axis=0), X.var(axis=0)
    batch_norm_x = ((X - batch_mn) / np.sqrt(batch_var + eps)).T

    out = gamma * batch_norm_x + beta
    cache = (X - batch_mn, batch_norm_x, batch_mn, batch_var+eps, gamma )

    # out, cache =  batchnorm_forward(x.T, gamma, beta,ln_param)
    # cache[1] = cache[1].T
    # out = out.T

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    xmu, xhat, batch_mn, bve, gamma = cache
    dbeta = dout.sum(axis=0)
    dgamma = (xhat * dout).sum(axis=0)
    dxhat = (dout * gamma).T
    xhat = xhat.T
    N, D = xhat.shape

    inv_var = 1 / (np.sqrt(bve))
    dx = (1. / N) * inv_var * (N * dxhat - np.sum(dxhat, axis=0)
                               - xhat * (dxhat * xhat).sum(axis=0))
    dx = dx.T
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param: np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # mask = (np.random.rand(*x.shape) < p) / p
        #mask = np.random.binomial(1, 1 - p, size=x.shape) * (1/(1-p))
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    elif mode == 'test':
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        mask = np.ones(x.shape)
        out = x
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cache = (dropout_param, mask)

    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dx = dout * mask# / (1/(1-dropout_param['p']))
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    elif mode == 'test':
        dx = dout
    return dx
E, R = enumerate, range

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    P, stride = conv_param['pad'], conv_param['stride']
    Hp = int(1 + (H + 2 * P - HH) / stride)
    Wp = int(1 + (W + 2 * P - WW) / stride)
    out = np.zeros((N, F, Hp, Wp))
    x_pad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')

    for n, img in E(x_pad):
        for f, filter in E(w):
            for hp in R(Hp):
                for wp in R(Wp):
                    xstart = hp * stride
                    ystart = wp * stride
                    region = img[:, xstart: xstart + HH, ystart: ystart + WW]
                    assert filter.shape == region.shape
                    out[n, f, hp, wp] = (filter * region).sum() + b[f]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param = cache
    N, F, Hp, Wp = dout.shape

    F, C, HH, WW = w.shape
    P, stride = conv_param['pad'], conv_param['stride']
    x_pad = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
    dx, dw  = np.zeros_like(x_pad), np.zeros_like(w)
    for n, img in E(x_pad):
        for f, filter in E(w):
            for hp in R(Hp):
                for wp in R(Wp):
                    xstart = hp * stride
                    ystart = wp * stride
                    s1, s2 = slice(xstart, xstart + HH), slice(ystart, ystart + WW)
                    region = img[:, s1, s2]

                    output_region = dout[n, f, hp, wp]
                    dx[n, :, s1, s2] += (output_region * filter)
                    dw[f] += (region * output_region)



    db = dout.sum(axis=(0,2,3))
    dx = dx[:, :, P:-P, P:-P]  # unpad

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      Hp = 1 + (H - pool_height) / stride
      Wp = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    ph, pw, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    Hp = int(1 + (H - ph) / stride)
    Wp = int(1 + (W - pw) / stride)
    out = np.zeros((N, C, Hp, Wp))
    for n, img in E(x):
        for c, channel in E(img):
            for hp in R(Hp):
                for wp in R(Wp):
                    xstart = hp * stride
                    ystart = wp * stride
                    s1, s2 = slice(xstart, xstart + Hp), slice(ystart, ystart + Wp)
                    region = channel[s1, s2]
                    out[n, c, hp, wp] = region.max()


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache

def nd_argmax(a):
    return np.unravel_index(np.argmax(a, axis=None), a.shape)

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """

    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, pool_param = cache
    N, C, H, W = x.shape
    ph, pw, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    Hp = int(1 + (H - ph) / stride)
    Wp = int(1 + (W - pw) / stride)
    dx= np.zeros_like(x)
    for n, img in E(x):
        for c, channel in E(img):
            for hp in R(Hp):
                for wp in R(Wp):
                    xstart = hp * stride
                    ystart = wp * stride
                    s1, s2 = slice(xstart, xstart + Hp), slice(ystart, ystart + Wp)
                    max_x, max_y = nd_argmax(channel[s1, s2])
                    dx[n, c, xstart + max_x, ystart + max_y] += dout[n, c, hp, wp]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    x2d = x.transpose([0, 2,3, 1]).reshape(N*H*W, C)

    out, cache = batchnorm_forward(x2d, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose([0, 3, 1, 2])  # TODO(SS): maybe overcomplex?
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

     ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    x2d = dout.transpose([0, 2, 3, 1]).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(x2d, cache)
    dx = dx.reshape(N, H, W, C).transpose([0, 3, 1, 2])
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that
    of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1,C,1,1).
    - beta: Shift parameter, of shape (1,C,1,1).
    - G: Integer number of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    eps = gn_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    assert C % G == 0
    def reshape_fn(x):  return x.reshape(N * G, -1).T  # (N*G, C/G * H * W)
    X = reshape_fn(x)

    batch_mn, batch_var = X.mean(axis=0), X.var(axis=0)
    batch_norm_x = (X - batch_mn) / np.sqrt(batch_var + eps)
    x_norm = batch_norm_x.T.reshape(*x.shape)
    out = gamma * x_norm + beta
    cache = (X - batch_mn, x_norm, batch_mn, batch_var, gamma, beta, gn_param, eps, reshape_fn)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    xmu, xhat, batch_mn, batch_var, gamma, beta, bn_param, eps, reshape_fn = cache
    not_c_axes = (0, 2, 3)
    C = dout.shape[1]
    _f = lambda x: x.sum(axis=not_c_axes).reshape(1, C, 1, 1)
    dbeta, dgamma = _f(dout), _f(xhat * dout)
    dxhat = reshape_fn(dout * gamma)
    xhat = reshape_fn(xhat)
    N_grp, grp_size = xhat.shape
    inv_var = 1 / (np.sqrt(batch_var + eps))
    dx = (1. / N_grp) * inv_var * (N_grp * dxhat - dxhat.sum(axis=0) - xhat * (dxhat * xhat).sum(axis=0))
    dx = dx.T.reshape(*dout.shape)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx



