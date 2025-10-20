"""
Three-Layer Convolutional Neural Network Implementation
======================================================

This implementation follows the structure described in the document:
- Input: RGB image (3 channels, 32x32)
- Conv1: 16 filters, 3x3, stride 1, padding 1, followed by BN, ReLU, and MaxPool(2x2)
- Conv2: 32 filters, 3x3, stride 1, padding 1, followed by BN, ReLU, and MaxPool(2x2)
- Conv3: 64 filters, 3x3, stride 1, padding 0, followed by BN and ReLU
- Global Average Pooling
- Fully Connected: 64 -> 10 classes
- Softmax classifier

All forward and backward propagation equations are implemented according to the mathematical
formulations provided in the document.
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle


# ---------------------------
# Utility: im2col / col2im
# ---------------------------
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """
    Get indices for im2col operation.
    """
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_width) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """
    Convert image to columns using indexing.
    Input:
    - x: Input data of shape (N, C, H, W)
    Output:
    - cols: 2D array of shape (C * field_height * field_width, N * out_height * out_width)
    """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """
    Convert columns back to image.
    Input:
    - cols: 2D array of shape (C * field_height * field_width, N * out_height * out_width)
    - x_shape: Shape of the input image (N, C, H, W)
    Output:
    - x: Image of shape (N, C, H, W)
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    # Use numpy's advanced indexing for faster operation
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# ---------------------------
# Batch Normalization
# ---------------------------
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    Input:
    - x: Input data of shape (N, C, H, W)
    - gamma, beta: Scale and shift parameters of shape (C,)
    - bn_param: Dictionary with parameters:
      - mode: 'train' or 'test'
      - eps: Small constant for numerical stability
      - momentum: Momentum for running mean/var computation
      - running_mean, running_var: Running statistics for inference

    Output:
    - out: Normalized data
    - cache: Values needed for backward pass
    """
    mode = bn_param.get('mode', 'train')
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, C, H, W = x.shape

    # Initialize running mean and variance if not present
    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.ones(C, dtype=x.dtype))

    if mode == 'train':
        # Step 1: Calculate mean across batch and spatial dimensions for each channel
        mu = np.mean(x, axis=(0, 2, 3), keepdims=True)

        # Step 2: Calculate variance
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        # Step 3: Normalize
        x_normalized = (x - mu) / np.sqrt(var + eps)

        # Step 4: Scale and shift
        out = gamma.reshape(1, C, 1, 1) * x_normalized + beta.reshape(1, C, 1, 1)

        # Update running mean and variance for test mode
        running_mean = momentum * running_mean + (1 - momentum) * mu.reshape(C)
        running_var = momentum * running_var + (1 - momentum) * var.reshape(C)

        # Save values for backprop
        cache = (x, x_normalized, mu, var, gamma, beta, eps)

    elif mode == 'test':
        # Use running mean and variance for inference
        x_normalized = (x - running_mean.reshape(1, C, 1, 1)) / np.sqrt(running_var.reshape(1, C, 1, 1) + eps)
        out = gamma.reshape(1, C, 1, 1) * x_normalized + beta.reshape(1, C, 1, 1)
        cache = None

    else:
        raise ValueError('Invalid batch normalization mode')

    # Store updated running mean and var
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    Input:
    - dout: Upstream derivatives of shape (N, C, H, W)
    - cache: Values from forward pass

    Output:
    - dx: Gradient with respect to inputs x
    - dgamma: Gradient with respect to scale parameter gamma
    - dbeta: Gradient with respect to shift parameter beta
    """
    x, x_normalized, mu, var, gamma, beta, eps = cache
    N, C, H, W = dout.shape

    # Gradient with respect to beta and gamma
    dbeta = np.sum(dout, axis=(0, 2, 3))
    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3))

    # Gradient with respect to x_normalized
    dx_normalized = dout * gamma.reshape(1, C, 1, 1)

    # Gradient with respect to x
    # We're computing gradients following the chain rule for batch normalization

    # Step 1: Gradient through the normalization
    dvar = np.sum(dx_normalized * (x - mu) * -0.5 * (var + eps) ** (-1.5), axis=(0, 2, 3), keepdims=True)
    dmu = np.sum(dx_normalized * -1 / np.sqrt(var + eps), axis=(0, 2, 3), keepdims=True)
    dmu += dvar * np.sum(-2 * (x - mu), axis=(0, 2, 3), keepdims=True) / (N * H * W)

    # Step 2: Combine gradients
    dx = dx_normalized / np.sqrt(var + eps)
    dx += dvar * 2 * (x - mu) / (N * H * W)
    dx += dmu / (N * H * W)

    return dx, dgamma, dbeta


# ---------------------------
# Convolution Layer Forward/Backward
# ---------------------------
def conv_forward(x, w, b, conv_param):
    """
    Forward pass for convolutional layer.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases of shape (F,)
    - conv_param: Dictionary with parameters:
      - 'stride': Stride for convolution
      - 'pad': Zero-padding size

    Output:
    - out: Output data of shape (N, F, H', W')
    - cache: Values needed for backward pass
    """
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Check dimensions
    assert (H + 2 * pad - HH) % stride == 0, 'Invalid height dimensions'
    assert (W + 2 * pad - WW) % stride == 0, 'Invalid width dimensions'

    # Output dimensions
    out_height = (H + 2 * pad - HH) // stride + 1
    out_width = (W + 2 * pad - WW) // stride + 1

    # im2col transformation
    x_cols = im2col_indices(x, HH, WW, padding=pad, stride=stride)

    # Reshape filters
    w_reshaped = w.reshape(F, -1)

    # Compute outputs
    out = w_reshaped @ x_cols + b.reshape(-1, 1)
    out = out.reshape(F, out_height, out_width, N)
    out = out.transpose(3, 0, 1, 2)  # (N, F, H', W')

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_backward(dout, cache):
    """
    Backward pass for convolutional layer.

    Input:
    - dout: Upstream derivatives of shape (N, F, H', W')
    - cache: Values from forward pass

    Output:
    - dx: Gradient with respect to input x
    - dw: Gradient with respect to weights w
    - db: Gradient with respect to biases b
    """
    x, w, b, conv_param, x_cols = cache
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    N, F, out_height, out_width = dout.shape
    _, C, HH, WW = w.shape

    # Reshape dout
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)

    # Compute gradients
    db = np.sum(dout, axis=(0, 2, 3))
    dw = dout_reshaped @ x_cols.T
    dw = dw.reshape(w.shape)

    # Compute gradient with respect to x
    dx_cols = w.reshape(F, -1).T @ dout_reshaped
    dx = col2im_indices(dx_cols, x.shape, HH, WW, padding=pad, stride=stride)

    return dx, dw, db


# ---------------------------
# MaxPool Forward/Backward
# ---------------------------
def maxpool_forward(x, pool_param):
    """
    Forward pass for max-pooling layer.

    Input:
    - x: Input data of shape (N, C, H, W)
    - pool_param: Dictionary with pool parameters:
      - 'pool_height': Height of pooling regions
      - 'pool_width': Width of pooling regions
      - 'stride': Distance between pooling regions

    Output:
    - out: Output data of shape (N, C, H', W')
    - cache: Values needed for backward pass
    """
    pool_height = pool_param.get('pool_height', 2)
    pool_width = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    N, C, H, W = x.shape

    # Output dimensions
    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    # Initialize output and max indices
    out = np.zeros((N, C, out_height, out_width))
    max_indices = np.zeros((N, C, out_height, out_width, 2), dtype=int)

    # Perform max pooling
    for n in range(N):
        for c in range(C):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * stride
                    w_start = j * stride

                    # Extract pooling region
                    pool_region = x[n, c, h_start:h_start + pool_height, w_start:w_start + pool_width]

                    # Find maximum value and its index
                    max_idx = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                    out[n, c, i, j] = pool_region[max_idx]

                    # Store indices for backprop
                    max_indices[n, c, i, j] = [h_start + max_idx[0], w_start + max_idx[1]]

    cache = (x, max_indices, pool_param)
    return out, cache


def maxpool_backward(dout, cache):
    """
    Backward pass for max-pooling layer.

    Input:
    - dout: Upstream derivatives of shape (N, C, H', W')
    - cache: Values from forward pass

    Output:
    - dx: Gradient with respect to input x
    """
    x, max_indices, pool_param = cache
    N, C, H, W = x.shape

    # Initialize gradient
    dx = np.zeros_like(x)

    # Distribute gradient only to max elements
    for n in range(N):
        for c in range(C):
            for i in range(dout.shape[2]):
                for j in range(dout.shape[3]):
                    h_idx, w_idx = max_indices[n, c, i, j]
                    dx[n, c, h_idx, w_idx] += dout[n, c, i, j]

    return dx


# ---------------------------
# Global Average Pooling Forward/Backward
# ---------------------------
def global_avg_pool_forward(x):
    """
    Forward pass for global average pooling layer.

    Input:
    - x: Input data of shape (N, C, H, W)

    Output:
    - out: Output data of shape (N, C)
    - cache: Values needed for backward pass
    """
    N, C, H, W = x.shape
    out = np.mean(x, axis=(2, 3))  # Average across spatial dimensions
    cache = (x.shape,)
    return out, cache


def global_avg_pool_backward(dout, cache):
    """
    Backward pass for global average pooling layer.

    Input:
    - dout: Upstream derivatives of shape (N, C)
    - cache: Values from forward pass

    Output:
    - dx: Gradient with respect to input x
    """
    x_shape = cache[0]
    N, C, H, W = x_shape

    # Distribute gradient evenly across the spatial dimensions
    dx = np.zeros(x_shape)
    for n in range(N):
        for c in range(C):
            dx[n, c] = np.ones((H, W)) * dout[n, c] / (H * W)

    return dx


# ---------------------------
# Fully Connected Forward/Backward
# ---------------------------
def fc_forward(x, w, b):
    """
    Forward pass for fully connected layer.

    Input:
    - x: Input data of shape (N, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Output:
    - out: Output data of shape (N, M)
    - cache: Values needed for backward pass
    """
    out = x @ w + b
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Backward pass for fully connected layer.

    Input:
    - dout: Upstream derivatives of shape (N, M)
    - cache: Values from forward pass

    Output:
    - dx: Gradient with respect to input x
    - dw: Gradient with respect to weights w
    - db: Gradient with respect to biases b
    """
    x, w, b = cache

    dx = dout @ w.T
    dw = x.T @ dout
    db = np.sum(dout, axis=0)

    return dx, dw, db


# ---------------------------
# ReLU Forward/Backward
# ---------------------------
def relu_forward(x):
    """
    Forward pass for ReLU activation.

    Input:
    - x: Input data

    Output:
    - out: ReLU(x)
    - cache: Values needed for backward pass
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Backward pass for ReLU activation.

    Input:
    - dout: Upstream derivatives
    - cache: Input data from forward pass

    Output:
    - dx: Gradient with respect to x
    """
    x = cache
    dx = dout * (x > 0)
    return dx


# ---------------------------
# Softmax + Cross Entropy Loss
# ---------------------------
def softmax_loss(scores, y):
    """
    Softmax loss function with L2 regularization.

    Input:
    - scores: Output of shape (N, C) where C is the number of classes
    - y: One-hot encoded labels of shape (N, C)

    Output:
    - loss: Scalar loss value
    - dscores: Gradient with respect to scores
    """
    # Compute softmax probabilities
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute cross-entropy loss
    N = scores.shape[0]
    loss = -np.sum(y * np.log(probs + 1e-15)) / N

    # Compute gradient
    dscores = (probs - y) / N

    return loss, dscores, probs


# ---------------------------
# Three-Layer CNN Model
# ---------------------------
class ThreeLayerCNN:
    """
    Three-layer convolutional neural network with the structure:
    - Conv1 (3x3, 16 filters) + BN + ReLU + MaxPool
    - Conv2 (3x3, 32 filters) + BN + ReLU + MaxPool
    - Conv3 (3x3, 64 filters) + BN + ReLU
    - Global Average Pooling
    - Fully Connected (64 -> 10)
    - Softmax Classifier
    """

    def __init__(self, input_dim=(3, 32, 32), num_classes=10, reg=1e-4, dtype=np.float32):
        """
        Initialize the model parameters.

        Parameters:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_classes: Number of classes to classify
        - reg: L2 regularization strength
        - dtype: numpy datatype to use for computation
        """
        self.reg = reg
        self.dtype = dtype
        self.params = {}
        self.bn_params = []
        self.conv_params = []
        self.pool_params = []

        C, H, W = input_dim

        # Conv1 parameters: 16 filters, 3x3, stride 1, padding 1
        self.params['W1'] = np.random.randn(16, C, 3, 3) * np.sqrt(2.0 / (C * 3 * 3))
        self.params['b1'] = np.zeros(16)
        self.params['gamma1'] = np.ones(16)
        self.params['beta1'] = np.zeros(16)
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.bn_params.append({'mode': 'train', 'eps': 1e-5, 'momentum': 0.9})
        self.pool_params.append({'pool_height': 2, 'pool_width': 2, 'stride': 2})

        # Conv2 parameters: 32 filters, 3x3, stride 1, padding 1
        self.params['W2'] = np.random.randn(32, 16, 3, 3) * np.sqrt(2.0 / (16 * 3 * 3))
        self.params['b2'] = np.zeros(32)
        self.params['gamma2'] = np.ones(32)
        self.params['beta2'] = np.zeros(32)
        self.conv_params.append({'stride': 1, 'pad': 1})
        self.bn_params.append({'mode': 'train', 'eps': 1e-5, 'momentum': 0.9})
        self.pool_params.append({'pool_height': 2, 'pool_width': 2, 'stride': 2})

        # Conv3 parameters: 64 filters, 3x3, stride 1, padding 0
        self.params['W3'] = np.random.randn(64, 32, 3, 3) * np.sqrt(2.0 / (32 * 3 * 3))
        self.params['b3'] = np.zeros(64)
        self.params['gamma3'] = np.ones(64)
        self.params['beta3'] = np.zeros(64)
        self.conv_params.append({'stride': 1, 'pad': 0})
        self.bn_params.append({'mode': 'train', 'eps': 1e-5, 'momentum': 0.9})

        # FC parameters: 64 -> num_classes
        self.params['W4'] = np.random.randn(64, num_classes) * np.sqrt(2.0 / 64)
        self.params['b4'] = np.zeros(num_classes)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def forward(self, X, y=None, mode='train'):
        """
        Forward pass of the network.

        Parameters:
        - X: Input data of shape (N, C, H, W)
        - y: One-hot encoded labels of shape (N, C)
        - mode: 'train' or 'test'

        Returns:
        - If y is None: scores
        - If y is not None: (loss, gradients)
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']

        # Set batchnorm modes
        for bn_param in self.bn_params:
            bn_param['mode'] = mode

        # Forward pass for each layer
        # Layer 1: Conv + BN + ReLU + MaxPool
        conv1, cache_conv1 = conv_forward(X, W1, b1, self.conv_params[0])
        bn1, cache_bn1 = batchnorm_forward(conv1, gamma1, beta1, self.bn_params[0])
        relu1, cache_relu1 = relu_forward(bn1)
        pool1, cache_pool1 = maxpool_forward(relu1, self.pool_params[0])

        # Layer 2: Conv + BN + ReLU + MaxPool
        conv2, cache_conv2 = conv_forward(pool1, W2, b2, self.conv_params[1])
        bn2, cache_bn2 = batchnorm_forward(conv2, gamma2, beta2, self.bn_params[1])
        relu2, cache_relu2 = relu_forward(bn2)
        pool2, cache_pool2 = maxpool_forward(relu2, self.pool_params[1])

        # Layer 3: Conv + BN + ReLU
        conv3, cache_conv3 = conv_forward(pool2, W3, b3, self.conv_params[2])
        bn3, cache_bn3 = batchnorm_forward(conv3, gamma3, beta3, self.bn_params[2])
        relu3, cache_relu3 = relu_forward(bn3)

        # Global Average Pooling
        gap, cache_gap = global_avg_pool_forward(relu3)

        # Fully Connected Layer
        fc, cache_fc = fc_forward(gap, W4, b4)

        if y is None:
            return fc

        # Compute loss and gradients
        loss, dscores, probs = softmax_loss(fc, y)

        # Add L2 regularization
        loss += 0.5 * self.reg * (
                np.sum(W1 * W1) + np.sum(W2 * W2) +
                np.sum(W3 * W3) + np.sum(W4 * W4)
        )

        # Backward pass
        # FC layer
        dgap, dW4, db4 = fc_backward(dscores, cache_fc)

        # Global Average Pooling
        drelu3 = global_avg_pool_backward(dgap, cache_gap)

        # Layer 3: ReLU + BN + Conv
        dbn3 = relu_backward(drelu3, cache_relu3)
        dconv3, dgamma3, dbeta3 = batchnorm_backward(dbn3, cache_bn3)
        dpool2, dW3, db3 = conv_backward(dconv3, cache_conv3)

        # Layer 2: MaxPool + ReLU + BN + Conv
        drelu2 = maxpool_backward(dpool2, cache_pool2)
        dbn2 = relu_backward(drelu2, cache_relu2)
        dconv2, dgamma2, dbeta2 = batchnorm_backward(dbn2, cache_bn2)
        dpool1, dW2, db2 = conv_backward(dconv2, cache_conv2)

        # Layer 1: MaxPool + ReLU + BN + Conv
        drelu1 = maxpool_backward(dpool1, cache_pool1)
        dbn1 = relu_backward(drelu1, cache_relu1)
        dconv1, dgamma1, dbeta1 = batchnorm_backward(dbn1, cache_bn1)
        dX, dW1, db1 = conv_backward(dconv1, cache_conv1)

        # Add regularization gradients
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4

        # Store gradients
        grads = {
            'W1': dW1, 'b1': db1, 'gamma1': dgamma1, 'beta1': dbeta1,
            'W2': dW2, 'b2': db2, 'gamma2': dgamma2, 'beta2': dbeta2,
            'W3': dW3, 'b3': db3, 'gamma3': dgamma3, 'beta3': dbeta3,
            'W4': dW4, 'b4': db4
        }

        return loss, grads, probs

    def train(self, X, y, X_val=None, y_val=None,
              learning_rate=1e-3, learning_rate_decay=0.95, reg=1e-4,
              num_epochs=10, batch_size=32, verbose=True):
        """
        Train the network using stochastic gradient descent.

        Parameters:
        - X: Training data of shape (N, C, H, W)
        - y: One-hot encoded training labels of shape (N, num_classes)
        - X_val: Validation data of shape (N_val, C, H, W)
        - y_val: One-hot encoded validation labels of shape (N_val, num_classes)
        - learning_rate: Starting learning rate
        - learning_rate_decay: Learning rate decay factor
        - reg: L2 regularization strength
        - num_epochs: Number of epochs to train for
        - batch_size: Size of minibatches
        - verbose: Print training progress

        Returns:
        - loss_history: List of losses at each training iteration
        - train_acc_history: List of training accuracies
        - val_acc_history: List of validation accuracies
        """
        N = X.shape[0]
        num_train = N
        iterations_per_epoch = max(N // batch_size, 1)

        # Initialize histories
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for epoch in range(num_epochs):
            # Shuffle data at the start of each epoch
            shuffle_indices = np.random.permutation(N)
            X_shuffled = X[shuffle_indices]
            y_shuffled = y[shuffle_indices]

            # Mini-batch training
            for it in range(iterations_per_epoch):
                # Get mini-batch
                batch_start = it * batch_size
                batch_end = min(batch_start + batch_size, N)
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]

                # Forward pass
                loss, grads, _ = self.forward(X_batch, y_batch, mode='train')
                loss_history.append(loss)

                # Update parameters
                for param_name in self.params:
                    self.params[param_name] -= learning_rate * grads[param_name]

                # Print progress
                if verbose and it % 100 == 0:
                    print(
                        f'Epoch {epoch + 1}/{num_epochs}, Iteration {it + 1}/{iterations_per_epoch}, Loss: {loss:.4f}')

            # Decay learning rate at the end of each epoch
            learning_rate *= learning_rate_decay

            # Compute training and validation accuracy
            y_pred = np.argmax(self.forward(X, y=None, mode='test'), axis=1)
            y_true = np.argmax(y, axis=1)
            train_acc = np.mean(y_pred == y_true)
            train_acc_history.append(train_acc)

            if X_val is not None and y_val is not None:
                y_val_pred = np.argmax(self.forward(X_val, y=None, mode='test'), axis=1)
                y_val_true = np.argmax(y_val, axis=1)
                val_acc = np.mean(y_val_pred == y_val_true)
                val_acc_history.append(val_acc)

                if verbose:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
            else:
                if verbose:
                    print(f'Epoch {epoch + 1}/{num_epochs}, Train Acc: {train_acc:.4f}')

        return loss_history, train_acc_history, val_acc_history

    def save(self, filename):
        """
        Save the model parameters to a file.
        """
        data = {
            'params': self.params,
            'bn_params': self.bn_params,
            'conv_params': self.conv_params,
            'pool_params': self.pool_params,
            'reg': self.reg,
            'dtype': self.dtype
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filename):
        """
        Load model parameters from a file.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        self.params = data['params']
        self.bn_params = data['bn_params']
        self.conv_params = data['conv_params']
        self.pool_params = data['pool_params']
        self.reg = data['reg']
        self.dtype = data['dtype']


# ---------------------------
# Create Synthetic Dataset
# ---------------------------
def create_synthetic_dataset(num_samples=1000, num_classes=10, image_size=32, seed=42):
    """
    Create a synthetic dataset with simple patterns for testing the CNN.

    Parameters:
    - num_samples: Number of samples to generate
    - num_classes: Number of classes
    - image_size: Size of the images (height and width)
    - seed: Random seed

    Returns:
    - X: Input data of shape (num_samples, 3, image_size, image_size)
    - y: One-hot encoded labels of shape (num_samples, num_classes)
    """
    np.random.seed(seed)

    # Initialize data and labels
    X = np.zeros((num_samples, 3, image_size, image_size), dtype=np.float32)
    y = np.zeros((num_samples, num_classes), dtype=np.float32)

    # Generate data with class-specific patterns
    for i in range(num_samples):
        label = i % num_classes
        y[i, label] = 1.0

        # Create patterns based on class
        if label < 3:
            # Horizontal stripes
            X[i, :, ::4, :] = 1.0
        elif label < 6:
            # Vertical stripes
            X[i, :, :, ::4] = 1.0
        else:
            # Diagonal pattern
            for p in range(min(image_size, image_size)):
                X[i, :, p, p] = 1.0

        # Add noise
        X[i] += np.random.randn(3, image_size, image_size) * 0.1

    # Clip values to [0, 1]
    X = np.clip(X, 0, 1)

    return X, y


# ---------------------------
# Run Example Training
# ---------------------------
def train_example_network():
    """
    Train a CNN on a synthetic dataset.
    """
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    X, y = create_synthetic_dataset(1000, 10, 32, 42)

    # Split into training, validation, and test sets
    num_train = 800
    num_val = 100
    num_test = 100

    indices = np.random.permutation(X.shape[0])
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Initialize and train model
    print("Initializing model...")
    model = ThreeLayerCNN(input_dim=(3, 32, 32), num_classes=10, reg=1e-4)

    print("Training model...")
    loss_history, train_acc_history, val_acc_history = model.train(
        X_train, y_train, X_val, y_val,
        learning_rate=1e-3, learning_rate_decay=0.95,
        reg=1e-4, num_epochs=10, batch_size=32, verbose=True
    )

    # Evaluate model on test set
    print("Evaluating model...")
    y_test_pred = np.argmax(model.forward(X_test, y=None, mode='test'), axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    test_acc = np.mean(y_test_pred == y_test_true)
    print(f"Test accuracy: {test_acc:.4f}")

    # Plot learning curves
    print("Plotting learning curves...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title('Loss History')
    plt.plot(loss_history, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.title('Classification Accuracy')
    plt.plot(train_acc_history, '-o', label='train')
    plt.plot(val_acc_history, '-o', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    return model, loss_history, train_acc_history, val_acc_history


# Run the code if executed as a script
if __name__ == "__main__":
    train_example_network()