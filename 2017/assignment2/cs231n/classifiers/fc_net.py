from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

from past.builtins import xrange

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

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
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
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        #pass
        self.params = {}
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


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
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #pass
        X_local = X.reshape(X.shape[0], -1)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X_local.shape
        q1 = X_local.dot(W1) + b1  #(N, H)  'H' is hidden_size
        relu1_out = np.maximum(0, q1)
        scores = relu1_out.dot(W2) + b2 #(N, C)
        H = W1.shape[1]
        C = W2.shape[1]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #pass
        #loss
        scores = scores - np.max(scores, axis = 1).reshape(-1,1)
        exp_scores = np.exp(scores)
        exp_sum = np.sum(exp_scores, axis = 1).reshape(-1,1)  #always remember to use reshape(-1,1)!!
        correct_class_score = scores[range(N), y].reshape(-1,1)  
        loss_each_sample = -correct_class_score + np.log(exp_sum)
        loss = np.sum(loss_each_sample) / N + 0.5* self.reg* (np.sum(W1*W1) + np.sum(W2*W2))
        #gradient
        dsoftmax = exp_scores / exp_sum
        dsoftmax[range(N), y] -= 1
        dsoftmax /= N
        grads['W2'] = relu1_out.T.dot(dsoftmax) + self.reg*W2
        grads['b2'] = np.sum(dsoftmax, axis = 0)
        drelu1_out = dsoftmax.dot(W2.T)
        #dq1 = np.maximum(0, drelu1_out) wrong
        dq1 = (relu1_out > 0) * drelu1_out
        grads['W1'] = X_local.T.dot(dq1) + self.reg*W1
        grads['b1'] = np.sum(dq1, axis = 0)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
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
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        #pass
        dim = input_dim
        for idx, hd in enumerate(hidden_dims):
            self.params['w%d' % (idx+1)] = weight_scale * np.random.randn(dim, hd)
            self.params['b%d' % (idx+1)] = np.zeros(hd)
            #self.params['w%d_out' % (idx+1)] = 0
            if self.use_batchnorm:
                self.params['bn_gamma%d' % (idx+1)] = np.ones(hd)
                self.params['bn_beta%d' % (idx+1)] = np.zeros(hd)
            dim = hd
        #output layer
        self.params['w%d' % self.num_layers] = weight_scale * np.random.randn(dim, num_classes)
        self.params['b%d' % self.num_layers] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

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
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


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
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
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
        #pass
        cache = {}
        dp_cache = {}
        data_input = X
        for i in range(1, self.num_layers):
            if self.use_batchnorm == False:
                data_input, cache[i] = affine_relu_forward(data_input, self.params['w%d' % i], self.params['b%d' % i])
            elif self.use_batchnorm == True:
                data_input, cache[i] = affine_bn_relu_forward(data_input, self.params['w%d' % i], self.params['b%d' % i], self.params['bn_gamma%d' % i], self.params['bn_beta%d' % i], self.bn_params[i-1])
            if self.use_dropout:
                data_input, dp_cache[i] = dropout_forward(data_input, self.dropout_param)
        scores, cache[self.num_layers] = affine_forward(data_input, self.params['w%d' % self.num_layers], self.params['b%d' % self.num_layers])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #pass
        loss, dout = softmax_loss(scores, y)
        loss = loss + 0.5 * self.reg * np.sum(self.params['w%d'%(self.num_layers)] * self.params['w%d'%(self.num_layers)])
        dx, dw, db = affine_backward(dout, cache[self.num_layers])
        grads['w%d'%(self.num_layers)] = dw + self.reg * self.params['w%d'%(self.num_layers)]
        grads['b%d'%(self.num_layers)] = db
        dout = dx
        for i in range(self.num_layers-1, 0, -1):
            loss = loss + 0.5 * self.reg * np.sum(self.params['w%d'% i] * self.params['w%d'% i])
            if self.use_dropout:
                dout = dropout_backward(dout, dp_cache[i])
            if self.use_batchnorm == False:
                dx, dw, db = affine_relu_backward(dout, cache[i])
                dout = dx
            elif self.use_batchnorm == True:
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, cache[i])
                dout = dx
                grads['bn_gamma%d' % i] = dgamma
                grads['bn_beta%d' % i] = dbeta
            grads['w%d'% i] = dw + self.reg * self.params['w%d'% i]
            grads['b%d'% i] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
