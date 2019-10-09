from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    bp_log = 0
    bp_exp = np.zeros((num_classes, 1))
    dW = np.zeros_like(W)

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        sum_exp = 0
        for j in range(num_classes):
            exp = np.exp(scores[j])
            bp_exp[j] = exp
            sum_exp += exp

        log = np.log(sum_exp)
        bp_log = 1 / sum_exp   #back_prop of log
        loss += log - correct_class_score

        for j in range(num_classes):
            dW[:, j] += X[i].T * bp_exp[j] * bp_log
        dW[:, y[i]] -= X[i].T

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    #pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_class = W.shape[1]
    dW = np.zeros_like(W)

    scores = X.dot(W)
    scores_exp = np.exp(scores)
    exp_sum = np.sum(scores_exp, axis=1)
    loss = np.sum(np.log(exp_sum)) - np.sum(scores[range(num_train), y])
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    d_log = np.reshape((1 / exp_sum), (-1,1)) * np.ones((1, num_class))
    d_exp = scores_exp
    a = d_log * d_exp
    a[range(num_train), y] -= 1
    dW = X.T.dot(a)

    dW /= num_train
    dW += reg * W
    # pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
