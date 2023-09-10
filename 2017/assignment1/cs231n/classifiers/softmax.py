import numpy as np
from random import shuffle
from past.builtins import xrange
import math

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
  #pass
  #my code
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    raw_scores = X[i].dot(W)
    scores = raw_scores - max(raw_scores)  # refer to lecture note, this is for data stable
    correct_class_score = scores[y[i]]
    loss_i = -correct_class_score + np.log(sum(np.exp(scores)))
    loss += loss_i
    for j in xrange(num_classes):
      softmax_output = np.exp(scores[j])/sum(np.exp(scores))
      if j == y[i]:
        dW[:,j] += (-1 + softmax_output) *X[i] 
      else: 
        dW[:,j] += softmax_output *X[i]

  loss /= num_train
  loss +=  0.5* reg * np.sum(W * W)
  dW = dW/X.shape[0] + reg* W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
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
  #pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  raw_scores = X.dot(W)
  scores = raw_scores - np.max(raw_scores, axis = 1).reshape(-1,1)
  exp_scores = np.exp(scores)
  exp_sum = np.sum(exp_scores, axis = 1).reshape(-1,1)  #always remember to use reshape(-1,1)!!
  correct_class_score = scores[range(num_train), y].reshape(-1,1)  
  loss_each_sample = -correct_class_score + np.log(exp_sum)
  loss = np.sum(loss_each_sample) / num_train + 0.5 * reg * np.sum(W * W)
  
  softmax_output = exp_scores / exp_sum
  softmax_output[range(num_train), y] -= 1
  dW = X.T.dot(softmax_output)
  dW = dW/num_train + reg* W 

  # from github
#  num_classes = W.shape[1]
#  num_train = X.shape[0]
#  scores = X.dot(W)
#  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
#  softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
#  loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
#  loss /= num_train 
#  loss +=  0.5* reg * np.sum(W * W)
#  
#  dS = softmax_output.copy()
#  dS[range(num_train), list(y)] += -1
#  dW = (X.T).dot(dS)
#  dW = dW/num_train + reg* W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

