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
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # e_score = np.exp(score - np.max(score))
  # class_score = e_score/ e_score.sum(axis = 0)

  for i in range(num_train):
    score = X[i].dot(W)
    shift_score = score - max(score)

    loss += -shift_score[y[i]] + np.log(np.sum(np.exp(shift_score)))

    for j in range(num_class):
      softmax_out = np.exp(shift_score[j])/np.sum(np.exp(shift_score))
      if j == y[i]:
        dW[:,j] += (-1 + softmax_out) * X[i]
      else:
        dW[:,j] += softmax_out*X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW /= num_train
  dW += reg*W


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
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  shift_score = (score.T - np.max(score, axis=1).T).T # to make broadcast work
  loss = -np.sum(shift_score[range(num_train),list(y)]) + np.sum(np.log(np.sum(np.exp(shift_score),axis=1)))
  loss = loss/num_train + 0.5 * reg * np.sum(W*W)

  coeff_mat = np.zeros((num_train,num_class))
  coeff_mat[range(num_train),list(y)] = -1
  soft_matrix = (np.exp(shift_score).T/ np.exp(shift_score).sum(axis=1).T).T
  # print("coeff", coeff_mat.shape)
  # print("X",X.shape)
  dW += (X.T).dot(coeff_mat)+(X.T).dot(soft_matrix)
  dW = dW/num_train + reg * W
  # good job~

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

