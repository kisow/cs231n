import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  delta = 1
  C = W.shape[1]
  N = X.shape[0]
  loss = 0.0
  for i in xrange(N):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(C):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + delta
      if margin > 0:
        loss += margin
        dW[:, j] += X[i].T
        dW[:, y[i]] -= X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by N.
  loss /= N
  dW /= N

  # Add regularization to the loss.
  loss += reg * 0.5 *np.sum(W * W)
  dW += reg * W

  #############################################################################
  # DONE:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # DONE:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  arange_N = np.arange(N)

  delta = 1
  scores = X.dot(W) # N * C
  c_scores = scores[arange_N, y] # N * 1
  # transpose to broadcast and then re-transpose
  margins = np.maximum(0, (scores.T - c_scores.T).T + delta) # N * C
  margins[arange_N, y] = 0 # ignore if j == yi
  loss = np.sum(margins) / N
  loss += reg * 0.5 * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # DONE:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  coeff = (margins > 0) * 1 # N * C, margins[i][j] == 1 if j != yi and activated
  j_counts = -1 * np.sum(coeff, axis=1) # N * 1
  coeff[arange_N, y] = j_counts # N * C, pairwise update to correct counts, j == yi
  dW = X.T.dot(coeff) / N # D * C, minus by correct times and plus by incorrect times.
  dW += reg * W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
