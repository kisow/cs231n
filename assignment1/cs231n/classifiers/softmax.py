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
  # DONE: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  D, C = W.shape
  N = y.shape[0]
  for i in range(N):
    f = np.zeros(C)
    for j in range(C):
      f[j] = X[i].dot(W.T[j])

    # for numeric stability
    m = np.finfo(np.float32).min;
    for j in range(C):
      if m < f[j]:
        m = f[j]
    for j in range(C):
      f[j] -= m;

    # calc Li to accmulate
    s = 0
    p = np.zeros(C)
    for j in range(C):
      p[j] = np.exp(f[j])
      s += p[j]
    for j in range(C):
      p[j] /= s
    loss += -1 * np.log(p[y[i]])

    p[y[i]] -= 1.0
    for j in range(C):
      dW[:, j] += p[j] * X[i]

  loss /= N
  dW /= N

  # Add regularization to the loss
  loss += reg * 0.5 * np.sum(W * W)
  dW += reg * W
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
  # DONE: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  D, C = W.shape
  N = X.shape[0]

  F = X.dot(W) # N * C
  F = (F.T - np.max(F.T, axis=0)).T # for numeric stability
  F = np.exp(F)
  P = (F.T / np.sum(F.T, axis=0)).T # N * C

  arange_N = np.arange(N)

  # negative sum of the probability of correct class.
  loss = -np.log(P[arange_N, y]).sum() 

  P[arange_N, y] -= 1 # P -= y, y=[00..1..00]
  dW = X.T.dot(P) # D * C

  # Add regularization to the loss
  loss /= N
  dW /= N
  
  # Add regularization to the loss
  loss += reg * 0.5 * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

