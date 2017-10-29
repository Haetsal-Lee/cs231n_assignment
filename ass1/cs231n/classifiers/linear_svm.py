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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dimension = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        #for k in xrange(num_dimension):#winbaram; in margin = scores[j] - correct_class_score + 1
        #  dW[k, j] += X[i, k] #winbaram; part of scores[j]
        #  dW[k, y[i]] -= X[i, k] #winbaram; part of correct_class_score
        dW[:,j]     +=  X[i,:].T
        dW[:, y[i]] -=  X[i, :].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train #winbaram
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += (reg * 2) *W #winbaram

  #############################################################################
  # TODO:                                                                     #
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dimension = X.shape[1]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_class_score = np.reshape( scores[np.arange(num_train),y], (num_train, 1) )
  margin = (scores - correct_class_score) + 1
  #apply class_mask
  class_mask = np.ones((num_train, num_classes))
  class_mask[np.arange(num_train), y] = 0;
  margin *= class_mask
  #mask margin<=1
  margin[margin<=0] = 0
  #sum all margin values
  loss += np.sum(margin)
  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin[margin>0]=1;#winbaram; part of scores[j]
  margin_sum  = np.sum(margin, axis=1) # shape=(500,1)
  margin[np.arange(num_train), y ] = -1 * margin_sum #winbaram; part of correct_class_score shape=(500,10)
  dW = np.dot(X.T, margin)
  
  dW /= num_train #winbaram
  dW += (reg * 2) *W #winbaram
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
