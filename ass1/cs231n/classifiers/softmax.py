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
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dimension = X.shape[1]
  loss = 0.0
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)
    exped_correct_class_score = np.exp(scores[y[i]])
    sum_exped_score_except_correct_class = 0.0
    sum_exped_score = 0.0
    for j in xrange(num_classes):
        sum_exped_score += np.exp(scores[j])
        if j==y[i]:
            continue
        else:
            sum_exped_score_except_correct_class += np.exp(scores[j])
    loss += -np.log(exped_correct_class_score/sum_exped_score)

    #dW[:,y[i]] += -(sum_exped_score_except_correct_class / sum_exped_score) * X[i,:].T
    dW[:,y[i]] += (np.exp(scores[y[i]])/ sum_exped_score -1) * X[i,:].T
    for j in xrange(num_classes):
        if j != y[i]:
            dW[:,j] += (np.exp(scores[j]) / sum_exped_score) * X[i,:].T
        #dW[:,j]     +=  X[i,:].T
        #dW[:, y[i]] -=  X[i, :].T
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train #winbaram
  dW += (reg * 2) *W #winbaram

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dimension = X.shape[1]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) # X=(N,D) * W=(D,C) => scores=(N,C)
  scores_exp = np.exp(scores) #(N,C)
  correct_scores_exp = scores_exp[np.arange(num_train), y] #(N,1)
  sum_scores_exp = np.sum(scores_exp, axis=1) #(N,1)
  loss = np.sum(-np.log( correct_scores_exp / sum_scores_exp))
  elm_over_sum_scores_exp = scores_exp / np.reshape(sum_scores_exp, (num_train, 1))#(N,C)
  elm_over_sum_scores_exp[np.arange(num_train), y] = elm_over_sum_scores_exp[np.arange(num_train), y]-1#(N,C)
  dW = np.dot(X.T, elm_over_sum_scores_exp)# X_t = (D,N) * elm_over_sum_scores_exp = (N,C) => dW = (D,C)

  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train #winbaram
  dW += (reg * 2) *W #winbaram
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

