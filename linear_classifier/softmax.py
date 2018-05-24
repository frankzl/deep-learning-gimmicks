"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np
import math

from .linear_classifier import LinearClassifier

def calc_percentage_scores(W, X):
    scores = dot_product(X, W)
    for idx, score in enumerate(scores):
        scores[idx] = calc_p(score)
    return (scores)

def calc_loss_data(percentage_score, y):
    loss_sum = 0
    for idx, score in enumerate(percentage_score):
        loss_sum += calc_loss_data_i(score, y[idx])
    return loss_sum/y.shape[0]

def calc_loss_data_i(percentage_score_i, y):
    return -1*math.log(percentage_score_i[y])
    
def calc_p(class_scores):
    val_sum = 0
    for val in class_scores:
        val_sum += math.exp(val)
    percentage_score = np.array(class_scores)
    for idx in range(0, len(class_scores)):
        percentage_score[idx] = math.exp(class_scores[idx])/val_sum
    return percentage_score

def calc_loss_reg(W, reg):
    loss_sum = 0
    for row in range(0,W.shape[0]):
        for col in range(0, W.shape[1]):
            loss_sum += W[row, col]**2
    loss_sum *= reg/2
    return loss_sum

def dot_product( W, X ):
    product = np.zeros((W.shape[0], X.shape[1]))
    for row in range(0, W.shape[0]):
        for col in range(0, X.shape[1]):
            row_sum = 0
            for row_ele in range(0, W.shape[1]):
                row_sum += W[row, row_ele] * X[row_ele, col]
            product[row, col] = row_sum
    return product

def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    p_scores = calc_percentage_scores(W,X)
    loss += calc_loss_reg( W, reg )
    loss += calc_loss_data( p_scores, y)

    for idx in range(0,len(p_scores)):
        p_scores[idx, y[idx]] -= 1
        
    # p_scores = np.dot(X.T, p_scores) 
    new_scores = dot_product( X.T, p_scores )
    p_scores = new_scores
    for idx in range(0,len(p_scores)):
        for i in range(0, len(p_scores[idx])):
            p_scores[idx, i] /= y.shape[0]
    dW = p_scores
    dW += reg*W

    return loss, dW

def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    amount = y.shape[0]
    exp_scores = np.exp(np.dot(X,W))
    p_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    log_scores = -np.log(p_scores[range(amount),y])
    loss += np.sum(log_scores)/amount

    loss += reg*np.sum(W*W)/2
    
    p_scores[range(amount), y] -= 1
    p_scores /= amount
    p_scores = np.dot(X.T, p_scores) 
    dW = p_scores
    dW += reg*W

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)

    
def get_accuracy(softmax, X_train, y_train, X_val, y_val):
    y_train_pred = softmax.predict(X_train)
    training_acc = np.mean(y_train == y_train_pred)
    y_val_pred = softmax.predict(X_val)
    val_acc = np.mean(y_val == y_val_pred)
    return training_acc, val_acc

def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [5e-6, 5e-7, 5e-8]
    regularization_strengths = [1.5e4, 1e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    
    for rate in learning_rates:
        for reggae in regularization_strengths:
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate=rate, reg=reggae,
                          num_iters=1500, batch_size=200, verbose=False)
            train_acc, val_acc = get_accuracy(softmax, X_train, y_train, X_val, y_val)
            results[(rate, reggae)] = (train_acc, val_acc)
            
            all_classifiers.append(softmax)
            if best_val < val_acc:
                best_val = val_acc
                best_softmax = softmax
    

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
