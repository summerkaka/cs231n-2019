import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    print('num_test is: %d, num_train is : %d' % (num_test, num_train))
    print('type of X[i]', type(X[0]))
    print('X[0]:', X[0], 'X[0][0]:', X[0][0])
    print('length of X[i], X_train[i]:', len(X[0]), len(self.X_train[0]))
    dists = np.zeros((num_test, num_train))
    print('start 2-loop.....')
    for i in xrange(num_test):
#      print('calculate %d test sample' % i)
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
#        pass
#        my code:
#        dist = [(a-b)**2 for a, b in zip(X[i], self.X_train[j])]
#        print('train%d test%d dist' % (j,i), 'length %d' % len(dist))
#        dists[i][j] = np.sum(np.array(dist))
#       based on syllabus: run speed is very fast
        dists[i][j] = np.sqrt(np.sum(np.square(self.X_train[j,:] - X[i,:]), axis = 0))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    print('finish 2-loop')
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
#      pass
      dists[i,:] = np.sqrt(np.sum(np.square(self.X_train - X[i,:]), axis = 1))
#      print('test %d data' % i)
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
#    pass
# my code
    # (a-b)**2 = a**2 - 2ab + b**2
    # x_train: 5000 x 3072, x_test: 500 x 3072
    A = np.sum(self.X_train**2, axis=1).reshape(num_train, -1)
    print('A shape', A.shape)
    B = np.sum(X**2, axis=1).reshape(-1, num_test)
    print('B shape', B.shape)
    AB = np.dot(self.X_train, X.T)
    print('AB shape', AB.shape)
    dists = A - 2*AB
    dists += B
    dists = dists.T
    dists = np.sqrt(dists)
    print('dists shape', dists.shape)
# from github
#    M = np.dot(X, self.X_train.T)
#    print M.shape
#    te = np.square(X).sum(axis = 1)
#    print te.shape
#    tr = np.square(self.X_train).sum(axis = 1)
#    print tr.shape
#    dists = np.sqrt(-2*M+tr+np.matrix(te).T) #tr add to line, te add to row
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
# my code (is right)
      k_lable = np.argsort(dists[i])[0:k]
      closest_y = [self.y_train[idx] for idx in k_lable]
# from github
#      closest_y = self.y_train[np.argsort(dists[i])[:k]]
#      pass
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
# my code
      y_pred[i] = max(closest_y, key = closest_y.count)
#      print('%d test data:' % i, closest_y, y_pred[i])
# from github
#      y_pred[i] = np.argmax(np.bincount(closest_y))
#      pass
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

