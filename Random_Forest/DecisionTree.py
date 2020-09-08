# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
# Decision Tree for continuous-vector input, binary output
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np

def entropy(y):
    # assume y is binary - 0 or 1
    N = len(y)
    pY = np.bincount(y)
    pY = pY[pY>0]/N
    return -pY.dot(np.log2(pY))

def gini(y):
    N = len(y)
    pY = np.bincount(y)
    pY = pY[pY>0]/N
    return 1-pY.dot(pY)

class TreeNode:
    def __init__(self, depth=1, max_depth=None):
        #print('depth:', depth)
        self.depth = depth
        self.max_depth = max_depth
        if self.max_depth is not None and self.max_depth < self.depth:
            raise Exception("depth > max_depth")

    def fit(self, X, Y, criterion = 'entropy'):
        if len(Y) == 1 or len(set(Y)) == 1:
            # base case, only 1 sample
            # another base case
            # this node only receives examples from 1 class
            # we can't make a split
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]

        else:
            D = X.shape[1]
            cols = range(D)

            max_ig = 0
            best_col = None
            best_split = None
            for col in cols:
                ig, split = self.find_split(X, Y, col, criterion)
                # print "ig:", ig
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split

            if max_ig == 0:
                # nothing we can do
                # no further splits
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.bincount(Y).argmax()
            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth:
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.bincount(Y[X[:,best_col] < self.split]).argmax(),
                        np.bincount(Y[X[:,best_col] >= self.split]).argmax(),
                    ]
                else:
                    # print "best split:", best_split
                    left_idx = (X[:,best_col] < best_split)
                    # print "left_idx.shape:", left_idx.shape, "len(X):", len(X)
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = TreeNode(self.depth + 1, self.max_depth)
                    self.left.fit(Xleft, Yleft)

                    right_idx = (X[:,best_col] >= best_split)
                    Xright = X[right_idx]
                    Yright = Y[right_idx]
                    self.right = TreeNode(self.depth + 1, self.max_depth)
                    self.right.fit(Xright, Yright)

    def find_split(self, X, Y, col, criterion = 'entropy'):
        # print "finding split for col:", col
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]

        # Note: optimal split is the midpoint between 2 points
        # Note: optimal split is only on the boundaries between 2 classes

        # if boundaries[i] is true
        # then y_values[i] != y_values[i+1]
        # nonzero() gives us indices where arg is true
        # but for some reason it returns a tuple of size 1
        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_ig = 0
        for b in boundaries:
            split = (x_values[b] + x_values[b+1]) / 2
            ig = self.information_gain(x_values, y_values, split, criterion)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split

    def information_gain(self, x, y, split, criterion = 'entropy'):
        # assume classes are 0 and 1
        # print "split:", split
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0 #float(len(y1)) / N
        # print "entropy(y):", entropy(y)
        # print "p0:", p0
        # print "entropy(y0):", entropy(y0)
        # print "p1:", p1
        # print "entropy(y1):", entropy(y1)
        if criterion == 'entropy':
            return entropy(y) - p0*entropy(y0) - p1*entropy(y1)
        else:
            return -(p0*gini(y0) + p1*gini(y1))

    def predict_one(self, x):
        # use "is not None" because 0 means False
        if self.col is not None and self.split is not None:
            feature = x[self.col]
            if feature < self.split:
                if self.left:
                    p = self.left.predict_one(x)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]
        else:
            # corresponds to having only 1 prediction
            p = self.prediction
        return p

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P


# This class is kind of redundant
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y, d = 0, criterion = 'entropy'):
        D = len(X[0])
        if d > 0:
            sel = np.random.choice(D, size = d, replace = False)
            Xsel = X[:,sel]
            Ysel = Y
        else:
            Xsel = X
            Ysel = Y
        self.root = TreeNode(max_depth=self.max_depth)
        self.root.fit(Xsel, Ysel, criterion)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
