import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fontTools.ttLib.tables.F__e_a_t import Feature
from numpy.ma.core import left_shift


# class Node:
#     def __init__(self, feature, threshold, left, right, split_value, leaf_value):
#         self.feature = feature
#         self.threshold = threshold
#         self.left = left
#         self.right = right
#         self.split_value = split_value
#         self.leaf_value = leaf_value

class RegressionTree:
    def __init__(self, max_depth = None, min_split = 2):
        self.max_depth = max_depth
        self.min_split = min_split
        self.tree = None

    class Node:
        def __init__(self, feature = None, threshold = None, left = None, right = None, split_value = None, leaf_value = None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.split_value = split_value
            self.leaf_value = leaf_value

    def split_data(self, x, y, feature, threshold):
        left = x[:, feature] <= threshold
        right = x[:, feature] > threshold
        return x[left], x[right], y[left], y[right]

    def find_best_split(self, x, y):
        best_feature = None
        best_threshold = None
        best_reduction = -np.inf
        for feature in range(x.shape[1]):
            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                x_left, x_right, y_left, y_right = self.split_data(x,y,feature, threshold)
                if len(y_left) < self.min_split or len(y_right) < self.min_split:
                    continue
                reduction = self.calculate_variance_reduction(y, y_left, y_right)
                if reduction > best_reduction:
                    best_threshold = threshold
                    best_feature = feature
                    best_reduction = reduction
        
        return best_feature, best_threshold
                
    def calculate_variance_reduction(self, y, y_left, y_right):
        total_var = np.var(y) * len(y)
        left_var = np.var(y_left) * len(y)
        right_var = np.var(y_right) * len(y)
        return total_var - (left_var + right_var)

    def build_tree(self, x, y, depth=0):
        # check for leaf nodes
        if len(y) < self.min_split or depth >= self.max_depth:
           return self.Node(leaf_value = np.mean(y)) 
        
        feature, threshold = self.find_best_split(x, y)
        if feature is None:
            return self.Node(leaf_value= np.mean(y))
        x_left, x_right, y_left, y_right = self.split_data(x, y, feature, threshold)
        left = self.build_tree(x_left, y_left)
        right = self.build_tree(x_right, y_right)
        return self.Node(feature= feature, threshold= threshold, left= left, right=right)

    def fit(self, x, y):
        self.tree = self.build_tree(x,y)

    # change
    def _predict_one(self, x, node):
        if node.leaf_value is not None:
            return node.leaf_value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

