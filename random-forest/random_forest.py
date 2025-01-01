import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature, threshold, left, right, value):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth = None):
        self.max_depth = max_depth
        self.tree = None

    def split_data(self, x, y, feature, threshold):
        NotImplemented()

    def calculate_variance_reduction(self, y, y_left, y_right):
        NotImplemented()

    def fit(self, X, y):
        NotImplemented()
