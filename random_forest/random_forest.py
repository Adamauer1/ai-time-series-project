import numpy as np

class RegressionTree:
    def __init__(self, max_depth = None, min_split_amount = 2):
        self.max_depth = max_depth
        self.min_split_amount = min_split_amount
        self.tree = None

    class Node:
        def __init__(self, feature = None, threshold = None, left = None, right = None, reduction_value = None, leaf_value = None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.reduction_value = reduction_value
            self.leaf_value = leaf_value
# This splits the data based on the feature and the threshold value
    def split_data(self, x, y, feature, threshold):
        left = x[:, feature] <= threshold
        right = x[:, feature] > threshold
        return x[left], x[right], y[left], y[right]

# Finding which feature and threshold combo will generate the best split
    def find_best_split(self, x, y):
        best_feature = None
        best_threshold = None
        best_reduction = -np.inf
        # loops through each feature
        for feature in range(x.shape[1]):
            # gets all the unique thresholds for the current feature
            # a threshold is the value of the feature
            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                # splits the data based on the feature / threshold combo
                # only the target values matter in this split
                _, _, y_left, y_right = self.split_data(x,y,feature, threshold)
                # checks if the split would lead to a side having a smaller amount of values than the min_split_amount
                if len(y_left) < self.min_split_amount or len(y_right) < self.min_split_amount:
                    continue
                # calculates the variance reduction
                # this number is the spread of the data between each side
                reduction = self.calculate_variance_reduction(y, y_left, y_right)
                # reassigning values if reduction value is better
                if reduction > best_reduction:
                    best_threshold = threshold
                    best_feature = feature
                    best_reduction = reduction
        
        return best_feature, best_threshold, best_reduction

    # function to apply the variance reduction formula
    def calculate_variance_reduction(self, y, y_left, y_right):
        return np.var(y) - ((len(y_left)/len(y)) * np.var(y_left) + (len(y_right) / len(y)) * np.var(y_right))

    # function to build out regression tree
    def build_tree(self, x, y, depth=0):
        # check if leaf node
        if len(y) < self.min_split_amount or depth >= self.max_depth:
            return self.Node(leaf_value = np.mean(y))

        # find best split data
        feature, threshold, best_reduction = self.find_best_split(x, y)
        # check to make sure there is a feature selected
        if feature is None:
            return self.Node(leaf_value= np.mean(y))
        # split data on base of the feature
        x_left, x_right, y_left, y_right = self.split_data(x, y, feature, threshold)

        # continue building the tree to the left
        left = self.build_tree(x_left, y_left, depth+1)

        # continue building the tree to the right
        right = self.build_tree(x_right, y_right, depth+1)

        return self.Node(feature= feature, threshold= threshold, left= left, right=right, reduction_value=best_reduction)
# This just builds the tree
    def fit(self, x, y):
        self.tree = self.build_tree(x,y)

# This function searches through the tree in order to find the predicted value
    def predict_value(self, x, node):
        if node.leaf_value is not None:
            return node.leaf_value
        if x[node.feature] <= node.threshold:
            return self.predict_value(x, node.left)
        else:
            return self.predict_value(x, node.right)

# This will loop through all the test values and make an array of predictions
    def predict(self, X):
        return np.array([self.predict_value(x, self.tree) for x in X])

class RandomForest:
    def __init__(self, n_trees = 3, max_depth = None, min_split_amount = 2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_split_amount = min_split_amount
        self.trees = []

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.n_trees):
            indices = np.random.choice(range(x.shape[0]), size=x.shape[0], replace=True)
            x_tree = x[indices]
            y_tree = y[indices]
            tree = RegressionTree(max_depth=self.max_depth, min_split_amount=self.min_split_amount)
            tree.fit(x_tree,y_tree)
            self.trees.append(tree)

    def predict(self, x):
        return np.mean(np.array([tree.predict(x) for tree in self.trees]), axis=0)


