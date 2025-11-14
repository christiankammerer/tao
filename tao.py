from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from sklearn import tree
import matplotlib.pyplot as plt

class TAOTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_leaf=5, random_state=None,
                 type = "oblique", max_passes=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.type = type
        self.max_passes = max_passes

    def fit(self, X, y):
        self.X = X
        self.y = y
        base = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )

        base.fit(self.X, self.y)
        self.base_tree_ = base
        self.tree_ = self.base_tree_.tree_
        self.care_set_indices = self.base_tree_.decision_path(self.X)
        self._compute_depths()
        
        for _ in range(self.max_passes):
            self.optimize()
            self.reroute()
        self.prune_tree()

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def plot_base_tree(self, feature_names=None, class_names=None):
        plt.figure(figsize=(12, 6))
        tree.plot_tree(
            self.base_tree_,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True
        ) 
        plt.show()

    def _compute_depths(self):
        # as taken from sklearn documentation
        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right

        depth = np.zeros(n_nodes, dtype=int)
        stack = [(0, 0)]  # (node_id, depth), root is 0

        while stack:
            node_id, d = stack.pop()
            depth[node_id] = d
            left = children_left[node_id]
            right = children_right[node_id]
            if left != right: # check if node is internal node
                stack.append((left, d + 1)) # add children to stack and increment depth
                stack.append((right, d + 1))

        self.node_depth_ = depth

    def optimize(self):
        """
        Optimizes the tree by optimizing each internal node. 
        Nodes at the same depth are independent and can be optimized in parallel.
        """
        print(self.node_depth_)
                
    
    def reroute(self):
        """ Reroute samples through the optimized tree structure """
        pass

    def optimize_oblique_node(self, node_id):
        """ Optimize an oblique split at the given node """
        pass

    def optimize_axis_aligned_node(self, node_id):
        """ Optimize an axis-aligned split at the given node """
        pass

    def prune_tree(self):
        """Prune the tree to remove dead or pure branches"""
        pass