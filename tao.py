from sklearn.base import BaseEstimator, ClassifierMixin # make basic sklearn classifier functions available through inheritance
from sklearn.preprocessing import StandardScaler # standard scale features to allow for convergence in logistic regression
from sklearn.tree import DecisionTreeClassifier # base decision tree classifier
from sklearn.linear_model import LogisticRegression # logistic regression for oblique splits
import numpy as np 

from sklearn import tree
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

class TAOTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_leaf=5, random_state=None,
                 type = "oblique", max_passes=5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.type = type
        self.max_passes = max_passes
    
    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        self.X_ = self.scaler_.fit_transform(X)
        self.y_ = y
        base = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )

        base.fit(self.X_, self.y_)
        self.base_tree_ = base
        self.tree_ = self.base_tree_.tree_
        
        node_indicator = self.base_tree_.decision_path(self.X_).tocsc()
        self.care_sets_ = [
            node_indicator[:, j].indices  # row indices of non-zero entries
            for j in range(node_indicator.shape[1])
        ]
        self._compute_depths()
        self.weights_ = np.zeros((self.base_tree_.tree_.node_count, X.shape[1]))  # placeholder
        self.biases_ = np.zeros(self.base_tree_.tree_.node_count)      # placeholder

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
        for depth in reversed(range(self.max_depth)):
            node_ids_at_depth = np.where(self.node_depth_ == depth)[0]
            results = Parallel(n_jobs=-1)(
                delayed(self._compute_oblique_params_for_node)(node_id)
                for node_id in node_ids_at_depth
            )
                
            for node_id, params in results:
                if self.type == "oblique":
                    self._apply_oblique_params(node_id, params)

                """
                else:
                    self._apply_axis_aligned_params(node_id, params)
                """
    
    def reroute(self):
        """ Reroute samples through the optimized tree structure """
        pass

    def _compute_oblique_params_for_node(self, node_id):
        """
        Compute the oblique parameters (w, b) for the given node, without modifying the tree to prevent race conditions
        """

        X_node = self.X_[self.care_sets_[node_id]]
        y_node = self.y_[self.care_sets_[node_id]]

        logreg = LogisticRegression().fit(X_node, y_node)
        w, b = logreg.coef_, logreg.intercept_

        return node_id, (w, b)

    def _apply_oblique_params(self, node_id, params):
        w, b = params
        # store them somewhere:
        self.weights_[node_id] = w
        self.biases_[node_id] = b

    def prune_tree(self):
        """Prune the tree to remove dead or pure branches"""
        pass