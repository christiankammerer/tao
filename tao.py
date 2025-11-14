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
                 type = "oblique", max_passes=5, C=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.type = type
        self.max_passes = max_passes
        self.C = C

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
        self.base_tree_ = base # store the original cart tree, purely archival
        self.tree_ = self.base_tree_.tree_
        
        n_nodes = self.base_tree_.tree_.node_count

        node_indicator = self.base_tree_.decision_path(self.X_).tocsc()
        self.node_sets_ = [
            node_indicator[:, j].indices  # row indices of non-zero entries
            for j in range(node_indicator.shape[1])
        ]
        self._compute_depths()
        self.weights_ = np.zeros((n_nodes, self.X_.shape[1]))  # placeholder
        self.biases_ = np.zeros(n_nodes)                       # placeholder
        self.oblique_active_ = np.zeros(n_nodes, dtype=bool) # tracks which nodes have been turned oblique, relevant for traversal

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
            
    
    def _descend_from(self, x, start_node):
        node_id = start_node

        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        feature = self.tree_.feature
        threshold = self.tree_.threshold

        while children_left[node_id] != -1:  # while not leaf
            if self.type == "oblique" and self.oblique_active_[node_id]:
                # use learned oblique split
                w = self.weights_[node_id]
                b = self.biases_[node_id]
                score = np.dot(w, x) + b      # or w @ x
                go_left = score <= 0.0
            else:
                # use original axis-aligned split
                fid = feature[node_id]
                thr = threshold[node_id]
                go_left = x[fid] <= thr

            if go_left:
                node_id = children_left[node_id]
            else:
                node_id = children_right[node_id]

        return node_id  # leaf node id


    def _leaf_pred_label(self, leaf_node_id):
        counts = self.tree_.value[leaf_node_id, 0]
        class_idx = np.argmax(counts)
        return self.base_tree_.classes_[class_idx]

    def _compute_care_set(self, node_id, node_set):
        """
        We are only interested in samples for which the left and right child have different losses.
        This function computes the so called care set for a given node and the corresponding targets.
        """
        if self.tree_.children_left[node_id] == -1:
            return np.array([], dtype=int), np.array([], dtype=int) # leaf nodes do not have care sets

        y_true = self.y_[node_set]

        leaves_l = [self._descend_from(self.X_[idx], self.tree_.children_left[node_id]) for idx in node_set]
        leaves_r = [self._descend_from(self.X_[idx], self.tree_.children_right[node_id]) for idx in node_set]
        labels_l = [self._leaf_pred_label(leaf_l) for leaf_l in leaves_l]
        labels_r = [self._leaf_pred_label(leaf_r) for leaf_r in leaves_r]

        losses_l = (labels_l != y_true).astype(int)
        losses_r = (labels_r != y_true).astype(int)

        care_mask = losses_l != losses_r               
        care_indices = node_set[care_mask]            
        targets = np.where(losses_l[care_mask] < losses_r[care_mask], 0, 1)
        return care_indices, targets

    def reroute(self):
        """ Reroute samples through the optimized tree structure """
        pass    

    def _compute_oblique_params_for_node(self, node_id):
        """
        Compute the oblique parameters (w, b) for the given node, without modifying the tree to prevent race conditions
        """

        care_indices , targets = self._compute_care_set(node_id, self.node_sets_[node_id])

        X_node = self.X_[care_indices]
        
        logreg = LogisticRegression(C=self.C).fit(X_node, targets)
        w, b = logreg.coef_, logreg.intercept_
        return node_id, (w, b)

    def _apply_oblique_params(self, node_id, params):
        w, b = params
        # store them somewhere:
        self.weights_[node_id] = w
        self.biases_[node_id] = b[0]
        self.oblique_active_[node_id] = True 

    def prune_tree(self):
        """Prune the tree to remove dead or pure branches"""
        pass