from sklearn.base import BaseEstimator, ClassifierMixin # make basic sklearn classifier functions available through inheritance
from sklearn.preprocessing import StandardScaler # standard scale features to allow for convergence in logistic regression
from sklearn.tree import DecisionTreeClassifier # base decision tree classifier
from sklearn.linear_model import LogisticRegression # logistic regression for oblique splits
import numpy as np 
from typing import Tuple
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler_ = StandardScaler()
        self.X_ = self.scaler_.fit_transform(X)
        self.y_ = y
        self.model_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )



        self.model_.fit(self.X_, self.y_)
        self.tree_ = self.model_.tree_ # will be mani
        self._compute_depths()
        

        # Compute which samples reach which nodes
        node_indicator = self.model_.decision_path(self.X_).tocsc()
        self.node_sets_ = [
            node_indicator[:, j].indices  # row indices of non-zero entries
            for j in range(node_indicator.shape[1])
        ]
        
        n_nodes = self.tree_.node_count

        # Will later store oblique weights, biases and indicator variable if node is oblique or still axis-aligned
        self.weights_, self.biases_, self.oblique_active_ = np.zeros((n_nodes, self.X_.shape[1])), np.zeros(n_nodes), np.zeros(n_nodes, dtype=bool)
        self.traverser_ = TreeTraversal(self.tree_, self.weights_, self.biases_, self.oblique_active_)

        for _ in range(self.max_passes):
            print(_)
            self.optimize()
            self.reroute()
        self.prune_tree()

    def _compute_depths(self):
        """
        Computes depth of each node in the tree.
        Only used once during set-up
        """
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        Args:
            X: (n_samples, n_features) array of input samples
        Returns:
            predictions: (n_samples,) array of predicted class labels
        """
        X_scaled = self.scaler_.transform(X) # scale features
        leaf_ids = self.traverser_.batch_descend_from(X_scaled, 0) # traverse to leaves
        class_indices = np.argmax(self.tree_.value[leaf_ids, 0, :], axis=1) # majority voting in leaves
        return self.model_.classes_[class_indices]

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
                if params is not None:  # Only apply if we computed parameters
                    self._apply_oblique_params(node_id, params)


    def _compute_care_set(self, node_id: int, node_set: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        We are only interested in samples for which the left and right child have different losses.
        This function computes the so called care set for a given node and the corresponding targets.

        Args:
            node_id: ID of the node to compute the care set for
            node_set: Set of sample indices that reach the given node
        Returns:
            care_indices: Indices of samples in the care set
            targets: Targets for the care set samples (0 for left, 1 for right)
        """
        if self.tree_.children_left[node_id] == -1:
            return np.array([], dtype=int), np.array([], dtype=int)

        y_true = self.y_[node_set]
        X_node = self.X_[node_set]
        
        # Use batch descent instead of loops
        leaves_l = self.traverser_.batch_descend_from(
            X_node, 
            self.tree_.children_left[node_id]
        )
        leaves_r = self.traverser_.batch_descend_from(
            X_node, 
            self.tree_.children_right[node_id]
        )
        
        # Vectorized label lookup
        labels_l = self.model_.classes_[
            np.argmax(self.tree_.value[leaves_l, 0, :], axis=1)
        ]
        labels_r = self.model_.classes_[
            np.argmax(self.tree_.value[leaves_r, 0, :], axis=1)
        ]

        losses_l = (labels_l != y_true).astype(int)
        losses_r = (labels_r != y_true).astype(int)

        care_mask = losses_l != losses_r               
        care_indices = node_set[care_mask]            
        targets = np.where(losses_l[care_mask] < losses_r[care_mask], 0, 1)
        return care_indices, targets

    def reroute(self):
        """ Reroute samples through the optimized tree structure """
        pass    

    def _compute_oblique_params_for_node(self, node_id: int) -> Tuple[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the oblique parameters (w, b) for the given node, without modifying the tree to prevent race conditions

        Args:
            node_id: ID of the node to compute parameters for
        Returns:
            node_id: ID of the node
            (w, b): Tuple of weight vector and bias term
        """

        # Compute care set and targets
        care_indices , targets = self._compute_care_set(node_id, self.node_sets_[node_id])

        if len(care_indices) < 2:  # Need at least 2 samples for logistic regression
            return node_id, None  # Return None to indicate no optimization needed
    
        X_node = self.X_[care_indices]
        
        # Apply logreg on care set
        logreg = LogisticRegression(C=self.C).fit(X_node, targets)
        w, b = logreg.coef_, logreg.intercept_
        return node_id, (w, b)

    def _apply_oblique_params(self, node_id: int, params: Tuple[np.ndarray, np.ndarray]) -> None:
        """Apply the oblique parameters (w, b) to the given node"""
        w, b = params
        self.weights_[node_id] = w
        self.biases_[node_id] = b[0]
        self.oblique_active_[node_id] = True 

    def prune_tree(self):
        """Prune the tree to remove dead or pure branches"""
        pass

class TreeTraversal:
    """
    Class for traversing a decision tree with oblique and axis-aligned splits.
    """

    def __init__(self, tree_: tree, weights: np.ndarray, biases: np.ndarray, oblique_active: np.ndarray):
        self.tree_ = tree_
        self.weights_ = weights
        self.biases_ = biases
        self.oblique_active_ = oblique_active

    def batch_descend_from(self, X_batch, start_node):
        """
        Vectorized batch descent: traverse multiple samples through the tree simultaneously.
        
        Args:
            X_batch: (n_samples, n_features) array
            start_node: Starting node ID for all samples
        
        Returns:
            leaf_ids: (n_samples,) array of final leaf node IDs
        """
        n_samples = X_batch.shape[0]
        current_nodes = np.full(n_samples, start_node, dtype=np.int32)
        
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        
        # Traverse until all samples reach leaves
        active_mask = np.ones(n_samples, dtype=bool)
        
        while np.any(active_mask):
            # Get current node info for active samples
            current_node_ids = current_nodes[active_mask]
            node_is_leaf = children_left[current_node_ids] == -1
            
            # Filter to only internal nodes
            internal_mask = ~node_is_leaf
            internal_indices = np.where(active_mask)[0][internal_mask]
            
            # All samples reached leaves
            if len(internal_indices) == 0:
                break
            
            # Extract nodes and data for active internal samples

            samples_at_internal = internal_indices
            nodes_at_internal = current_nodes[samples_at_internal]
            X_samples = X_batch[samples_at_internal]
            
            # Compute go_left decisions for all active internal nodes
            go_left = self.vectorized_split_decision(
                X_samples, nodes_at_internal
            )
            
            # Update node IDs based on decisions
            left_children = children_left[nodes_at_internal]
            right_children = children_right[nodes_at_internal]
            current_nodes[samples_at_internal] = np.where(
                go_left, left_children, right_children
            )
            
            # All samples at internal nodes are still active
        
        return current_nodes
    
    def vectorized_split_decision(self, X_batch, node_ids):
        """
        Vectorized split decision for multiple samples at different nodes.
        
        Args:
            X_batch: (n_samples, n_features) array
            node_ids: (n_samples,) array of node IDs
        
        Returns:
            go_left: (n_samples,) boolean array
        """
        n_samples = len(node_ids)
        go_left = np.zeros(n_samples, dtype=bool)
        
        # Check which nodes have oblique splits active
        oblique_mask = self.oblique_active_[node_ids]
        
        # Handle oblique splits (vectorized)
        if np.any(oblique_mask):
            oblique_indices = np.where(oblique_mask)[0]
            X_oblique = X_batch[oblique_indices]
            oblique_nodes = node_ids[oblique_indices]
            
            # Batch matrix multiplication for oblique splits
            scores = np.sum(
                self.weights_[oblique_nodes] * X_oblique, axis=1
            ) + self.biases_[oblique_nodes]
            go_left[oblique_indices] = scores <= 0.0
        
        # Handle axis-aligned splits (vectorized)
        axis_mask = ~oblique_mask
        if np.any(axis_mask):
            axis_indices = np.where(axis_mask)[0]
            X_axis = X_batch[axis_indices]
            axis_nodes = node_ids[axis_indices]
            
            features = self.tree_.feature[axis_nodes]
            thresholds = self.tree_.threshold[axis_nodes]
            
            # Vectorized feature comparison
            go_left[axis_indices] = X_axis[np.arange(len(axis_indices)), features] <= thresholds
        
        return go_left