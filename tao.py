from sklearn.base import BaseEstimator, ClassifierMixin # make basic sklearn classifier functions available through inheritance
from sklearn.preprocessing import StandardScaler # standard scale features to allow for convergence in logistic regression
from sklearn.tree import DecisionTreeClassifier # base decision tree classifier
from sklearn.linear_model import LogisticRegression # logistic regression for oblique splits
import numpy as np 
from typing import Tuple, Optional, List
from sklearn import tree
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

class TAOTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_leaf=5, random_state=None,
                 type = "oblique", max_passes=5, C=1, reroute_every: int = 1, njobs: int = -1):
        if reroute_every < 1:
            raise ValueError("reroute_every must be >= 1")
        if type not in ["oblique", "axis-aligned"]:
            raise ValueError("type must be either 'oblique' or 'axis-aligned'")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.type = type
        self.max_passes = max_passes
        self.C = C
        self.reroute_every = reroute_every
        self.njobs = njobs
    
    def _init_base_tree(self, X, y):
        """
        Initialize the base decision tree classifier.
        Args:
            X: (n_samples, n_features) array of input samples
            y: (n_samples,) array of class labels
        """
        self.model_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.scaler_ = StandardScaler()
        self.X_ = self.scaler_.fit_transform(X)
        self.y_ = y
        self.model_.fit(self.X_, self.y_)
        self.tree_ = self.model_.tree_ 

    def __init_params(self):
        """
        Initialize parameters for alternating optimization.
        """
        self._compute_depths() # compute depth of every node
        # Compute which samples reach which nodes
        node_indicator = self.model_.decision_path(self.X_).tocsc()
        self.node_sets_ = [
            node_indicator[:, j].indices  # row indices of non-zero entries
            for j in range(node_indicator.shape[1])
        ]
        
        n_nodes = self.tree_.node_count

        # Will later store oblique weights, biases and indicator variable if node is oblique or still axis-aligned
        self.weights_, self.biases_, self.oblique_active_ = np.zeros((n_nodes, self.X_.shape[1])), np.zeros(n_nodes), np.zeros(n_nodes, dtype=bool)
        self.traverser_: TreeTraversal = TreeTraversal(self.tree_, self.weights_, self.biases_, self.oblique_active_)
        


    def _optimize_tree(self):
        for pass_num in range(self.max_passes):
            for depth_batch in self.get_depth_batch(self.node_depth_, self.reroute_every):
                self._optimize_depth(depth_batch)
                if np.any(depth_batch > 0):
                    self.reroute()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits initial decision tree and optimizes oblique splits using alternating optimization.
        Args:
            X: (n_samples, n_features) array of input samples
            y: (n_samples,) array of class labels
        """
        # Initialize base decision tree
        self._init_base_tree(X, y)
        # Initialize parameters for alternating optimization
        self.__init_params()
        # Perform alternating optimization
        self._optimize_tree()
        # Prune tree to remove dead or pure branches
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

    def get_depth_batch(self, node_depths: np.ndarray, reroute_every: int):
        """
        Yield batches of unique depth values (processes deepest first).
        Reroute_every dictates how often data is re-routed. If it does not happen after optimizing each depth level,
        nodes across different depth levels become indepenedent and can be optimized in parallel.
        Args:
            node_depths: (n_nodes,) array of node depths
            reroute_every: Number of depth levels to include in each batch
        """
        if reroute_every < 1:
            raise ValueError("reroute_every must be >= 1")

        unique_depths = np.unique(node_depths)
        sorted_depths = np.sort(unique_depths)[::-1]
        for start in range(0, len(sorted_depths), reroute_every):
            yield sorted_depths[start:start + reroute_every]

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

    def _optimize_depth(self, depths: list) -> None:
        """
        Optimize all nodes at one or multiple depth levels in parallel
        
        Args:
            depths: Single depth value or iterable of depth values to optimize
        """

        # Extract all nodes at the given depth(s)
        depth_array = np.atleast_1d(depths).astype(int)
        node_mask = np.isin(self.node_depth_, depth_array)
        node_ids_at_depth = np.where(node_mask)[0]

        if node_ids_at_depth.size == 0:
            return

        # Optimize each node in parallel
        results = Parallel(n_jobs=self.njobs)(
            delayed(self._compute_oblique_params_for_node)(node_id)
            for node_id in node_ids_at_depth
        )

        # Apply the computed oblique parameters, after all optimizations are done to avoid race conditions
        for node_id, params in results:
            if params is not None:
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
        
        # Use TreeTraversal's batch prediction for both subtrees
        labels_l, labels_r = self.traverser_.batch_predict_subtrees(
            X_node, 
            self.tree_.children_left[node_id],
            self.tree_.children_right[node_id], 
            self.model_.classes_
        )

        # Compute losses and identify care set
        losses_l = (labels_l != y_true).astype(int)
        losses_r = (labels_r != y_true).astype(int)
        care_mask = losses_l != losses_r               
        
        care_indices = node_set[care_mask]            
        targets = np.where(losses_l[care_mask] < losses_r[care_mask], 0, 1)
        return care_indices, targets

    def reroute(self):
        """
        Update node_sets based on current oblique splits.
        """
        self.node_sets_ = self.traverser_.compute_all_node_sets(self.X_)



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
        unique_classes = np.unique(targets)

        if len(unique_classes) < 2: # All targets are the same, no need to optimize
            return node_id, None  
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

    def __init__(self, tree_: tree, weights: np.ndarray, biases: np.ndarray, oblique_active: np.ndarray) -> None:
        self.tree_ = tree_
        self.weights_ = weights
        self.biases_ = biases
        self.oblique_active_ = oblique_active

    def batch_descend_from(self, X_batch: np.ndarray, start_node: int) -> np.ndarray:
        """
        Vectorized batch descent: traverse multiple samples through the tree simultaneously.
        
        Args:
            X_batch: (n_samples, n_features) array
            start_node: Starting node ID for all samples
        
        Returns:
            leaf_ids: (n_samples,) array of final leaf node IDs
        """
        current_nodes = self._initialize_sample_positions(X_batch.shape[0], start_node)
        active_mask = np.ones(X_batch.shape[0], dtype=bool)
        
        while np.any(active_mask):
            internal_indices = self._get_active_internal_samples(current_nodes, active_mask)
            
            if len(internal_indices) == 0:
                break
            
            # Navigate samples to next level
            next_nodes = self._compute_next_nodes(
                X_batch[internal_indices], 
                current_nodes[internal_indices]
            )
            current_nodes[internal_indices] = next_nodes
            
            # Update active mask - samples at leaves become inactive
            active_mask[internal_indices] = ~self._are_leaf_nodes(next_nodes)
        
        return current_nodes
    
    def compute_all_node_sets(self, X_batch: np.ndarray) -> list:
        """
        Compute node sets for all nodes in the tree - tracks which samples visit each node.
        
        Args:
            X_batch: (n_samples, n_features) array of input data
            
        Returns:
            List of arrays, where each array contains sample indices that visit that node
        """
        current_nodes, active_mask, node_visits = self._init_traversal_state(X_batch)
        
        while np.any(active_mask):
            step = self._descend_active_samples(X_batch, current_nodes, active_mask)
            if step is None:
                break
            sample_indices, next_nodes = step
            self._record_node_visits(node_visits, sample_indices, next_nodes)
            
        return [np.array(indices, dtype=int) for indices in node_visits]
    
    def _init_traversal_state(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Initialize traversal state for node set computation.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Tuple of (current_nodes, active_mask, node_visits)
        """
        n_samples = X.shape[0]
        current_nodes = np.zeros(n_samples, dtype=np.int32)
        active_mask = np.ones(n_samples, dtype=bool)
        node_visits = [[] for _ in range(self.tree_.node_count)]
        for idx in range(n_samples):
            node_visits[0].append(idx)
        return current_nodes, active_mask, node_visits
    
    def _descend_active_samples(self, X: np.ndarray, current_nodes: np.ndarray, 
                               active_mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Descend active samples by one depth level.
        
        Args:
            X: Input feature matrix
            current_nodes: Array of current node indices for each sample
            active_mask: Boolean array indicating which samples are still active
            
        Returns:
            Tuple of (sample_indices, next_nodes) or None if no more samples to process
        """
        internal_indices = self._get_active_internal_samples(current_nodes, active_mask)
        
        if len(internal_indices) == 0:
            # Mark remaining active samples as inactive (they've reached leaves)
            active_indices = np.where(active_mask)[0]
            if active_indices.size > 0:
                active_mask[active_indices] = False
            return None
        
        # Navigate samples to next level
        next_nodes = self._compute_next_nodes(X[internal_indices], current_nodes[internal_indices])
        current_nodes[internal_indices] = next_nodes
        active_mask[internal_indices] = ~self._are_leaf_nodes(next_nodes)
        
        return internal_indices, next_nodes
    
    def _record_node_visits(self, node_visits: list, sample_indices: np.ndarray, 
                           next_nodes: np.ndarray) -> None:
        """
        Record visits of samples to nodes during traversal.
        
        Args:
            node_visits: List of lists containing sample indices for each node
            sample_indices: Indices of samples being processed
            next_nodes: Node IDs that samples will visit next
        """
        for sample_idx, node_id in zip(sample_indices, next_nodes):
            node_visits[node_id].append(sample_idx)
    
    def _initialize_sample_positions(self, n_samples: int, start_node: int) -> np.ndarray:
        """
        Initialize array of current node positions for all samples.
        
        Args:
            n_samples: Number of samples
            start_node: Starting node ID for all samples
            
        Returns:
            Array of current node positions
        """
        return np.full(n_samples, start_node, dtype=np.int32)
    
    def _get_active_internal_samples(self, current_nodes: np.ndarray, 
                                    active_mask: np.ndarray) -> np.ndarray:
        """
        Get indices of samples that are active and at internal (non-leaf) nodes.
        
        Args:
            current_nodes: Array of current node positions for each sample (n_samples,)
            active_mask: Boolean mask indicating which samples are active (n_samples,)
            
        Returns:
            Indices of samples at active internal nodes
        """
        active_indices = np.where(active_mask)[0]
        if active_indices.size == 0:
            return np.array([], dtype=int)
        
        current_node_ids = current_nodes[active_indices]
        is_leaf = self._are_leaf_nodes(current_node_ids)
        return active_indices[~is_leaf]
    
    def _are_leaf_nodes(self, node_ids: np.ndarray) -> np.ndarray:
        """
        Evaluate if given node IDs correspond to leaf nodes. Just for code legibility.
        
        Args:
            node_ids: Array of node IDs
            
        Returns:
            Boolean array indicating which nodes are leaves
        """
        return self.tree_.children_left[node_ids] == -1
    
    def _compute_next_nodes(self, X_samples: np.ndarray, current_node_ids: np.ndarray) -> np.ndarray:
        """
        Compute the next node IDs for samples based on split decisions.
        
        Args:
            X_samples: Feature data for samples
            current_node_ids: Current node IDs for samples
            
        Returns:
            Next node IDs for samples
        """
        go_left = self.vectorized_split_decision(X_samples, current_node_ids) # get pathing decision
        left_children = self.tree_.children_left[current_node_ids]
        right_children = self.tree_.children_right[current_node_ids]
        return np.where(go_left, left_children, right_children)

    def vectorized_split_decision(self, X_batch: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
        """
        Vectorized split decision for multiple samples at different nodes.
        Handles both oblique and axis-aligned splits. Returns boolean array indicating whether to go left.
        Args:
            X_batch: (n_samples, n_features) array
            node_ids: (n_samples,) array of node IDs
        
        Returns:
            go_left: (n_samples,) boolean array
        """
        n_samples = len(node_ids)
        go_left = np.zeros(n_samples, dtype=bool)
        
        # Separate oblique and axis-aligned nodes
        oblique_mask = self.oblique_active_[node_ids]
        
        # Process oblique splits
        if np.any(oblique_mask):
            oblique_indices = np.where(oblique_mask)[0]
            go_left[oblique_indices] = self._compute_oblique_splits(
                X_batch[oblique_indices], node_ids[oblique_indices]
            )
        
        # Process axis-aligned splits
        axis_mask = ~oblique_mask
        if np.any(axis_mask):
            axis_indices = np.where(axis_mask)[0]
            go_left[axis_indices] = self._compute_axis_aligned_splits(
                X_batch[axis_indices], node_ids[axis_indices]
            )
        
        return go_left
    
    def _compute_oblique_splits(self, X_batch: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
        """
        Compute oblique split decisions for given samples and nodes.
        
        Args:
            X_batch: Feature data for samples
            node_ids: Node IDs with oblique splits
            
        Returns:
            Boolean array indicating go_left decisions
        """
        scores = np.sum(self.weights_[node_ids] * X_batch, axis=1) + self.biases_[node_ids]
        return scores <= 0.0
    
    def _compute_axis_aligned_splits(self, X_batch: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
        """
        Compute axis-aligned split decisions for given samples and nodes.
        
        Args:
            X_batch: Feature data for samples
            node_ids: Node IDs with axis-aligned splits
            
        Returns:
            Boolean array indicating go_left decisions
        """
        features = self.tree_.feature[node_ids]
        thresholds = self.tree_.threshold[node_ids]
        return X_batch[np.arange(len(node_ids)), features] <= thresholds
    

    
    def batch_predict_from_leaves(self, leaf_ids: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """
        Convert leaf node IDs to class predictions using majority voting.
        Used in batch_predict_subtrees.

        Args:
            leaf_ids: Array of leaf node IDs
            classes: Array of class labels from the original model
            
        Returns:
            Array of predicted class labels
        """
        class_indices = np.argmax(self.tree_.value[leaf_ids, 0, :], axis=1)
        return classes[class_indices]
    
    def batch_predict_subtrees(self, X_batch: np.ndarray, left_root: int, 
                              right_root: int, classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels for samples using both left and right subtrees.
        Used in care set computation.
        
        Args:
            X_batch: Feature data for samples 
            left_root: Root node ID of left subtree
            right_root: Root node ID of right subtree
            classes: Array of class labels from the original model
            
        Returns:
            Tuple of (left_predictions, right_predictions)
        """
        leaves_l = self.batch_descend_from(X_batch, left_root)
        leaves_r = self.batch_descend_from(X_batch, right_root)
        
        labels_l = self.batch_predict_from_leaves(leaves_l, classes)
        labels_r = self.batch_predict_from_leaves(leaves_r, classes)
        
        return labels_l, labels_r
