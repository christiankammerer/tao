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
                 type = "oblique", max_passes=5, C=1, reroute_every: int = 1, njobs: int = -1,
                 change_threshold=0.01, selective_reroute=True):
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
        self.change_threshold = change_threshold
        self.selective_reroute = selective_reroute
    
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
        # Compute which samples reach which nodes from the initial tree
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
                changed_nodes = self._optimize_depth(depth_batch)
                if changed_nodes and np.any(depth_batch > 0):
                    self.reroute(changed_nodes)

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

    def _optimize_depth(self, depths: list) -> List[int]:
        """
        Optimize all nodes at one or multiple depth levels in parallel
        
        Args:
            depths: Single depth value or iterable of depth values to optimize
            
        Returns:
            List of node IDs that had significant parameter changes
        """

        # Extract all nodes at the given depth(s)
        depth_array = np.atleast_1d(depths).astype(int)
        node_mask = np.isin(self.node_depth_, depth_array)
        node_ids_at_depth = np.where(node_mask)[0]

        if node_ids_at_depth.size == 0:
            return []

        # Store old parameters for change detection
        old_params = {}
        for node_id in node_ids_at_depth:
            old_params[node_id] = (self.weights_[node_id].copy(), self.biases_[node_id])

        # Optimize each node in parallel (does not provide significant speed-up in most cases due to overhead)
        results = Parallel(n_jobs=self.njobs)(
            delayed(self._compute_oblique_params_for_node)(node_id)
            for node_id in node_ids_at_depth
        )

        # Apply the computed oblique parameters and track significant changes
        changed_nodes = []
        for node_id, params in results:
            if params is not None:
                # Check if parameters changed significantly
                if self._params_changed_significantly(old_params[node_id], params, node_id):
                    self._apply_oblique_params(node_id, params)
                    changed_nodes.append(node_id)
        
        return changed_nodes

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
        # If leaf node, no care set
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

        # Compute losses for both subtrees (0-1 loss)
        losses_l = (labels_l != y_true).astype(int)
        losses_r = (labels_r != y_true).astype(int)

        # Care set: samples where left and right losses differ
        care_mask = losses_l != losses_r               
        
        # Care-indice = index of care sample, target = which subtree has lower loss
        care_indices = node_set[care_mask]            
        targets = np.where(losses_l[care_mask] < losses_r[care_mask], 0, 1)
        return care_indices, targets

    def reroute(self, changed_nodes=None):
        """
        Update node_sets based on current oblique splits.
        
        Args:
            changed_nodes: List of node IDs that changed parameters. If None, performs full rerouting.
        """
        if not self.selective_reroute or changed_nodes is None or len(changed_nodes) == 0:
            # Full rerouting - recompute all node assignments
            self.node_sets_ = self.traverser_.compute_all_node_sets(self.X_)
        else:
            # Selective rerouting - only update affected samples
            affected_samples = self._get_affected_samples(changed_nodes)
            if len(affected_samples) > 0:
                self._selective_reroute(affected_samples)
    
    def _params_changed_significantly(self, old_params, new_params, node_id):
        """
        Check if oblique parameters changed significantly enough to warrant rerouting.
        Uses relative change thresholds for both weights and bias to determine significance.
        
        Args:
            old_params: Tuple of (old_weights, old_bias)
            new_params: Tuple of (new_weights, new_bias)
            node_id: ID of the node being checked
            
        Returns:
            bool: True if parameters changed significantly
        """
        old_w, old_b = old_params
        new_w, new_b = new_params
        
        # Handle edge case: first time setting oblique parameters
        if np.allclose(old_w, 0) and abs(old_b) < 1e-8:
            return True  # Always significant when going from axis-aligned to oblique
        
        # Relative change for weights
        w_norm_old = np.linalg.norm(old_w)
        w_norm_diff = np.linalg.norm(new_w.flatten() - old_w)
        w_rel_change = w_norm_diff / (w_norm_old + 1e-8)
        
        # Relative change for bias
        b_rel_change = abs(new_b[0] - old_b) / (abs(old_b) + 1e-8)
        
        # Consider significant if either weight or bias changed substantially
        return w_rel_change > self.change_threshold or b_rel_change > self.change_threshold
    
    def _selective_reroute(self, affected_samples):
        """
        Selective rerouting for samples affected by oblique parameter changes.
        Recomputes paths only for affected samples and updates node_sets accordingly.
        
        Args:
            affected_samples: Array of sample indices that need rerouting
        """
        if len(affected_samples) == 0:
            return
        
        # Remove affected samples from all node sets
        for node_id in range(len(self.node_sets_)):
            old_samples = self.node_sets_[node_id]
            self.node_sets_[node_id] = old_samples[~np.isin(old_samples, affected_samples)]
        
        # Recompute paths for affected samples
        X_affected = self.X_[affected_samples]
        new_node_assignments = self.traverser_.compute_node_sets_for_samples(X_affected, affected_samples)
        
        # Update node_sets with new assignments
        for node_id, new_samples in new_node_assignments.items():
            if len(new_samples) > 0:
                self.node_sets_[node_id] = np.concatenate([self.node_sets_[node_id], new_samples])
    
    def _get_affected_samples(self, changed_nodes):
        """Find samples that would route differently due to changed nodes."""
        affected_samples = []
        
        for node_id in changed_nodes:
            # Get samples that reach this node
            samples_at_node = self.node_sets_[node_id]
            
            if len(samples_at_node) == 0:
                continue
            
            # Check which samples would route differently
            routing_changes = self._check_routing_changes_vectorized(samples_at_node, node_id)
            changed_samples = samples_at_node[routing_changes]
            affected_samples.extend(changed_samples)
        
        return np.unique(affected_samples) if affected_samples else np.array([], dtype=int)
    
    def _check_routing_changes_vectorized(self, sample_indices, node_id):
        """
        Check which samples would route differently after oblique parameter changes.
        Compares original axis-aligned decisions with new oblique decisions.
        
        Args:
            sample_indices: Array of sample indices to check
            node_id: ID of the node with changed parameters
            
        Returns:
            Boolean mask indicating which samples changed routing decisions
        """
        if not self.oblique_active_[node_id] or self.tree_.children_left[node_id] == -1:
            return np.zeros(len(sample_indices), dtype=bool)
        
        X_samples = self.X_[sample_indices]
        
        # Original axis-aligned decisions (vectorized)
        feature = self.tree_.feature[node_id]
        threshold = self.tree_.threshold[node_id]
        original_goes_left = X_samples[:, feature] <= threshold
        
        # New oblique decisions (vectorized)
        oblique_scores = X_samples @ self.weights_[node_id] + self.biases_[node_id]
        oblique_goes_left = oblique_scores <= 0.0
        
        # Return mask where decisions differ
        return original_goes_left != oblique_goes_left

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
    
    def compute_node_sets_for_samples(self, X_batch: np.ndarray, sample_indices: np.ndarray) -> dict:
        """Compute which nodes the given samples visit and return as dictionary."""
        n_samples = len(X_batch)
        current_nodes = np.zeros(n_samples, dtype=np.int32)
        active_mask = np.ones(n_samples, dtype=bool)
        
        # Dictionary to track which samples visit each node
        node_visits = {}
        
        # All samples start at root
        if 0 not in node_visits:
            node_visits[0] = []
        node_visits[0].extend(sample_indices)
        
        while np.any(active_mask):
            active_indices = np.where(active_mask)[0]
            if len(active_indices) == 0:
                break
            
            current_node_ids = current_nodes[active_indices]
            is_leaf = self.tree_.children_left[current_node_ids] == -1
            
            # Samples at leaves are done
            active_mask[active_indices[is_leaf]] = False
            
            # Continue with samples at internal nodes
            internal_mask = ~is_leaf
            if not np.any(internal_mask):
                break
            
            internal_indices = active_indices[internal_mask]
            X_internal = X_batch[internal_indices]
            node_ids_internal = current_node_ids[internal_mask]
            
            # Compute next nodes using current parameters
            go_left = self.vectorized_split_decision(X_internal, node_ids_internal)
            left_children = self.tree_.children_left[node_ids_internal]
            right_children = self.tree_.children_right[node_ids_internal]
            next_nodes = np.where(go_left, left_children, right_children)
            
            # Update positions and record visits
            for i, internal_idx in enumerate(internal_indices):
                next_node = next_nodes[i]
                current_nodes[internal_idx] = next_node
                
                # Record visit
                if next_node not in node_visits:
                    node_visits[next_node] = []
                node_visits[next_node].append(sample_indices[internal_idx])
        
        # Convert lists to numpy arrays
        for node_id in node_visits:
            node_visits[node_id] = np.array(node_visits[node_id], dtype=int)
            
        return node_visits
