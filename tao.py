from sklearn.base import BaseEstimator, ClassifierMixin # make basic sklearn classifier functions available through inheritance
from sklearn.preprocessing import StandardScaler # standard scale features to allow for convergence in logistic regression
from sklearn.tree import DecisionTreeClassifier # base decision tree classifier
from sklearn.linear_model import LogisticRegression # logistic regression for oblique splits
import numpy as np 
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

class TAOTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_leaf=5, random_state=None,
                 type = "oblique", max_passes=5, C=1, reroute_every: int = 1, njobs: int = 1, niter = 100,
                 change_threshold=0.01, selective_reroute=True, early_stopping=True):
        """
        Initialize TAO Tree Classifier.
        Args:
            max_depth: Maximum depth of the tree (as trained by initial decision tree)
            min_samples_leaf: Minimum samples required to be at a leaf node
            random_state: Random seed for reproducibility
            type: Type of splits to use ("oblique" or "axis-aligned") # axis aligned not yet implemented
            max_passes: Maximum number of alternating optimization passes
            C: Inverse regularization strength for logistic regression
            reroute_every: Number of depth levels to optimize before rerouting data (increases training speed)
            njobs: Number of parallel jobs to use for optimizing nodes at the same depth (parallelism overhead may outweigh benefits)
            niter: Number of iterations for logistic regression solver (some logreg solvers may not converge well with few iterations)
            change_threshold: Relative change threshold to determine if rerouting is needed
            selective_reroute: Whether to selectively reroute only changed nodes or all nodes at a depth
            early_stopping: Whether to stop optimization early if no nodes change in a complete pass
        """
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
        self.niter = niter
        self.change_threshold = change_threshold
        self.selective_reroute = selective_reroute
        self.early_stopping = early_stopping
    
    # The two methods below initialize the base decision tree and extract its structure
    def _init_base_tree(self, X, y):
        """
        Initialize the base decision tree classifier.
        Args:
            X: (n_samples, n_features) array of input samples
            y: (n_samples,) array of class labels
        """
        sklearn_model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        
        # Only scale for oblique splits (logistic regression needs it)
        # Axis-aligned splits work better with raw features, especially for sparse data
        if self.type == "oblique":
            self.scaler_ = StandardScaler()
            self.X_ = self.scaler_.fit_transform(X)
        else:  # axis-aligned
            from sklearn.preprocessing import FunctionTransformer
            self.scaler_ = FunctionTransformer()  # Identity transform for predict()
            self.X_ = X.copy()
        
        self.y_ = y
        
        # Pass sample weights to sklearn tree if provided
        if self.sample_weight_ is not None:
            sklearn_model.fit(self.X_, self.y_, sample_weight=self.sample_weight_)
        else:
            sklearn_model.fit(self.X_, self.y_)
        
        # Extract tree structure (sklearn model no longer needed after this)
        self._extract_tree_structure(sklearn_model) 

    def _extract_tree_structure(self, sklearn_model):
        """Extract essential attributes from sklearn tree for independent operation."""
        tree = sklearn_model.tree_
        
        # Core tree structure
        self.node_count_ = tree.node_count
        self.children_left_ = tree.children_left.copy()
        self.children_right_ = tree.children_right.copy()
        self.features_ = tree.feature.copy()
        self.thresholds_ = tree.threshold.copy()
        self.values_ = tree.value.copy()
        
        # Class information
        self.classes_ = sklearn_model.classes_.copy()
        
        # Build node sets using sklearn's decision_path (only time we need sklearn functionality)
        node_indicator = sklearn_model.decision_path(self.X_).tocsc()
        self.node_sets_ = [
            node_indicator[:, j].indices
            for j in range(node_indicator.shape[1])
        ]
    
    # The two methods below initialize parameters and structures for the main algorithm
    def _init_params(self):
        """
        Initialize parameters for alternating optimization.
        """
        self._compute_depths() # compute depth of every node
        
        n_nodes = self.node_count_

        # Will later store oblique weights, biases and indicator variable if node is oblique or still axis-aligned
        self.weights_, self.biases_, self.oblique_active_ = np.zeros((n_nodes, self.X_.shape[1])), np.zeros(n_nodes), np.zeros(n_nodes, dtype=bool)
        self.traverser_: TreeTraversal = TreeTraversal(
            self.node_count_, self.children_left_, self.children_right_, 
            self.features_, self.thresholds_, self.values_, self.classes_,
            self.weights_, self.biases_, self.oblique_active_
        )

    def _compute_depths(self):
        """
        Computes depth of each node in the tree.
        Only used once during set-up
        """
        n_nodes = self.node_count_
        children_left = self.children_left_
        children_right = self.children_right_

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

    # Main execution methods: fit and predict
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> None:
        """
        Fits initial decision tree and optimizes oblique splits using alternating optimization.
        Args:
            X: (n_samples, n_features) array of input samples
            y: (n_samples,) array of class labels
            sample_weight: (n_samples,) array of sample weights (optional, for boosting)
        """
        # Store sample weights for use in tree building
        self.sample_weight_ = sample_weight
        
        # Initialize base decision tree and extract structure
        self._init_base_tree(X, y)
        # Initialize parameters for alternating optimization
        self._init_params()
        # Perform alternating optimization
        self._optimize_tree()

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
        class_indices = np.argmax(self.values_[leaf_ids, 0, :], axis=1) # majority voting in leaves
        return self.classes_[class_indices]
    
    def _optimize_tree(self):
        for pass_num in range(self.max_passes):
            pass_had_changes = False  # Track if any nodes changed in this pass
            
            for depth_batch in self.get_depth_batch(self.node_depth_, self.reroute_every):
                changed_nodes = self._optimize_depth(depth_batch)
                if changed_nodes:
                    pass_had_changes = True
                    if np.any(depth_batch > 0):
                        self.reroute(changed_nodes)
            
            # Early stopping if converged
            if self.early_stopping and not pass_had_changes:
                break


    # The two methods below handle depth batching and optimization at a given depth
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

        # Store old parameters to check for significant changes
        old_params = {}
        for node_id in node_ids_at_depth:
            if self.type == "oblique":
                old_params[node_id] = (self.weights_[node_id].copy(), self.biases_[node_id])
            else:  # axis-aligned
                old_params[node_id] = (self.features_[node_id], self.thresholds_[node_id])

        # Choose optimization function based on split type
        if self.type == "oblique":
            compute_func = self._compute_oblique_params_for_node
        else:  # axis-aligned
            compute_func = self._compute_axis_aligned_params_for_node

        # Optimize each node in parallel (does not provide significant speed-up in most cases due to overhead)
        results = Parallel(n_jobs=self.njobs)(
            delayed(compute_func)(node_id)
            for node_id in node_ids_at_depth
        )

        # Apply the computed parameters and track significant changes
        changed_nodes = []
        for node_id, params in results:
            if params is not None:
                # Check if parameters changed significantly
                if self._params_changed_significantly(old_params[node_id], params, node_id):
                    if self.type == "oblique":
                        self._apply_oblique_params(node_id, params)
                    else:  # axis-aligned
                        self._apply_axis_aligned_params(node_id, params)
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
        if self.children_left_[node_id] == -1:
            return np.array([], dtype=int), np.array([], dtype=int)

        y_true = self.y_[node_set]
        X_node = self.X_[node_set]
        
        # Use TreeTraversal's batch prediction for both subtrees
        labels_l, labels_r = self.traverser_.batch_predict_subtrees(
            X_node, 
            self.children_left_[node_id],
            self.children_right_[node_id], 
            self.classes_
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

    # The five methods below handle re-routing of samples, either fully or selectively
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
        Check if parameters changed significantly enough to warrant rerouting.
        Handles both oblique (weights, bias) and axis-aligned (feature, threshold) parameters.
        
        Args:
            old_params: Tuple of old parameters (weights, bias) for oblique or (feature, threshold) for axis-aligned
            new_params: Tuple of new parameters
            node_id: ID of the node being checked
            
        Returns:
            bool: True if parameters changed significantly
        """
        if self.type == "oblique":
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
        
        else:  # axis-aligned
            old_feature, old_threshold = old_params
            new_feature, new_threshold = new_params
            
            # Feature change is always significant
            if old_feature != new_feature:
                return True
            
            # Threshold change - use relative change
            threshold_rel_change = abs(new_threshold - old_threshold) / (abs(old_threshold) + 1e-8)
            return threshold_rel_change > self.change_threshold
    
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
                if len(self.node_sets_[node_id]) == 0:
                    # Direct assignment for empty arrays (faster than concatenation)
                    self.node_sets_[node_id] = new_samples
                else:
                    # Concatenate when both arrays have elements
                    self.node_sets_[node_id] = np.concatenate([self.node_sets_[node_id], new_samples])
    
    def _get_affected_samples(self, changed_nodes: List[int]) -> np.ndarray:
        """Find samples that would route differently due to changed nodes.
        Args:
            changed_nodes: List of node IDs that have significantly changed oblique parameters
        Returns:
            np.ndarray: Array of sample indices that would route differently
        """
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
        if not self.oblique_active_[node_id] or self.children_left_[node_id] == -1:
            return np.zeros(len(sample_indices), dtype=bool)
        
        X_samples = self.X_[sample_indices]
        
        # Original axis-aligned decisions (vectorized)
        feature = self.features_[node_id]
        threshold = self.thresholds_[node_id]
        original_goes_left = X_samples[:, feature] <= threshold
        
        # New oblique decisions (vectorized)
        oblique_scores = X_samples @ self.weights_[node_id] + self.biases_[node_id]
        oblique_goes_left = oblique_scores <= 0.0
        
        # Return mask where decisions differ
        return original_goes_left != oblique_goes_left
    
    def _compute_axis_aligned_params_for_node(self, node_id: int) -> Tuple[int, Optional[Tuple[int, float]]]:
        """
        Compute the optimal axis-aligned split (feature, threshold) for a given node.
        Searches over all features and thresholds to minimize weighted misclassification on care set.

        Args:
            node_id: ID of the node to optimize
        Returns:
            node_id: ID of the node
            (feature, threshold): Tuple of feature index and threshold, or None if no improvement
        """
        # Compute care set and targets
        care_indices, targets = self._compute_care_set(node_id, self.node_sets_[node_id])

        if len(care_indices) < 2:
            return node_id, None
        
        unique_classes = np.unique(targets)
        if len(unique_classes) < 2:
            return node_id, None
        
        X_node = self.X_[care_indices]
        
        # Get sample weights if they exist
        if hasattr(self, 'sample_weight_') and self.sample_weight_ is not None:
            weights_node = self.sample_weight_[care_indices]
            weight_sum = weights_node.sum()
            if weight_sum > 0:
                weights_node = weights_node / weight_sum
            else:
                weights_node = np.ones(len(care_indices)) / len(care_indices)
        else:
            weights_node = np.ones(len(care_indices)) / len(care_indices)
        
        best_loss = float('inf')
        best_feature = None
        best_threshold = None
        
        # Search over all features
        for feature_idx in range(X_node.shape[1]):
            feature_values = X_node[:, feature_idx]
            
            # Get unique values for potential thresholds
            unique_values = np.unique(feature_values)
            if len(unique_values) < 2:
                continue
            
            # Try thresholds between consecutive unique values
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                # Split samples based on this threshold
                goes_left = feature_values <= threshold
                
                # Calculate weighted misclassification loss
                # targets: 0 means left child is better (sample should go left)
                #          1 means right child is better (sample should go right)
                # So we want: goes_left == True when targets == 0
                #             goes_left == False when targets == 1
                correct = goes_left == (targets == 0)
                loss = np.sum(weights_node[~correct])
                
                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None:
            return node_id, None
        
        return node_id, (best_feature, best_threshold)

    # The two methods below handle oblique parameter computation and application
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
        
        # Get sample weights for the care set if they exist
        if hasattr(self, 'sample_weight_') and self.sample_weight_ is not None:
            weights_node = self.sample_weight_[care_indices]
            # Renormalize weights to sum to len(care_indices) to maintain proper scale
            # AdaBoost normalizes weights to sum to 1 over entire dataset, but LogisticRegression
            # expects weights scaled to the subset size for proper regularization
            weight_sum = weights_node.sum()
            if weight_sum > 0:
                weights_node = weights_node * (len(care_indices) / weight_sum)
                logreg = LogisticRegression(C=self.C, max_iter=self.niter).fit(X_node, targets, sample_weight=weights_node)
            else:
                # If all weights are 0, treat as unweighted
                logreg = LogisticRegression(C=self.C, max_iter=self.niter).fit(X_node, targets)
        else:
            logreg = LogisticRegression(C=self.C, max_iter=self.niter).fit(X_node, targets)
        
        w, b = logreg.coef_, logreg.intercept_
        return node_id, (w, b)

    def _apply_oblique_params(self, node_id: int, params: Tuple[np.ndarray, np.ndarray]) -> None:
        """Apply the oblique parameters (w, b) to the given node
        
        Args:
            node_id: ID of the node to update
            params: Tuple of (weight vector, bias term)
        """
        w, b = params
        self.weights_[node_id] = w.flatten()
        self.biases_[node_id] = b[0]
        self.oblique_active_[node_id] = True
    
    def _apply_axis_aligned_params(self, node_id: int, params: Tuple[int, float]) -> None:
        """Apply the axis-aligned parameters (feature, threshold) to the given node
        
        Args:
            node_id: ID of the node to update
            params: Tuple of (feature index, threshold)
        """
        feature, threshold = params
        self.features_[node_id] = feature
        self.thresholds_[node_id] = threshold
        # Keep oblique_active_ as False for axis-aligned splits

    def prune_tree(self):
        """
        Prune the tree by removing dead branches, pure subtrees, and pass-through nodes.
        This delegates to the TreePruner class for the actual pruning logic.
        Stores tree state before pruning for analysis.
        """
        # Store tree structure before pruning for analysis
        self._store_tree_before_pruning()
        
        pruner = TreePruner(self)
        pruner.prune()
    
    def _store_tree_before_pruning(self):
        """Store a copy of the tree structure before pruning."""
        self.tree_before_pruning_ = []
        for node_id in range(self.node_count_):
            node_dict = {
                'is_leaf': self.children_left_[node_id] == -1,
                'left_child': self.children_left_[node_id],
                'right_child': self.children_right_[node_id],
                'feature': self.features_[node_id],
                'threshold': self.thresholds_[node_id],
                'n_samples': len(self.node_sets_[node_id]) if node_id < len(self.node_sets_) else 0,
                'values': self.values_[node_id].copy(),
                'samples_per_class': self.values_[node_id, 0, :].copy()
            }
            self.tree_before_pruning_.append(node_dict)
    
    def _update_traverser(self):
        """
        Update the TreeTraversal object with current tree structure.
        """
        self.traverser_ = TreeTraversal(
            self.node_count_, self.children_left_, self.children_right_, 
            self.features_, self.thresholds_, self.values_, self.classes_,
            self.weights_, self.biases_, self.oblique_active_
        )


class TreePruner:
    """
    Class for pruning decision trees by removing dead branches, pure subtrees,
    and handling pass-through nodes.
    """
    
    def __init__(self, tree_classifier):
        """
        Initialize pruner with reference to the tree classifier.
        
        Parameters
        ----------
        tree_classifier : TAOTreeClassifier
            The tree classifier to prune
        """
        self.tree = tree_classifier
    
    def prune(self):
        """
        Physically prune dead branches, pure subtrees, and pass-through nodes.
        Reduces memory usage by actually removing redundant nodes.
        """
        if not hasattr(self.tree, 'node_sets_') or not hasattr(self.tree, 'y_'):
            return
        
        # Step 1: Identify nodes to remove
        nodes_to_remove = self._identify_nodes_to_remove()
        
        if not nodes_to_remove:
            return
        
        # Step 2: Handle promotions before removal
        self._handle_promotions(nodes_to_remove)
        
        # Step 3: Physically reconstruct tree without removed nodes
        self._reconstruct_tree(nodes_to_remove)
        
        # Step 4: Update traverser with new structure
        self.tree._update_traverser()
    
    def _identify_nodes_to_remove(self) -> set:
        """Identify all nodes that need to be removed: dead branches and pure subtrees."""
        nodes_to_remove = set()
        
        # Find dead branches (entire dead subtrees)
        for node_id in range(self.tree.node_count_):
            if len(self.tree.node_sets_[node_id]) == 0 and self.tree.children_left_[node_id] == -1:
                # Dead leaf - traverse up to find entire dead branch
                nodes_to_remove.update(self._find_dead_branch(node_id))
        
        # Find pure subtrees (keep only highest pure ancestor)
        for node_id in range(self.tree.node_count_):
            if (node_id not in nodes_to_remove and 
                self._is_pure(node_id) and 
                self.tree.children_left_[node_id] == -1):
                # Pure leaf - find pure subtree to remove
                nodes_to_remove.update(self._find_pure_subtree(node_id))
        
        return nodes_to_remove
    
    def _is_pure(self, node_id: int) -> bool:
        """Check if node is pure (all samples have same class)."""
        samples = self.tree.node_sets_[node_id]
        return len(samples) > 0 and len(np.unique(self.tree.y_[samples])) == 1
    
    def _find_dead_branch(self, dead_leaf: int) -> set:
        """Find entire dead branch starting from dead leaf."""
        dead_nodes = set()
        current = dead_leaf
        
        # Traverse up while nodes are dead
        while current is not None and len(self.tree.node_sets_[current]) == 0:
            dead_nodes.add(current)
            current = self._find_parent(current)
        
        return dead_nodes
    
    def _find_pure_subtree(self, pure_leaf: int) -> set:
        """Find pure subtree, keeping highest pure ancestor."""
        # Find highest pure ancestor
        current = pure_leaf
        highest_pure = pure_leaf
        
        while current is not None and self._is_pure(current):
            highest_pure = current
            current = self._find_parent(current)
        
        # Remove all descendants of highest pure node
        to_remove = set()
        self._collect_descendants(highest_pure, to_remove)
        
        # Convert highest pure to leaf with correct pure class values
        if to_remove:
            self.tree.children_left_[highest_pure] = -1
            self.tree.children_right_[highest_pure] = -1
            # Set correct pure class prediction
            samples = self.tree.node_sets_[highest_pure]
            if len(samples) > 0:
                pure_class = self.tree.y_[samples[0]]
                self.tree.values_[highest_pure] = np.zeros((1, len(self.tree.classes_)))
                self.tree.values_[highest_pure][0, pure_class] = 1.0
        
        return to_remove
    
    def _find_parent(self, node_id: int):
        """Find parent of given node."""
        for i in range(self.tree.node_count_):
            if self.tree.children_left_[i] == node_id or self.tree.children_right_[i] == node_id:
                return i
        return None
    
    def _collect_descendants(self, node_id: int, descendants: set):
        """Recursively collect all descendants of a node."""
        left = self.tree.children_left_[node_id]
        right = self.tree.children_right_[node_id]
        
        if left != -1:
            descendants.add(left)
            self._collect_descendants(left, descendants)
        
        if right != -1:
            descendants.add(right)
            self._collect_descendants(right, descendants)
    
    def _handle_promotions(self, nodes_to_remove: set):
        """Handle pass-through nodes by promoting surviving children."""
        # Find pass-through nodes (nodes with exactly one child being removed)
        promoted_nodes = set()  # Track nodes that get promoted to avoid double promotion
        
        for node_id in range(self.tree.node_count_):
            if node_id in nodes_to_remove:
                continue
                
            left = self.tree.children_left_[node_id]
            right = self.tree.children_right_[node_id]
            
            left_removed = left in nodes_to_remove
            right_removed = right in nodes_to_remove
            
            # Pass-through: exactly one child removed
            if left_removed and not right_removed and right != -1:
                self._promote_subtree(node_id, right)
                promoted_nodes.add(right)  # Mark child as promoted
            elif right_removed and not left_removed and left != -1:
                self._promote_subtree(node_id, left)
                promoted_nodes.add(left)  # Mark child as promoted
        
        # Add promoted nodes to removal set (they've been copied to their parents)
        nodes_to_remove.update(promoted_nodes)
    
    def _promote_subtree(self, parent_id: int, child_id: int):
        """Promote entire subtree: copy all data from child to parent."""
        # Copy tree structure
        self.tree.children_left_[parent_id] = self.tree.children_left_[child_id]
        self.tree.children_right_[parent_id] = self.tree.children_right_[child_id]
        self.tree.features_[parent_id] = self.tree.features_[child_id]
        self.tree.thresholds_[parent_id] = self.tree.thresholds_[child_id]
        self.tree.values_[parent_id] = self.tree.values_[child_id].copy()
        
        # Copy oblique parameters
        if hasattr(self.tree, 'weights_'):
            self.tree.weights_[parent_id] = self.tree.weights_[child_id].copy()
            self.tree.biases_[parent_id] = self.tree.biases_[child_id]
            self.tree.oblique_active_[parent_id] = self.tree.oblique_active_[child_id]
        
        # Copy node sets
        self.tree.node_sets_[parent_id] = self.tree.node_sets_[child_id].copy()
    
    def _reconstruct_tree(self, nodes_to_remove: set):
        """Physically reconstruct tree arrays without removed nodes."""
        # Create mapping from old indices to new indices
        active_nodes = [i for i in range(self.tree.node_count_) if i not in nodes_to_remove]
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(active_nodes)}
        new_count = len(active_nodes)
        
        # Create new arrays with reduced size
        new_children_left = np.full(new_count, -1, dtype=np.int32)
        new_children_right = np.full(new_count, -1, dtype=np.int32)
        new_features = np.zeros(new_count, dtype=np.int32)
        new_thresholds = np.zeros(new_count, dtype=np.float64)
        new_values = np.zeros((new_count, 1, len(self.tree.classes_)), dtype=np.float64)
        new_weights = np.zeros((new_count, self.tree.X_.shape[1]), dtype=np.float64)
        new_biases = np.zeros(new_count, dtype=np.float64)
        new_oblique_active = np.zeros(new_count, dtype=bool)
        new_node_sets = [None] * new_count
        
        # Copy data for active nodes and remap indices
        for new_id, old_id in enumerate(active_nodes):
            # Copy node data
            new_features[new_id] = self.tree.features_[old_id]
            new_thresholds[new_id] = self.tree.thresholds_[old_id]
            new_values[new_id] = self.tree.values_[old_id].copy()
            new_weights[new_id] = self.tree.weights_[old_id].copy()
            new_biases[new_id] = self.tree.biases_[old_id]
            new_oblique_active[new_id] = self.tree.oblique_active_[old_id]
            new_node_sets[new_id] = self.tree.node_sets_[old_id].copy()
            
            # Remap children indices
            left = self.tree.children_left_[old_id]
            right = self.tree.children_right_[old_id]
            
            new_children_left[new_id] = old_to_new[left] if left in old_to_new else -1
            new_children_right[new_id] = old_to_new[right] if right in old_to_new else -1
        
        # Replace all tree arrays
        self.tree.node_count_ = new_count
        self.tree.children_left_ = new_children_left
        self.tree.children_right_ = new_children_right
        self.tree.features_ = new_features
        self.tree.thresholds_ = new_thresholds
        self.tree.values_ = new_values
        self.tree.weights_ = new_weights
        self.tree.biases_ = new_biases
        self.tree.oblique_active_ = new_oblique_active
        self.tree.node_sets_ = new_node_sets


class TreeTraversal:
    """
    Class for traversing a decision tree with oblique and axis-aligned splits.
    """

    def __init__(self, node_count: int, children_left: np.ndarray, children_right: np.ndarray, 
                 features: np.ndarray, thresholds: np.ndarray, values: np.ndarray, classes: np.ndarray,
                 weights: np.ndarray, biases: np.ndarray, oblique_active: np.ndarray) -> None:
        self.node_count_ = node_count
        self.children_left_ = children_left
        self.children_right_ = children_right
        self.features_ = features
        self.thresholds_ = thresholds
        self.values_ = values
        self.classes_ = classes
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
        node_visits = [[] for _ in range(self.node_count_)]
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
        return self.children_left_[node_ids] == -1
    
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
        left_children = self.children_left_[current_node_ids]
        right_children = self.children_right_[current_node_ids]
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
        features = self.features_[node_ids]
        thresholds = self.thresholds_[node_ids]
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
        class_indices = np.argmax(self.values_[leaf_ids, 0, :], axis=1)
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
            is_leaf = self.children_left_[current_node_ids] == -1
            
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
            left_children = self.children_left_[node_ids_internal]
            right_children = self.children_right_[node_ids_internal]
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
