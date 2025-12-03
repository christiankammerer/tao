"""
TAO Ensemble Methods: Forest (Bagging) and Boosting

Wrapper classes that use sklearn's ensemble methods with TAO trees.
"""
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from tao import TAOTreeClassifier


class TAOForest(BaggingClassifier):
    """
    Random Forest using TAO trees with sklearn's BaggingClassifier.
    
    Wrapper around sklearn.ensemble.BaggingClassifier that uses TAO trees
    as the base estimator with sensible defaults for forest construction.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int or None, default=None
        Maximum depth of each tree
    max_passes : int, default=2
        Number of alternating optimization passes for each tree
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node
    random_state : int or None, default=None
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs (-1 means use all processors)
    **kwargs : dict
        Additional arguments passed to BaggingClassifier
    """
    
    def __init__(self, n_estimators=100, max_depth=None, max_passes=2, 
                 min_samples_leaf=1, random_state=None, n_jobs=-1, **kwargs):
        # Store TAO-specific parameters
        self.max_depth = max_depth
        self.max_passes = max_passes
        self.min_samples_leaf = min_samples_leaf
        
        base_estimator = TAOTreeClassifier(
            max_depth=max_depth,
            max_passes=max_passes,
            min_samples_leaf=min_samples_leaf
        )
        
        super().__init__(
            estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )


class TAOBoost(AdaBoostClassifier):
    """
    AdaBoost using TAO trees with sklearn's AdaBoostClassifier.
    
    Wrapper around sklearn.ensemble.AdaBoostClassifier that uses TAO trees
    as weak learners with sensible defaults for boosting.
    
    Parameters
    ----------
    n_estimators : int, default=50
        Number of boosting iterations
    max_depth : int or None, default=3
        Maximum depth of each weak learner tree
    max_passes : int, default=1
        Number of alternating optimization passes (typically 1 for weak learners)
    learning_rate : float, default=1.0
        Learning rate shrinks the contribution of each classifier
    random_state : int or None, default=None
        Random seed for reproducibility
    **kwargs : dict
        Additional arguments passed to AdaBoostClassifier
    """
    
    def __init__(self, n_estimators=50, max_depth=3, max_passes=1, 
                 learning_rate=1.0, random_state=None, **kwargs):
        # Store TAO-specific parameters
        self.max_depth = max_depth
        self.max_passes = max_passes
        
        base_estimator = TAOTreeClassifier(
            max_depth=max_depth,
            max_passes=max_passes
        )
        
        super().__init__(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )
