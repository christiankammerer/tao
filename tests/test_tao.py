import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tao import TAOTreeClassifier

def test_fit_and_predict_digits():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    clf = TAOTreeClassifier(max_depth=3, max_passes=2, random_state=0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert preds.shape == (len(X_test),)
    # sanity check accuracy above random chance (~1/10)
    assert np.mean(preds == y_test) > 0.3
