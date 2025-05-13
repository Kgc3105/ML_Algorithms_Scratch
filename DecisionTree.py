import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = int(np.sqrt(self.n_features))
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _gini_idx(self, y):
        classes = np.unique(y)
        gini = 1.0
        for cls in classes:
            p = np.sum(y == cls) / len(y)
            gini -= p ** 2
        return gini

    def _best_split(self, X, y, features):
        best_gini = float('inf')
        best_col, best_val = None, None
        for col in features:
            values = np.unique(X[:, col])
            for val in values:
                left_mask = X[:, col] <= val
                right_mask = ~left_mask
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                gini = (
                    len(y[left_mask]) / len(y) * self._gini_idx(y[left_mask])
                    + len(y[right_mask]) / len(y) * self._gini_idx(y[right_mask])
                )
                if gini < best_gini:
                    best_gini = gini
                    best_col, best_val = col, val
        return best_col, best_val

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or depth >= self.max_depth:
            return {'class': Counter(y).most_common(1)[0][0]}

        feature_indices = np.random.choice(self.n_features, self.max_features, replace=False)
        col, val = self._best_split(X, y, feature_indices)

        if col is None:
            return {'class': Counter(y).most_common(1)[0][0]}

        left_mask = X[:, col] <= val
        right_mask = ~left_mask
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'col': col, 'val': val, 'left': left_subtree, 'right': right_subtree}

    def _predict_tree(self, x, tree):
        if 'class' in tree:
            return tree['class']
        if x[tree['col']] <= tree['val']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])
