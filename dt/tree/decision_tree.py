import numpy as np


class MyDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples < self.min_samples_split or len(
                np.unique(y)) == 1:
            return {"value": self._most_common_label(y)}

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return {"value": self._most_common_label(y)}

        left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
        left_tree = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_tree = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left_tree, "right": right_tree}

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gini = self._calculate_gini(y, X[:, feature] <= threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gini(self, y, condition):
        left_y = y[condition]
        right_y = y[~condition]
        left_gini = 1 - sum((np.sum(left_y == c) / len(left_y)) ** 2 for c in np.unique(y)) if len(left_y) > 0 else 0
        right_gini = 1 - sum((np.sum(right_y == c) / len(right_y)) ** 2 for c in np.unique(y)) if len(
            right_y) > 0 else 0
        return len(left_y) / len(y) * left_gini + len(right_y) / len(y) * right_gini

    def _split(self, feature, threshold):
        return feature <= threshold, feature > threshold

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if "value" in node:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])
