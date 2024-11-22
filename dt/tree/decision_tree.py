import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        data = np.c_[X, y]
        self.tree = self._build_tree(data, depth=0)

    def _build_tree(self, data, depth):
        X, y = data[:, :-1], data[:, -1]
        num_samples, num_features = X.shape

        if depth == self.max_depth or num_samples < self.min_samples_split or np.unique(y).size == 1:
            return {"type": "leaf", "value": np.mean(y) > 0.5}

        best_feature, best_threshold, best_gain = None, None, -np.inf
        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(X[:, feature_idx], y, threshold)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature_idx, threshold

        if best_gain == -np.inf:
            return {"type": "leaf", "value": np.mean(y) > 0.5}

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        if sum(left_indices) < self.min_samples_leaf or sum(right_indices) < self.min_samples_leaf:
            return {"type": "leaf", "value": np.mean(y) > 0.5}

        left_subtree = self._build_tree(data[left_indices], depth + 1)
        right_subtree = self._build_tree(data[right_indices], depth + 1)

        return {
            "type": "node",
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _information_gain(self, feature, y, threshold):
        parent_entropy = self._entropy(y)
        left = y[feature <= threshold]
        right = y[feature > threshold]
        if len(left) == 0 or len(right) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left), len(right)

        child_entropy = (n_left / n) * self._entropy(left) + (n_right / n) * self._entropy(right)
        return parent_entropy - child_entropy

    @staticmethod
    def _entropy(y):
        proportions = np.bincount(y.astype(int)) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if tree["type"] == "leaf":
            return tree["value"]
        feature, threshold = tree["feature"], tree["threshold"]
        if sample[feature] <= threshold:
            return self._predict_sample(sample, tree["left"])
        return self._predict_sample(sample, tree["right"])
