import numpy as np


class MyDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, x, y):
        data = np.c_[x, y]
        self.tree = self._build_tree(data, depth=0)

    def _build_tree(self, data, depth):
        x, y = data[:, :-1], data[:, -1]
        result = self._get_leaf_or_params(x, y, depth)

        if isinstance(result, dict):
            return result

        best_feature = result[0]
        best_threshold = result[1]
        left_indices = result[2]
        right_indices = result[3]
        left_subtree = self._build_tree(data[left_indices], depth + 1)
        right_subtree = self._build_tree(data[right_indices], depth + 1)

        return {
            "type": "node",
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _get_leaf_or_params(self, x, y, depth):
        num_samples, num_features = x.shape
        params = []
        leaf = self._try_to_set_leaf(y, depth=depth, num_samples=num_samples)
        if leaf is not None:
            return leaf

        best_feature, best_threshold, best_gain = self._get_self_best_params(num_features, x, y)
        params.append(best_feature)
        params.append(best_threshold)

        leaf = self._try_to_set_leaf(y, best_gain=best_gain)
        if leaf is not None:
            return leaf

        left_indices = x[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        params.append(left_indices)
        params.append(right_indices)

        leaf = self._try_to_set_leaf(y, left_indices=left_indices, right_indices=right_indices)
        if leaf is not None:
            return leaf

        return params

    def _try_to_set_leaf(self, y, depth=None, num_samples=None, best_gain=None, left_indices=None, right_indices=None):
        leaf = {"type": "leaf", "value": np.mean(y) > 0.5}

        if depth is not None and num_samples is not None:
            if depth == self.max_depth or num_samples < self.min_samples_split or np.unique(y).size == 1:
                return leaf

        if best_gain is not None:
            if best_gain == -np.inf:
                return leaf

        if left_indices is not None and right_indices is not None:
            if sum(left_indices) < self.min_samples_leaf or sum(right_indices) < self.min_samples_leaf:
                return leaf

        return None

    def _get_self_best_params(self, num_features, x, y):
        best_feature, best_threshold, best_gain = None, None, -np.inf
        for feature_idx in range(num_features):
            thresholds = np.unique(x[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(x[:, feature_idx], y, threshold)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature_idx, threshold

        return best_feature, best_threshold, best_gain

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

    def predict(self, x):
        return np.array([self._predict_sample(sample, self.tree) for sample in x])

    def _predict_sample(self, sample, tree):
        if tree["type"] == "leaf":
            return tree["value"]
        feature, threshold = tree["feature"], tree["threshold"]
        if sample[feature] <= threshold:
            return self._predict_sample(sample, tree["left"])
        return self._predict_sample(sample, tree["right"])

    def get_tree_height(self, tree=None, depth=0):
        if tree is None:
            tree = self.tree
        if tree["type"] == "leaf":
            return depth
        return max(self.get_tree_height(tree["left"], depth + 1),
                   self.get_tree_height(tree["right"], depth + 1))
