import numpy as np

from dt.tree.decision_tree import MyDecisionTree


class MyRandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.forest = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, x, y):
        n_samples = x.shape[0]
        self._set_forest(x, y, n_samples)

    def _set_forest(self, x, y, n_samples):
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_sample = x[indices]
            y_sample = y.to_numpy()[indices]
            tree = MyDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                  min_samples_leaf=self.min_samples_leaf)
            tree.fit(x_sample, y_sample)
            self.forest.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.forest])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds)
