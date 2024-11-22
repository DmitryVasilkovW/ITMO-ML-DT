import numpy as np

from dt.tree.decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
            X_sample = X[bootstrap_indices]
            y_sample = y[bootstrap_indices]
            tree = DecisionTree(self.max_depth, self.min_samples_split, self.min_samples_leaf)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return (np.mean(predictions, axis=0) > 0.5).astype(int)
