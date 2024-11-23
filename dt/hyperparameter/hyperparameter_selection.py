import numpy as np
from sklearn.metrics import accuracy_score

from dt.dataset.axis_repo import DataRepoImpl
from dt.tree.decision_tree import MyDecisionTree

repo = DataRepoImpl
X_train = repo.get_axis("x", "train")
X_test = repo.get_axis("x", "test")
y_train = repo.get_axis("y", "train")
y_test = repo.get_axis("y", "test")


class HyperparameterSelection:
    best_params = None
    best_score = -np.inf

    param_grid = {
        'max_depth': [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    @classmethod
    def get_best_max_depth(cls):
        if cls.best_params is None:
            cls._set_hyperparameters()
        return cls.best_params['max_depth']

    @classmethod
    def get_best_min_samples_split(cls):
        if cls.best_params is None:
            cls._set_hyperparameters()
        return cls.best_params['min_samples_split']

    @classmethod
    def get_best_min_samples_leaf(cls):
        if cls.best_params is None:
            cls._set_hyperparameters()
        return cls.best_params['min_samples_leaf']

    @classmethod
    def get_best_score(cls):
        if cls.best_score is -np.inf:
            cls._set_hyperparameters()
        return cls.best_score

    @classmethod
    def set_grid(cls, param_grid):
        cls.param_grid = param_grid
        cls._set_hyperparameters()

    @classmethod
    def _set_hyperparameters(cls):
        for max_depth in cls.param_grid['max_depth']:
            for min_samples_split in cls.param_grid['min_samples_split']:
                for min_samples_leaf in cls.param_grid['min_samples_leaf']:
                    model = MyDecisionTree(max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           min_samples_leaf=min_samples_leaf)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)

                    if score > cls.best_score:
                        cls.best_score = score
                        cls.best_params = {
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf
                        }
