import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from dt.dataset.axis_repo import DataRepoImpl
from dt.tree.decision_tree import MyDecisionTree

repo = DataRepoImpl
X_train = repo.get_axis("x", "train")
X_test = repo.get_axis("x", "test")
y_train = repo.get_axis("y", "train")
y_test = repo.get_axis("y", "test")


def tune_hyperparameters(model_class, param_grid):
    best_params = None
    best_score = -np.inf

    # Перебор всех комбинаций гиперпараметров
    for max_depth in param_grid['max_depth']:
        for min_samples_split in param_grid['min_samples_split']:
            for min_samples_leaf in param_grid['min_samples_leaf']:
                # Инициализация модели с текущими гиперпараметрами
                model = model_class(max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf)
                # Обучение модели
                model.fit(X_train, y_train)

                # Оценка на тестовых данных
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)

                # Обновление лучшего результата
                if score > best_score:
                    best_score = score
                    best_params = {
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf
                    }

    return best_params, best_score


param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Вызов функции
best_params, best_score = tune_hyperparameters(MyDecisionTree, param_grid)

print("Лучшие параметры:", best_params)
print("Лучшая точность:", best_score)
