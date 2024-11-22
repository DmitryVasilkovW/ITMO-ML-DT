from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from dt.dataset.axis_repo import DataRepoImpl
from dt.tree.decision_tree import MyDecisionTree
from dt.tree.random_forest import MyRandomForest

repo = DataRepoImpl
X_train = repo.get_axis("x", "train")
X_test = repo.get_axis("x", "test")
y_train = repo.get_axis("y", "train")
y_test = repo.get_axis("y", "test")


def show_my_decision_tree_impl():
    depths = range(1, 11)
    train_scores = []
    test_scores = []

    for depth in depths:
        model = MyDecisionTree(max_depth=depth)
        model.fit(X_train, y_train)
        train_scores.append((model.predict(X_train) == y_train).mean())
        test_scores.append((model.predict(X_test) == y_test).mean())

    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_scores, label="Train Accuracy", marker="o")
    plt.plot(depths, test_scores, label="Test Accuracy", marker="o")
    plt.title("Dependence of Accuracy on Tree Depth (My Implementation)")
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def show_my_random_forest_impl():
    n_trees = range(1, 21)
    train_rf_scores = []
    test_rf_scores = []

    for n in n_trees:
        rf = MyRandomForest(n_estimators=n)
        rf.fit(X_train, y_train)
        train_rf_scores.append((rf.predict(X_train) == y_train).mean())
        test_rf_scores.append((rf.predict(X_test) == y_test).mean())

    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, train_rf_scores, label="Train Accuracy", marker="o")
    plt.plot(n_trees, test_rf_scores, label="Test Accuracy", marker="o")
    plt.title("Dependence of Accuracy on Number of Trees (My Random Forest)")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def show_decision_tree_classifier_lib_impl():
    depths = range(1, 11)
    train_scores = []
    test_scores = []

    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_scores, label="Train Accuracy", marker="o")
    plt.plot(depths, test_scores, label="Test Accuracy", marker="o")
    plt.title("Dependence of Accuracy on Tree Depth (Library Implementation)")
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def show_random_forest_classifier_lib_impl():
    n_trees = range(1, 21)
    train_rf_scores = []
    test_rf_scores = []

    for n in n_trees:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        train_rf_scores.append(rf.score(X_train, y_train))
        test_rf_scores.append(rf.score(X_test, y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, train_rf_scores, label="Train Accuracy", marker="o")
    plt.plot(n_trees, test_rf_scores, label="Test Accuracy", marker="o")
    plt.title("Dependence of Accuracy on Number of Trees (Random Forest)")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def show_gradient_boost_classifier_lib_impl():
    n_trees = range(1, 21)
    boosting_scores_train = []
    boosting_scores_test = []

    for n in n_trees:
        boost = GradientBoostingClassifier(n_estimators=n, random_state=42)
        boost.fit(X_train, y_train)
        boosting_scores_train.append(boost.score(X_train, y_train))
        boosting_scores_test.append(boost.score(X_test, y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, boosting_scores_train, label="Train Accuracy", marker="o")
    plt.plot(n_trees, boosting_scores_test, label="Test Accuracy", marker="o")
    plt.title("Dependence of Accuracy on Number of Trees (Boosting)")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
