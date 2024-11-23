import unittest
import numpy as np
import pandas as pd
from dt.tree.decision_tree import MyDecisionTree
from dt.tree.random_forest import MyRandomForest


class TestMyRandomForest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = pd.Series([0, 1, 0, 1])
        self.model = MyRandomForest(n_estimators=5, max_depth=3, min_samples_split=2, min_samples_leaf=1)

    def test_initialization(self):
        self.assertEqual(self.model.n_estimators, 5)
        self.assertEqual(self.model.max_depth, 3)
        self.assertEqual(self.model.min_samples_split, 2)
        self.assertEqual(self.model.min_samples_leaf, 1)
        self.assertEqual(len(self.model.forest), 0)

    def test_fit(self):
        self.model.fit(self.X, self.y)
        self.assertEqual(len(self.model.forest), 5)
        for tree in self.model.forest:
            self.assertIsInstance(tree, MyDecisionTree)

    def test_predict(self):
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertTrue(all(isinstance(pred, np.integer) for pred in predictions))

    def test_predict_majority_voting(self):
        class MockTree:
            def __init__(self, preds):
                self.preds = preds

            def predict(self, X):
                return self.preds

        self.model.forest = [
            MockTree(np.array([0, 1, 0, 1])),
            MockTree(np.array([0, 0, 0, 1])),
            MockTree(np.array([1, 1, 0, 1])),
        ]
        predictions = self.model.predict(self.X)
        expected = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(predictions, expected)


if __name__ == "__main__":
    unittest.main()
