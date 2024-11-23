import unittest
import numpy as np

from dt.tree.decision_tree import MyDecisionTree


class TestMyDecisionTree(unittest.TestCase):

    def setUp(self):
        self.x_train = np.array([[0, 0],
                                 [0, 1],
                                 [1, 0],
                                 [1, 1]])
        self.y_train = np.array([0, 0, 0, 1])

        self.x_test = np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]])

    def test_entropy(self):
        y = np.array([0, 0, 1, 1])
        expected_entropy = 1.0
        calculated_entropy = MyDecisionTree._entropy(y)
        self.assertAlmostEqual(calculated_entropy, expected_entropy)

    def test_information_gain(self):
        y = np.array([0, 0, 1, 1])
        feature = np.array([0, 0, 1, 1])
        threshold = 0.5
        tree = MyDecisionTree()
        gain = tree._information_gain(feature, y, threshold)

        expected_gain = 1.0
        self.assertAlmostEqual(gain, expected_gain)


if __name__ == '__main__':
    unittest.main()
