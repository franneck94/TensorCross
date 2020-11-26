"""Test code for the random search.
"""
import unittest

import numpy as np
import tensorflow as tf

from tensorcross.utils import dataset_split


np.random.seed(0)
tf.random.set_seed(0)


def f(x: np.ndarray) -> np.ndarray:
    return 2 * x + 1


class DATA:
    def __init__(self) -> None:
        x = np.random.uniform(low=-10.0, high=10.0, size=100)
        y = f(x) + np.random.normal(size=100)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (x.reshape(-1, 1), y.reshape(-1, 1))
        )


class RandomSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        data = DATA()
        self.train_dataset = data.train_dataset

    def test_train_validation_split(self) -> None:
        split_fraction = 0.3
        train_dataset, val_dataset = dataset_split(
            self.train_dataset,
            split_fraction=split_fraction
        )
        self.assertEqual(
            len(val_dataset),
            int(len(self.train_dataset) * split_fraction)
        )
        self.assertEqual(
            len(val_dataset) + len(train_dataset),
            len(self.train_dataset)
        )


if __name__ == '__main__':
    unittest.main()
