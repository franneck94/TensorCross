"""Test code for the random search.
"""
import unittest

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorcross.utils import dataset_split


np.random.seed(0)
tf.random.set_seed(0)


def f(x: np.ndarray) -> np.ndarray:
    return 2 * x + 1


class DATA:
    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.33
    ) -> None:
        # User-definen constants
        self.num_targets = 1
        self.batch_size = 128
        # Load the dataset
        x = np.random.uniform(low=-10.0, high=10.0, size=100)
        y = f(x) + np.random.normal(size=100)
        x = x.reshape(-1, 1).astype(np.float32)
        y = y.reshape(-1, 1).astype(np.float32)
        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=validation_size
        )
        # Dataset attributes
        self.train_size = x_train.shape[0]
        self.test_size = x_test.shape[0]
        self.val_size = x_val.shape[0]
        self.num_features = x_train.shape[1]
        # tf.data Datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)
        )
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (x_val, y_val)
        )


class RandomSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DATA()

    def test_train_validation_split(self) -> None:
        split_size = 0.3
        train_dataset, val_dataset = dataset_split(
            self.data.train_dataset,
            split_size=split_size
        )
        self.assertEqual(
            len(val_dataset),
            int(len(self.data.train_dataset) * split_size)
        )
        self.assertEqual(
            len(val_dataset) + len(train_dataset),
            len(self.data.train_dataset)
        )


if __name__ == '__main__':
    unittest.main()
