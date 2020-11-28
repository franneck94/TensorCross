"""Test code for the random search.
"""
import os
import unittest

import numpy as np
import tensorflow as tf

from tensorcross.utils import dataset_split


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


np.random.seed(0)
tf.random.set_seed(0)


class UtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = tf.data.Dataset.from_tensor_slices(
            ([1, 2, 3],  #
             [-1, -2, -3])  # y
        )

    def test_train_validation_split(self) -> None:
        split_fraction = 0.3
        train_dataset, val_dataset = dataset_split(
            self.dataset,
            split_fraction=split_fraction,
            fold=0
        )
        self.assertEqual(
            len(val_dataset),
            int(len(self.dataset) * split_fraction)
        )
        self.assertEqual(
            len(val_dataset) + len(train_dataset),
            len(self.dataset)
        )

    @staticmethod
    def _dataset_to_list(dataset: tf.data.Dataset) -> list:
        return [(it[0].numpy(), it[1].numpy()) for it in dataset]

    def test_cross_validation_split(self) -> None:
        split_fraction = 1 / 3
        # First cross-validation split
        first_train_dataset, first_val_dataset = dataset_split(
            dataset=self.dataset,
            split_fraction=split_fraction,
            fold=0
        )
        self.assertEqual(
            len(first_train_dataset) + len(first_val_dataset),
            len(self.dataset)
        )
        self.assertEqual(
            set(
                self._dataset_to_list(first_train_dataset) +
                self._dataset_to_list(first_val_dataset)
            ),
            set(
                self._dataset_to_list(self.dataset)
            )
        )
        # Second cross-validation split
        second_train_dataset, second_val_dataset = dataset_split(
            dataset=self.dataset,
            split_fraction=split_fraction,
            fold=1
        )
        self.assertEqual(
            len(second_train_dataset) + len(second_val_dataset),
            len(self.dataset)
        )
        self.assertEqual(
            set(
                self._dataset_to_list(second_train_dataset) +
                self._dataset_to_list(second_val_dataset)
            ),
            set(
                self._dataset_to_list(self.dataset)
            )
        )
        # Third cross-validation split
        third_train_dataset, third_val_dataset = dataset_split(
            dataset=self.dataset,
            split_fraction=split_fraction,
            fold=2
        )
        self.assertEqual(
            len(third_train_dataset) + len(third_val_dataset),
            len(self.dataset)
        )
        self.assertEqual(
            set(
                self._dataset_to_list(third_train_dataset) +
                self._dataset_to_list(third_val_dataset)
            ),
            set(
                self._dataset_to_list(self.dataset)
            )
        )


if __name__ == '__main__':
    unittest.main()
