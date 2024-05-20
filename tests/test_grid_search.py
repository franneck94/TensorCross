"""Test code for the grid search.
"""
import os
import unittest

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import Optimizer
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from tensorcross.model_selection import GridSearch


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


np.random.seed(0)
tf.random.set_seed(0)


def f(x: np.ndarray) -> np.ndarray:
    return 2 * x + 1


class DATA:
    def __init__(self, test_size: float = 0.2) -> None:
        x = np.random.uniform(low=-10.0, high=10.0, size=100)
        y = f(x) + np.random.normal(size=100)
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=test_size
        )
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train.reshape(-1, 1), y_train.reshape(-1, 1))
        )
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (x_val.reshape(-1, 1), y_val.reshape(-1, 1))
        )


def build_model(
    num_features: int,
    num_targets: int,
    optimizer: Optimizer,
    learning_rate: float,
) -> Model:
    """Build the test model."""
    x_input = Input(shape=(num_features,))
    y_pred = Dense(units=num_targets)(x_input)
    model = Model(inputs=[x_input], outputs=[y_pred])

    opt = optimizer(learning_rate=learning_rate)

    model.compile(loss="mse", optimizer=opt, metrics=["mse"])

    return model


class GridSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        data = DATA()
        self.train_dataset = data.train_dataset
        self.val_dataset = data.val_dataset

        self.param_grid = {
            "optimizer": [Adam, RMSprop],
            "learning_rate": [0.001, 0.0001],
        }

        self.grid_search = GridSearch(
            model_fn=build_model,
            param_grid=self.param_grid,
            verbose=1,
            num_features=1,
            num_targets=1,
        )

    def test_grid_search(self) -> None:
        self.grid_search.fit(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            epochs=1,
            verbose=1,
        )

        self.grid_search.summary()


if __name__ == "__main__":
    unittest.main()
