"""Test code for the random search.
"""
import unittest

import numpy as np
import tensorflow as tf
from scipy.stats import uniform
from sklearn.model_selection import train_test_split

from tensorcross.model_selection import RandomSearch


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
        # Load the data set
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

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset


def build_model(
    num_features: int,
    num_targets: int,
    optimizer: tf.keras.optimizers.Optimizer,
    learning_rate: float
) -> tf.keras.models.Model:
    x_input = tf.keras.layers.Input(shape=num_features)

    x = tf.keras.layers.Dense(units=10)(x_input)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(units=num_targets)(x)
    y_pred = tf.keras.layers.Activation("softmax")(x)

    model = tf.keras.models.Model(inputs=[x_input], outputs=[y_pred])

    opt = optimizer(learning_rate=learning_rate)

    model.compile(
        loss="mse", optimizer=opt, metrics=["mse"]
    )

    return model


class RandomSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DATA()

        self.train_dataset = self.data.get_train_set()
        self.val_dataset = self.data.get_val_set()

        self.num_features = self.data.num_features
        self.num_targets = self.data.num_targets

        self.epochs = 3

        self.param_distributions = {
            "optimizer": [
                tf.keras.optimizers.Adam,
                tf.keras.optimizers.RMSprop
            ],
            "learning_rate": uniform(0.001, 0.0001)
        }

        self.build_model = build_model

        self.rand_search = RandomSearch(
            model_fn=self.build_model,
            param_distributions=self.param_distributions,
            n_iter=2,
            verbose=False,
            num_features=self.num_features,
            num_targets=self.num_targets
        )

    def test_random_search(self) -> None:
        self.assertEqual(True, True)  # Dummy Test


if __name__ == '__main__':
    unittest.main()
