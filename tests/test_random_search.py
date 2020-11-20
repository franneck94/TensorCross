"""Test code.
"""
import unittest

import numpy as np
import tensorflow as tf
from scipy.stats import randint
from scipy.stats import uniform

from tensorcross import RandomSearch

from dummyData import DATA


np.random.seed(0)
tf.random.set_seed(0)


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
        pass


if __name__ == '__main__':
    unittest.main()
