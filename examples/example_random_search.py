import numpy as np
import tensorflow as tf
from scipy.stats import uniform
from sklearn.model_selection import train_test_split

from tensorcross import RandomSearch


np.random.seed(0)
tf.random.set_seed(0)


def f(x: np.ndarray) -> np.ndarray:
    return 2 * x + 1


class DATA:
    def __init__(
        self,
        test_size: float = 0.2,
        split_size: float = 0.33
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
            x_train, y_train, test_size=split_size
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
    """Build the test model.
    """
    x_input = tf.keras.layers.Input(shape=num_features)
    y_pred = tf.keras.layers.Dense(units=num_targets)(x_input)
    model = tf.keras.models.Model(inputs=[x_input], outputs=[y_pred])

    opt = optimizer(learning_rate=learning_rate)

    model.compile(
        loss="mse", optimizer=opt, metrics=["mse"]
    )

    return model


if __name__ == "__main__":
    data = DATA()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()

    num_features = data.num_features
    num_targets = data.num_targets

    epochs = 1

    param_distributions = {
        "optimizer": [
            tf.keras.optimizers.Adam,
            tf.keras.optimizers.RMSprop
        ],
        "learning_rate": uniform(0.001, 0.0001)
    }

    build_model = build_model

    rand_search = RandomSearch(
        model_fn=build_model,
        param_distributions=param_distributions,
        n_iter=2,
        verbose=1,
        num_features=num_features,
        num_targets=num_targets
    )

    rand_search.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=3,
        verbose=1
    )

    rand_search.summary()
