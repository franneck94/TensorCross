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
        test_size: float = 0.2
    ) -> None:
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
        loss="mse", optimizer=opt, metrics=["mae"]
    )

    return model


if __name__ == "__main__":
    data = DATA()

    train_dataset = data.train_dataset
    val_dataset = data.val_dataset

    param_distributions = {
        "optimizer": [
            tf.keras.optimizers.Adam,
            tf.keras.optimizers.RMSprop
        ],
        "learning_rate": uniform(0.001, 0.0001)
    }

    rand_search = RandomSearch(
        model_fn=build_model,
        param_distributions=param_distributions,
        n_iter=2,
        verbose=1,
        num_features=1,
        num_targets=1
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="logs",
            histogram_freq=0,  # How often to log histogram visualizations
            embeddings_freq=0,  # How often to log embedding visualizations
            update_freq="epoch",
        )
    ]

    rand_search.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=3,
        callbacks=callbacks,
        verbose=1
    )

    rand_search.summary()
