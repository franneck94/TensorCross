import numpy as np
import tensorflow as tf

from tensorcross.model_selection import GridSearchCV


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
    train_dataset = data.train_dataset

    param_grid = {
        "optimizer": [
            tf.keras.optimizers.Adam,
            tf.keras.optimizers.RMSprop
        ],
        "learning_rate": [0.001, 0.0001]
    }

    grid_search_cv = GridSearchCV(
        model_fn=build_model,
        param_grid=param_grid,
        n_folds=2,
        verbose=1,
        num_features=1,
        num_targets=1
    )

    grid_search_cv.fit(
        train_dataset=train_dataset,
        epochs=1,
        verbose=1
    )

    grid_search_cv.summary()
