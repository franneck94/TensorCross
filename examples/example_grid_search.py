import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import Optimizer
from keras.optimizers import RMSprop

from tensorcross.model_selection import GridSearch
from tensorcross.utils import dataset_split


def build_model(
    optimizer: Optimizer,
    learning_rate: float,
) -> Model:
    """Build the test model."""
    x_input = Input(shape=(1,))
    y_pred = Dense(units=1)(x_input)
    model = Model(inputs=[x_input], outputs=[y_pred])

    opt = optimizer(learning_rate=learning_rate)

    model.compile(loss="mse", optimizer=opt, metrics=["mae"])

    return model


if __name__ == "__main__":
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            np.array([1, 2, 3]).reshape(-1, 1),
            np.array([-1, -2, -3]).reshape(-1, 1),
        )
    )

    train_dataset, val_dataset = dataset_split(
        dataset=dataset, split_fraction=(1 / 3)
    )

    param_grid = {
        "optimizer": [Adam, RMSprop],
        "learning_rate": [0.001, 0.0001],
    }

    grid_search = GridSearch(
        model_fn=build_model,
        param_grid=param_grid,
        verbose=1,
    )

    grid_search.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=1,
        verbose=1,
    )

    grid_search.summary()
