import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import Optimizer
from keras.optimizers import RMSprop
from scipy.stats import uniform

from tensorcross.model_selection import RandomSearchCV


def build_model(
    optimizer: Optimizer,
    learning_rate: float,
) -> Model:
    """Build the test model."""
    x_input = Input(shape=(1,))
    y_pred = Dense(units=1)(x_input)
    model = Model(inputs=[x_input], outputs=[y_pred])

    opt = optimizer(learning_rate=learning_rate)

    model.compile(
        loss="mse",
        optimizer=opt,
        metrics=["mse"],
    )

    return model


if __name__ == "__main__":
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            np.array([1, 2, 3, 4]).reshape(-1, 1),  # x
            np.array([-1, -2, -3, -4]).reshape(-1, 1),  # y
        )
    )

    param_distributions = {
        "optimizer": [Adam, RMSprop],
        "learning_rate": uniform(0.001, 0.0001),
    }

    rand_search_cv = RandomSearchCV(
        model_fn=build_model,
        param_distributions=param_distributions,
        n_iter=2,
        n_folds=4,
        verbose=1,
    )

    rand_search_cv.fit(
        dataset=dataset,
        epochs=1,
        verbose=1,
    )

    rand_search_cv.summary()
