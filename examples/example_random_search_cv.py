import numpy as np
import tensorflow as tf
from scipy.stats import uniform
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers import RMSprop

from tensorcross.model_selection import RandomSearchCV


np.random.seed(0)
tf.random.set_seed(0)


def build_model(
    num_features: int,
    num_targets: int,
    optimizer: Optimizer,
    learning_rate: float,
) -> Model:
    """Build the test model."""
    x_input = Input(shape=num_features)
    y_pred = Dense(units=num_targets)(x_input)
    model = Model(inputs=[x_input], outputs=[y_pred])

    opt = optimizer(learning_rate=learning_rate)

    model.compile(loss="mse", optimizer=opt, metrics=["mse"])

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
        num_features=1,
        num_targets=1,
    )

    rand_search_cv.fit(dataset=dataset, epochs=1, verbose=1)

    rand_search_cv.summary()
