from typing import Tuple

import numpy as np
import tensorflow as tf
from scipy.stats import randint
from scipy.stats import uniform
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop

from tensorcross.random_search import RandomSearch

from dummyData import DATA


np.random.seed(0)
tf.random.set_seed(0)


def build_model(
    num_features: int,
    num_targets: int,
    optimizer: tf.keras.optimizers.Optimizer,
    learning_rate: float
) -> Model:
    x_input = Input(shape=num_features)

    x = Dense(units=10)(x_input)
    x = Activation("relu")(x)
    x = Dense(units=num_targets)(x)
    y_pred = Activation("softmax")(x)

    model = Model(inputs=[x_input], outputs=[y_pred])

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

    epochs = 3

    param_distributions = {
        "optimizer": [Adam, RMSprop],
        "learning_rate": uniform(0.001, 0.0001)
    }

    rand_search = RandomSearch(
        model_fn=build_model,
        param_distributions=param_distributions,
        n_iter=2,
        verbose=True,
        num_features=num_features,
        num_targets=num_targets
    )

    rand_search.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        verbose=1
    )

    rand_search.summary()
