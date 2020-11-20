import numpy as np
import tensorflow as tf

from scipy.stats import randint
from scipy.stats import uniform

from tensorcross import GridSearch

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


if __name__ == "__main__":
    data = DATA()

    train_dataset = data.get_train_set()
    val_dataset = data.get_val_set()

    num_features = data.num_features
    num_targets = data.num_targets

    epochs = 3

    param_grid = {
        "optimizer": [
            tf.keras.optimizers.Adam,
            tf.keras.optimizers.RMSprop
        ],
        "learning_rate": [0.001, 0.0001]
    }

    build_model = build_model

    grid_search = GridSearch(
        model_fn=build_model,
        parameter_grid=param_grid,
        n_iter=2,
        verbose=False,
        num_features=num_features,
        num_targets=num_targets
    )

    grid_search.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=3,
        verbose=1
    )

    grid_search.summary()
