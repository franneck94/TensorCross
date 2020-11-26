# TensorCross

![Python](https://img.shields.io/badge/python-%203.8-blue)
[![License](https://camo.githubusercontent.com/890acbdcb87868b382af9a4b1fac507b9659d9bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667)](https://github.com/franneck94/TensorCross/blob/main/LICENSE)
[![Build](https://github.com/franneck94/TensorCross/workflows/Test/badge.svg)](https://github.com/franneck94/TensorCross/actions?query=workflow%3A%22Test+and+Coverage%22)
[![codecov](https://codecov.io/gh/franneck94/TensorCross/branch/main/graph/badge.svg)](https://codecov.io/gh/franneck94/TensorCross)
[![Documentation](https://img.shields.io/badge/ref-Documentation-blue)](https://franneck94.github.io/TensorCross-Docs/)

```bash
pip install tensorcross
```

Cross Validation, Grid Search and Random Search for tf.data.Datasets in TensorFlow 2.3 and Python 3.8.

## Motivation

Currently, there is the tf.keras.wrapper.KerasClassifier/KerasRegressor class,
which can be used to transform your tf.keras model into a sklearn estimator.
However, this approach is only applicable if your dataset is a numpy.ndarray
for your x and y data.
If you want to use the new tf.data.Dataset class, you cannot use the sklearn
wrappers.
This python package aims to help with this use-case.

## API

- [GridSearch](#GridSearch-Example)
- [RandomSearch](#RandomSearch-Example)
- For more examples see: [here](examples/)

### GridSearch Example

```python
    import tensorflow as tf
    from tensorcross.model_selection GridSearch

    data = tf.data.Dataset()

    def build_model(
        optimizer: tf.keras.optimizers.Optimizer,
        learning_rate: float
    ) -> tf.keras.models.Model:
        x_input = tf.keras.layers.Input(shape=2)
        y_pred = tf.keras.layers.Dense(units=1)(x_input)
        model = tf.keras.models.Model(inputs=[x_input], outputs=[y_pred])

        opt = optimizer(learning_rate=learning_rate)

        model.compile(
            loss="mse", optimizer=opt, metrics=["mse"]
        )

        return model

    param_grid = {
        "optimizer": [
            tf.keras.optimizers.Adam,
            tf.keras.optimizers.RMSprop
        ],
        "learning_rate": [0.001, 0.0001]
    }

    grid_search = GridSearch(
        model_fn=build_model,
        param_grid=param_grid,
        verbose=1,
        num_features=1,
        num_targets=1
    )

    grid_search.fit(
        train_dataset=data.train_dataset,
        val_dataset=data.val_dataset,
        epochs=1,
        verbose=1
    )

    grid_search.summary()
```

### RandomSearch Example

```python
    import tensorflow as tf
    from tensorcross.model_selection RandomSearch

    data = tf.data.Dataset()

    def build_model(
        optimizer: tf.keras.optimizers.Optimizer,
        learning_rate: float
    ) -> tf.keras.models.Model:
        x_input = tf.keras.layers.Input(shape=2)
        y_pred = tf.keras.layers.Dense(units=1)(x_input)
        model = tf.keras.models.Model(inputs=[x_input], outputs=[y_pred])

        opt = optimizer(learning_rate=learning_rate)

        model.compile(
            loss="mse", optimizer=opt, metrics=["mse"]
        )

        return model

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
```

## Issues

If you want to open an issue, it must be a bug, a feature request, or a significant problem with the documentation.
For more information see [here](.github/ISSUE_TEMPLATE/).

## Pull Requests

You are free to make contributions to the repository.
For more information see [here](.github/PULL_REQUEST_TEMPLATE/).
