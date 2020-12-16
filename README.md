# TensorCross

![Python](https://img.shields.io/badge/python-%203.8-blue)
[![License](https://camo.githubusercontent.com/890acbdcb87868b382af9a4b1fac507b9659d9bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667)](https://github.com/franneck94/TensorCross/blob/main/LICENSE)
[![Build](https://github.com/franneck94/TensorCross/workflows/Test/badge.svg)](https://github.com/franneck94/TensorCross/actions?query=workflow%3A%22Test+and+Coverage%22)
[![codecov](https://codecov.io/gh/franneck94/TensorCross/branch/main/graph/badge.svg)](https://codecov.io/gh/franneck94/TensorCross)
[![Documentation](https://img.shields.io/badge/ref-Documentation-blue)](https://franneck94.github.io/TensorCross-Docs/)

```bash
pip install tensorcross
```

Cross Validation, Grid Search and Random Search for tf.data.Datasets in TensorFlow 2.0-2.4 and Python 3.8.

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
- [GridSearchCV](#GridSearchCV-Example)
- For more examples see: [here](examples/)

### Dataset and TensorFlow Model for the Examples

```python
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(
        (np.array([1, 2, 3]).reshape(-1, 1),  # x
         np.array([-1, -2, -3]).reshape(-1, 1))  # y
    )

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
```

The dataset must be a tf.data.Dataset object and you have to define a
function/callable that returns a compiled tf.keras.models.Model object.
This object will then be trained in e.g. the GridSearch.

### GridSearch Example

Assuming you have a tf.data.Dataset object and a build_model function,
defined as above. You can run a GridSearch as below:

```python
    from tensorcross.model_selection GridSearch

    train_dataset, val_dataset = dataset_split(
        dataset=dataset,
        split_fraction=(1 / 3)
    )

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
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=1,
        verbose=1
    )

    grid_search.summary()
```

This would result in the following console output:

```console
    --------------------------------------------------
    Best score: 1.1800532341003418 using params: {
        'learning_rate': 0.001, 'optimizer': 'RMSprop'
    }
    --------------------------------------------------
    Idx: 0 - Score: 0.2754371166229248 with param: {
        'learning_rate': 0.001, 'optimizer': 'Adam'
    }
    Idx: 1 - Score: 1.1800532341003418 with param: {
        'learning_rate': 0.001, 'optimizer': 'RMSprop'
    }
    Idx: 2 - Score: 0.055416107177734375 with param: {
        learning_rate': 0.0001, 'optimizer': 'Adam'
    }
    Idx: 3 - Score: 0.12417340278625488 with param: {
        'learning_rate': 0.0001, 'optimizer': 'RMSprop'
    }
    --------------------------------------------------
```

### GridSearchCV Example

Assuming you have a tf.data.Dataset object and a build_model function,
defined as above. You can run a GridSearchCV as below:

```python
    from tensorcross.model_selection GridSearchCV

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
        dataset=dataset,
        epochs=1,
        verbose=1
    )

    grid_search_cv.summary()
```

This would result in the following console output:

```console
    --------------------------------------------------
    Best score: 1.1800532341003418 using params: {
        'learning_rate': 0.001, 'optimizer': 'RMSprop'
    }
    --------------------------------------------------
    Idx: 0 - Score: 0.2754371166229248 with param: {
        'learning_rate': 0.001, 'optimizer': 'Adam'
    }
    Idx: 1 - Score: 1.1800532341003418 with param: {
        'learning_rate': 0.001, 'optimizer': 'RMSprop'
    }
    Idx: 2 - Score: 0.055416107177734375 with param: {
        learning_rate': 0.0001, 'optimizer': 'Adam'
    }
    Idx: 3 - Score: 0.12417340278625488 with param: {
        'learning_rate': 0.0001, 'optimizer': 'RMSprop'
    }
    --------------------------------------------------
```
