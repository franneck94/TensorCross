[![Python](https://img.shields.io/badge/python-%203.8-blue)]()
[![License](https://camo.githubusercontent.com/890acbdcb87868b382af9a4b1fac507b9659d9bf/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667)](https://github.com/franneck94/TensorCross/blob/main/LICENSE)
[![Build](https://github.com/franneck94/TensorCross/workflows/Test%20and%20Coverage/badge.svg?branch=main)](https://github.com/franneck94/TensorCross/actions?query=workflow%3A%22Test+and+Coverage%22)
[![codecov](https://codecov.io/gh/franneck94/TensorCross/branch/main/graph/badge.svg)](https://codecov.io/gh/franneck94/TensorCross)

# TensorCross

Cross Validation, Grid Search and Random Search for tf.data.Datasets in
TensorFlow 2.3 and Python 3.8.

## Motivation

Currently, there is the tf.keras.wrapper.KerasClassifier/KerasRegressor class,
which can be used to transform your tf.keras model into a sklearn estimator.
However, this approach is only applicable if your dataset is a numpy.ndarray
for your x and y data.
If you want to use the new tf.data.Dataset class, you cannot use the sklearn
wrappers.

## Implemented Features

- Random Search (with one validation set)
- Grid Search (with one validation set)

### TODO

- Random search with cross validation
- Grid search with cross validation
