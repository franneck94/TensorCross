Welcome to TensorCross's documentation!
=======================================

Cross Validation, Grid Search and Random Search for tf.data.Datasets in
TensorFlow 2.3 and Python 3.8.


Motivation
==========

Currently, there is the tf.keras.wrapper.KerasClassifier/KerasRegressor class,
which can be used to transform your tf.keras model into a sklearn estimator.
However, this approach is only applicable if your dataset is a numpy.ndarray
for your x and y data.
If you want to use the new tf.data.Dataset class, you cannot use the sklearn
wrappers.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
