from __future__ import annotations

import logging
import os
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.models import Model
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler

from tensorcross._types import ResultsDict
from tensorcross.utils import dataset_split


logger = tf.get_logger()


class BaseSearchCV(ABC):
    """Abstract BaseSearchCV class for the grid or random search
    with cross validation.

    Args:
        model_fn (Callable[..., Model]): Function that
            builds and compiles a Model object.
        n_folds (int): How many folds. Defaults to 3.
        verbose (int): Whether to show information in terminal.
            Defaults to 0.
        kwargs (Any): Keyword arguments for the model_fn function.
    """

    @abstractmethod
    def __init__(
        self,
        model_fn: Callable[..., Model],
        n_folds: int = 3,
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        self.model_fn = model_fn
        self.verbose = verbose
        self.n_folds = n_folds
        self.model_fn_kwargs = kwargs
        self.results_: ResultsDict = {
            "best_score": -np.inf,
            "best_params": {},
            "best_model" : [],
            "val_scores": [],
            "params": [],
            "models" : []
        }

    def _run_search(  # noqa: PLR0912
        self,
        dataset: tf.data.Dataset,
        parameter_obj: Union[ParameterGrid, ParameterSampler],
        **kwargs: Any,
    ) -> None:
        """Runs the exhaustive grid search over the parameter grid.

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset object for the training.
            parameter_obj (ParameterGrid | ParameterSampler): Object to iterate
                over, to generate hyperparameter combinations.
            kwargs (Any): Keyword arguments for the fit method of the
                Model or tf.keras.models.Sequential model.
        """
        if len(dataset) == 0:
            error_msg = "Validation Set is empty!"
            raise ValueError(error_msg)

        maximize = True
        tensorboard_callback: Optional[TensorBoard] = None
        tensorboard_log_dir = ""

        for param, value in kwargs.items():
            if param == "callbacks":
                for callback in value:
                    if isinstance(callback, TensorBoard):
                        tensorboard_callback = callback

        if tensorboard_callback:
            tensorboard_log_dir = tensorboard_callback.log_dir

        split_fraction = 1.0 / self.n_folds

        tf_log_level = logger.level
        logger.setLevel(logging.ERROR)  # Issue 30: Ignore warnings for training

        for idx, grid_combination in enumerate(parameter_obj):
            if self.verbose:
                print(f"Running Comb: {idx}")

            val_scores = np.zeros(shape=(self.n_folds,), dtype=np.float32)

            for fold in range(self.n_folds):
                if self.verbose:
                    print(f"Running Fold: {fold}")

                model = self.model_fn(
                    **grid_combination, **self.model_fn_kwargs
                )

                train_dataset, val_dataset = dataset_split(
                    dataset=dataset, split_fraction=split_fraction, fold=fold
                )
                if len(val_dataset) == 0:
                    error_msg = "Validation Set is empty!"
                    raise ValueError(error_msg)

                if tensorboard_callback:
                    if not os.path.exists(tensorboard_log_dir):
                        os.mkdir(tensorboard_log_dir)
                    new_log_dir = os.path.join(
                        tensorboard_log_dir, f"model_{idx}_fold_{fold}"
                    )
                    os.mkdir(new_log_dir)
                    tensorboard_callback.log_dir = new_log_dir

                model.fit(train_dataset, validation_data=val_dataset, **kwargs)

                if len(model.metrics) > 1:
                    val_score = model.evaluate(val_dataset, verbose=0)[-1]
                else:
                    maximize = False
                    val_score = model.evaluate(val_dataset, verbose=0)
                val_scores[fold] = val_score

            self.results_["val_scores"].append(val_scores)
            self.results_["params"].append(grid_combination)
            self.results_["models"].append(model)

        logger.setLevel(tf_log_level)  # Issue 30

        mean_val_scores = np.mean(self.results_["val_scores"], axis=1)
        if maximize:
            best_run_idx = np.argmax(mean_val_scores)
        else:
            best_run_idx = np.argmin(mean_val_scores)
        self.results_["best_score"] = self.results_["val_scores"][best_run_idx]
        self.results_["best_params"] = self.results_["params"][best_run_idx]
        self.results_["best_model"] = self.results_["models"][best_run_idx]

    def summary(self) -> str:
        """Prints the summary of the search to the console.

        Assuming the *RandomSearch* had n iterations or the
        *GridSearch* had n combinations in total, the output
        would have the following structure:
        ```
            --------------------------------------------------
            Best score: ``float`` using params: ``dict``
            --------------------------------------------------
            Idx: 0   - Score: ``float`` using params: ``dict``
            ...
            Idx: n-1 - Score: ``float`` using params: ``dict``
            --------------------------------------------------
        ```

        Returns:
            Full string of the summary that was printed.
        """
        best_params_str = (
            f"Best score: {self.results_['best_score']} "
            f"using params: {self.results_['best_params']}"
        )
        dashed_line = "".join(
            map(lambda x: "-", best_params_str)  # noqa: C417, ARG005
        )

        current_line = f"\n{dashed_line}\n{best_params_str}\n{dashed_line}"
        results_str = current_line
        print(current_line)

        scores = self.results_["val_scores"]
        params = self.results_["params"]

        for idx, (score, param) in enumerate(zip(scores, params)):
            current_line = f"Idx: {idx} - Score: {score} with param: {param}"
            results_str += current_line
            print(current_line)

        current_line = f"{dashed_line}\n"
        results_str += current_line
        print(current_line)

        return results_str


class GridSearchCV(BaseSearchCV):
    def __init__(
        self,
        model_fn: Callable[..., Model],
        param_grid: Mapping[str, Iterable],
        n_folds: int = 3,
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        """GridSearchCV for a given parameter grid.

        The grid search iterates over all combinations of the param_grid
        dictionary, which defines the hyperparameter values for a key that
        is a parameter name of the model_fn.
        For example, if the model_fn has the parameter "num_units" a dictionary
        could look like this:

        ``` python
            def model_fn(num_units: int):
                pass

            param_distributions = {"num_units": [10, 20 ,30]}
        ```

        Note: Inside the model_fn it is expected that the model is compiled.

        The grid search is evaluated by:
        - The validation loss value, if no metrics are passed to model.compile()
        - The validation score of the last defined metric in model.compile()

        ``` python
            model.compile(loss="mse", metrics=["mse", "mae"])
        ```

        This would sort the grid search combinations based on the validation
        mae score.

        Args:
            model_fn (Callable[..., Model]): Function that
                builds and compiles a Model object.
            param_grid (Mapping[str, Iterable]): Dict of str, iterable
            hyperparameter, where the str is the parameter name of the.
            n_folds (int): How many folds. Defaults to 3.
            verbose (int): Whether to show information in terminal.
                Defaults to 0.
            kwargs (Any): Keyword arguments for the model_fn function.
        """
        super().__init__(
            model_fn=model_fn, n_folds=n_folds, verbose=verbose, **kwargs
        )
        self.param_grid = ParameterGrid(param_grid)

    def fit(self, dataset: tf.data.Dataset, **kwargs: Any) -> None:
        """Runs the exhaustive grid search over the parameter grid.

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset object for the training.
            kwargs (Any): Keyword arguments for the fit method of the
                Model or tf.keras.models.Sequential model.
        """
        super()._run_search(
            dataset=dataset, parameter_obj=self.param_grid, **kwargs
        )


class RandomSearchCV(BaseSearchCV):
    def __init__(  # noqa: PLR0913
        self,
        model_fn: Callable[..., Model],
        param_distributions: Dict[str, Callable],
        n_iter: int = 10,
        n_folds: int = 3,
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        """RandomSearchCV for a given parameter distribution.

        The random search randomly iterates over the param_distributions
        dictionary, which defines the hyperparameter value range for a key that
        is a parameter name of the model_fn.
        For example, if the model_fn has the parameter "num_units" a dictionary
        could look like this:

        ``` python
            def model_fn(num_units: int):
                pass

            param_distributions = {"num_units": [10, 20 ,30]}
        ```

        Note: Inside the model_fn it is expected that the model is compiled.

        The random search is evaluated by:
        - The validation loss value, if no metrics are passed to model.compile()
        - The validation score of the last defined metric in model.compile()

        ``` python
            model.compile(loss="mse", metrics=["mse", "mae"])
        ```

        This would sort the grid search combinations based on the validation
        mae score.

        Args:
            model_fn (Callable[..., Model]): Function that
                builds and compiles a Model object.
            param_distributions (Dict[str, Callable]): Dict of str, callable
                pairs, where the str is the parameter name of the.
            n_iter (int): Number of random models. Defaults to 10.
            n_folds (int): How many folds. Defaults to 3.
            verbose (int): Whether to show information in terminal.
                Defaults to 0.
            kwargs (Any): Keyword arguments for the model_fn function.
        """
        super().__init__(
            model_fn=model_fn, n_folds=n_folds, verbose=verbose, **kwargs
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_sampler = ParameterSampler(
            self.param_distributions, n_iter=self.n_iter
        )

    def fit(self, dataset: tf.data.Dataset, **kwargs: Any) -> None:
        """Runs the random search over the parameter distributions.

        Args:
            dataset (tf.data.Dataset): tf.data.Dataset object for the training.
            kwargs (Any): Keyword arguments for the fit method of the
                Model or tf.keras.models.Sequential model.
        """
        super()._run_search(
            dataset=dataset, parameter_obj=self.random_sampler, **kwargs
        )
