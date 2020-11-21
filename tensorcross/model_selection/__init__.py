from typing import Any
from typing import Callable
from typing import Dict
from typing import Mapping

import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler


class BaseSearch:
    def __init__(self) -> None:
        self.results_ = {
            "best_score": -np.inf,
            "best_params": {},
            "val_scores": [],
            "params": [],
        }

    def summary(self) -> None:
        print(
            f"Best score: {self.results_['best_score']} "
            f"using params: {self.results_['best_params']}\n"
        )

        scores = self.results_["val_scores"]
        params = self.results_["params"]

        for idx, (score, param) in enumerate(zip(scores, params)):
            print(f"Idx: {idx} - Score: {score} with param: {param}")


class GridSearch(BaseSearch):
    def __init__(
        self,
        model_fn: Callable,
        parameter_grid: Mapping,
        n_iter: int = 10,
        verbose: int = 0,
        **kwargs: Any
    ) -> None:
        """RandomSearch for a given parameter distribution.

        Args:
            model_fn (Callable): Function that builds and compiles a
                tf.keras.Model or tf.keras.Sequential object.
            parameter_grid (Dict[str, Iterable]): Dict of str, iterable
                hyperparameter, where the str is the parameter name of the.
            n_iter (int, optional): Number of random models. Defaults to 10.
            verbose (int, optional): Whether to show information in terminal.
                Defaults to 0.
        """
        super().__init__()
        self.model_fn = model_fn
        self.parameter_grid = ParameterGrid(parameter_grid)
        self.n_iter = n_iter
        self.verbose = verbose
        self.model_fn_kwargs = kwargs

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        **kwargs: Any
    ) -> None:
        """[summary]

        Args:
            train_dataset (tf.data.Dataset): tf.data.Dataset object for the
                training.
            val_dataset (tf.data.Dataset, optional): tf.data.Dataset object for
                the validation.
            kwargs (Any): Keyword arguments for the build model_fn.
        """
        for idx, grid_combination in enumerate(self.parameter_grid):
            if self.verbose:
                print(f"Running Comb: {idx}")
            model = self.model_fn(
                **grid_combination,
                **self.model_fn_kwargs
            )

            model.fit(
                train_dataset,
                validation_data=val_dataset,
                **kwargs,
            )

            val_metric = model.evaluate(
                val_dataset,
                verbose=0
            )[1]
            self.results_["val_scores"].append(val_metric)
            self.results_["params"].append(grid_combination)

        best_run_idx = np.argmax(self.results_["val_scores"])
        self.results_["best_score"] = self.results_["val_scores"][best_run_idx]
        self.results_["best_params"] = self.results_["params"][best_run_idx]


class RandomSearch(BaseSearch):
    def __init__(
        self,
        model_fn: Callable,
        param_distributions: Dict[str, Callable],
        n_iter: int = 10,
        verbose: int = 0,
        **kwargs: Any
    ) -> None:
        """RandomSearch for a given parameter distribution.

        Args:
            model_fn (Callable): Function that builds and compiles a
                tf.keras.Model or tf.keras.Sequential object.
            param_distributions (Dict[str, Callable]): Dict of str, callable
                pairs, where the str is the parameter name of the.
            n_iter (int, optional): Number of random models. Defaults to 10.
            verbose (int, optional): Whether to show information in terminal.
                Defaults to 0.
        """
        super().__init__()
        self.model_fn = model_fn
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_sampler = ParameterSampler(
            self.param_distributions,
            n_iter=self.n_iter
        )
        self.model_fn_kwargs = kwargs

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        **kwargs: Any
    ) -> None:
        """[summary]

        Args:
            train_dataset (tf.data.Dataset): tf.data.Dataset object for the
                training.
            val_dataset (tf.data.Dataset, optional): tf.data.Dataset object for
                the validation.
            kwargs (Any): Keyword arguments for the build model_fn.
        """
        for idx, random_combination in enumerate(self.random_sampler):
            if self.verbose:
                print(f"Running Comb: {idx}")
            model = self.model_fn(
                **random_combination,
                **self.model_fn_kwargs
            )

            model.fit(
                train_dataset,
                validation_data=val_dataset,
                **kwargs,
            )

            val_metric = model.evaluate(
                val_dataset,
                verbose=0
            )[1]
            self.results_["val_scores"].append(val_metric)
            self.results_["params"].append(random_combination)

        best_run_idx = np.argmax(self.results_["val_scores"])
        self.results_["best_score"] = self.results_["val_scores"][best_run_idx]
        self.results_["best_params"] = self.results_["params"][best_run_idx]
