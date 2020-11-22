from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Dict
from typing import Mapping
from typing import Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler


class BaseSearch(metaclass=ABCMeta):
    """RandomSearch for a given parameter distribution.

        Args:
            model_fn (Callable): Function that builds and compiles a
                tf.keras.Model or tf.keras.Sequential object.
            verbose (int, optional): Whether to show information in terminal.
                Defaults to 0.
            kwargs (Any): Keyword arguments for the model_fn function.
        """
    @abstractmethod
    def __init__(
        self,
        model_fn: Callable,
        verbose: int = 0,
        **kwargs: Any
    ) -> None:
        self.model_fn = model_fn
        self.verbose = verbose
        self.model_fn_kwargs = kwargs
        self.results_ = {
            "best_score": -np.inf,
            "best_params": {},
            "val_scores": [],
            "params": [],
        }

    def _run_search(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        parameter_obj: Union[ParameterGrid, ParameterSampler],
        **kwargs: Any
    ) -> None:
        """Runs the exhaustive grid search over the parameter grid.

        Args:
            train_dataset (tf.data.Dataset): tf.data.Dataset object for the
                training.
            val_dataset (tf.data.Dataset): tf.data.Dataset object for
                the validation.
            parameter_obj (ParameterGrid | ParameterSampler): Object to iterate
                over, to generate hyperparameter combinations.
            kwargs (Any): Keyword arguments for the fit method of the
                tf.keras.models.Model or tf.keras.models.Sequential model.
        """
        for idx, grid_combination in enumerate(parameter_obj):
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

    def summary(self) -> None:
        """Prints the summary of the search to the console.
        """
        best_params_str = (f"Best score: {self.results_['best_score']} "
                           f"using params: {self.results_['best_params']}")
        dashed_line = "".join(map(lambda x: "-", best_params_str))
        print(f"\n{dashed_line}\n{best_params_str}\n{dashed_line}")

        scores = self.results_["val_scores"]
        params = self.results_["params"]

        for idx, (score, param) in enumerate(zip(scores, params)):
            print(f"Idx: {idx} - Score: {score} with param: {param}")

        print(f"{dashed_line}\n")


class GridSearch(BaseSearch):
    def __init__(
        self,
        model_fn: Callable,
        param_grid: Mapping,
        n_iter: int = 10,
        verbose: int = 0,
        **kwargs: Any
    ) -> None:
        """RandomSearch for a given parameter distribution.

        Args:
            model_fn (Callable): Function that builds and compiles a
                tf.keras.Model or tf.keras.Sequential object.
            param_grid (Dict[str, Iterable]): Dict of str, iterable
                hyperparameter, where the str is the parameter name of the.
            n_iter (int, optional): Number of random models. Defaults to 10.
            verbose (int, optional): Whether to show information in terminal.
                Defaults to 0.
            kwargs (Any): Keyword arguments for the model_fn function.
        """
        super().__init__(
            model_fn=model_fn,
            verbose=verbose,
            **kwargs
        )
        self.param_grid = ParameterGrid(param_grid)
        self.n_iter = n_iter

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        **kwargs: Any
    ) -> None:
        """Runs the exhaustive grid search over the parameter grid.

        Args:
            train_dataset (tf.data.Dataset): tf.data.Dataset object for the
                training.
            val_dataset (tf.data.Dataset): tf.data.Dataset object for
                the validation.
            kwargs (Any): Keyword arguments for the fit method of the
                tf.keras.models.Model or tf.keras.models.Sequential model.
        """
        super()._run_search(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            parameter_obj=self.param_grid,
            **kwargs
        )


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
            kwargs (Any): Keyword arguments for the model_fn function.
        """
        super().__init__(
            model_fn=model_fn,
            verbose=verbose,
            **kwargs
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_sampler = ParameterSampler(
            self.param_distributions,
            n_iter=self.n_iter
        )

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        **kwargs: Any
    ) -> None:
        """Runs the random search over the parameter distributions.

        Args:
            train_dataset (tf.data.Dataset): tf.data.Dataset object for the
                training.
            val_dataset (tf.data.Dataset): tf.data.Dataset object for
                the validation.
            kwargs (Any): Keyword arguments for the fit method of the
                tf.keras.models.Model or tf.keras.models.Sequential model.
        """
        super()._run_search(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            parameter_obj=self.random_sampler,
            **kwargs
        )
