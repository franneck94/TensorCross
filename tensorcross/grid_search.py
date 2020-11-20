import numpy as np
import tensorflow as tf

from typing import Any, Callable, Dict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid


class GridSearch:
    def __init__(
        self,
        model_fn: Callable,
        parameter_grid: Dict[str, Any],
        n_iter: int = 10,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        self.model_fn = model_fn
        self.parameter_grid = ParameterGrid(parameter_grid)
        self.n_iter = n_iter
        self.verbose = verbose
        self.results_ = {
            "best_score": -np.inf,
            "best_params": {},
            "val_scores": [],
            "params": [],
        }
        self.model_fn_kwargs = kwargs

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset = None,
        **kwargs: Any
    ) -> None:
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

    def summary(self) -> None:
        print(
            f"Best score: {self.results_['best_score']} "
            f"using params: {self.results_['best_params']}\n"
        )

        scores = self.results_["val_scores"]
        params = self.results_["params"]

        for idx, (score, param) in enumerate(zip(scores, params)):
            print(f"Idx: {idx} - Score: {score} with param: {param}")