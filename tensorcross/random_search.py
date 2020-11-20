from typing import Any
from typing import Callable
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterSampler


class RandomSearch:
    def __init__(
        self,
        model_fn: Callable,
        param_distributions: Dict[str, Callable],
        n_iter: int = 10,
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        """RandomSearch for a given parameter distribution.

        Args:
            model_fn (Callable): Function that builds and compiles a keras Model
            param_distributions (Dict[str, Callable]): Dict of str, callable
                pairs, where the str is the parameter name of the
            n_iter (int, optional): Number of random models. Defaults to 10.
        """
        self.model_fn = model_fn
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_sampler = ParameterSampler(
            self.param_distributions,
            n_iter=self.n_iter
        )
        self.results = {
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
        """[summary]

        Args:
            train_dataset (tf.data.Dataset): tf.data.Dataset object for the
                training.
            val_dataset (tf.data.Dataset, optional): tf.data.Dataset object for
                the validation.. Defaults to None.
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
            self.results["val_scores"].append(val_metric)
            self.results["params"].append(random_combination)

        best_run_idx = np.argmax(self.results["val_scores"])
        self.results["best_score"] = self.results["val_scores"][best_run_idx]
        self.results["best_params"] = self.results["params"][best_run_idx]
