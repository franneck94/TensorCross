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
        verbose: bool = False,
        **kwargs: Any
    ) -> None:
        self.model_fn = model_fn
        self.parameter_grid = ParameterGrid(parameter_grid)
        self.results = {
            "best_score": -np.inf,
            "best_params": {},
            "val_scores": [],
            "params": [],
        }


    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset = None
    ) -> None:
        pass