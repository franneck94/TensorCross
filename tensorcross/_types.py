from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import TypedDict


class ResultsDict(TypedDict):
    best_score: float
    best_params: Dict[str, Any]
    val_scores: List[Any]
    params: List[Any]
