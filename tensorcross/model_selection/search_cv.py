from abc import ABCMeta


class BaseSearchCV(metaclass=ABCMeta):
    def __init__(self) -> None:
        raise NotImplementedError


class GridSearchCV(BaseSearchCV):
    def __init__(self) -> None:
        raise NotImplementedError


class RandomSearchCV(BaseSearchCV):
    def __init__(self) -> None:
        raise NotImplementedError
