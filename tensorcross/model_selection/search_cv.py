from abc import ABCMeta


class BaseSearchCV(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass


class GridSearchCV(BaseSearchCV):
    def __init__(self) -> None:
        pass


class RandomSearchCV(BaseSearchCV):
    def __init__(self) -> None:
        pass
