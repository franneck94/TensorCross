from abc import ABCMeta


class BaseSearchCV(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass


class GridSearchCV(BaseSearchCV):
    def __init__(self) -> None:
        super().__init__()


class RandomSearchCV(BaseSearchCV):
    def __init__(self) -> None:
        super().__init__()
