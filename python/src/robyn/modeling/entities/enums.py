#pyre-strict

from enum import Enum

class Models(Enum):
    """
    Enum class representing different models.

    Attributes:
        RIDGE (str): Ridge model.
    """
    RIDGE: str = "Ridge"

class NevergradAlgorithm(Enum):
    """
    Enum class representing different optimization algorithms from Nevergrad.

    Attributes:
        TWO_POINTS_DE (str): Two Points Differential Evolution algorithm.
    """
    TWO_POINTS_DE: str = "TwoPointsDE"