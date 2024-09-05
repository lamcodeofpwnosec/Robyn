#pyre-strict

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.enums import Models, NevergradAlgorithm
from robyn.modeling.entities.modeloutput import ModelOutputs
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig


class BaseModelExecutor(ABC):
    """
    Abstract base class for executing a model.
    """

    def __init__(self,
        mmmdata: MMMData,
        holidays_data: HolidaysData,
        hyperparameters: Hyperparameters,
        calibration_input: CalibrationInput,
        trials_config: TrialsConfig,
        #model: Models = Models.RIDGE, # Ride, Hierarchical, or MMM
    ) -> ModelOutputs:
        pass