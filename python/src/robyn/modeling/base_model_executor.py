#pyre-strict

from abc import abstractmethod
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutput import ModelOutputs
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig


class BaseModelExecutor(ABC):
    
    @abstractmethod
    def model_run(
        self,
        mmmdata: MMMData,
        hyperparameters: Hyperparameters,
        calibration_input: CalibrationInput,
        holidays_data: HolidaysData,
        trials_config: TrialsConfig,
        #model: Models = Models.RIDGE, # Ride, Hierarchical, or MMM
    ) -> ModelOutputs:
        pass