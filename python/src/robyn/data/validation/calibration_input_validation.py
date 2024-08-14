from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.validation import Validation, ValidationResult
from typing import List

class CalibrationInputValidation(Validation):
    def __init__(self, mmmdata: MMMData, calibration_input: CalibrationInput) -> None:
        self.mmmdata = mmmdata
        self.calibration_input = calibration_input

    def check_calibration_input(
        self, mmm_data: MMMData, time_window: TimeWindow, day_interval: int
    ) -> ValidationResult:
        """
        This function checks the calibration input data for consistency and correctness.
        It verifies that the input data contains the required columns, that the date range
        is within the modeling window, and that the spend values match the input data.
        """
        # method implementation goes here
        raise NotImplementedError("Not yet implemented")

    def check_objective_weights(
        self, objective_weights: Optional[List[float]], refresh: bool
    ) -> ValidationResult:
        """
        Check if the objective weights are valid.
        """
        raise NotImplementedError("Not yet implemented")

    def validate(self) -> ValidationResult:
        pass
        #raise NotImplementedError("Not yet implemented")
