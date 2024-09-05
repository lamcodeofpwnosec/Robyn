# pyre-strict

# TODO This needs to be rewritten to match the new structure of the codebase
# TODO Add separate methods if state is loaded from robyn_object or json_file for each method

import logging

from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.calibration_input_validation import CalibrationInputValidation
from robyn.data.validation.holidays_data_validation import HolidaysDataValidation
from robyn.data.validation.hyperparameter_validation import HyperparametersValidation
from robyn.data.validation.mmmdata_validation import MMMDataValidation

#This needs to be rewritten to match the new structure of the codebase

from robyn.analysis.budget_allocation_result import BudgetAllocationResult
from robyn.analysis.budgetallocator_config import BudgetAllocatorConfig
from robyn.data.entities.mmmdata_collection import MMMDataCollection
from robyn.data.validation.calibration_input_validation import CalibrationInputValidation
from robyn.data.validation.holidays_data_validation import HolidaysDataValidation
from robyn.data.validation.hyperparameter_validation import HyperparametersValidation
from robyn.data.validation.mmmdata_validation import MMMDataValidation
from robyn.modeling.entities.model_refresh_config import ModelRefreshConfig
from robyn.modeling.entities.modeloutput import ModelOutput
from robyn.modeling.entities.modeloutput_collection import ModelOutputCollection
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig


class Robyn:
    def __init__(self, working_dir: str):
        """
        Initializes the Robyn object with a working directory.

        Args:
            working_dir (str): The path to the working directory.
        """
        self.working_dir = working_dir
        logger = logging.getLogger()
        logger.info(f"Robyn initialized with working directory: {working_dir}")

    # Load input data for the first time and validates
    def initialize(
        self,
        mmm_data: MMMData,
        holidays_data: HolidaysData,
        hyperparameters: Hyperparameters,
        calibration_input: CalibrationInput,
    ) -> None:
        """
        Loads input data for the first time and validates it.
        Calls validate from MMMDataValidation, HolidaysDataValidation, HyperparametersValidation, and CalibrationInputValidation.
        """
        mmm_data_validation = MMMDataValidation(mmm_data)
        holidays_data_validation = HolidaysDataValidation(holidays_data)
        hyperparameters_validation = HyperparametersValidation(hyperparameters)
        calibration_input_validation = CalibrationInputValidation(mmm_data, calibration_input)

        mmm_data_validation.validate()
        holidays_data_validation.validate()
        hyperparameters_validation.validate()
        calibration_input_validation.validate()


    # Load previous state from Json file
    def reinitialize_from_json(self, robyn_object_json_file: str) -> None:
        """
        Loads the previous state from a JSON file and validates it.

        Args:
            robyn_object_json_file (str): The path to the JSON file containing the previous state.
        """
        pass

    # Run models for all trials and iterations, using num of cores
    def model_run(
        self,
        num_of_cores: int,
        trials_config: TrialsConfig,
        plot: bool = False,
        export: bool = False,
    ) -> ModelOutput:
        """
        Runs the models for all trials and iterations using the specified number of cores.

        Args:
            mmmdata_collection (MMMDataCollection): The MMM data collection object.
            num_of_cores (int): The number of cores to use for running the models.
            trials_config (TrialsConfig): The trials configuration object.

        Returns:
            OutputModels: The output models object.
        """
        pass

    # Run budget allocator for given MMMDataCollection and ModelOutputsCollection
    def budget_allocator(
        self,
        select_model: str,
        budger_allocator_config: BudgetAllocatorConfig,
        report: bool = False,
        plot: bool = False,
        onepager: bool = False,
        export: bool = False,
    ) -> BudgetAllocationResult:
        """
        Runs the budget allocator for the given MMMDataCollection and ModelOutputsCollection.

        Args:
            mmmdata_colllection (MMMDataCollection): The MMM data collection object.
            model_output_collection (ModelOutputsCollection): The model output collection object.
            select_model (str): The selected model to use for budget allocation.
            budger_allocator_config (BudgetAllocatorConfig): The budget allocator configuration object.

        Returns:
            BudgetAllocationResult: The budget allocation result object.
        """
        pass

    # Run budget allocator - load from json file? # Using json file from robyn_write() for allocation
    def budget_allocator_from_json(
        self,
        robyn_object_json: str,
        select_model: str,
        budger_allocator_config: BudgetAllocatorConfig,
        report: bool = False,
        plot: bool = False,
        onepager: bool = False,
        export: bool = False,
    ) -> BudgetAllocationResult:
        """
        Runs the budget allocator using the specified JSON file.

        Args:
            robyn_object_json (str): The path to the JSON file containing the robyn object.
            select_model (str): The selected model to use for budget allocation.
            budger_allocator_config (BudgetAllocatorConfig): The budget allocator configuration object.

        Returns:
            BudgetAllocationResult: The budget allocation result object.
        """
        pass

    # Refresh
    def model_refresh(
        self,
        refresh_config: ModelRefreshConfig,
        objective_weights=None,
    ) -> BudgetAllocationResult:
        """
        Refreshes the model using the specified configuration and input.

        Args:
            refresh_config (ModelRefreshConfig): The refresh configuration object.
            objective_weights (dict, optional): The objective weights dictionary. Defaults to None.

        Returns:
            BudgetAllocationResult: The budget allocation result object.
        """
        pass


    # Model Evaluate (outputs.R from Robyn)
    def model_evaluate(self,
        pareto_fronts : str="auto",
        calibration_constraint : float=0.1,
        plot_pareto: bool=True,
        clusters: bool=True,
        plot:bool=False,
        select_model:str="clusters",
        ) -> Any: #TODO Update return type
        """
        Evaluate the model using the given data collection and output models.

        Parameters:
        - mmmdata_collection (MMMDataCollection): The collection of MMMData objects.
        - output_models (ModelOutput): The output models to evaluate.
        - pareto_fronts (str, optional): The method to calculate pareto fronts. Defaults to "auto".
        - calibration_constraint (float, optional): The calibration constraint value. Defaults to 0.1.
        - plot_pareto (bool, optional): Whether to plot the pareto fronts. Defaults to True.
        - clusters (bool, optional): Whether to use clustering. Defaults to True.
        - plot (bool, optional): Whether to plot the results. Defaults to False.
        - select_model (str, optional): The method to select the model. Defaults to "clusters".

        Returns:
        - ModelOutputCollection: The collection of model outputs.
        """
        pass

    # If json_file has everything always including MMMDataCollection, ModelOutputCollection then use reinitialize_from_json
    def robyn_response_from_json(
        self,
        json_file=None,
        select_build=None,
        select_model=None,
        metric_name=None,
        metric_value=None,
        date_range=None,
        dt_hyppar=None,
        dt_coef=None,
    ):
        """
        Generates a response for the given input parameters using a JSON file.

        Args:
            json_file (_type_, optional): The path to the JSON file. Defaults to None.
            select_build (_type_, optional): The selected build. Defaults to None.
            select_model (_type_, optional): The selected model. Defaults to None.
            metric_name (_type_, optional): The metric name. Defaults to None.
            metric_value (_type_, optional): The metric value. Defaults to None.
            date_range (_type_, optional): The date range. Defaults to None.
            dt_hyppar (_type_, optional): The dt hyppar. Defaults to None.
            dt_coef (_type_, optional): The dt coef. Defaults to None.
        """
        pass
