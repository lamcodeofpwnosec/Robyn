# pyre-strict

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.enums import AdStockType
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.modeloutput import ModelOutputs, Trial
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class ModelRefitOutput:
    """
    Represents the output of a model refit process.

    Attributes:
        rsq_train (float): The R-squared value for the training data.
        rsq_val (Optional[float]): The R-squared value for the validation data (optional).
        rsq_test (Optional[float]): The R-squared value for the test data (optional).
        nrmse_train (float): The normalized root mean squared error for the training data.
        nrmse_val (Optional[float]): The normalized root mean squared error for the validation data (optional).
        nrmse_test (Optional[float]): The normalized root mean squared error for the test data (optional).
        coefs (np.ndarray): The coefficients of the model.
        y_train_pred (np.ndarray): The predicted values for the training data.
        y_val_pred (Optional[np.ndarray]): The predicted values for the validation data (optional).
        y_test_pred (Optional[np.ndarray]): The predicted values for the test data (optional).
        y_pred (np.ndarray): The predicted values for all data.
        mod (Ridge): The fitted Ridge model.
        df_int (int): The degrees of freedom for the intercept.
    """

    rsq_train: float
    rsq_val: Optional[float]
    rsq_test: Optional[float]
    nrmse_train: float
    nrmse_val: Optional[float]
    nrmse_test: Optional[float]
    coefs: np.ndarray
    y_train_pred: np.ndarray
    y_val_pred: Optional[np.ndarray]
    y_test_pred: Optional[np.ndarray]
    y_pred: np.ndarray
    mod: Ridge
    df_int: int


@dataclass(frozen=True)
class HyperCollectorOutput:
    """
    Represents the output of the HyperCollector.

    Attributes:
        hyper_list_all (Hyperparameters): The list of all hyperparameters.
        hyper_bound_list_updated (Hyperparameters): The list of updated hyperparameters.
        hyper_bound_list_fixed (Hyperparameters): The list of fixed hyperparameters.
        dt_hyper_fixed_mod (pd.DataFrame): The modified fixed hyperparameters.
        all_fixed (bool): Indicates if all hyperparameters are fixed.
    """

    hyper_list_all: Hyperparameters
    hyper_bound_list_updated: Hyperparameters
    hyper_bound_list_fixed: Hyperparameters
    dt_hyper_fixed_mod: pd.DataFrame
    all_fixed: bool


@dataclass(frozen=True)
class ModelDecompOutput:
    """
    Represents the output of the model decomposition process.

    Attributes:
        x_decomp_vec (pd.DataFrame): The decomposed vector of x.
        x_decomp_vec_scaled (pd.DataFrame): The scaled decomposed vector of x.
        x_decomp_agg (pd.DataFrame): The aggregated decomposed vector of x.
        coefs_out_cat (pd.DataFrame): The coefficients of the categorical variables.
        media_decomp_immediate (pd.DataFrame): The immediate media decomposition.
        media_decomp_carryover (pd.DataFrame): The carryover media decomposition.
    """

    x_decomp_vec: pd.DataFrame
    x_decomp_vec_scaled: pd.DataFrame
    x_decomp_agg: pd.DataFrame
    coefs_out_cat: pd.DataFrame
    media_decomp_immediate: pd.DataFrame
    media_decomp_carryover: pd.DataFrame


class RidgeModelBuilder:
    """
    Class responsible for building and running the model.

    Args:
        mmm_data (MMMData): The MMM data.
        holiday_data (HolidaysData): The holiday data.
        calibration_input (CalibrationInput): The calibration input.
        hyperparameters (Hyperparameters): The hyperparameters.

    Attributes:
        mmm_data (MMMData): The MMM data.
        holiday_data (HolidaysData): The holiday data.
        calibration_input (CalibrationInput): The calibration input.
        hyperparameters (Hyperparameters): The hyperparameters.

    Methods:
        model_run: Runs the model.
        _model_train: Trains the model.
        run_nevergrad_optimization: Runs the Nevergrad optimization.
        model_decomp: Performs model decomposition.
        _model_refit: Refits the model.
        _lambda_seq: Generates a sequence of lambda values.
        hyper_collector: Collects hyperparameters.

    """

    def __init__(
        self,
        mmm_data: MMMData,
        holiday_data: HolidaysData,
        calibration_input: CalibrationInput,
        hyperparameters: Hyperparameters,
    ) -> None:
        self.mmm_data = mmm_data
        self.holiday_data = holiday_data
        self.calibration_input = calibration_input
        self.hyperparameters = hyperparameters

    def build_models(
        self,
        trials_config: TrialsConfig,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        seed: int = 123,
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[List[float]] = None,
        nevergrad_algo: str = "TwoPointsDE",  # TODO: Enum?
        intercept: bool = True,
        intercept_sign: str = "non_negative",  # TODO: Enum?
        ad_stock: AdStockType = AdStockType,
    ) -> ModelOutputs:
        """
        Run the model with the given configuration.

        Args:
            trials_config (TrialsConfig): Configuration for the trials.
            dt_hyper_fixed (Optional[pd.DataFrame], optional): Fixed hyperparameters. Defaults to None.
            ts_validation (bool, optional): Flag indicating whether to perform time series validation. Defaults to False.
            add_penalty_factor (bool, optional): Flag indicating whether to add penalty factor. Defaults to False.
            seed (int, optional): Seed for reproducibility. Defaults to 123.
            rssd_zero_penalty (bool, optional): Flag indicating whether to apply zero penalty for RSSD. Defaults to True.
            objective_weights (Optional[List[float]], optional): Weights for the objective function. Defaults to None.
            nevergrad_algo (str, optional): Algorithm for optimization. Defaults to "TwoPointsDE".
            intercept (bool, optional): Flag indicating whether to include an intercept term. Defaults to True.
            intercept_sign (str, optional): Sign of the intercept term. Defaults to "non_negative".

        Returns:
            ModelOutputs: Outputs of the model.
        """
        pass

    def _model_train(
        self,
        hyper_collect: Hyperparameters,
        trials_config: TrialsConfig,
        cores: int,
        intercept_sign: str,
        intercept: bool,
        nevergrad_algo: str,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        ts_validation: bool = True,
        add_penalty_factor: bool = False,
        objective_weights: Optional[Dict[str, float]] = None,
        rssd_zero_penalty: bool = True,
        seed: int = 123,
    ) -> ModelOutputs:
        # Implementation
        pass

    # model.R robyn_mmm
    def run_nevergrad_optimization(self,
                      hyper_collect: Hyperparameters,
                      iterations: int,
                      cores: int,
                      nevergrad_algo: str = "TwoPointsDE",
                      intercept:bool = True,
                      intercept_sign: str = "non_negative",
                      ts_validation = True,
                      add_penalty_factor = False,
                      objective_weights: Optional[Dict[str, float]] = None,
                      dt_hyper_fixed: Optional[pd.DataFrame] = None,
                      rssd_zero_penalty: bool = True,
                      trial: int = 1,
                      seed: int = 123,
                      ):
        # Implementation
        pass

        def _iteration(self) -> Trial:
            # Implementation
            # Inner method to run nevergrad optimization
            pass

    @staticmethod
    def model_decomp(
        coefs: Dict[str, float],
        y_pred: np.ndarray,
        dt_mod_saturated: pd.DataFrame,
        dt_saturated_immediate: pd.DataFrame,
        dt_saturated_carryover: pd.DataFrame,
        dt_mod_roll_wind: pd.DataFrame,
        refresh_added_start: str,
    ) -> ModelDecompOutput:
        # Implementation here
        pass

    @staticmethod
    def _model_refit(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        lambda_: float = 1.0,
        lower_limits: Optional[List[float]] = None,
        upper_limits: Optional[List[float]] = None,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
    ) -> ModelRefitOutput:
        # Implementation here
        pass

    @staticmethod
    def _lambda_seq(
        x: np.ndarray,
        y: np.ndarray,
        seq_len: int = 100,
        lambda_min_ratio: float = 0.0001,
    ) -> np.ndarray:
        # Implementation
        pass

    @staticmethod
    def _hyper_collector(
        adstock: str,
        all_media: List[str],
        paid_media_spends: List[str],
        organic_vars: List[str],
        prophet_vars: List[str],
        context_vars: List[str],
        dt_mod: pd.DataFrame,
        hyper_in: Dict[str, Any],
        ts_validation: bool,
        add_penalty_factor: bool,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        cores: int = 1,
    ) -> HyperCollectorOutput:
        # Implementation here
        pass
