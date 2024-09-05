#pyre-strict

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ResultHypParam:
    """
    Represents the result of hyperparameter tuning for a model.

    Attributes:
        sol_id (str): The solution ID.
        nrmse (float): The normalized root mean squared error.
        decomp_rssd (float): The decomposed residual sum of squares difference.
        mape (float): The mean absolute percentage error.
        rsq_train (float): The R-squared value for the training set.
        rsq_val (Optional[float]): The R-squared value for the validation set (optional).
        rsq_test (Optional[float]): The R-squared value for the test set (optional).
        nrmse_train (float): The normalized root mean squared error for the training set.
        nrmse_val (Optional[float]): The normalized root mean squared error for the validation set (optional).
        nrmse_test (Optional[float]): The normalized root mean squared error for the test set (optional).
        lambda_ (float): The lambda value.
        lambda_max (float): The maximum lambda value.
        lambda_min_ratio (float): The ratio of minimum lambda to maximum lambda.
        iterations (int): The number of iterations.
        trial (int): The trial number.
        iter_ng (int): The number of iterations for NG.
        iter_par (int): The number of iterations for Par.
        elapsed_accum (float): The accumulated elapsed time.
        elapsed (float): The elapsed time.
        pos (float): The position.
        error_score (float): The error score.
    """
    sol_id: str
    nrmse: float
    decomp_rssd: float
    mape: float
    rsq_train: float
    rsq_val: Optional[float]
    rsq_test: Optional[float]
    nrmse_train: float
    nrmse_val: Optional[float]
    nrmse_test: Optional[float]
    lambda_: float
    lambda_max: float
    lambda_min_ratio: float
    iterations: int
    trial: int
    iter_ng: int
    iter_par: int
    elapsed_accum: float
    elapsed: float
    pos: float
    error_score: float
    # Add other hyperparameters as needed

@dataclass
class XDecompAgg:
    """
    Represents an aggregated model output for X decomposition.

    Attributes:
        sol_id (str): The solution ID.
        rn (str): The RN value.
        coef (float): The coefficient value.
        decomp (float): The decomposition value.
        total_spend (float): The total spend value.
        mean_spend (float): The mean spend value.
        roi_mean (float): The ROI mean value.
        roi_total (float): The ROI total value.
        cpa_total (float): The CPA total value.
    """
    sol_id: str
    rn: str
    coef: float
    decomp: float
    total_spend: float
    mean_spend: float
    roi_mean: float
    roi_total: float
    cpa_total: float

@dataclass
class DecompSpendDist:
    """
    Represents the decomposition of spending distribution.

    Attributes:
        rn (str): The rn attribute.
        coefs (float): The coefs attribute.
        x_decomp_agg (float): The x_decomp_agg attribute.
        x_decomp_perc (float): The x_decomp_perc attribute.
        x_decomp_mean_non0 (float): The x_decomp_mean_non0 attribute.
        x_decomp_mean_non0_perc (float): The x_decomp_mean_non0_perc attribute.
        pos (float): The pos attribute.
        total_spend (float): The total_spend attribute.
        mean_spend (float): The mean_spend attribute.
        spend_share (float): The spend_share attribute.
        effect_share (float): The effect_share attribute.
        roi_mean (float): The roi_mean attribute.
        roi_total (float): The roi_total attribute.
        cpa_mean (float): The cpa_mean attribute.
        cpa_total (float): The cpa_total attribute.
        sol_id (str): The sol_id attribute.
        trial (int): The trial attribute.
        iter_ng (int): The iter_ng attribute.
        iter_par (int): The iter_par attribute.
    """
    rn: str
    coefs: float
    x_decomp_agg: float
    x_decomp_perc: float
    x_decomp_mean_non0: float
    x_decomp_mean_non0_perc: float
    pos: float
    total_spend: float
    mean_spend: float
    spend_share: float
    effect_share: float
    roi_mean: float
    roi_total: float
    cpa_mean: float
    cpa_total: float
    sol_id: str
    trial: int
    iter_ng: int
    iter_par: int

@dataclass
class ResultCollect:
    """
    Represents the result collection of a model output.

    Attributes:
        result_hyp_param (List[ResultHypParam]): A list of ResultHypParam objects.
        x_decomp_agg (List[XDecompAgg]): A list of XDecompAgg objects.
        decomp_spend_dist (List[DecompSpendDist]): A list of DecompSpendDist objects.
        elapsed_min (float): The elapsed time in minutes.
    """
    result_hyp_param: List[ResultHypParam]
    x_decomp_agg: List[XDecompAgg]
    decomp_spend_dist: List[DecompSpendDist]
    elapsed_min: float

@dataclass
class Trial:
    """
    Represents a trial in the modeling entities.

    Attributes:
        result_collect (ResultCollect): The result collect of the trial.
        hyper_bound_ng (Dict[str, List[float]]): The non-grid hyperparameter bounds of the trial.
        hyper_bound_fixed (Dict[str, List[float]]): The fixed hyperparameter bounds of the trial.
        trial (int): The trial number.
        name (str): The name of the trial.
    """
    result_collect: ResultCollect
    hyper_bound_ng: Dict[str, List[float]]
    hyper_bound_fixed: Dict[str, List[float]]
    trial: int
    name: str


@dataclass
class Metadata:
    """
    Represents the metadata for model output.

    Attributes:
        hyper_fixed (bool): Indicates if hyperparameters are fixed.
        bootstrap (Optional[Any]): Optional bootstrap value.
        refresh (bool): Indicates if the model output needs to be refreshed.
        train_timestamp (float): The timestamp of the training.
        cores (int): The number of cores used for training.
        iterations (int): The number of iterations used for training.
        trials (int): The number of trials used for training.
        intercept (bool): Indicates if an intercept is used.
        intercept_sign (str): The sign of the intercept.
        nevergrad_algo (str): The algorithm used for hyperparameter optimization.
        ts_validation (bool): Indicates if time series validation is used.
        add_penalty_factor (bool): Indicates if a penalty factor is added.
        hyper_updated (Dict[str, List[float]]): A dictionary containing updated hyperparameters.
    """
    hyper_fixed: bool
    bootstrap: Optional[Any]
    refresh: bool
    train_timestamp: float
    cores: int
    iterations: int
    trials: int
    intercept: bool
    intercept_sign: str
    nevergrad_algo: str
    ts_validation: bool
    add_penalty_factor: bool
    hyper_updated: Dict[str, List[float]]

@dataclass
class ModelOutputs:
    """
    Represents the outputs of a model.

    Attributes:
        trials (List[Trial]): A list of Trial objects representing the model trials.
        metadata (Metadata): The metadata associated with the model outputs.
        seed (int): The seed used for generating the model outputs.
    """
    trials: List[Trial]
    metadata: Metadata
    seed: int
