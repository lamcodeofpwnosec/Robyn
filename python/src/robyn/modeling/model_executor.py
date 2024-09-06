#pyre-strict

from typing import Optional, Dict, Any

from robyn.modeling.base_model_executor import BaseModelExecutor

class ModelExecutor(BaseModelExecutor):
    
    def model_run(
        self,
        dt_hyper_fixed: Optional[Dict[str, Any]] = None,
        json_file: Optional[str] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        refresh: bool = False,
        seed: int = 123,
        quiet: bool = False,
        cores: Optional[int] = None,
        trials: int = 5,
        iterations: int = 2000,
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[Dict[str, float]] = None,
        nevergrad_algo: str = "TwoPointsDE",
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        lambda_control: Optional[float] = None,
        outputs: bool = False,
    ) -> None:
        """
        Run the Robyn model with the specified parameters.

        Args:
            InputCollect: Input data collection.
            dt_hyper_fixed: Fixed hyperparameters.
            json_file: JSON file path.
            ts_validation: Enable time-series validation.
            add_penalty_factor: Add penalty factor.
            refresh: Refresh the model.
            seed: Random seed.
            quiet: Suppress output.
            cores: Number of cores to use.
            trials: Number of trials.
            iterations: Number of iterations.
            rssd_zero_penalty: Enable RSSD zero penalty.
            objective_weights: Objective weights.
            nevergrad_algo: Nevergrad algorithm to use.
            intercept: Include intercept term.
            intercept_sign: Sign of the intercept term.
            lambda_control: Lambda control value.
            outputs: Output results.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    #Call build_models from model_builder.py
    #Evaluator, clustering, and plotting
