# pyre-strict
from dataclasses import dataclass

@dataclass
class TrialsConfig:
    def __init__(
        self,
        num_trials: int,
        num_iterations_per_trial: int,
        timeseries_validation: bool, # TODO Does this belong in this class?
        add_penalty_factor: bool     # TODO Does this belong in this class?
    ) -> None:
        self.num_trials: int = num_trials
        self.num_iterations_per_trial: int = num_iterations_per_trial
        self.timeseries_validation: bool = timeseries_validation
        self.add_penalty_factor: bool = add_penalty_factor

    def __str__(self) -> str:
        return (
            f"TrialsConfig("
            f"num_trials={self.num_trials}, "
            f"num_iterations_per_trial={self.num_iterations_per_trial}, "
            f"timeseries_validation={self.timeseries_validation}, "
            f"add_penalty_factor={self.add_penalty_factor}"
            f")"
        )

    def update(
        self,
        num_trials: int,
        num_iterations_per_trial: int,
        timeseries_validation: bool,
        add_penalty_factor: bool
    ) -> None:
        """
        Update the TrialsConfig parameters.

        :param num_trials: The new number of trials.
        :param num_iterations_per_trial: The new number of iterations per trial.
        :param timeseries_validation: Whether to use timeseries validation.
        :param add_penalty_factor: Whether to add a penalty factor.
        """
        self.num_trials = num_trials
        self.num_iterations_per_trial = num_iterations_per_trial
        self.timeseries_validation = timeseries_validation
        self.add_penalty_factor = add_penalty_factor
