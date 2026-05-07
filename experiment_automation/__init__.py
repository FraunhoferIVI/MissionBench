"""
Experiment automation package.
"""

from experiment_automation.config import ExperimentConfig, GridSearchDefinition
from experiment_automation.runner import ExperimentRunner
from experiment_automation.executor import BatchExecutor
from experiment_automation.aggregator import ResultsAggregator

__all__ = [
    "ExperimentConfig",
    "GridSearchDefinition",
    "ExperimentRunner",
    "BatchExecutor",
    "ResultsAggregator",
]
