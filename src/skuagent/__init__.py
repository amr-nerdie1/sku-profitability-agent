"""SKU profitability agent: weekly diagnosis of unprofitable SKUs, routed to owning teams as Jira tickets."""
from .simulate import simulate
from .pipeline import run_pipeline
from .taxonomy import ACTIONS, diagnose, accuracy
from .agent import Agent

__all__ = ["simulate", "run_pipeline", "ACTIONS", "diagnose", "accuracy", "Agent"]
