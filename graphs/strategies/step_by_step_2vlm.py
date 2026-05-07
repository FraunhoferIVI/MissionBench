"""
Single-Stage High-Level Planning Graph

This module provides a function to build a LangGraph workflow for
single-stage high-level action planning.

Workflow:
    START → generate_actions → parse_actions → evaluate → END

The graph takes an initial state with image, instruction, and model_name,
then flows through each node to produce parsed actions and evaluation results.
"""

from langgraph.graph import END, StateGraph
from functools import partial
from graphs.step_by_step_nodes import (
    bb_and_depth_predictor,
    node_generate_step_by_step_action,
    node_parse_step_by_step_action,
    node_generate_waypoints,
    should_continue_navigation,
    node_parse_waypoints,
    node_evaluate_step_by_step
)

from graphs.state import StepByStepState


def build_step_by_step_graph(verbose: bool = False):
    """
    Build a step-by-step graph.

    The graph executes three nodes in sequence:
    1. generate_step_by_step_action

    Returns:
        Compiled StateGraph ready to execute with .invoke(initial_state)
    """
    # Initialize graph with StepByStepState so waypoint outputs are preserved
    workflow = StateGraph(StepByStepState)
    # Helper to inject verbosity using functools.partial
    def _run_with_verbose(fn, verbose_flag, state):
        state["verbose"] = verbose_flag
        return fn(state)

    # Add nodes bound with verbose flag
    workflow.add_node("bb_and_depth_prediction", partial(_run_with_verbose, bb_and_depth_predictor, verbose))
    workflow.add_node("generate_step_by_step_action", partial(_run_with_verbose, node_generate_step_by_step_action, verbose))
    workflow.add_node("parse_step_by_step_action", partial(_run_with_verbose, node_parse_step_by_step_action, verbose))
    workflow.add_node("generate_waypoints", partial(_run_with_verbose, node_generate_waypoints, verbose))
    workflow.add_node("parse_waypoints", partial(_run_with_verbose, node_parse_waypoints, verbose))
    workflow.add_node("should_continue_navigation", partial(_run_with_verbose, should_continue_navigation, verbose))
    workflow.add_node("evaluate", partial(_run_with_verbose, node_evaluate_step_by_step, verbose))
    # Define edges (sequential flow)
    workflow.set_entry_point("bb_and_depth_prediction")
    workflow.add_edge("bb_and_depth_prediction", "generate_step_by_step_action")
    workflow.add_edge("generate_step_by_step_action", "parse_step_by_step_action")
    workflow.add_edge("parse_step_by_step_action", "generate_waypoints")
    workflow.add_edge("generate_waypoints", "parse_waypoints")
    workflow.add_edge("parse_waypoints", "should_continue_navigation")
    workflow.add_conditional_edges(
        "should_continue_navigation",
        lambda state: state["nav_status"],
        {
            "continue": "generate_step_by_step_action",
            "done": "evaluate",
        },
    )
    workflow.add_edge("evaluate", END)

    # Compile and return
    return workflow.compile()