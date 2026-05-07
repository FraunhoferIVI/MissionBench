"""
Single-Stage High-Level Planning Graph

This module provides a function to build a LangGraph workflow for
single-stage high-level action planning.

Workflow:
        START
            → generate_step_by_step_action
            → parse_step_by_step_action
            → (if continue) generate_waypoints → parse_waypoints → should_continue_navigation (fetch new image)
            → loop back to generate_step_by_step_action
            → (if done) evaluate → END

The graph takes an initial state with image, instruction, and model_name,
then flows through each node to produce parsed actions and evaluation results.
"""

from langgraph.graph import END, StateGraph
from functools import partial
from graphs.step_by_step_nodes import (
    node_generate_step_by_step_action,
    node_parse_step_by_step_action,
    node_generate_waypoints,
    should_continue_navigation,
    node_parse_waypoints,
    node_evaluate_step_by_step
)

from graphs.state import StepByStepState


def build_step_by_step_1vlmgraph(verbose: bool = False):
    """
    Build the 1-VLM step-by-step graph.

    Order:
    1) Get high-level action
    2) If action says continue, generate + parse waypoint
    3) Fetch new image in should_continue_navigation
    4) Run high-level action again (loop)
    5) If action says navigation done (or budget/collision stop), evaluate

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
    workflow.add_node("generate_step_by_step_action", partial(_run_with_verbose, node_generate_step_by_step_action, verbose))
    workflow.add_node("parse_step_by_step_action", partial(_run_with_verbose, node_parse_step_by_step_action, verbose))
    workflow.add_node("generate_waypoints", partial(_run_with_verbose, node_generate_waypoints, verbose))
    workflow.add_node("parse_waypoints", partial(_run_with_verbose, node_parse_waypoints, verbose))
    workflow.add_node("should_continue_navigation", partial(_run_with_verbose, should_continue_navigation, verbose))
    workflow.add_node("evaluate", partial(_run_with_verbose, node_evaluate_step_by_step, verbose))
    # Define edges (sequential flow)
    workflow.set_entry_point("generate_step_by_step_action")
    workflow.add_edge("generate_step_by_step_action", "parse_step_by_step_action")
    workflow.add_conditional_edges(
        "parse_step_by_step_action",
        lambda state: state["nav_status"],
        {
            "continue": "generate_waypoints",
            "done": "evaluate",
        },
    )
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