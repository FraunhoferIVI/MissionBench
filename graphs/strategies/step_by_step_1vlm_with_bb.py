from functools import partial

from langgraph.graph import END, StateGraph

from graphs.step_by_step_nodes import (
    node_generate_step_by_step_action,
    node_parse_step_by_step_action_with_bb,
    node_generate_waypoints,
    node_parse_waypoints,
    should_continue_navigation,
    node_evaluate_step_by_step,
)
from graphs.state import StepByStepState


def build_step_by_step_1vlm_with_bb_graph(verbose: bool = False):
    """Build the 1-VLM step-by-step graph with BB-aware parsing.

    This keeps the exact same graph topology as ``step_by_step_1vlm`` and only
    changes the parse node so it also extracts ``<BoundingBox>`` information.

    Returns:
        Compiled StateGraph ready to execute with ``.invoke(initial_state)``.
    """
    workflow = StateGraph(StepByStepState)

    def _run(fn, verbose_flag, state):
        state["verbose"] = verbose_flag
        return fn(state)

    # ── nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node(
        "generate_step_by_step_action",
        partial(_run, node_generate_step_by_step_action, verbose),
    )
    workflow.add_node(
        "parse_step_by_step_action_with_bb",
        partial(_run, node_parse_step_by_step_action_with_bb, verbose),
    )
    workflow.add_node(
        "generate_waypoints",
        partial(_run, node_generate_waypoints, verbose),
    )
    workflow.add_node(
        "parse_waypoints",
        partial(_run, node_parse_waypoints, verbose),
    )
    workflow.add_node(
        "should_continue_navigation",
        partial(_run, should_continue_navigation, verbose),
    )
    workflow.add_node(
        "evaluate",
        partial(_run, node_evaluate_step_by_step, verbose),
    )

    # ── edges ─────────────────────────────────────────────────────────────────
    workflow.set_entry_point("generate_step_by_step_action")
    workflow.add_edge("generate_step_by_step_action", "parse_step_by_step_action_with_bb")
    workflow.add_conditional_edges(
        "parse_step_by_step_action_with_bb",
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

    return workflow.compile()
