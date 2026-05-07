import time
from typing import Dict


def stream_graph_with_progress(
    graph,
    initial_state: Dict,
    recursion_limit: int = 1000,
    show_node_details: bool = True,
    show_state_updates: bool = False,
) -> Dict:
    """Execute graph with real-time progress streaming to terminal."""
    node_counter = {}
    final_state = None

    try:
        for output in graph.stream(initial_state, config={"recursion_limit": recursion_limit}):
            for node_name, node_output in output.items():
                node_counter[node_name] = node_counter.get(node_name, 0) + 1
                count = node_counter[node_name]

                display_name = node_name.replace("_", " ").title()

                if show_node_details:
                    print(f"\n   📍 [{count}] {display_name}")

                    if isinstance(node_output, dict):
                        for key in list(node_output.keys())[:3]:
                            value = node_output[key]
                            if isinstance(value, str) and len(value) > 100:
                                print(f"      • {key}: {value[:100]}...")
                            elif isinstance(value, list) and len(value) > 3:
                                print(f"      • {key}: {value[:2]} ... ({len(value)} items)")
                            else:
                                print(f"      • {key}: {value}")
                else:
                    print(f"   ✓ {display_name} (#{count})", end="", flush=True)
                    print("")

                final_state = {**initial_state, **node_output}

                if show_state_updates and isinstance(node_output, dict):
                    print(f"      State updated with {len(node_output)} keys")

        print(f"\n   ✓ Workflow complete after {sum(node_counter.values())} node executions")
        print(
            f"      Nodes executed: {', '.join(f'{k} ({v}x)' for k, v in node_counter.items())}"
        )

        return final_state or initial_state

    except Exception as e:
        print(f"\n   ❌ Error during graph execution: {e}")
        raise


def execute_graph_with_logging(
    graph,
    initial_state: Dict,
    recursion_limit: int = 1000,
    verbose: bool = True,
) -> Dict:
    """Execute graph using invoke() with enhanced logging."""
    if verbose:
        print("🎬 Invoking graph (standard execution)...")

    start_time = time.time()
    final_state = graph.invoke(initial_state, {"recursion_limit": recursion_limit})
    elapsed = time.time() - start_time

    if verbose:
        print(f"✓ Execution completed in {elapsed:.2f}s")
        print(f"  Total actions: {len(final_state.get('parsed_actions', []))}")
        print(f"  Total waypoints: {len(final_state.get('all_parsed_waypoints', []))}")

    return final_state


def execute_with_progress_spinner(
    graph,
    initial_state: Dict,
    recursion_limit: int = 1000,
) -> Dict:
    """Execute graph with a simple spinner showing activity."""
    import sys
    import threading

    spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    stop_spinner = False

    def spin():
        i = 0
        while not stop_spinner:
            sys.stdout.write(f"\r   🤖 Processing {spinner_chars[i % len(spinner_chars)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    spinner_thread = threading.Thread(target=spin, daemon=True)
    spinner_thread.start()

    try:
        final_state = graph.invoke(initial_state, {"recursion_limit": recursion_limit})
    finally:
        stop_spinner = True
        spinner_thread.join(timeout=1)
        print("\r   ✓ Processing complete                    ")

    return final_state
