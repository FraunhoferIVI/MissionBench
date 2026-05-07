import requests
import sys
import time

from rpds import List

_STEP_TIMEOUT_S = 120   # seconds to wait for a single /step response
_STEP_MAX_RETRIES = 1   # retry once on timeout/connection error
_RESET_TIMEOUT_S = 120  # seconds to wait for a single /reset response
_RESET_MAX_RETRIES = 1  # retry once on timeout/connection error


def fetch_image_from_simulator(x, y, z, yaw) -> str:
    """Fetch an image from the AirSim simulator at the specified drone pose.

    Raises RuntimeError after _STEP_MAX_RETRIES failed attempts so the caller
    can decide how to handle the failure instead of hanging indefinitely.
    """
    payload = {
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
    }
    last_exc = None
    for attempt in range(1, _STEP_MAX_RETRIES + 2):  # +2 so range gives 1..retries+1
        try:
            step_response = requests.post(
                "http://localhost:5001/step",
                json=payload,
                timeout=_STEP_TIMEOUT_S,
            )
            step_response.raise_for_status()
            result = step_response.json()
            return result
        except requests.exceptions.Timeout as e:
            print(f"   ⚠ /step timed out after {_STEP_TIMEOUT_S}s (attempt {attempt}/{_STEP_MAX_RETRIES + 1})")
            if attempt <= _STEP_MAX_RETRIES:
                print("   Retrying in 3s...")
                time.sleep(3)
            else:
                print("   ✘ Simulator /step timed out after all retries.")
                sys.exit(1)
        except requests.exceptions.ConnectionError as e:
            print(f"   ⚠ Cannot connect to simulator server on attempt {attempt}: {e}")
            print("   Make sure you're running: python airsim_api_server.py")
            if attempt <= _STEP_MAX_RETRIES:
                time.sleep(3)
            else:
                print("   ✘ Simulator unreachable after all retries.")
                sys.exit(1)
        except Exception as e:
            print(f"   ⚠ Error communicating with simulator: {e}")
            sys.exit(1)

    print("   ✘ Simulator /step failed after all retries. Aborting.")
    sys.exit(1)


def reset_simulator(center_position: dict | None = None) -> dict:
    """Reset simulator and optionally set center/start pose for this mission.

    Args:
        center_position: Optional dict with keys x,y,z and optional yaw, roll,
            pitch. Example: {"x": 10, "y": -5, "z": -3, "yaw": 90}

    Returns:
        Parsed JSON response from /reset.
    """
    payload = {}
    if center_position is not None:
        payload["center_position"] = center_position

    for attempt in range(1, _RESET_MAX_RETRIES + 2):
        try:
            reset_response = requests.post(
                "http://localhost:5001/reset",
                json=payload,
                timeout=_RESET_TIMEOUT_S,
            )
            reset_response.raise_for_status()
            result = reset_response.json()
            reset_pose_error = result.get("reset_pose_error")
            if reset_pose_error is not None:
                print(f"   reset_pose_error={reset_pose_error}")
            return result
        except requests.exceptions.Timeout:
            print(
                f"   ⚠ /reset timed out after {_RESET_TIMEOUT_S}s "
                f"(attempt {attempt}/{_RESET_MAX_RETRIES + 1})"
            )
            if attempt <= _RESET_MAX_RETRIES:
                print("   Retrying /reset in 3s...")
                time.sleep(3)
            else:
                raise RuntimeError("Simulator /reset timed out after all retries")
        except requests.exceptions.ConnectionError as e:
            print(f"   ⚠ Cannot connect to simulator server on attempt {attempt}: {e}")
            print("   Make sure you're running: python airsim_api_server.py")
            if attempt <= _RESET_MAX_RETRIES:
                time.sleep(3)
            else:
                raise RuntimeError("Simulator /reset unreachable after all retries")
        except requests.exceptions.HTTPError as e:
            body = ""
            if e.response is not None:
                try:
                    body = e.response.text
                except Exception:
                    body = "<unavailable>"
            raise RuntimeError(
                "Error communicating with simulator /reset: "
                f"{e}. Response body: {body}"
            )
        except Exception as e:
            raise RuntimeError(f"Error communicating with simulator /reset: {e}")

    raise RuntimeError("Simulator /reset failed after all retries")

def create_airsim_langchain_tools() -> List:
    """
    Create list of LangChain tools for interacting with AirSim simulator.

    Returns:
        List of LangChain Tool objects that can be bound to an LLM.

    Usage:
        from langchain_openai import ChatOpenAI
        tools = create_airsim_langchain_tools()
        llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
    """
    return [fetch_image_from_simulator]
