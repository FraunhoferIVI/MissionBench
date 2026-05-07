"""Simulator interface package.

Keep imports lightweight so modules that only need helpers (e.g. image fetch)
don't require AirSim runtime deps at import time.
"""

try:
	from .airsim_client_wrapper import AirSimClientWrapper
except ModuleNotFoundError:
	AirSimClientWrapper = None