import logging
import pathlib
import os
from datetime import datetime

def setup_project_logger(log_dir=None):
    if log_dir is None:
        env_log_dir = os.getenv("MISSIONBENCH_LOG_DIR")
        if env_log_dir:
            log_dir = pathlib.Path(env_log_dir)
        else:
            log_dir = pathlib.Path(__file__).parent.parent / "project_logs"
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / datetime.now().strftime("%Y-%m-%d-%H-%M-%S-Unreal.log")
    logger = logging.getLogger("project_logger")
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger