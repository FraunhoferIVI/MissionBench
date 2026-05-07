from datetime import datetime
import pathlib
import subprocess
from time import sleep
from typing import IO, Optional, Union
import os
import time
import logging
import sys
import psutil
import shutil
import json
import yaml
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

try:
    from utils.logging_config import setup_project_logger
    project_logger = setup_project_logger()
    from utils.utils import get_active_settings_file
except Exception as e:
    raise ImportError(f"Could not import setup_project_logger from utils.logging_config, {e}")


class UnrealProcessManager:
    """Manages the lifecycle of an Unreal Engine process"""
    def __init__(
        self,
        unreal_binary_path: Union[str, pathlib.Path],
        startup_wait_sec: int = 10,
        settings_file_path: Optional[Union[str, pathlib.Path]] = None
    ):
        self.unreal_binary_path = unreal_binary_path
        self.process: Optional[subprocess.Popen[str]] = None
        self.startup_wait_sec = startup_wait_sec
        self.settings_file_path = settings_file_path
        self.logger = project_logger
        self.sim_mode = None
        self.vehicles = None
        self._verify_inputs()
        if settings_file_path:
            self.sim_mode, self.vehicles = self._get_sim_mode_and_vehicles_from_settings(settings_file_path)

    def _verify_inputs(self):
        # Check if binary exists
        if not os.path.exists(self.unreal_binary_path):
            self.logger.error(f"Binary not found at: {self.unreal_binary_path}")
            raise FileNotFoundError(
                f"Binary not found at: {self.unreal_binary_path}"
            )

        # Check if settings file exists (if provided)
        if self.settings_file_path and not os.path.exists(self.settings_file_path):
            self.logger.error(f"Settings file not found at: {self.settings_file_path}")
            raise FileNotFoundError(
                f"Settings file not found at: {self.settings_file_path}"
            )

    def _get_sim_mode_and_vehicles_from_settings(self, settings_file_path):
        with open(settings_file_path, "r") as f:
            settings = json.load(f)
        sim_mode = settings.get("SimMode", "Multirotor")
        vehicles = settings.get('Vehicles', {})
        return sim_mode, vehicles

    def _setup_airsim_settings(self):
        """Copy AirSim settings file to the correct location"""
        if not self.settings_file_path:
            self.logger.info("No settings file specified, using AirSim defaults")
            return

        # AirSim looks for settings.json in ~/Documents/AirSim/
        home_dir = os.path.expanduser("~")
        airsim_dir = os.path.join(home_dir, "Documents", "AirSim")
        settings_dest = os.path.join(airsim_dir, "settings.json")

        # Create AirSim directory if it doesn't exist
        os.makedirs(airsim_dir, exist_ok=True)

        # Copy settings file
        try:
            shutil.copy2(self.settings_file_path, settings_dest)
            self.logger.info(f"Copied settings file from {self.settings_file_path} to {settings_dest}")

            # Log some settings info
            with open(self.settings_file_path, 'r') as f:
                settings = json.load(f)
                sim_mode = settings.get('SimMode', 'Unknown')
                vehicles = list(settings.get('Vehicles', {}).keys())
                self.logger.info(f"AirSim configured for SimMode: {sim_mode}, Vehicles: {vehicles}")

        except Exception as e:
            self.logger.error(f"Failed to copy settings file: {e}")
            raise

    def _free_rpc_port(self, port: int = 41451):
        """Kill any process using the AirSim RPC port."""
        import psutil
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                try:
                    proc = psutil.Process(conn.pid)
                    proc.kill()
                    self.logger.info(f"Killed process {conn.pid} using port {port}")
                except Exception as e:
                    self.logger.warning(f"Could not kill process {conn.pid} on port {port}: {e}")

    def start_process(self, rendering: bool = False):
        # Free the AirSim RPC port before starting
        self._free_rpc_port(port=41451)
        # Setup AirSim settings before starting
        self._setup_airsim_settings()

        args = [str(self.unreal_binary_path), "-nosound"]
        if not rendering:
          args.append("-RenderOffscreen")

        self.process = subprocess.Popen(
            args,
            stdout=self.logger.handlers[0].stream,
            stderr=subprocess.STDOUT,
        )

        # Give the process time to initialize and start the UnrealCV server.
        self.logger.info(f"Waiting {self.startup_wait_sec}s for Unreal server...")
        sleep(self.startup_wait_sec)
        if not self.is_alive:
            raise RuntimeError("Unreal process failed to start.")
        self.logger.info("Unreal process started successfully.")
        print("Environment created successfully.")

    @property
    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop_process(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            finally:
                self.process = None
            self.logger.info("Unreal process terminated.")
        else:
            self.process = None
            self.logger.info("Unreal process was not running.")

    def reset(self, rendering: bool = False) -> None:
        # Kill all processes with the binary name
        binary_name = os.path.basename(self.unreal_binary_path)
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] == binary_name:
                    proc.kill()
                    self.logger.info(f"Killed process {proc.info['pid']} ({binary_name})")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        self.logger.info("Unreal process is resetting.")
        self.stop_process()
        self.start_process(rendering=rendering)

if __name__ == "__main__":
    # Hardcoded paths
    config_file = "./configs/config.yaml"
    env_yaml_path = "./configs/environments.yaml"
    settings_file_path = get_active_settings_file(config_file)
    # Load config to get environment name
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    env_name = config.get("environment", "AirSimNH")
    rendering = config.get("rendering", False)
    # Load environments from YAML
    with open(env_yaml_path, "r") as f:
        envs = yaml.safe_load(f)

    # Get binary path for the environment
    unreal_binary = envs.get(env_name)
    if not unreal_binary:
        print(f"✗ Environment '{env_name}' not found in {env_yaml_path}")
        exit(1)

    print(f"Standalone test of UnrealProcessManager at {datetime.now().isoformat()}")
    print(f"Environment: {env_name}")
    print(f"Binary: {unreal_binary}")
    

    # Create and test the process manager
    manager = UnrealProcessManager(
        unreal_binary_path=unreal_binary,
        startup_wait_sec=15,
        settings_file_path=settings_file_path
    )
    manager.start_process(rendering=rendering)

    # Check if process started successfully
    if manager.process and manager.process.poll() is None:
        # Check if still running
        if manager.process.poll() is None:
            print("✓ Unreal is running...")
        else:
            exit_code = manager.process.poll()
            print(f"✗ Unreal exited with code: {exit_code}")
    else:
        print("✗ Failed to start Unreal process")

    input("Press Enter to stop Unreal...")

    manager.stop_process()

    print("Test completed.")