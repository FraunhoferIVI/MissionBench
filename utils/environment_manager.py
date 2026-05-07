"""
Environment Manager: Handles dynamic AirSim environment switching for missions.

Responsibilities:
- Track current running environment
- Detect environment requirements from missions
- Stop current server if environment changes
- Start new server with correct environment
- Handle graceful shutdown
- Provide safety checks and logging
"""

import subprocess
import time
import signal
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import sys
import queue as _queue


class SimulatorServerStartupError(RuntimeError):
    """Raised when the simulator API server cannot be started/restarted."""


class EnvironmentManager:
    """
    Manages AirSim environment lifecycle and switching.
    
    Ensures that:
    1. Only one simulator server runs at a time
    2. Environment switches are clean (proper shutdown before new start)
    3. Missions always run with correct environment
    4. Server is properly terminated on exit
    """
    
    def __init__(self, project_root: Path | str):
        """
        Initialize environment manager.
        
        Args:
            project_root: Path to MissionBench project root
        """
        self.project_root = Path(project_root)
        self.current_process: Optional[subprocess.Popen] = None
        self.current_environment: Optional[str] = None
        self.log_file = None
        self.log_path: Optional[Path] = None
        self.server_lock = threading.Lock()
        self._atexit_registered = False
        self.logs_dir = Path(
            os.getenv("MISSIONBENCH_LOG_DIR", str(self.project_root / "project_logs"))
        )
        self.server_port = int(os.getenv("AIRSIM_API_PORT", "5001"))
        
        # Valid environments
        self.valid_environments = {"NHEnv", "CityEnv", "ForestEnv"}

    def normalize_environment_name(self, env: Optional[str]) -> Optional[str]:
        """Normalize heterogeneous environment labels to canonical names."""
        if env is None:
            return None

        raw = str(env).strip()
        if not raw:
            return None

        # Fast path for exact values.
        if raw in self.valid_environments:
            return raw

        compact = raw.replace("_", "").replace("-", "").replace(" ", "")
        lower = compact.lower()

        alias_map = {
            "nhenv": "NHEnv",
            "airsimnhenv": "NHEnv",
            "airsimnh": "NHEnv",
            "cityenv": "CityEnv",
            "airsimcityenv": "CityEnv",
            "airsimcity": "CityEnv",
            "forestenv": "ForestEnv",
            "airsimforestenv": "ForestEnv",
            "airsimforest": "ForestEnv",
        }

        return alias_map.get(lower)
        
    def get_mission_environment(self, mission_config: Dict[str, Any]) -> Optional[str]:
        """
        Extract environment from mission config.
        
        Args:
            mission_config: Mission config dict
            
        Returns:
            Environment name (e.g., "NHEnv") or None if not specified
        """
        metadata = mission_config.get("metadata", {})
        env = metadata.get("environment")
        normalized_env = self.normalize_environment_name(env)

        if env is not None and normalized_env is None:
            raise ValueError(
                f"Invalid environment in mission: {env}. "
                f"Valid: {self.valid_environments}"
            )

        return normalized_env
    
    def is_environment_switch_needed(self, target_environment: Optional[str]) -> bool:
        """
        Check if environment switch is needed.
        
        Args:
            target_environment: Target environment (None if not required by mission)
            
        Returns:
            True if switch needed, False otherwise
        """
        # If mission doesn't specify environment, no switch needed
        if target_environment is None:
            return False
        
        # If no server running, switch needed
        if self.current_process is None or self.current_process.poll() is not None:
            return True
        
        # If environment differs, switch needed
        return self.current_environment != target_environment
    
    def ensure_environment(self, target_environment: Optional[str]) -> None:
        """
        Ensure the target environment is running.
        
        If environment differs from current, gracefully stops current and starts new.
        If mission doesn't specify environment, does nothing.
        
        Args:
            target_environment: Target environment or None
            
        Raises:
            ValueError: If environment is invalid
            RuntimeError: If server fails to start
        """
        target_environment = self.normalize_environment_name(target_environment)

        if target_environment not in self.valid_environments and target_environment is not None:
            raise ValueError(
                f"Invalid environment: {target_environment}. "
                f"Valid: {self.valid_environments}"
            )
        
        # Mission doesn't require specific environment. Ensure server is up anyway.
        if target_environment is None:
            if self.current_process is None or self.current_process.poll() is not None:
                fallback_env = self.current_environment or self._infer_runtime_environment() or "NHEnv"
                with self.server_lock:
                    print(f"\n   🔁 API server down; restarting with environment: {fallback_env}")
                    self._start_server(fallback_env)
                    self.current_environment = fallback_env
            return
        
        # Check if switch needed
        if not self.is_environment_switch_needed(target_environment):
            print(f"   ✓ Environment {self.current_environment} already running")
            return
        
        # Switch environment
        with self.server_lock:
            print(f"\n   🔄 Environment switch: {self.current_environment} → {target_environment}")
            
            # Stop current server
            if self.current_process and self.current_process.poll() is None:
                self._stop_server_gracefully()
                cooldown_sec = float(os.getenv("MISSIONBENCH_ENV_SWITCH_COOLDOWN_SEC", "3"))
                if cooldown_sec > 0:
                    print(f"   ⏳ Cooldown {cooldown_sec:.1f}s before starting {target_environment}")
                    time.sleep(cooldown_sec)
            
            # Start new server
            self._start_server(target_environment)
            
            self.current_environment = target_environment
            print(f"   ✓ Environment ready: {target_environment}")
    
    def _start_server(self, environment: str) -> None:
        """
        Start AirSim API server for specific environment.
        
        Args:
            environment: Environment name (NHEnv, CityEnv, ForestEnv)
            
        Raises:
            FileNotFoundError: If Python interpreter not found
            RuntimeError: If server fails to start
        """
        # Prepare environment variables
        env_vars = os.environ.copy()
        env_vars["AIRSIM_ENVIRONMENT"] = environment
        env_vars["MISSIONBENCH_LOG_DIR"] = str(self.logs_dir)
        
        # Use system Python
        python_exe = Path(sys.executable)
        if not python_exe.exists():
            raise FileNotFoundError(
                f"Python interpreter not found at {python_exe}"
            )
        
        # Prepare log file
        logs_dir = self.logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.log_path = logs_dir / f"{ts}-airsim_api_server-{environment}.log"
        
        max_retries = int(os.getenv("MISSIONBENCH_SERVER_START_RETRIES", "1"))
        retry_backoff_sec = float(os.getenv("MISSIONBENCH_SERVER_RETRY_BACKOFF_SEC", "10"))

        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(
                    f"   ♻ Retry {attempt}/{max_retries} starting {environment} "
                    f"after SIGKILL; waiting {retry_backoff_sec:.1f}s"
                )
                time.sleep(retry_backoff_sec)

            print(f"   Starting API server for {environment}...")
            print(f"   Log: {self.log_path}")

            # Close previous log handle before replacing it.
            if self.log_file:
                try:
                    self.log_file.close()
                except Exception:
                    pass

            self.log_file = open(self.log_path, "w", buffering=1)

            try:
                self.current_process = subprocess.Popen(
                    [str(python_exe), "airsim_api_server.py"],
                    cwd=self.project_root,
                    env=env_vars,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                )
            except Exception as e:
                raise SimulatorServerStartupError(
                    f"Failed to start API server: {e}"
                ) from e

            # Start tee thread to mirror output to log
            threading.Thread(
                target=self._tee_output,
                args=(self.current_process,),
                daemon=True,
                name=f"airsim-tee-{environment}"
            ).start()

            try:
                # Wait for server to start
                self._wait_for_server_ready(environment)
                break
            except SimulatorServerStartupError as e:
                rc = self.current_process.returncode if self.current_process else None
                if rc == -9 and attempt < max_retries:
                    continue
                raise
        
        # Register cleanup
        if not self._atexit_registered:
            import atexit
            atexit.register(self.cleanup)
            self._atexit_registered = True
    
    def _tee_output(self, process: subprocess.Popen) -> None:
        """Mirror process output to log file."""
        try:
            for line in process.stdout:
                if self.log_file:
                    self.log_file.write(line)
                    self.log_file.flush()
        except Exception:
            pass
    
    def _wait_for_server_ready(self, environment: str, timeout: int = 120) -> None:
        """
        Wait for server to be ready to accept connections.
        
        Args:
            environment: Environment name for logging
            timeout: Maximum seconds to wait
            
        Raises:
            TimeoutError: If server doesn't start within timeout
        """
        print(f"   Waiting for {environment} server to start", end="", flush=True)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if process is still running
            if self.current_process.poll() is not None:
                log_tail = self._read_recent_log_lines(max_lines=25)
                raise SimulatorServerStartupError(
                    "Server process exited prematurely with code "
                    f"{self.current_process.returncode}.\n"
                    f"Recent server log tail:\n{log_tail}"
                )
            
            # Try to connect
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("127.0.0.1", self.server_port))
                sock.close()
                if result == 0:
                    print(" ✓")
                    time.sleep(1)  # Give server extra time to initialize
                    return
            except Exception:
                pass
            
            print(".", end="", flush=True)
            time.sleep(0.5)
        
        raise SimulatorServerStartupError(
            f"Server for {environment} failed to start within {timeout}s"
        )

    def _read_recent_log_lines(self, max_lines: int = 25) -> str:
        """Return recent server log lines for failure diagnostics."""
        if not self.log_path or not self.log_path.exists():
            return "<log file unavailable>"

        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail = lines[-max_lines:]
            content = "".join(tail).strip()
            return content if content else "<log file empty>"
        except Exception as e:
            return f"<failed to read log: {e}>"

    def _infer_runtime_environment(self) -> Optional[str]:
        """Infer runtime environment from configs/config.yaml when available."""
        cfg_path = self.project_root / "configs" / "config.yaml"
        if not cfg_path.exists():
            return None
        try:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            env_name = self.normalize_environment_name(cfg.get("environment"))
            if env_name in self.valid_environments:
                return env_name
        except Exception:
            return None
        return None
    
    def _stop_server_gracefully(self, timeout: int = 10) -> None:
        """
        Stop server gracefully with timeout, forcefully kill if needed.
        
        Args:
            timeout: Seconds to wait before force kill
        """
        if self.current_process is None or self.current_process.poll() is not None:
            return
        
        print(f"   Stopping {self.current_environment} server...", end="", flush=True)
        
        try:
            # Graceful termination
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=timeout)
                print(" ✓")
                return
            except subprocess.TimeoutExpired:
                # Force kill
                print(" (force kill)", end="", flush=True)
                self.current_process.kill()
                self.current_process.wait(timeout=5)
                print(" ✓")
        except Exception as e:
            print(f" ⚠ {e}")
        finally:
            self.current_process = None
    
    def cleanup(self) -> None:
        """Clean shutdown of environment manager."""
        print("\n[Cleanup] Shutting down environment manager...")
        
        with self.server_lock:
            # Stop server
            if self.current_process and self.current_process.poll() is None:
                self._stop_server_gracefully()
            
            # Close log file
            if self.log_file:
                self.log_file.close()
            
            self.current_process = None
            self.current_environment = None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current environment status.
        
        Returns:
            Status dict with environment, process info, etc.
        """
        is_running = (
            self.current_process is not None and 
            self.current_process.poll() is None
        )
        
        return {
            "environment": self.current_environment,
            "is_running": is_running,
            "process_id": self.current_process.pid if is_running else None,
            "log_path": str(self.log_path) if self.log_path else None,
        }


# Global instance (one per process)
_global_env_manager: Optional[EnvironmentManager] = None


def get_environment_manager(project_root: Path | str) -> EnvironmentManager:
    """
    Get or create global environment manager instance.
    
    Args:
        project_root: Path to MissionBench project root
        
    Returns:
        EnvironmentManager instance
    """
    global _global_env_manager
    
    if _global_env_manager is None:
        _global_env_manager = EnvironmentManager(project_root)
    
    return _global_env_manager
