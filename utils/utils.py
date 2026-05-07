import math
import shutil
import yaml
import pathlib
import os
import re
import zipfile
import subprocess
import time
import atexit
from urllib.parse import parse_qs, urlparse
from typing import Dict, List, Optional
from pathlib import Path
import json
import requests
import sys


def load_mission_index(
    mission_index_path: str | Path,
    project_root: str | Path | None = None,
) -> Dict[str, str]:
    """Load a mission index YAML and recursively expand nested mission-list YAMLs.

    Returns a flat mapping of ``scenario_name -> mission_config_path``.
    Mission paths are returned as stored in the leaf YAML files.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    else:
        project_root = Path(project_root)

    start_path = Path(mission_index_path)
    if not start_path.is_absolute():
        start_path = project_root / start_path
    start_path = start_path.resolve()

    visited: set[Path] = set()

    def _resolve_target(base_file: Path, raw_target: str) -> Path:
        target = Path(raw_target)
        if target.is_absolute():
            return target

        candidates = [
            (base_file.parent / target),
            (project_root / target),
            (project_root / "dataset" / target),
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return (base_file.parent / target).resolve()

    def _load(path: Path) -> Dict[str, str]:
        if path in visited:
            raise ValueError(f"Circular mission index include detected: {path}")
        visited.add(path)

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Mission index must be a mapping: {path}")

        merged: Dict[str, str] = {}
        for scenario_name, raw_value in data.items():
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue

            value = raw_value.strip()
            value_path = Path(value)
            is_yaml = value_path.suffix.lower() in {".yaml", ".yml"}
            is_leaf_mission = value_path.name == "mission_config.yaml"

            if is_yaml and not is_leaf_mission:
                nested_path = _resolve_target(path, value)
                if not nested_path.exists():
                    raise FileNotFoundError(
                        f"Nested mission index not found: {value} (from {path})"
                    )
                nested = _load(nested_path)
                for nested_name, nested_mission in nested.items():
                    if nested_name in merged and merged[nested_name] != nested_mission:
                        raise ValueError(
                            "Duplicate scenario name with different mission paths: "
                            f"{nested_name}"
                        )
                    merged[nested_name] = nested_mission
                continue

            if scenario_name in merged and merged[scenario_name] != value:
                raise ValueError(
                    "Duplicate scenario name with different mission paths: "
                    f"{scenario_name}"
                )
            merged[scenario_name] = value

        visited.remove(path)
        return merged

    if not start_path.exists():
        raise FileNotFoundError(f"Mission index file not found: {start_path}")

    return _load(start_path)


def _extract_google_drive_file_id(drive_url: str) -> str:
    """Extract a Google Drive file id from common share URL formats."""
    parsed = urlparse(drive_url)

    # Format: https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path)
    if match:
        return match.group(1)

    # Format: https://drive.google.com/open?id=<FILE_ID>
    # Format: https://drive.google.com/uc?id=<FILE_ID>&export=download
    query_id = parse_qs(parsed.query).get("id", [])
    if query_id:
        return query_id[0]

    raise ValueError(f"Could not parse Google Drive file id from URL: {drive_url}")


def _download_google_drive_file(drive_url: str, output_path: Path) -> Path:
    """Download a file from Google Drive, including large files requiring confirm token."""
    file_id = _extract_google_drive_file_id(drive_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(base_url, params={"id": file_id}, stream=True, timeout=120)
    response.raise_for_status()

    confirm_token = None
    for cookie_key, cookie_val in response.cookies.items():
        if cookie_key.startswith("download_warning"):
            confirm_token = cookie_val
            break

    if confirm_token:
        response = session.get(
            base_url,
            params={"id": file_id, "confirm": confirm_token},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
    else:
        # Some Google Drive responses return a virus-scan warning HTML page.
        # Parse the embedded download form and resubmit it.
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            html_text = response.text[:300000]

            form_action_match = re.search(
                r'<form[^>]+id=["\']download-form["\'][^>]+action=["\']([^"\']+)["\']',
                html_text,
            )
            hidden_inputs = re.findall(
                r'<input[^>]+type=["\']hidden["\'][^>]+name=["\']([^"\']+)["\'][^>]+value=["\']([^"\']*)["\']',
                html_text,
            )
            form_params = {k: v for k, v in hidden_inputs}

            if form_action_match and ("confirm" in form_params or "uuid" in form_params):
                form_action_url = form_action_match.group(1).replace("&amp;", "&")
                response = session.get(
                    form_action_url,
                    params=form_params,
                    stream=True,
                    timeout=120,
                )
                response.raise_for_status()
            else:
                # Fallback token extraction from links in the HTML.
                html_token_match = re.search(r"confirm=([0-9A-Za-z_\-]+)&amp;id=", html_text)
                if not html_token_match:
                    html_token_match = re.search(r"confirm=([0-9A-Za-z_\-]+)&id=", html_text)
                if html_token_match:
                    response = session.get(
                        base_url,
                        params={"id": file_id, "confirm": html_token_match.group(1)},
                        stream=True,
                        timeout=120,
                    )
                    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return output_path


def _assert_zip_file(zip_path: Path, source_url: str) -> None:
    """Validate downloaded file is a real zip and raise helpful error otherwise."""
    if zipfile.is_zipfile(zip_path):
        return

    snippet = ""
    try:
        with open(zip_path, "rb") as f:
            head = f.read(512)
        snippet = head.decode("utf-8", errors="ignore").strip().replace("\n", " ")
    except Exception:
        snippet = "<unable to read downloaded file>"

    raise ValueError(
        "Downloaded file is not a valid zip archive. "
        f"URL: {source_url}. "
        "This usually means the Google Drive link is not directly downloadable "
        "(not public / permission page / quota page). "
        f"File: {zip_path}. Header preview: {snippet[:200]}"
    )


def _normalize_env_permissions(env_dir: Path) -> None:
    """Ensure downloaded env files are readable and executables are runnable."""
    for root, _, files in os.walk(env_dir):
        for name in files:
            p = Path(root) / name
            try:
                mode = p.stat().st_mode
                # Ensure user can read everything.
                mode |= 0o400
                # Mark likely executables as executable.
                if p.name == "Blocks" or p.suffix in {".sh", ".run"}:
                    mode |= 0o100
                p.chmod(mode)
            except Exception:
                continue


def _ensure_binary_executable(binary_path: Path) -> None:
    """Make binary executable if it exists but has no execute permission."""
    if not binary_path.exists():
        return
    if os.access(binary_path, os.X_OK):
        return
    mode = binary_path.stat().st_mode
    binary_path.chmod(mode | 0o100)


def download_unpack_and_rename_env(
    drive_url: str,
    destination_root: str | Path,
    target_env_name: str,
    zip_filename: str = "packaged_env.zip",
) -> Path:
    """
    Download a packaged env zip from Google Drive, unzip it, and rename it.

    Args:
        drive_url: Google Drive share link for the zip file.
        destination_root: Directory where env folder should be created.
        target_env_name: Final folder name (e.g., NHEnv, CityEnv, ForestEnv).
        zip_filename: Saved zip filename under destination_root.

    Returns:
        Path to the final renamed environment directory.
    """
    if target_env_name not in {"NHEnv", "CityEnv", "ForestEnv"}:
        raise ValueError("target_env_name must be one of: NHEnv, CityEnv, ForestEnv")

    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    zip_path = destination_root / zip_filename
    extract_dir = destination_root / "_tmp_packaged_env"
    final_dir = destination_root / target_env_name

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    if final_dir.exists():
        shutil.rmtree(final_dir)

    _download_google_drive_file(drive_url, zip_path)
    _assert_zip_file(zip_path, drive_url)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    extracted_items = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        extracted_root = extracted_items[0]
    else:
        extracted_root = extract_dir

    shutil.move(str(extracted_root), str(final_dir))
    _normalize_env_permissions(final_dir)

    if zip_path.exists():
        zip_path.unlink()
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    return final_dir


def download_unpack_and_rename_three_envs(
    drive_urls: list[str],
    destination_root: str | Path,
    env_names: tuple[str, ...] = ("NHEnv", "CityEnv", "ForestEnv"),
) -> dict[str, Path]:
    """Download and prepare 3 packaged env archives, mapped respectively to NHEnv/CityEnv/ForestEnv."""
    # if len(drive_urls) != 3:
    #     raise ValueError("drive_urls must contain exactly 3 Google Drive links")

    results: dict[str, Path] = {}
    for url, env_name in zip(drive_urls, env_names):
        results[env_name] = download_unpack_and_rename_env(
            drive_url=url,
            destination_root=destination_root,
            target_env_name=env_name,
            zip_filename=f"{env_name}.zip",
        )

    return results


def download_unpack_dataset(
    drive_url: str,
    destination_root: str | Path,
    zip_filename: str = "dataset.zip",
) -> Path:
    """Download a dataset zip from Google Drive and extract it into destination_root."""
    destination_root = Path(destination_root)
    destination_root.parent.mkdir(parents=True, exist_ok=True)

    download_root = destination_root.parent / "_tmp_dataset_download"
    extract_dir = download_root / "_tmp_extract"
    zip_path = download_root / zip_filename

    if download_root.exists():
        shutil.rmtree(download_root)
    extract_dir.mkdir(parents=True, exist_ok=True)

    _download_google_drive_file(drive_url, zip_path)
    _assert_zip_file(zip_path, drive_url)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    extracted_items = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
    if len(extracted_items) == 1 and extracted_items[0].is_dir():
        extracted_root = extracted_items[0]
    else:
        extracted_root = extract_dir

    if destination_root.exists():
        shutil.rmtree(destination_root)

    if extracted_root.name == destination_root.name and extracted_root.is_dir():
        shutil.move(str(extracted_root), str(destination_root))
    else:
        destination_root.mkdir(parents=True, exist_ok=True)
        for item in extracted_root.iterdir():
            shutil.move(str(item), str(destination_root / item.name))

    if download_root.exists():
        shutil.rmtree(download_root)

    return destination_root


def ensure_required_envs_present(project_root: str | Path) -> None:
    """Ensure NHEnv/CityEnv/ForestEnv binaries exist, download if missing."""
    project_root = Path(project_root)
    env_yaml = project_root / "configs" / "environments.yaml"
    if not env_yaml.exists():
        raise FileNotFoundError(f"environments.yaml not found at {env_yaml}")

    with open(env_yaml, "r") as f:
        env_data = yaml.safe_load(f) or {}

    drive_links_cfg = env_data.get("drive_links", {})
    env_binary_paths = {k: v for k, v in env_data.items() if isinstance(v, str)}

    target_envs = ["ForestEnv", "NHEnv", "CityEnv"]
    missing_target_envs = []
    for env_name in target_envs:
        binary_rel = env_binary_paths.get(env_name)
        if not binary_rel:
            continue
        binary_abs = project_root / binary_rel
        _ensure_binary_executable(binary_abs)
        if not binary_abs.exists():
            missing_target_envs.append(env_name)

    if not missing_target_envs:
        print("✓ Environment binaries found for NHEnv/CityEnv/ForestEnv")
        return

    print(
        "⚠ Missing environments detected: "
        f"{', '.join(missing_target_envs)}. "
        "Attempting auto-download..."
    )

    env_links = {
        "NHEnv": drive_links_cfg.get("NHEnv") or os.getenv("NHENV_DRIVE_LINK"),
        "CityEnv": drive_links_cfg.get("CityEnv") or os.getenv("CITYENV_DRIVE_LINK"),
        "ForestEnv": drive_links_cfg.get("ForestEnv") or os.getenv("FORESTENV_DRIVE_LINK"),
    }

    missing_links = [env_name for env_name in missing_target_envs if not env_links.get(env_name)]
    if missing_links:
        raise ValueError(
            "Missing Google Drive links for env download: "
            f"{', '.join(missing_links)}. "
            "Provide drive_links.{NHEnv,CityEnv,ForestEnv} in "
            "configs/environments.yaml or set env vars "
            "NHENV_DRIVE_LINK, CITYENV_DRIVE_LINK, FORESTENV_DRIVE_LINK."
        )

    envs_root = project_root / "envs"
    urls_to_download = [env_links[env_name] for env_name in missing_target_envs]
    download_unpack_and_rename_three_envs(
        drive_urls=urls_to_download,
        destination_root=envs_root,
        env_names=tuple(missing_target_envs),
    )

    still_missing = []
    for env_name in missing_target_envs:
        binary_abs = project_root / env_binary_paths[env_name]
        _ensure_binary_executable(binary_abs)
        if not binary_abs.exists():
            still_missing.append(env_name)

    if still_missing:
        raise FileNotFoundError(
            "Environment download completed but binaries are still missing for: "
            f"{', '.join(still_missing)}"
        )

    print("✓ Environments downloaded and prepared successfully")


def ensure_required_dataset_present(project_root: str | Path) -> None:
    """Ensure the dataset folder exists, download it from Google Drive if missing."""
    project_root = Path(project_root)
    dataset_root = project_root / "dataset"

    if dataset_root.exists() and any(dataset_root.iterdir()):
        print(f"✓ Dataset folder found: {dataset_root}")
        return

    env_yaml = project_root / "configs" / "environments.yaml"
    if not env_yaml.exists():
        raise FileNotFoundError(f"environments.yaml not found at {env_yaml}")

    with open(env_yaml, "r") as f:
        env_data = yaml.safe_load(f) or {}

    drive_links_cfg = env_data.get("drive_links", {})
    dataset_link = drive_links_cfg.get("test_dataset") or os.getenv("test_dataset_DRIVE_LINK")

    if not dataset_link:
        raise ValueError(
            "Missing Google Drive link for dataset download. "
            "Provide drive_links.test_dataset in configs/environments.yaml "
            "or set test_dataset_DRIVE_LINK."
        )

    print(f"⚠ Dataset folder missing: {dataset_root}. Attempting auto-download...")
    download_unpack_dataset(
        drive_url=dataset_link,
        destination_root=dataset_root,
        zip_filename="dataset.zip",
    )

    if not dataset_root.exists() or not any(dataset_root.iterdir()):
        raise FileNotFoundError(
            f"Dataset download completed but the dataset folder is still missing or empty: {dataset_root}"
        )

    print(f"✓ Dataset downloaded and prepared successfully: {dataset_root}")


def start_airsim_api_server(project_root: str | Path, logs_dir: str | Path | None = None) -> subprocess.Popen:
    """Start the AirSim API server as a background subprocess.

    All stdout/stderr from the server is tee'd to
    ``<logs_dir>/<timestamp>-airsim_api_server.log``.
    You can follow it live with::

        tail -f project_logs/<timestamp>-airsim_api_server.log

    Args:
        project_root: Path to project root
        logs_dir: Optional log directory. Defaults to ``project_logs``.

    Returns:
        subprocess.Popen: The running server process

    Raises:
        FileNotFoundError: If airsim_env Python interpreter not found
        RuntimeError: If server process terminates unexpectedly
        TimeoutError: If server doesn't start within timeout
    """
    import threading
    import queue as _queue

    project_root = Path(project_root)

    # airsim_env_python = project_root / "airsim_env" / "bin" / "python"
    airsim_env_python = Path(sys.executable)
    if not airsim_env_python.exists():
        raise FileNotFoundError(
            f"airsim_env Python not found at {airsim_env_python}. "
            f"Run ./env_setup.sh airsim to create the environment."
        )

    # Prepare timestamped log file in selected logs dir.
    logs_dir = Path(logs_dir) if logs_dir else (project_root / "project_logs")
    logs_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_path = logs_dir / f"{ts}-airsim_api_server.log"
    log_file = open(log_path, "w", buffering=1)
    print(f"  API server log → {log_path}")

    def _launch() -> subprocess.Popen:
        proc_env = os.environ.copy()
        # Inherit log directory so UnrealProcessManager logs there as well.
        proc_env["MISSIONBENCH_LOG_DIR"] = str(logs_dir)
        return subprocess.Popen(
            [str(airsim_env_python), "airsim_api_server.py"],
            cwd=project_root,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

    # Queue: tee-thread feeds lines here for the startup monitor.
    line_queue: _queue.Queue = _queue.Queue()

    def _tee(proc: subprocess.Popen) -> None:
        """Copy every stdout line to the log file and the queue."""
        try:
            for raw in proc.stdout:
                log_file.write(raw)
                log_file.flush()
                line_queue.put(raw)
        except Exception:
            pass
        finally:
            line_queue.put(None)  # sentinel: stdout closed

    api_server_process = _launch()
    threading.Thread(target=_tee, args=(api_server_process,),
                     daemon=True, name="airsim-tee").start()

    # Register cleanup
    def _cleanup():
        if api_server_process.poll() is None:
            print("\n[Cleanup] Shutting down API server...")
            api_server_process.terminate()
            try:
                api_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_server_process.kill()
        log_file.close()

    atexit.register(_cleanup)

    # Wait for Flask "Running on" message
    print("  Waiting for server to start", end="", flush=True)
    server_started = False
    max_wait_time = 240
    start_time = time.time()
    restart_attempted = False
    first_attempt_output: list[str] = []

    while time.time() - start_time < max_wait_time:
        try:
            line = line_queue.get(timeout=0.5)
        except _queue.Empty:
            continue

        if line is None:
            # stdout closed → process exited; drain any queued remnants
            leftover: list[str] = []
            try:
                while True:
                    item = line_queue.get_nowait()
                    if item:
                        leftover.append(item)
            except _queue.Empty:
                pass

            if not restart_attempted:
                restart_attempted = True
                first_attempt_output = leftover
                print(
                    "\n⚠ API server terminated on first attempt. "
                    "Retrying once..."
                )
                print(f"  See log for details: {log_path}")
                new_proc = _launch()
                # Re-assign so cleanup closure captures the latest handle
                api_server_process.__dict__.update(new_proc.__dict__)
                threading.Thread(
                    target=_tee, args=(new_proc,),
                    daemon=True, name="airsim-tee-retry",
                ).start()
                start_time = time.time()
                print("  Waiting for server to start", end="", flush=True)
                continue

            raise RuntimeError(
                "API server process terminated unexpectedly.\n"
                f"Exit code: {api_server_process.returncode}\n"
                f"Log file: {log_path}\n"
                f"First attempt tail: {''.join(first_attempt_output[-20:])}\n"
                f"Second attempt tail: {''.join(leftover[-20:])}"
            )

        if "Running on" in line or "Server running on" in line:
            print()
            print(f"✓ API server started (PID: {api_server_process.pid})")
            print(f"  {line.strip()}")
            server_started = True
            break

        print(".", end="", flush=True)

    if not server_started:
        print()
        raise TimeoutError(
            f"API server did not start within {max_wait_time}s. "
            f"Check log: {log_path}"
        )

    return api_server_process


def ensure_model_api_keys_present(config_path: str | Path) -> None:
    """Verify API keys for configured models exist in .env."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"experiment_configs.yaml not found")
    
    with open(config_path, "r") as f:
        exp_cfg = yaml.safe_load(f) or {}
    
    main_cfg = exp_cfg.get("main_driver", {})
    model_related_cfg = (
        main_cfg.get("model_related")
        if isinstance(main_cfg.get("model_related"), dict)
        else main_cfg.get("models_related")
        if isinstance(main_cfg.get("models_related"), dict)
        else {}
    )

    model_names = model_related_cfg.get("model_names", main_cfg.get("model_names", []))
    task_gt_eval_model = model_related_cfg.get("task_gt_eval_model", main_cfg.get("task_gt_eval_model"))

    models_to_check = list(model_names)
    if isinstance(task_gt_eval_model, str) and task_gt_eval_model.strip():
        models_to_check.append(task_gt_eval_model.strip())

    # Preserve order while removing duplicates
    models_to_check = list(dict.fromkeys(models_to_check))
    
    if not models_to_check:
        return
    
    gemini_keys = get_available_gemini_api_keys()
    missing = []
    for model in models_to_check:
        if model_uses_gemini(model):
            if not gemini_keys:
                missing.append("GOOGLE_API_KEY/GEMINI_API_KEY (or suffixed variants like GOOGLE_API_KEY_1)")
        elif model.startswith("gpt"):
            if not os.getenv("OPENAI_API_KEY"):
                missing.append("OPENAI_API_KEY")
        elif model.startswith("aws") or model.startswith("us.amazon"):
            if not os.getenv("AWS_ACCESS_KEY_ID"):
                missing.append("AWS_ACCESS_KEY_ID")
            if not os.getenv("AWS_SECRET_ACCESS_KEY"):
                missing.append("AWS_SECRET_ACCESS_KEY")
        elif model.startswith("anthropic") or model.startswith("us.anthropic"):
            if not os.getenv("ANTHROPIC_API_KEY"):
                missing.append("ANTHROPIC_API_KEY")
        elif model.startswith("qwen"):
            if not os.getenv("DASHSCOPE_API_KEY"):
                missing.append("DASHSCOPE_API_KEY")
    
    if missing:
        raise ValueError(
            f"Missing API keys: {', '.join(set(missing))}\n"
            f"Models: {', '.join(models_to_check)}\n"
            f"Set these in your .env file"
        )
    
    print(f"✓ API keys verified for {', '.join(models_to_check)}")


def model_uses_gemini(model_name: Optional[str]) -> bool:
    """Return True when the model string points to a Google GenAI model/provider."""
    if not isinstance(model_name, str):
        return False
    token = model_name.strip().lower()
    return token.startswith("gemini") or token.startswith("google-") or token.startswith("gemma-")


def get_available_gemini_api_keys() -> List[str]:
    """Collect Gemini API keys from environment in deterministic priority order."""
    key_pattern = re.compile(r"^(GOOGLE|GEMINI)_API_KEY(?:_?(.+))?$")
    ranked: List[tuple[int, int, str, str]] = []

    def _sanitize_env_value(raw_value: str) -> str:
        # Handle values loaded via manual .env parsing where trailing comments may remain.
        value = (raw_value or "").strip()
        value = re.sub(r"#.*$", "", value).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1].strip()
        return value

    def _looks_like_gemini_api_key(value: str) -> bool:
        # Google AI Studio API keys are expected to start with "AIza".
        return isinstance(value, str) and value.startswith("AIza") and len(value) >= 20

    for env_name, env_value in os.environ.items():
        match = key_pattern.fullmatch(env_name)
        if not match:
            continue

        value = _sanitize_env_value(env_value)
        if not value:
            continue
        if not _looks_like_gemini_api_key(value):
            continue

        suffix = (match.group(2) or "").strip()
        provider_rank = 0 if match.group(1) == "GOOGLE" else 1
        if not suffix:
            suffix_rank = 0
            suffix_num = -1
        elif suffix.isdigit():
            suffix_rank = 1
            suffix_num = int(suffix)
        else:
            suffix_rank = 2
            suffix_num = 10**9

        ranked.append((suffix_rank, suffix_num, provider_rank, env_name))

    ranked.sort()

    keys: List[str] = []
    seen_values = set()
    for _, _, _, env_name in ranked:
        value = _sanitize_env_value(os.environ.get(env_name, ""))
        if value and _looks_like_gemini_api_key(value) and value not in seen_values:
            keys.append(value)
            seen_values.add(value)
    return keys


def is_gemini_quota_error(error_message: Optional[str]) -> bool:
    """Detect Gemini errors that should trigger key failover (quota/auth issues)."""
    if not error_message:
        return False
    text = str(error_message).lower()
    tokens = [
        "resource_exhausted",
        "quota",
        "rate limit",
        "429",
        "too many requests",
        "daily limit",
        "exceeded",
        "insufficient_quota",
        "unauthenticated",
        "invalid authentication credentials",
        "api key not valid",
        "api key not found",
        "api_key_invalid",
        "access_token_type_unsupported",
        "permission_denied",
        "401",
        "403",
    ]
    return any(token in text for token in tokens)


class GeminiApiKeyRotator:
    """Manages sequential Gemini API key failover in-process."""

    def __init__(self, keys: List[str]):
        self._keys = [k for k in keys if isinstance(k, str) and k.strip()]
        self._active_index = -1

    @classmethod
    def from_environment(cls) -> "GeminiApiKeyRotator":
        return cls(get_available_gemini_api_keys())

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    @property
    def active_index(self) -> int:
        return self._active_index

    @property
    def active_key_masked(self) -> str:
        if self._active_index < 0 or self._active_index >= len(self._keys):
            return "<none>"
        key = self._keys[self._active_index]
        return f"***{key[-4:]}" if len(key) >= 4 else "***"

    def has_any_keys(self) -> bool:
        return bool(self._keys)

    def activate_first_key(self) -> bool:
        if not self._keys:
            return False
        self._activate_index(0)
        return True

    def rotate_to_next_key(self) -> bool:
        next_index = self._active_index + 1
        if next_index >= len(self._keys):
            return False
        self._activate_index(next_index)
        return True

    def _activate_index(self, index: int) -> None:
        selected = self._keys[index]
        # Keep a single active variable to avoid repeated SDK warnings when both are set.
        os.environ["GOOGLE_API_KEY"] = selected
        os.environ.pop("GEMINI_API_KEY", None)
        self._active_index = index

        # Persist active-key metadata so users can inspect it from a separate terminal.
        try:
            status_path = Path(
                os.getenv("MISSIONBENCH_GEMINI_KEY_STATUS_FILE", "/tmp/missionbench_gemini_key_status.json")
            )
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status = {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "active_index_1based": self._active_index + 1,
                "total_keys": len(self._keys),
                "active_key_masked": self.active_key_masked,
                "active_env_var": "GOOGLE_API_KEY",
            }
            with open(status_path, "w") as f:
                json.dump(status, f, indent=2)
        except Exception:
            # Status persistence is best-effort and should not block experiment execution.
            pass


def sync_runtime_sim_config(
    project_root: str | Path,
    main_driver_cfg: dict,
    runtime_config_relpath: str = "configs/config.yaml",
) -> Path:
    """Update runtime simulator config from experiment main_driver config.

    This keeps backward compatibility for components that still read
    ``configs/config.yaml`` while allowing central control from
    ``configs/experiment_configs*.yaml``.

        Expected optional structure under main_driver:

    simulator:
      rendering: false
            CollisionCheckMode: false

        Runtime ``ActiveSimMode`` is derived automatically:
            - CollisionCheckMode=true  -> Multirotor
            - CollisionCheckMode=false -> ComputerVision

            Runtime ``environment`` is inferred from mission metadata when omitted.
    """
    project_root = Path(project_root)
    runtime_cfg_path = project_root / runtime_config_relpath

    simulator_cfg = main_driver_cfg.get("simulator", {}) or {}
    if not isinstance(simulator_cfg, dict):
        return runtime_cfg_path

    def _infer_environment_from_mission_data() -> Optional[str]:
        """Infer environment from mission config referenced by val_mission_file."""
        val_mission_file = main_driver_cfg.get("val_mission_file")
        if not val_mission_file:
            return None

        try:
            try:
                mission_map = load_mission_index(val_mission_file, project_root=project_root)
            except Exception:
                # Backward compatibility: legacy direct YAML mission map.
                val_path = Path(val_mission_file)
                if not val_path.is_absolute():
                    val_path = project_root / val_path
                if not val_path.exists():
                    return None
                with open(val_path, "r") as f:
                    mission_map = yaml.safe_load(f) or {}
            if not isinstance(mission_map, dict) or not mission_map:
                return None

            requested = main_driver_cfg.get("scenario_names") or []
            scenario = requested[0] if requested else next(iter(mission_map.keys()))
            mission_rel = mission_map.get(scenario)
            if not mission_rel:
                mission_rel = next(iter(mission_map.values()))

            mission_path = Path(mission_rel)
            if not mission_path.is_absolute():
                if str(mission_path).startswith("dataset/"):
                    mission_path = project_root / mission_path
                else:
                    mission_path = project_root / "dataset" / mission_path
            if not mission_path.exists():
                return None

            with open(mission_path, "r") as f:
                mission_cfg = yaml.safe_load(f) or {}

            metadata = mission_cfg.get("metadata", {}) or {}
            env = metadata.get("environment")
            if isinstance(env, str) and env.strip():
                return env.strip()

            for part in mission_path.parts:
                if part in {
                    "NHEnv",
                    "CityEnv",
                    "ForestEnv",
                    "AirSimNHEnv",
                }:
                    return part
        except Exception:
            return None

        return None

    key_values = {}
    if "environment" in simulator_cfg:
        key_values["environment"] = simulator_cfg.get("environment")
    else:
        inferred_env = _infer_environment_from_mission_data()
        if inferred_env:
            key_values["environment"] = inferred_env
            print(f"✓ Inferred simulator environment from mission data: {inferred_env}")
    if "rendering" in simulator_cfg:
        key_values["rendering"] = bool(simulator_cfg.get("rendering"))

    if "CollisionCheckMode" in simulator_cfg:
        collision_check_mode = bool(simulator_cfg.get("CollisionCheckMode"))
        key_values["CollisionCheckMode"] = collision_check_mode
        key_values["ActiveSimMode"] = (
            "Multirotor" if collision_check_mode else "ComputerVision"
        )

    # Backward compatibility: allow explicit ActiveSimMode if provided.
    # CollisionCheckMode takes precedence whenever present.
    if (
        "ActiveSimMode" in simulator_cfg
        and "ActiveSimMode" not in key_values
    ):
        key_values["ActiveSimMode"] = simulator_cfg.get("ActiveSimMode")

    # Nothing to sync; preserve existing config.yaml as-is.
    if not key_values:
        return runtime_cfg_path

    existing = {}
    if runtime_cfg_path.exists():
        with open(runtime_cfg_path, "r") as f:
            existing = yaml.safe_load(f) or {}

    existing.update(key_values)

    runtime_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runtime_cfg_path, "w") as f:
        yaml.safe_dump(existing, f, sort_keys=False)

    print(f"✓ Synced runtime simulator config: {runtime_cfg_path}")
    print(
        "  - environment={env}, rendering={rendering}, "
        "CollisionCheckMode={collision}, ActiveSimMode={mode}".format(
            env=existing.get("environment"),
            rendering=existing.get("rendering"),
            collision=existing.get("CollisionCheckMode"),
            mode=existing.get("ActiveSimMode"),
        )
    )

    return runtime_cfg_path


def _append_step_payload(base_dir, step_index: int, payload: Dict, filename: str="step_by_step_intermediate_results.json") -> None:
    """Persist per-node debug payload into a single JSON file keyed by step index.
    
    Long text responses are saved as separate .txt files to improve readability.
    """
    # print(f"   💾 Saving debug file {filename} for step {step_index}, {payload}...")
    if not base_dir:
        return
    try:
        # Create debug_outputs subdirectory
        target_dir = Path(base_dir) / "debug_outputs"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract long text fields and save as separate files
        payload_for_json = payload.copy()
        # Move any time-related fields into a timing sub-dictionary
        timing_bucket = payload_for_json.get("timing") if isinstance(payload_for_json.get("timing"), dict) else {}
        #         {
        #     "raw_response_hlap_step_by_step": raw_response,
        #     "time_total_hlap_node": round(step_elapsed, 2)
            
        # },
        # Move all time-related fields (int/float) into timing_bucket
        for key in list(payload_for_json.keys()):
            if key == "timing":
                continue
            value = payload_for_json.get(key)
            if "time" in key and isinstance(value, (int, float)):
                timing_bucket[key] = value
                payload_for_json.pop(key, None)
        # Always include timing_bucket, even if empty
        payload_for_json["timing"] = timing_bucket
        long_text_fields = ["raw_response_hlap_step_by_step", "raw_waypoint_response", "should_continue_response", "reasoning_hlap_step_by_step", "parsed_reasoning_should_continue"]
        
        for field in long_text_fields:
            if field in payload_for_json and isinstance(payload_for_json[field], str):
                text = payload_for_json[field]
                # If text is longer than 120 chars, save separately
                if len(text) > 120:
                    text_file = target_dir / f"step_{step_index}_{field}.txt"
                    with open(text_file, "w") as f:
                        # Write with word wrapping at 120 chars
                        lines = []
                        words = text.split()
                        current_line = []
                        for word in words:
                            if sum(len(w) for w in current_line) + len(word) + len(current_line) > 120:
                                lines.append(" ".join(current_line))
                                current_line = [word]
                            else:
                                current_line.append(word)
                        if current_line:
                            lines.append(" ".join(current_line))
                        f.write("\n".join(lines))
                    # Replace in payload with reference
                    payload_for_json[field] = f"[See step_{step_index}_{field}.txt]"
        
        file_path = target_dir / filename

        # Load existing content
        if file_path.exists():
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            data = {}

        # Store by step index
        if str(step_index) in data:
            data[str(step_index)].update(payload_for_json)
        else:
            data[str(step_index)] = payload_for_json

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:  # Best-effort logging only
        print(f"⚠ Failed to write debug file {filename}: {exc}")


def get_active_settings_file(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    sim_mode = config.get("ActiveSimMode")
    if sim_mode is None:
        collision_check_mode = bool(config.get("CollisionCheckMode", False))
        sim_mode = "Multirotor" if collision_check_mode else "ComputerVision"
    # Hardcoded mapping
    if sim_mode == "Multirotor":
        return str(pathlib.Path(config_path).parent / "settings_multirotor.json")
    elif sim_mode == "ComputerVision":
        return str(pathlib.Path(config_path).parent / "settings_computer_vision_mode.json")
    else:
        raise ValueError(f"Unknown SimMode: {sim_mode}")

def get_unreal_binary(env_name, env_yaml_path):
    with open(env_yaml_path, "r") as f:
        envs = yaml.safe_load(f)
    unreal_binary = envs.get(env_name)
    if not unreal_binary:
        raise ValueError(f"Environment '{env_name}' not found in {env_yaml_path}")
    return unreal_binary

def create_mission_config(
    mission_name, 
    start_location, 
    end_location, 
    instruction, 
    target_object, 
    env_name, 
    category="Visual_Inspection",
    template_path="dataset/mission_template.yaml",
    start_img=None,
    end_img=None,
    task_gt="",
    airsim_rec_txt_path=None,
    difficulty_level="",
    difficulty_description="",
    task_type="other",
    task_type_description="",
):
    """
    Create and save a mission configuration file.
    
    Args:
        mission_name (str): Name of the mission
        start_location (list): [x, y, z, yaw] start pose (4 elements)
        end_location (list): [x, y, z, yaw] end pose (4 elements)
        instruction (str): Mission instruction text
        target_object (str): Target object name
        env_name (str): Environment name
        category (str): Category name (Visual_Inspection, Patrol, Manipulation)
        template_path (str): Path to mission template YAML
    
    Returns:
        tuple: (yaml_path, mission_config) - Path to saved config and the config dict
    """
    
    # Create directory
    directory = pathlib.Path("./dataset") / category / f"{env_name}" / mission_name
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Directory {directory} already exists. Files may be overwritten.")
    
    if category == "Patrol":
        # copy the airsim_rec.txt_path into the directory
        if airsim_rec_txt_path is not None:
            shutil.copy(airsim_rec_txt_path, directory / "airsim_rec.txt")
    if start_img is not None and end_img is not None:
        print(f"Start image: {start_img}, End image: {end_img}")    
        # copy the files into the directory called imags and name then as start_fp_image.png, gt_fp_image.png
        
        images_dir = directory / "imgs"
        images_dir.mkdir(exist_ok=True)
        shutil.copy(start_img, images_dir / "start_fp_image.png")
        shutil.copy(end_img, images_dir / "gt_fp_image.png")
    # Validate inputs
    assert len(start_location) == 4 and len(end_location) == 4, \
        "Start and end locations must have 4 elements: x, y, z, yaw"
    

    
    # Load template
    with open(template_path, 'r') as f:
        mission_config = yaml.safe_load(f)
    start_x = start_location[0]
    start_y = start_location[1]
    start_z = start_location[2]
    start_yaw = start_location[3]
    gt_x = end_location[0]
    gt_y = end_location[1]
    gt_z = end_location[2]
    gt_yaw = end_location[3]
    # Update mission config
    mission_config['name'] = mission_name
    mission_config['instruction'] = instruction

    mission_config['drone_start_pose']["x"] = start_x
    mission_config['drone_start_pose']["y"] = start_y
    mission_config['drone_start_pose']["z"] = start_z
    mission_config['drone_start_pose']["yaw"] = start_yaw
    mission_config['ground_truth']['waypoint_gt']["x"] = gt_x
    mission_config['ground_truth']['waypoint_gt']["y"] = gt_y
    mission_config['ground_truth']['waypoint_gt']["z"] = gt_z
    mission_config['ground_truth']['waypoint_gt']["yaw"] = gt_yaw
    mission_config['ground_truth']['task_gt'] = task_gt
    # Compute depth (distance from start to ground truth)

    depth = ((gt_x - start_x) ** 2 + (gt_y - start_y) ** 2 + (gt_z - start_z) ** 2) ** 0.5
    mission_config['ground_truth']['depth'] = depth
    mission_config['target_object'] = target_object
    mission_config['metadata']['environment'] = env_name
    mission_config['metadata']['task_category'] = category.lower()
    mission_config['metadata']['target_object'] = target_object.lower()
    mission_config['metadata']['difficulty_level'] = difficulty_level
    mission_config['metadata']['difficulty_description'] = difficulty_description
    mission_config['metadata']['task'] = task_type
    mission_config['metadata']['task_description'] = task_type_description
    mission_config["tolerances"] = {"MAX_ACTIONS": 20,
                                    "MAX_STEP_SIZE": 3,
                                    "x": 5,
                                    "y": 5,
                                    "yaw": 15,
                                    "z": 5}
    max_action = mission_config["tolerances"]["MAX_ACTIONS"]
    mission_config["tolerances"]["MAX_STEP_SIZE"] = math.ceil(depth / 4)
    # Save config
    yaml_path = directory / "mission_config.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(mission_config, f, default_flow_style=False)
    
    return yaml_path, mission_config


# voxel grid related(copied from Taeyoung's work)
import numpy as np

class Voxels(object):
    def __init__(self, data, dims, translate, scale, axis_order):
        self.data = data
        self.dims = dims
        self.translate = translate
        self.scale = scale
        assert (axis_order in ('xzy', 'xyz'))
        self.axis_order = axis_order

    def clone(self):
        data = self.data.copy()
        dims = self.dims[:]
        translate = self.translate[:]
        return Voxels(data, dims, translate, self.scale, self.axis_order)
    
def binvox_read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale
    
def binvox_read_as_3d_array(fp, fix_coords=True):
    dims, translate, scale = binvox_read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(bool)
    data = data.reshape(dims)
    if fix_coords:
        data = np.transpose(data, (0, 2, 1))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    return Voxels(data, dims, translate, scale, axis_order)

def get_next_mission_number(category, env_name):
    """
    Automatically determine the next mission number by scanning existing missions.
    
    Args:
        category: Mission category (e.g., "Visual_Inspection")
        env_name: Environment name (e.g., "ForestEnv")
        target_object: Target object name (e.g., "deer_statue")
    
    Returns:
        Next available mission number
    """
    dataset_dir = pathlib.Path(f"dataset/{category}/{env_name}")
    
    if not dataset_dir.exists():
        return 1
    
    # Find all existing mission directories for this target object
    dir_len = 0
    
    for item in dataset_dir.iterdir():
        if item.is_dir():
            dir_len += 1
    
    # Return next number (max + 1, or 1 if none exist)
    return dir_len + 1 if dir_len else 1

def quaternion_to_eulerian_angle(qw, qx, qy, qz):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    import math

    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    # convert to degrees
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    return roll_deg, pitch_deg, yaw_deg

def get_start_and_end_details(airsim_rec_txt_path: str):
    """
    Extract data from the text file
    
    Args:
        airsim_rec_txt_path (str): Path to the AirSim recording text file containing dataset collection details.
    Returns:
        tuple: (start_location, end_location, img_start_path, img_end_path) as lists
    """
    with open(airsim_rec_txt_path, 'r') as f:
        lines = f.readlines()
    start_line = lines[1].strip().split("\t")
    end_line = lines[-1].strip().split("\t")
    img_dir = "/".join(airsim_rec_txt_path.split("/")[:-1]) + "/images/"
    img_start_path = img_dir + start_line[-1].strip()
    img_end_path = img_dir + end_line[-1].strip()
    
    start_poses = list(map(float, start_line[2:9]))
    end_poses = list(map(float, end_line[2:9]))
    start_yaw = quaternion_to_eulerian_angle(start_poses[3], start_poses[4], start_poses[5], start_poses[6])[2]
    end_yaw = quaternion_to_eulerian_angle(end_poses[3], end_poses[4], end_poses[5], end_poses[6])[2]
    start_location = [start_poses[0], start_poses[1], start_poses[2], start_yaw]
    end_location = [end_poses[0], end_poses[1], end_poses[2], end_yaw]
    return start_location, end_location, img_start_path, img_end_path

def calculate_IoU(bb1, bb2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bb1: List or tuple of (x_min, y_min, x_max, y_max) for the first bounding box
        bb2: List or tuple of (x_min, y_min, x_max, y_max) for the second bounding box
    
    Returns:
        IoU value between 0 and 1
    """
    x_min1, y_min1, x_max1, y_max1 = bb1
    x_min2, y_min2, x_max2, y_max2 = bb2
    
    # Calculate intersection
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)
    
    if x_max_inter < x_min_inter or y_max_inter < y_min_inter:
        return 0.0  # No overlap
    
    intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    
    # Calculate union
    area_bb1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area_bb2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    
    union_area = area_bb1 + area_bb2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_depth_RMSE(pred_depth, gt_depth):
    """
    Calculate Root Mean Squared Error (RMSE) for depth prediction.
    Args:
        pred_depth: Predicted depth value (float)
        gt_depth: Ground truth depth value (float)
    Returns:
        RMSE value (float)
    """    
    return math.sqrt((pred_depth - gt_depth) ** 2)
        
if __name__ == "__main__":
    # Example usage
    # 0.979413	-0.0181512	0.10294	0.172697
    qw, qx, qy, qz = 0.999391,	-0,	0,	0.034883
    roll, pitch, yaw = quaternion_to_eulerian_angle(qw, qx, qy, qz)
    print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")
    airsim_rec_txt_path = "/home/nava/Documents/AirSim/deer_statue_NH_Environment/airsim_rec.txt"
    start_location, end_location, start_img_path, end_img_path = get_start_and_end_location(airsim_rec_txt_path)
    print(start_location, end_location, start_img_path, end_img_path)