from typing import Callable, Optional
import argparse
import importlib
import json
import os
import re
import pathlib
import shutil
import sys

# Ensure this project root is in path so we can resolve types correctly if needed
try:
    project_root = pathlib.Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception:
    pass

try:
    dotenv_module = importlib.import_module("dotenv")
    load_dotenv = getattr(dotenv_module, "load_dotenv")
    missionbench_root = pathlib.Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=missionbench_root / ".env", override=False)
except ImportError:
    pass


CATEGORY_CHOICES = {
    "1": "Visual_Inspection",
    "2": "Patrol",
    "3": "Manipulation",
}


def prompt_non_empty(prompt: str, validator: Optional[Callable[[str], bool]] = None, error_message: str = "Invalid input.") -> str:
    while True:
        value = input(prompt).strip()
        if not value:
            print("Input is required. Please provide a value.")
            continue
        if validator is not None and not validator(value):
            print(error_message)
            continue
        return value


def prompt_with_default(prompt: str, default_value: str = "") -> str:
    value = input(prompt).strip()
    if value:
        return value
    return default_value


def prompt_yes_no(prompt: str) -> bool:
    while True:
        value = input(prompt).strip().lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def prompt_category() -> str:
    print("\nSelect mission category:")
    for option_key, option_label in CATEGORY_CHOICES.items():
        print(f"  {option_key}) {option_label}")
    return prompt_non_empty(
        "Enter category number (1/2/3): ",
        validator=lambda value: value in CATEGORY_CHOICES,
        error_message="Please select one of: 1, 2, 3.",
    )


def get_environment_choices(environments_yaml_path: pathlib.Path) -> dict[str, str]:
    with open(environments_yaml_path, "r", encoding="utf-8") as f:
        env_config = importlib.import_module("yaml").safe_load(f) or {}

    env_names = [
        key for key, value in env_config.items()
        if key != "drive_links" and isinstance(value, str)
    ]

    if not env_names:
        raise ValueError(f"No environments found in: {environments_yaml_path}")

    return {str(idx): env_name for idx, env_name in enumerate(env_names, start=1)}


def prompt_environment_choice(environments_yaml_path: pathlib.Path, default_env: Optional[str] = None) -> str:
    environment_choices = get_environment_choices(environments_yaml_path)
    env_names = list(environment_choices.values())

    print("\nSelect environment:")
    for option_key, env_name in environment_choices.items():
        marker = " (default)" if default_env is not None and env_name == default_env else ""
        print(f"  {option_key}) {env_name}{marker}")

    if default_env is not None and default_env in env_names:
        prompt = f"Enter environment number [default {default_env}]: "
    else:
        prompt = "Enter environment number: "

    while True:
        choice = input(prompt).strip()
        if not choice and default_env is not None and default_env in env_names:
            assert default_env is not None
            return default_env
        if choice in environment_choices:
            return environment_choices[choice]
        print(f"Please select one of: {', '.join(environment_choices.keys())}.")


def get_mission_difficulty_choices(difficulty_yaml_path: pathlib.Path) -> dict[str, str]:
    with open(difficulty_yaml_path, "r", encoding="utf-8") as f:
        difficulty_config = importlib.import_module("yaml").safe_load(f) or {}

    if not isinstance(difficulty_config, dict) or not difficulty_config:
        raise ValueError(f"No difficulty levels found in: {difficulty_yaml_path}")

    normalized: dict[str, str] = {}
    for level, description in difficulty_config.items():
        level_key = str(level).strip().lower()
        normalized[level_key] = str(description).strip()
    return normalized


def prompt_mission_difficulty(difficulty_yaml_path: pathlib.Path) -> tuple[str, str]:
    difficulty_choices = get_mission_difficulty_choices(difficulty_yaml_path)
    difficulty_link = difficulty_yaml_path.resolve().as_uri()
    print(f"Mission difficulty reference: {difficulty_link}")

    option_map: dict[str, str] = {}
    print("\nSelect mission difficulty:")
    for idx, difficulty_level in enumerate(difficulty_choices.keys(), start=1):
        option_key = str(idx)
        option_map[option_key] = difficulty_level
        print(f"  {option_key}) {difficulty_level}")

    while True:
        choice = input("Enter difficulty number: ").strip()
        if choice in option_map:
            selected_level = option_map[choice]
            return selected_level, difficulty_choices[selected_level]
        print(f"Please select one of: {', '.join(option_map.keys())}.")


def get_mission_task_choices(task_yaml_path: pathlib.Path) -> dict[str, str]:
    with open(task_yaml_path, "r", encoding="utf-8") as f:
        task_config = importlib.import_module("yaml").safe_load(f) or {}

    if not isinstance(task_config, dict) or not task_config:
        raise ValueError(f"No task options found in: {task_yaml_path}")

    normalized: dict[str, str] = {}
    for task_name, description in task_config.items():
        normalized[str(task_name).strip().lower()] = str(description).strip()
    return normalized


def prompt_mission_task(task_yaml_path: pathlib.Path, default_task: str = "other") -> tuple[str, str]:
    task_choices = get_mission_task_choices(task_yaml_path)
    task_link = task_yaml_path.resolve().as_uri()
    print(f"Mission task reference: {task_link}")

    option_map: dict[str, str] = {}
    print("\nSelect mission task type:")
    option_index = 1
    default_option_key: Optional[str] = None
    for task_name in task_choices.keys():
        key = str(option_index)
        option_map[key] = task_name
        marker = " (default)" if task_name == default_task else ""
        if task_name == default_task:
            default_option_key = key
        print(f"  {key}) {task_name}{marker}")
        option_index += 1

    while True:
        prompt_text = "Enter task number"
        if default_option_key is not None:
            prompt_text += f" [default {default_task}]"
        prompt_text += ": "
        choice = input(prompt_text).strip()

        if not choice and default_option_key is not None:
            selected_task = option_map[default_option_key]
            return selected_task, task_choices[selected_task]

        if choice in option_map:
            selected_task = option_map[choice]
            return selected_task, task_choices[selected_task]

        print(f"Please select one of: {', '.join(option_map.keys())}.")


def build_unique_mission_name(
    target_object: str,
    mission_num: int,
    category: str,
    env_name: str,
    dataset_root: pathlib.Path = pathlib.Path("./dataset"),
) -> tuple[str, str]:
    sanitized_target_object = "_".join(target_object.strip().split())
    if not sanitized_target_object:
        sanitized_target_object = "target"

    base_mission_name = f"{sanitized_target_object}_mission{mission_num}"
    mission_dir = dataset_root / category / env_name / base_mission_name
    if not mission_dir.exists():
        return base_mission_name, sanitized_target_object

    suffix = 2
    while True:
        candidate_target_object = f"{sanitized_target_object}_{suffix}"
        candidate_mission_name = f"{candidate_target_object}_mission{mission_num}"
        candidate_dir = dataset_root / category / env_name / candidate_mission_name
        if not candidate_dir.exists():
            return candidate_mission_name, candidate_target_object
        suffix += 1


def detect_airsim_root() -> Optional[pathlib.Path]:
    home_dir = pathlib.Path.home()
    candidate_documents = [home_dir / "Documents", home_dir / "documents"]

    for documents_dir in candidate_documents:
        candidate = documents_dir / "AirSim"
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


def find_latest_airsim_rec_txt(root_dir: pathlib.Path) -> Optional[pathlib.Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return None

    candidates = [path for path in root_dir.rglob("airsim_rec.txt") if path.is_file()]
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


def prompt_airsim_root() -> pathlib.Path:
    auto_detected = detect_airsim_root()

    if auto_detected is not None:
        print(f"\nAuto-detected AirSim root in Documents: {auto_detected}")
        if prompt_yes_no("Use this AirSim root? (y/n): "):
            return auto_detected

    while True:
        root_str = prompt_non_empty("Enter AirSim root directory path: ")
        root_path = pathlib.Path(root_str).expanduser().resolve()
        if root_path.exists() and root_path.is_dir():
            return root_path
        print("Provided AirSim root path does not exist or is not a directory.")


def prompt_airsim_rec_txt_path(root_dir: pathlib.Path) -> pathlib.Path:
    latest_rec_txt = find_latest_airsim_rec_txt(root_dir)

    if latest_rec_txt is not None:
        latest_recording_dir = latest_rec_txt.parent
        print(f"\nLatest detected recording folder: {latest_recording_dir}")
        if prompt_yes_no("Proceed with this recording folder? (y/n): "):
            return latest_recording_dir

    while True:
        manual_path_str = prompt_non_empty("Enter recording folder path (folder containing airsim_rec.txt): ")
        manual_path = pathlib.Path(manual_path_str).expanduser().resolve()
        rec_txt_path = manual_path / "airsim_rec.txt"
        if manual_path.exists() and manual_path.is_dir() and rec_txt_path.exists() and rec_txt_path.is_file():
            return manual_path
        print("Path is invalid. Please provide a valid folder containing 'airsim_rec.txt'.")


def copy_recording_assets(recording_dir_path: pathlib.Path, destination_dir: pathlib.Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)

    recording_txt_path = recording_dir_path / "airsim_rec.txt"
    if not recording_txt_path.exists() or not recording_txt_path.is_file():
        raise FileNotFoundError(f"Missing airsim_rec.txt in recording folder: {recording_dir_path}")

    rec_txt_dest = destination_dir / "airsim_rec.txt"
    shutil.copy2(recording_txt_path, rec_txt_dest)
    print(f"Copied recording file: {rec_txt_dest}")

    source_images_dir = recording_dir_path / "images"
    if source_images_dir.exists() and source_images_dir.is_dir():
        dest_images_dir = destination_dir / "images"
        if dest_images_dir.exists():
            shutil.rmtree(dest_images_dir)
        shutil.copytree(source_images_dir, dest_images_dir)
        print(f"Copied images folder: {dest_images_dir}")
    else:
        print(f"No images folder found in recording folder at: {source_images_dir}")


def _category_aliases_for_drive_keys(category: Optional[str]) -> list[str]:
    if not category:
        return []

    normalized = category.strip()
    compact = "".join(ch for ch in normalized if ch.isalnum() or ch == "_")
    upper_compact = compact.upper()

    aliases = [compact, upper_compact]
    category_map = {
        "Visual_Inspection": ["VI", "VISUAL_INSPECTION"],
        "Patrol": ["P", "PATROL"],
        "Manipulation": ["M", "MANIPULATION"],
    }
    aliases.extend(category_map.get(normalized, []))

    unique_aliases: list[str] = []
    seen = set()
    for alias in aliases:
        if alias and alias not in seen:
            unique_aliases.append(alias)
            seen.add(alias)
    return unique_aliases


def _get_google_drive_destination_from_env(env_name: Optional[str] = None, category: Optional[str] = None) -> Optional[str]:
    env_candidates: list[str] = []
    category_aliases = _category_aliases_for_drive_keys(category)

    if env_name:
        normalized = "".join(ch for ch in env_name if ch.isalnum() or ch == "_")
        env_aliases = [normalized, normalized.upper()]
        for env_alias in env_aliases:
            for category_alias in category_aliases:
                env_candidates.extend(
                    [
                        f"GOOGLE_DRIVE_FOLDER_URL_{env_alias}_{category_alias}",
                        f"GOOGLE_DRIVE_DESTINATION_{env_alias}_{category_alias}",
                    ]
                )

            env_candidates.extend(
                [
                    f"GOOGLE_DRIVE_FOLDER_URL_{env_alias}",
                    f"GOOGLE_DRIVE_DESTINATION_{env_alias}",
                ]
            )

    for category_alias in category_aliases:
        env_candidates.extend(
            [
                f"GOOGLE_DRIVE_FOLDER_URL_{category_alias}",
                f"GOOGLE_DRIVE_DESTINATION_{category_alias}",
            ]
        )

    env_candidates.extend(["GOOGLE_DRIVE_FOLDER_URL", "GOOGLE_DRIVE_DESTINATION"])

    for env_key in env_candidates:
        value = os.getenv(env_key)
        if value and value.strip():
            return value.strip()

    return None


def resolve_google_drive_destination_for_mission(
    env_name: str,
    category: str,
    configured_destination: Optional[str],
) -> Optional[str]:
    if configured_destination and configured_destination.strip():
        return configured_destination.strip()

    return _get_google_drive_destination_from_env(env_name=env_name, category=category)


def prompt_google_drive_sync_destination(env_name: Optional[str] = None) -> Optional[str]:
    if not prompt_yes_no("Sync each created mission folder to Google Drive? (y/n): "):
        return None

    env_destination = _get_google_drive_destination_from_env(env_name=env_name)
    if env_destination:
        env_destination = env_destination.strip()
        if _extract_google_drive_folder_id(env_destination) is not None:
            print(f"Using Google Drive destination from .env: {env_destination}")
            return env_destination

        env_destination_dir = pathlib.Path(env_destination).expanduser().resolve()
        if env_destination_dir.exists() and env_destination_dir.is_dir():
            print(f"Using Google Drive destination from .env: {env_destination_dir}")
            return str(env_destination_dir)

        print("Google Drive destination in .env is invalid; please provide it manually.")

    while True:
        destination_str = input(
            "Enter Google Drive destination (local mounted path OR Drive folder URL/ID), or press Enter to use per-category .env mapping: "
        ).strip()
        if not destination_str:
            return ""
        if _extract_google_drive_folder_id(destination_str) is not None:
            return destination_str

        destination_dir = pathlib.Path(destination_str).expanduser().resolve()
        if destination_dir.exists() and destination_dir.is_dir():
            return str(destination_dir)

        print("Invalid destination. Provide either a valid local directory path or a Google Drive folder URL/ID.")


def _extract_google_drive_folder_id(destination: str) -> Optional[str]:
    destination = destination.strip()

    folder_match = re.search(r"/folders/([a-zA-Z0-9_-]+)", destination)
    if folder_match:
        return folder_match.group(1)

    id_param_match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", destination)
    if id_param_match:
        return id_param_match.group(1)

    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", destination):
        return destination

    return None


def _build_drive_service():
    scopes = ["https://www.googleapis.com/auth/drive"]

    oauth_client_secret_path = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE")
    oauth_token_path = os.getenv("GOOGLE_DRIVE_OAUTH_TOKEN_FILE")
    if oauth_client_secret_path:
        return _build_drive_service_with_oauth(scopes, oauth_client_secret_path, oauth_token_path)

    return _build_drive_service_with_service_account(scopes)


def _build_drive_service_with_service_account(scopes: list[str]):
    try:
        service_account_module = importlib.import_module("google.oauth2.service_account")
        discovery_module = importlib.import_module("googleapiclient.discovery")
    except ImportError as exc:
        raise ImportError(
            "Google Drive upload requires 'google-api-python-client' and 'google-auth'."
        ) from exc

    credential_path = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credential_path:
        raise ValueError(
            "Set GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE (or GOOGLE_APPLICATION_CREDENTIALS) to your service-account JSON file."
        )

    credential_file = pathlib.Path(credential_path).expanduser().resolve()
    if not credential_file.exists() or not credential_file.is_file():
        raise FileNotFoundError(f"Service-account credential file not found: {credential_file}")

    credentials = service_account_module.Credentials.from_service_account_file(str(credential_file), scopes=scopes)
    return discovery_module.build("drive", "v3", credentials=credentials)


def _build_drive_service_with_oauth(scopes: list[str], client_secret_path: str, token_path: Optional[str] = None):
    try:
        credentials_module = importlib.import_module("google.oauth2.credentials")
        request_module = importlib.import_module("google.auth.transport.requests")
        flow_module = importlib.import_module("google_auth_oauthlib.flow")
        discovery_module = importlib.import_module("googleapiclient.discovery")
    except ImportError as exc:
        raise ImportError(
            "Google Drive OAuth requires 'google-auth-oauthlib', 'google-auth', and 'google-api-python-client'."
        ) from exc

    client_secret_file = pathlib.Path(client_secret_path).expanduser().resolve()
    if not client_secret_file.exists() or not client_secret_file.is_file():
        raise FileNotFoundError(f"OAuth client secret file not found: {client_secret_file}")

    if token_path:
        token_file = pathlib.Path(token_path).expanduser().resolve()
    else:
        missionbench_workspace_root = pathlib.Path(__file__).resolve().parents[1]
        token_file = missionbench_workspace_root / ".gdrive_oauth_token.json"

    credentials = None
    credentials_cls = getattr(credentials_module, "Credentials")
    if token_file.exists() and token_file.is_file():
        credentials = credentials_cls.from_authorized_user_file(str(token_file), scopes)

    if not credentials or not getattr(credentials, "valid", False):
        if credentials and getattr(credentials, "expired", False) and getattr(credentials, "refresh_token", None):
            request_cls = getattr(request_module, "Request")
            try:
                credentials.refresh(request_cls())
            except Exception:
                # Refresh tokens can be revoked/expired server-side (invalid_grant);
                # remove stale token and force a fresh OAuth login.
                credentials = None
                try:
                    token_file.unlink(missing_ok=True)
                except OSError:
                    pass

        if not credentials or not getattr(credentials, "valid", False):
            flow_cls = getattr(flow_module, "InstalledAppFlow")
            flow = flow_cls.from_client_secrets_file(str(client_secret_file), scopes)
            credentials = flow.run_local_server(port=0)

        token_file.parent.mkdir(parents=True, exist_ok=True)
        with open(token_file, "w", encoding="utf-8") as token_writer:
            token_writer.write(credentials.to_json())

    return discovery_module.build("drive", "v3", credentials=credentials)


def _get_service_account_email() -> Optional[str]:
    credential_path = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credential_path:
        return None

    credential_file = pathlib.Path(credential_path).expanduser().resolve()
    if not credential_file.exists() or not credential_file.is_file():
        return None

    try:
        with open(credential_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        email = payload.get("client_email")
        if isinstance(email, str) and email.strip():
            return email.strip()
    except Exception:
        return None
    return None


def _ensure_drive_folder(drive_service, parent_folder_id: str, folder_name: str) -> str:
    safe_name = folder_name.replace("'", "\\'")
    query = (
        f"name = '{safe_name}' and "
        f"'{parent_folder_id}' in parents and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    response = drive_service.files().list(
        q=query,
        spaces="drive",
        fields="files(id,name)",
        pageSize=1,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute()
    existing = response.get("files", [])
    if existing:
        return existing[0]["id"]

    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    created = drive_service.files().create(
        body=metadata,
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def _upload_file_to_drive(drive_service, local_file: pathlib.Path, parent_folder_id: str) -> None:
    try:
        http_module = importlib.import_module("googleapiclient.http")
    except ImportError as exc:
        raise ImportError(
            "Google Drive upload requires 'google-api-python-client'."
        ) from exc

    media_file_upload = getattr(http_module, "MediaFileUpload")

    metadata = {"name": local_file.name, "parents": [parent_folder_id]}
    media = media_file_upload(str(local_file), resumable=True)
    drive_service.files().create(
        body=metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()


def _should_skip_results_sync_file(local_file: pathlib.Path) -> bool:
    """Skip raw image artifacts during results sync; keep stitched outputs like .webp."""
    image_suffixes = {
        ".png",
        ".jpg",
        ".jpeg"
    }
    suffix = local_file.suffix.lower()
    return suffix in image_suffixes


def _count_files_for_upload(local_dir: pathlib.Path) -> int:
    file_count = 0
    for child in local_dir.rglob("*"):
        if child.is_file() and not _should_skip_results_sync_file(child):
            file_count += 1
    return file_count


def _upload_file_to_drive_with_progress(
    drive_service,
    local_file: pathlib.Path,
    parent_folder_id: str,
    progress_state: dict[str, int],
) -> None:
    _upload_file_to_drive(drive_service, local_file, parent_folder_id)
    progress_state["uploaded_files"] += 1
    uploaded = progress_state["uploaded_files"]
    total = progress_state["total_files"]
    filename = local_file.name
    if len(filename) > 48:
        filename = f"...{filename[-45:]}"
    status_text = f"[Drive Upload] {uploaded}/{total} files uploaded | latest: {filename}"
    previous_length = progress_state.get("last_line_length", 0)
    padding = " " * max(0, previous_length - len(status_text))
    print(f"\r{status_text}{padding}", end="", flush=True)
    progress_state["last_line_length"] = len(status_text)
    if uploaded == total:
        print()


def _upload_directory_to_drive(
    drive_service,
    local_dir: pathlib.Path,
    parent_folder_id: str,
    progress_state: Optional[dict[str, int]] = None,
) -> str:
    current_folder_id = _ensure_drive_folder(drive_service, parent_folder_id, local_dir.name)

    if progress_state is None:
        total_files = _count_files_for_upload(local_dir)
        progress_state = {"uploaded_files": 0, "total_files": total_files, "last_line_length": 0}

    for child in local_dir.iterdir():
        if child.is_dir():
            _upload_directory_to_drive(drive_service, child, current_folder_id, progress_state)
        elif child.is_file():
            if _should_skip_results_sync_file(child):
                continue
            _upload_file_to_drive_with_progress(drive_service, child, current_folder_id, progress_state)
    return current_folder_id


def _build_sync_ignore_for_copy(src_dir: str, names: list[str]) -> set[str]:
    """Ignore raw image files when syncing to local destination."""
    src_path = pathlib.Path(src_dir)
    ignored: set[str] = set()
    for name in names:
        child = src_path / name
        if child.is_file() and _should_skip_results_sync_file(child):
            ignored.add(name)
    return ignored


def validate_google_drive_destination_for_sync(google_drive_destination: str) -> None:
    destination = google_drive_destination.strip()
    folder_id = _extract_google_drive_folder_id(destination)

    if folder_id is None:
        local_destination_dir = pathlib.Path(destination).expanduser().resolve()
        if not local_destination_dir.exists() or not local_destination_dir.is_dir():
            raise FileNotFoundError(
                f"Google Drive destination folder/ID invalid or local path not found: {destination}"
            )

    drive_service = _build_drive_service()
    try:
        folder_info = drive_service.files().get(
            fileId=folder_id,
            fields="id,name,driveId,capabilities(canAddChildren)",
            supportsAllDrives=True,
        ).execute()

        if not folder_info:
            raise FileNotFoundError(f"Drive folder not found: {folder_id}")

        print(f"✓ Validated Google Drive target: {folder_info.get('name')} ({folder_id})")

    except Exception as e:
        raise ValueError(
            f"Failed to validate Google Drive destination {folder_id}. Check permissions/auth."
        ) from e


def upload_or_update_file_in_drive(local_path: str, folder_id: str) -> None:
    """Uploads a file to Google Drive folder, updating it if it already exists."""
    local_file = pathlib.Path(local_path)
    if not local_file.exists():
        print(f"File not found: {local_file}")
        return

    try:
        drive_service = _build_drive_service()
        http_module = importlib.import_module("googleapiclient.http")
        media_file_upload = getattr(http_module, "MediaFileUpload")
        
        # Check if file exists
        query = (
            f"name = '{local_file.name}' and "
            f"'{folder_id}' in parents and "
            "trashed = false"
        )
        response = drive_service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            pageSize=1,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        files = response.get("files", [])

        media = media_file_upload(str(local_file), resumable=True)

        if files:
            # Update existing file
            file_id = files[0]["id"]
            drive_service.files().update(
                fileId=file_id,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()
        else:
            # Create new file
            metadata = {"name": local_file.name, "parents": [folder_id]}
            drive_service.files().create(
                body=metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()


    except Exception as e:
        print(f"Failed to upload/update file on Drive: {e}")


def sync_mission_folder_to_google_drive(mission_folder: pathlib.Path, google_drive_destination: str) -> str:
    destination = google_drive_destination.strip()
    folder_id = _extract_google_drive_folder_id(destination)

    if folder_id is not None:
        total_files = _count_files_for_upload(mission_folder)
        print(f"[Drive Upload] Starting upload: {mission_folder.name} ({total_files} files)")
        drive_service = _build_drive_service()
        try:
            progress_state = {"uploaded_files": 0, "total_files": total_files}
            uploaded_folder_id = _upload_directory_to_drive(drive_service, mission_folder, folder_id, progress_state)
            return f"https://drive.google.com/drive/folders/{uploaded_folder_id}"
        except Exception as exc:
            error_text = str(exc)
            quota_error = (
                "storageQuotaExceeded" in error_text
                or "Service Accounts do not have storage quota" in error_text
            )
            if quota_error:
                oauth_client_secret_path = os.getenv("GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE")
                if oauth_client_secret_path:
                    oauth_token_path = os.getenv("GOOGLE_DRIVE_OAUTH_TOKEN_FILE")
                    oauth_drive_service = _build_drive_service_with_oauth(
                        ["https://www.googleapis.com/auth/drive"],
                        oauth_client_secret_path,
                        oauth_token_path,
                    )
                    print("[Drive Upload] Retrying with OAuth user credentials...")
                    progress_state = {"uploaded_files": 0, "total_files": total_files}
                    uploaded_folder_id = _upload_directory_to_drive(oauth_drive_service, mission_folder, folder_id, progress_state)
                    return f"https://drive.google.com/drive/folders/{uploaded_folder_id}"

                raise PermissionError(
                    "Google Drive upload failed because service-account storage quota is unavailable for this destination. "
                    "Use a Shared Drive destination or configure OAuth user credentials with "
                    "GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE (and optional GOOGLE_DRIVE_OAUTH_TOKEN_FILE)."
                ) from exc

            if "insufficientParentPermissions" in error_text or "Insufficient permissions for the specified parent" in error_text:
                service_account_email = _get_service_account_email()
                email_hint = (
                    f"Service account email: {service_account_email}. "
                    if service_account_email is not None
                    else ""
                )
                raise PermissionError(
                    "Google Drive upload failed due to parent folder permissions. "
                    f"Share the destination folder with Editor/Content manager access for the service account. {email_hint}"
                    f"Destination: {google_drive_destination}"
                ) from exc
            raise

    local_destination_dir = pathlib.Path(destination).expanduser().resolve()
    if not local_destination_dir.exists() or not local_destination_dir.is_dir():
        raise FileNotFoundError(f"Local destination for Google Drive sync does not exist or is not a directory: {local_destination_dir}")

    destination_dir = local_destination_dir / mission_folder.name
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(mission_folder, destination_dir, ignore=_build_sync_ignore_for_copy)
    return str(destination_dir)


def sync_folder_to_configured_google_drive(
    folder_path: str | pathlib.Path,
    google_drive_destination: Optional[str] = None,
) -> str:
    """Sync an arbitrary local folder to Google Drive using configured destination.

    Args:
        folder_path: Local folder to upload.
        google_drive_destination: Optional destination override. When omitted,
            uses GOOGLE_DRIVE_RESULTS_FOLDER_URL from environment.

    Returns:
        Uploaded folder URL (Drive mode) or destination path (local mode).
    """
    source_folder = pathlib.Path(folder_path).expanduser().resolve()
    if not source_folder.exists() or not source_folder.is_dir():
        raise FileNotFoundError(f"Source folder does not exist or is not a directory: {source_folder}")

    destination = (google_drive_destination or os.getenv("GOOGLE_DRIVE_RESULTS_FOLDER_URL", "")).strip()
    if not destination:
        raise ValueError(
            "Google Drive destination is not configured. "
            "Set GOOGLE_DRIVE_RESULTS_FOLDER_URL in .env or pass google_drive_destination explicitly."
        )

    validate_google_drive_destination_for_sync(destination)
    return sync_mission_folder_to_google_drive(source_folder, destination)


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync a local folder to Google Drive using MissionBench utilities."
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to local folder to upload.",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default=None,
        help=(
            "Optional destination override (Google Drive folder URL/ID or local path). "
            "Defaults to GOOGLE_DRIVE_RESULTS_FOLDER_URL from .env"
        ),
    )
    args = parser.parse_args()

    try:
        uploaded_to = sync_folder_to_configured_google_drive(
            folder_path=args.folder_path,
            google_drive_destination=args.destination,
        )
        print(f"[Drive Upload] Completed: {uploaded_to}")
        return 0
    except Exception as exc:
        print(f"[Drive Upload] Failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
