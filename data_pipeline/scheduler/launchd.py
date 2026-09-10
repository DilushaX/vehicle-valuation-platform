import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

LAUNCHD_LABEL = "com.vehicle_valuation.collection"


def parse_schedule_time(time_str: str) -> Tuple[int, int]:
    """
    Parses and validates a 24-hour HH:MM time string into (hour, minute).
    Example: '02:00' -> (2, 0), '23:30' -> (23, 30).
    """
    match = re.match(r"^(\d{1,2}):(\d{2})$", time_str.strip())
    if not match:
        raise ValueError(
            f"Invalid time format: '{time_str}'. Must be in 24-hour 'HH:MM' format (e.g. '02:00')."
        )
    hour = int(match.group(1))
    minute = int(match.group(2))

    if not (0 <= hour <= 23):
        raise ValueError(f"Hour must be between 0 and 23. Got: {hour}")
    if not (0 <= minute <= 59):
        raise ValueError(f"Minute must be between 0 and 59. Got: {minute}")

    return hour, minute


def get_plist_path() -> Path:
    """Returns the default user LaunchAgents plist path on macOS."""
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def generate_launchd_plist(
    python_bin: Optional[Path] = None,
    project_dir: Optional[Path] = None,
    collection_time: Optional[str] = None,
) -> str:
    """
    Generates a standard macOS LaunchAgent property list (XML) configured to
    run the vehicle collection script daily at the specified time.
    """
    proj_dir = Path(project_dir or Path(__file__).resolve().parent.parent.parent).resolve()
    py_bin = Path(python_bin or (proj_dir / ".venv" / "bin" / "python"))
    script_path = proj_dir / "scripts" / "scheduled_collection.py"
    time_str = collection_time or settings.COLLECTION_TIME
    hour, minute = parse_schedule_time(time_str)

    log_dir = proj_dir / "logs"
    stdout_log = log_dir / "scheduled_collection.log"
    stderr_log = log_dir / "scheduled_collection_error.log"

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{py_bin}</string>
        <string>{script_path}</string>
    </array>

    <key>WorkingDirectory</key>
    <string>{proj_dir}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>{proj_dir}</string>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{proj_dir / ".venv" / "bin"}</string>
    </dict>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{stdout_log}</string>

    <key>StandardErrorPath</key>
    <string>{stderr_log}</string>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""
    return plist_content


def install_launchd_agent(
    python_bin: Optional[Path] = None,
    project_dir: Optional[Path] = None,
    collection_time: Optional[str] = None,
) -> Path:
    """
    Installs and loads the LaunchAgent plist into ~/Library/LaunchAgents.
    """
    plist_content = generate_launchd_plist(
        python_bin=python_bin,
        project_dir=project_dir,
        collection_time=collection_time,
    )
    plist_path = get_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure log directory exists
    proj_dir = Path(project_dir or Path(__file__).resolve().parent.parent.parent).resolve()
    (proj_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Unload existing agent if currently loaded
    if plist_path.exists():
        try:
            subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
        except Exception:
            pass

    # Write plist
    plist_path.write_text(plist_content)
    logger.info(f"Generated LaunchAgent plist at: {plist_path}")

    # Load agent
    res = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)
    if res.returncode != 0:
        logger.warning(f"launchctl load warning: {res.stderr.strip()}")
    else:
        logger.info(f"Successfully loaded LaunchAgent: {LAUNCHD_LABEL}")

    return plist_path


def uninstall_launchd_agent() -> bool:
    """
    Unloads and removes the LaunchAgent plist from ~/Library/LaunchAgents.
    """
    plist_path = get_plist_path()
    if not plist_path.exists():
        logger.info(f"LaunchAgent plist not found at: {plist_path}")
        return False

    # Unload
    try:
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
    except Exception as e:
        logger.warning(f"Error unloading launchctl agent: {e}")

    # Remove
    try:
        plist_path.unlink()
        logger.info(f"Removed LaunchAgent plist at: {plist_path}")
        return True
    except Exception as e:
        logger.error(f"Error deleting plist {plist_path}: {e}")
        return False


def get_launchd_status() -> Dict[str, object]:
    """
    Checks the status of the macOS LaunchAgent.
    """
    plist_path = get_plist_path()
    is_installed = plist_path.exists()
    is_loaded = False

    if is_installed:
        try:
            out = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
            is_loaded = LAUNCHD_LABEL in out.stdout
        except Exception:
            is_loaded = False

    return {
        "label": LAUNCHD_LABEL,
        "plist_path": str(plist_path),
        "is_installed": is_installed,
        "is_loaded": is_loaded,
        "configured_time": settings.COLLECTION_TIME,
        "schedule": settings.COLLECTION_SCHEDULE,
    }
