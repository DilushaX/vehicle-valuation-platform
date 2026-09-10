import fcntl
import logging
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


class CollectionLock:
    """
    Lightweight filesystem/process lock using POSIX fcntl.flock to prevent
    concurrent or overlapping scheduled vehicle collection runs.

    Guarantees:
    - Atomic non-blocking acquisition (LOCK_EX | LOCK_NB)
    - Records holding PID, host, and start timestamp in lockfile
    - Safe cleanup on normal termination or unhandled exception
    - Automatic OS-level lock release upon process termination/crash
    """

    def __init__(self, lock_file_path: Optional[Path] = None):
        self.lock_file_path = (
            Path(lock_file_path)
            if lock_file_path
            else getattr(settings, "LOCK_FILE_PATH", Path("data/.collection.lock"))
        )
        self.file_obj: Optional[object] = None
        self.is_locked: bool = False

    def acquire(self) -> bool:
        """
        Attempts to acquire an exclusive, non-blocking lock.
        Returns True if acquired successfully, False if already held by another process.
        """
        try:
            self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_obj = open(self.lock_file_path, "a+")

            # Attempt exclusive non-blocking lock
            fcntl.flock(self.file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Lock acquired: record metadata
            self.file_obj.seek(0)
            self.file_obj.truncate(0)
            now_iso = datetime.now(timezone.utc).isoformat()
            self.file_obj.write(
                f"pid={os.getpid()}\n"
                f"started_at={now_iso}\n"
                f"host={socket.gethostname()}\n"
                f"process=scheduled_collection\n"
            )
            self.file_obj.flush()
            self.is_locked = True
            logger.debug(f"Collection lock acquired: {self.lock_file_path} (PID {os.getpid()})")
            return True

        except (BlockingIOError, IOError, OSError):
            # Lock is held by an active process
            lock_info = self.get_lock_info()
            pid_str = lock_info.get("pid", "unknown")
            started_str = lock_info.get("started_at", "unknown")
            logger.warning(
                f"Collection lock already held by PID {pid_str} (started at {started_str}). "
                "Another collection run is currently active."
            )
            if self.file_obj:
                try:
                    self.file_obj.close()
                except Exception:
                    pass
                self.file_obj = None
            self.is_locked = False
            return False

    def release(self):
        """Releases the lock and removes the lockfile."""
        if not self.is_locked:
            if self.file_obj:
                try:
                    self.file_obj.close()
                except Exception:
                    pass
                self.file_obj = None
            return

        try:
            if self.file_obj:
                fcntl.flock(self.file_obj.fileno(), fcntl.LOCK_UN)
                self.file_obj.close()
                self.file_obj = None
        except Exception as e:
            logger.debug(f"Error unlocking file descriptor: {e}")
        finally:
            self.is_locked = False
            try:
                if self.lock_file_path.exists():
                    self.lock_file_path.unlink()
            except Exception as e:
                logger.debug(f"Error removing lock file {self.lock_file_path}: {e}")

    def get_lock_info(self) -> dict:
        """Reads lockholder metadata from the lockfile if accessible."""
        info = {}
        if not self.lock_file_path.exists():
            return info
        try:
            content = self.lock_file_path.read_text().strip()
            for line in content.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k.strip()] = v.strip()
        except Exception:
            pass
        return info

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
