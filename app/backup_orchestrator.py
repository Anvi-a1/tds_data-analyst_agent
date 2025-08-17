import os
import uuid
import shutil
import threading
import time
import logging
from typing import Dict, Optional, Any

from app.tools.backup_script import run_full_workflow

logger = logging.getLogger(__name__)


class BackupJob:
    def __init__(
        self,
        question_text: str,
        file_paths: Dict[str, str],
        base_root: Optional[str] = None,
    ):
        self.question_text = question_text
        self.file_paths = dict(file_paths or {})
        self.base_root = base_root or os.path.abspath(
            os.path.join("agent_workspaces", "backup_requests")
        )
        self.request_id = str(uuid.uuid4())[:8]
        self.request_dir = os.path.join(self.base_root, f"req_{self.request_id}")
        self._thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[Exception] = None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        finished = self._done.wait(timeout=timeout)
        if not finished:
            return None
        if self._error:
            logger.error(f"Backup job {self.request_id} failed: {self._error}")
            return None
        return self._result

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"backup-job-{self.request_id}", daemon=True
        )
        self._thread.start()

    def _prepare_workspace(self):
        os.makedirs(self.request_dir, exist_ok=True)
        q_path = os.path.join(self.request_dir, "questions.txt")
        with open(q_path, "w", encoding="utf-8") as f:
            f.write(self.question_text or "")

        for name, abs_path in self.file_paths.items():
            try:
                fname = (
                    name
                    if os.path.basename(name) == name
                    else os.path.basename(abs_path)
                )
                dst = os.path.join(self.request_dir, fname)
                if os.path.isfile(abs_path):
                    shutil.copy2(abs_path, dst)
            except Exception as e:
                logger.warning(
                    f"Failed to copy file '{name}' from '{abs_path}' to backup workspace: {e}"
                )

    def _run(self):
        try:
            self._prepare_workspace()

            start = time.time()
            logger.info(
                f"▶️  Starting backup workflow for req {self.request_id} (workspace: {self.request_dir})"
            )
            self._result = run_full_workflow(
                base_dir=self.request_dir, use_temp_workspace=True
            )
            elapsed = time.time() - start
            logger.info(
                f"✅ Backup workflow completed for req {self.request_id} in {elapsed:.2f}s"
            )
        except Exception as e:
            self._error = e
            logger.exception(f"💥 Backup workflow error for req {self.request_id}: {e}")
        finally:
            self._done.set()


class BackupOrchestrator:
    """
    Fire-and-forget wrapper around backup_script.run_full_workflow with async fallback retrieval.
    """

    def __init__(self, base_root: Optional[str] = None):
        self.base_root = base_root or os.path.abspath(
            os.path.join("agent_workspaces", "backup_requests")
        )
        os.makedirs(self.base_root, exist_ok=True)

    def start_async(self, question_text: str, file_paths: Dict[str, str]) -> BackupJob:
        job = BackupJob(
            question_text=question_text, file_paths=file_paths, base_root=self.base_root
        )
        job.start()
        return job
