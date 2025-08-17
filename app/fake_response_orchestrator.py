import os
import uuid
import threading
import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class FakeResponseJob:
    def __init__(self, question_text: str, base_root: Optional[str] = None):
        self.question_text = question_text or ""
        self.base_root = base_root or os.path.abspath(
            os.path.join("agent_workspaces", "fake_requests")
        )
        self.request_id = str(uuid.uuid4())[:8]
        self.request_dir = os.path.join(self.base_root, f"req_{self.request_id}")
        self._thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._result: Optional[Any] = None
        self._error: Optional[Exception] = None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def result(self, timeout: Optional[float] = None) -> Optional[Any]:
        finished = self._done.wait(timeout=timeout)
        if not finished:
            return None
        if self._error:
            logger.error(f"FakeResponse job {self.request_id} failed: {self._error}")
            return None
        return self._result

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"fake-response-job-{self.request_id}", daemon=True
        )
        self._thread.start()

    def _prepare_workspace(self) -> str:
        os.makedirs(self.request_dir, exist_ok=True)
        q_path = os.path.join(self.request_dir, "questions.txt")
        with open(q_path, "w", encoding="utf-8") as f:
            f.write(self.question_text)
        return q_path

    def _run(self):
        try:
            q_path = self._prepare_workspace()
            # Import here to avoid import-time issues
            try:
                from app.tools.fake_response import main as fake_main
            except Exception as e:
                self._error = e
                logger.exception(f"Failed to import fake_response module: {e}")
                return

            logger.info(
                f"▶️  Starting fake_response for req {self.request_id} (workspace: {self.request_dir})"
            )
            self._result = fake_main(q_path)
            logger.info(f"✅ fake_response completed for req {self.request_id}")
        except Exception as e:
            self._error = e
            logger.exception(f"💥 fake_response error for req {self.request_id}: {e}")
        finally:
            self._done.set()


class FakeResponseOrchestrator:
    def __init__(self, base_root: Optional[str] = None):
        self.base_root = base_root or os.path.abspath(
            os.path.join("agent_workspaces", "fake_requests")
        )
        os.makedirs(self.base_root, exist_ok=True)

    def start_async(self, question_text: str) -> FakeResponseJob:
        job = FakeResponseJob(question_text=question_text, base_root=self.base_root)
        job.start()
        return job
