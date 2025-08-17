import pytest
import subprocess
import time
from unittest.mock import MagicMock, patch

from app.tools.python_runner import run_python_with_uv, run_python_with_packages


@pytest.fixture
def mock_popen():
    """Fixture to mock subprocess.Popen."""
    mock_proc = MagicMock()
    mock_proc.communicate.return_value = (b"output", b"")
    mock_proc.returncode = 0
    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        yield mock_popen, mock_proc


def test_run_python_with_uv_success(mock_popen):
    """
    Test successful execution of Python code with uv.
    """
    mock_popen_fn, mock_proc = mock_popen
    code = "print('hello')"
    result = run_python_with_uv(code, time.time(), 60)

    assert result["success"]
    assert result["output"] == "output"
    mock_popen_fn.assert_called_once()
    args, kwargs = mock_popen_fn.call_args
    assert "uv" in args[0]
    assert "run" in args[0]
    assert "--isolated" in args[0]


def test_run_python_with_uv_failure(mock_popen):
    """
    Test failed execution of Python code with uv.
    """
    mock_popen_fn, mock_proc = mock_popen
    mock_proc.returncode = 1
    mock_proc.communicate.return_value = (b"", b"error")

    code = "print('hello')"
    result = run_python_with_uv(code, time.time(), 60)

    assert not result["success"]
    assert "error" in result["error"]


def test_run_python_with_uv_timeout(mock_popen):
    """
    Test timeout during Python code execution with uv.
    """
    mock_popen_fn, mock_proc = mock_popen
    mock_proc.communicate.side_effect = subprocess.TimeoutExpired(cmd="uv", timeout=1)

    code = "import time; time.sleep(2)"
    result = run_python_with_uv(code, time.time(), 60)

    assert not result["success"]
    assert "timed out" in result["error"]


def test_run_python_with_packages_success(mock_popen):
    """
    Test successful execution of Python code with packages.
    """
    mock_popen_fn, mock_proc = mock_popen
    code = "import pandas; print('hello')"
    packages = ["pandas", "numpy"]
    result = run_python_with_packages(code, packages, time.time(), 60)

    assert result["success"]
    assert result["output"] == "output"
    mock_popen_fn.assert_called_once()
    args, kwargs = mock_popen_fn.call_args
    assert "--with" in args[0]
    assert "pandas" in args[0]
    assert "numpy" in args[0]


def test_run_python_with_packages_failure(mock_popen):
    """
    Test failed execution of Python code with packages.
    """
    mock_popen_fn, mock_proc = mock_popen
    mock_proc.returncode = 1
    mock_proc.communicate.return_value = (b"", b"error")

    code = "import pandas; print('hello')"
    packages = ["pandas"]
    result = run_python_with_packages(code, packages, time.time(), 60)

    assert not result["success"]
    assert "error" in result["error"]
