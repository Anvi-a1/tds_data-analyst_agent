import subprocess
import tempfile
import os
import time
import json
from typing import Any


class SandboxTimeout(Exception):
    pass


class SandboxRunner:
    def __init__(self):
        self.docker_image = os.getenv(
            "SANDBOX_DOCKER_IMAGE", "data-agent-sandbox:latest"
        )

    def run_pandas_code(self, code: str, start_time: float, time_limit: float) -> dict:
        """
        Run the user-generated pandas code in an isolated environment.
        The code should read/write from/to predefined variables or files.
        Return JSON-serializable results.
        """
        script = f"""
import json, sys
import pandas as pd
import numpy as np
result = None
{code}
print(json.dumps(result, default=str))
"""
        return self._run_in_docker(script, start_time, time_limit)

    def run_duckdb_query(self, sql: str, start_time: float, time_limit: float) -> dict:
        script = f"""
import json
import duckdb
con = duckdb.connect(database=':memory:')
df = con.execute({json.dumps(sql)}).df()
records = df.to_dict(orient='records')
print(json.dumps(records, default=str))
"""
        return self._run_in_docker(script, start_time, time_limit)

    def run_plot_code(self, code: str, start_time: float, time_limit: float) -> bytes:
        """
        Run matplotlib code to produce a plot, and return raw PNG bytes.
        The code must save figure to a known path (e.g., 'output.png').
        """
        script = f"""
import matplotlib.pyplot as plt
import numpy as np
{code}
"""
        return self._run_in_docker_and_get_file(
            script, "output.png", start_time, time_limit
        )

    def _run_in_docker(self, script: str, start_time: float, time_limit: float) -> Any:
        """
        Write script to a temp file, run it in Docker with strict CPU/memory/time limits.
        Return parsed JSON output.
        """
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            raise SandboxTimeout()

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(script)

            cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-v",
                f"{os.path.abspath(tmpdir)}:/work:ro",
                "-w",
                "/work",
                "--cpus",
                "1.0",
                "--memory",
                "1g",
                self.docker_image,
                "timeout",
                "30s",  # limit execution time per script
                "python",
                "script.py",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                stdout, stderr = proc.communicate(timeout=max(1, time_limit - elapsed))
            except subprocess.TimeoutExpired:
                proc.kill()
                raise SandboxTimeout("Script execution timed out")
            if proc.returncode != 0:
                raise Exception(f"Sandbox script error: {stderr.decode()}")
            out = stdout.decode().strip()
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                raise Exception(f"Sandbox returned non-JSON: {out}")

    def _run_in_docker_and_get_file(
        self, script: str, filename: str, start_time: float, time_limit: float
    ) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            output_path = os.path.join(tmpdir, filename)
            with open(script_path, "w") as f:
                f.write(script)
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "-v",
                f"{os.path.abspath(tmpdir)}:/work",
                "-w",
                "/work",
                "--cpus",
                "1.0",
                "--memory",
                "1g",
                self.docker_image,
                "timeout",
                "30s",
                "python",
                "script.py",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elapsed = time.time() - start_time
            try:
                stdout, stderr = proc.communicate(timeout=max(1, time_limit - elapsed))
            except subprocess.TimeoutExpired:
                proc.kill()
                raise SandboxTimeout("Plot script timed out")
            if proc.returncode != 0:
                raise Exception(f"Sandbox plot error: {stderr.decode()}")
            # After successful run, read the file
            if not os.path.exists(output_path):
                raise Exception(f"Plot file not found: {filename}")
            with open(output_path, "rb") as f:
                data = f.read()
            return data
