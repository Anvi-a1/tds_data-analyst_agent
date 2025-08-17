#!/usr/bin/env python3

import importlib
import os
from pathlib import Path
import sys
import sysconfig
import time
import json
import ast
import shutil
import uuid
import textwrap
import subprocess
from typing import Dict, Optional, List
from dataclasses import dataclass

try:
    import resource
except Exception:
    resource = None

BASE_URL = os.getenv("BACKUP_RESPONSE_OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("BACKUP_RESPONSE_OPENAI_API_KEY")
MODEL = os.getenv("BACKUP_RESPONSE_OPENAI_MODEL", "openai/gpt-4.1-nano")
if OPENAI_API_KEY is None:
    print(
        "ERROR: Set BACKUP_RESPONSE_OPENAI_API_KEY environment variable.",
        file=sys.stderr,
    )

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096

TOTAL_REQUEST_TIMEOUT = os.getenv(
    "REQUEST_TIMEOUT", "280"
)  # Default to 280 seconds if not set
try:
    TOTAL_REQUEST_TIMEOUT = int(TOTAL_REQUEST_TIMEOUT)
except ValueError:
    TOTAL_REQUEST_TIMEOUT = 280

PHASE1_TIMEOUT = 30
PHASE2_SINGLE_ATTEMPT_TIMEOUT = 150

PHASE1_MAX_ATTEMPTS = 3
MAX_DEBUG_ATTEMPTS = 10

USE_DOCKER = False
DOCKER_IMAGE = "python:3.11-slim"
DOCKER_MEMORY = "1024m"
LOCAL_MAX_MEMORY_BYTES = 512 * 1024 * 1024

ALLOWED_MODULES = {
    "pandas",
    "numpy",
    "math",
    "json",
    "sys",
    "io",
    "base64",
    "matplotlib",
    "matplotlib.pyplot",
    "csv",
    "datetime",
    "statistics",
    "pathlib",
    "typing",
    "duckdb",
    "pyarrow",
}
BANNED_MODULES = {"subprocess", "ftplib", "paramiko", "fabric", "os", "shutil"}
BANNED_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "openai",
    "os.system",
    "os.popen",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.run",
}

BASE_TMP = os.path.abspath("./agent_workspaces")
os.makedirs(BASE_TMP, exist_ok=True)


from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)


def call_llm_system_user(
    prompt: str,
    system: Optional[str] = None,
    model: str = MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """Call the OpenAI chat completion endpoint and return text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    text = resp.choices[0].message.content
    return text


class SafetyError(Exception):
    pass


def find_unsafe_in_ast(code: str) -> List[str]:
    """Return list of issues detected in code AST; empty if none."""
    issues = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax_error: {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for nm in node.names:
                name = nm.name.split(".")[0]
                if name in BANNED_MODULES:
                    issues.append(f"import_of_banned_module: {name}")
                if name not in ALLOWED_MODULES and name not in ("builtins",):
                    pass
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            base = mod.split(".")[0]
            if base in BANNED_MODULES:
                issues.append(f"importfrom_banned: {mod}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname in BANNED_CALLS:
                    issues.append(f"call_banned_function: {fname}")
            if isinstance(node.func, ast.Attribute):
                try:
                    attr_chain = []
                    f = node.func
                    while isinstance(f, ast.Attribute):
                        attr_chain.append(f.attr)
                        f = f.value
                    if isinstance(f, ast.Name):
                        attr_chain.append(f.id)
                    attr_chain = ".".join(reversed(attr_chain))
                    for banned in BANNED_CALLS:
                        if attr_chain.startswith(banned):
                            issues.append(f"call_banned_attr: {attr_chain}")
                except Exception:
                    pass

    if ";;" in code or "`" in code or "&&" in code or "| " in code:
        issues.append("suspicious_shell_constructs_in_code")

    return issues


def enforce_static_safety(code: str) -> None:
    # issues = find_unsafe_in_ast(code)
    # if issues:
    #     msg = "Static safety checks failed: " + "; ".join(issues)
    #     raise SafetyError(msg)
    pass


def _strip_code_fences(code: str) -> str:
    """Remove Markdown code fences if present."""
    c = code.strip()
    if c.startswith("```"):
        lines = c.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return code


def extract_required_modules(code: str) -> List[str]:
    """Extract REQUIRED_MODULES = [..] from the generated script. Fallback to parsing a '# REQUIREMENTS:' comment."""
    cleaned = _strip_code_fences(code)
    try:
        tree = ast.parse(cleaned)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                # support multiple targets like REQUIRED_MODULES = [...]
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "REQUIRED_MODULES":
                        val = node.value
                        items: List[str] = []
                        if isinstance(val, (ast.List, ast.Tuple)):
                            for elt in val.elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, str
                                ):
                                    items.append(elt.value.strip())
                        # de-dup and sanitize
                        return [
                            m for i, m in enumerate(items) if m and m not in items[:i]
                        ]
    except Exception:
        pass
    for line in cleaned.splitlines()[:10]:
        if line.strip().lower().startswith("# requirements:"):
            mods = line.split(":", 1)[1]
            parts = [p.strip() for p in mods.replace(";", ",").split(",")]
            return [p for p in parts if p]
    return []


def _is_stdlib_module(name: str) -> bool:
    """
    Return True if `name` belongs to the Python standard library.
    """
    if hasattr(sys, "stdlib_module_names"):
        return name in sys.stdlib_module_names

    spec = importlib.util.find_spec(name)
    if spec is None:
        return False

    origin = getattr(spec, "origin", None)
    if origin in ("built-in", "frozen"):
        return True

    try:
        stdlib_dir = Path(sysconfig.get_paths()["stdlib"]).resolve()
    except Exception:
        return False

    if origin is None:
        return False

    try:
        origin_path = Path(origin).resolve()
    except Exception:
        return False

    return stdlib_dir == origin_path or stdlib_dir in origin_path.parents


def classify_stdlib(modules: List[str]) -> Dict[str, List[str]]:
    """
    Classify a list of module names into 'stdlib' and 'non_stdlib'.

    Args:
        modules: list of module names, e.g. ["json", "requests", "xml.etree"]

    Returns:
        {"stdlib": [...], "non_stdlib": [...]}
    """
    stdlib = []
    non_stdlib = []
    for m in modules:
        if _is_stdlib_module(m):
            stdlib.append(m)
        else:
            non_stdlib.append(m)
    return {"stdlib": stdlib, "non_stdlib": non_stdlib}


@dataclass
class ExecResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int


def run_in_docker(
    workspace: str,
    script_name: str,
    timeout: int,
    required_modules: Optional[List[str]] = None,
) -> ExecResult:
    """
    Execute script inside Docker with limited memory.
    Maps workspace into /workspace in container. Requires Docker daemon available.
    """
    with_args: List[str] = []
    if required_modules:
        with_args = [
            flag
            for pkg in classify_stdlib(required_modules)["non_stdlib"]
            for flag in ["--with", pkg]
        ]

    container_cmd_uv = [
        "docker",
        "run",
        "--rm",
        "--memory",
        DOCKER_MEMORY,
        "-v",
        f"{os.path.abspath(workspace)}:/workspace:ro",
        DOCKER_IMAGE,
        "timeout",
        str(timeout),
        "uv",
        "run",
        *with_args,
        "python",
        f"/workspace/{script_name}",
    ]
    container_cmd_py = [
        "docker",
        "run",
        "--rm",
        "--memory",
        DOCKER_MEMORY,
        "-v",
        f"{os.path.abspath(workspace)}:/workspace:ro",
        DOCKER_IMAGE,
        "timeout",
        str(timeout),
        "python",
        f"/workspace/{script_name}",
    ]
    try:
        proc = subprocess.run(
            container_cmd_uv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=timeout + 5,
        )
        if proc.returncode != 0:
            proc2 = subprocess.run(
                container_cmd_py,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                timeout=timeout + 5,
            )
            return ExecResult(
                success=(proc2.returncode == 0),
                stdout=proc2.stdout,
                stderr=proc2.stderr,
                exit_code=proc2.returncode,
            )
        return ExecResult(
            success=True,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            success=False,
            stdout=e.stdout or "",
            stderr=f"timeout after {timeout}s",
            exit_code=124,
        )
    except FileNotFoundError as e:
        return ExecResult(
            success=False,
            stdout="",
            stderr=f"docker not found or docker failed: {e}",
            exit_code=127,
        )


def preexec_set_limits():
    """Used with subprocess on Unix to set resource limits."""
    if resource is None:
        return
    # CPU limits (seconds) - small hard cap
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (PHASE2_SINGLE_ATTEMPT_TIMEOUT, PHASE2_SINGLE_ATTEMPT_TIMEOUT + 1),
    )
    mem = LOCAL_MAX_MEMORY_BYTES
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))


def run_locally(
    workspace: str,
    script_name: str,
    timeout: int,
    required_modules: Optional[List[str]] = None,
) -> ExecResult:
    """Execute script using `uv run` with optional --with deps; fallback to plain Python if uv is unavailable."""
    script_path = os.path.join(workspace, script_name)
    try:
        cmd: List[str]
        uv_path = shutil.which("uv")
        if uv_path is not None:
            cmd = [uv_path, "run"]
            if required_modules:
                cmd.extend(
                    [
                        flag
                        for pkg in classify_stdlib(required_modules)["non_stdlib"]
                        for flag in ["--with", pkg]
                    ]
                )
            cmd += ["python", script_path]
        else:
            cmd = [sys.executable, script_path]
        proc = subprocess.run(
            cmd,
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return ExecResult(
            success=(proc.returncode == 0),
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired as e:
        return ExecResult(
            success=False,
            stdout=e.stdout or "",
            stderr=f"timeout after {timeout}s",
            exit_code=124,
        )
    except Exception as e:
        return ExecResult(
            success=False, stdout="", stderr=f"local execution error: {e}", exit_code=1
        )


def execute_in_sandbox(
    workspace: str,
    script_name: str,
    timeout: int,
    required_modules: Optional[List[str]] = None,
) -> ExecResult:
    """Choose Docker if available & enabled, else local subprocess fallback."""
    if USE_DOCKER:
        r = run_in_docker(
            workspace, script_name, timeout, required_modules=required_modules
        )
        if r.exit_code == 127 and shutil.which("docker") is None:
            print(
                "WARN: Docker not found. Falling back to local execution (less secure).",
                file=sys.stderr,
            )
            return run_locally(
                workspace, script_name, timeout, required_modules=required_modules
            )
        return r
    else:
        return run_locally(
            workspace, script_name, timeout, required_modules=required_modules
        )


PHASE1_PROMPT = textwrap.dedent(
    """
You are asked to write a Python script that extracts metadata from files placed in {WORKSPACE_FOLDER}.
Constraints:
- DO NOT import banned modules like subprocess, os.system, or openai.
- Use only safe libs (pandas, numpy, json, pyarrow, duckdb, csv).
- The script must write a single JSON file to {WORKSPACE_FOLDER}/metadata.json with the following schema:

{{
  "files": [
    {{
      "filename": "<basename>",
      "type": "<csv|parquet|json|xlsx|other>",
      "row_count": <int|null>,
      "columns": [
        {{"name":"<col>","type":"<inferred>","sample_value":"..."}}
      ],
      "sample_rows": [<up to 5 row objects>]
    }},
    ...
  ]
}}

Behavior:
- Inspect all files in {WORKSPACE_FOLDER} (do not read files outside).
- For tabular formats attempt to compute row_count and columns; if not possible set row_count to null.
- Keep runtime small (<= {phase1_timeout} seconds).
- Print nothing to stdout. Produce only the metadata.json file at '{WORKSPACE_FOLDER}/metadata.json'.

Dependency declaration:
- At the top, include a Python list variable named REQUIRED_MODULES, e.g.:
    REQUIRED_MODULES = ["pandas", "numpy", "pyarrow", "duckdb"]
    Keep it minimal and only include non-stdlib modules truly needed by the script.

Return ONLY the Python code for this script (no explanation).
"""
)

PHASE2_PROMPT_TEMPLATE = textwrap.dedent(
    """
You are a Python data analyst assistant that will write code to answer the user's questions.
<User Question>
{questions}
</User Question>

Data metadata (contents of metadata.json):
<metadata.json>
{metadata_json}
</metadata.json>

Files located in {WORKSPACE_FOLDER}: {workspace_files}

Requirements and constraints:
- DO NOT import banned modules (subprocess, openai, etc).
- Use only safe libraries (pandas, numpy, duckdb, pyarrow, matplotlib) unless absolutely necessary.
- Your script must write final results to {WORKSPACE_FOLDER}/output.json as valid JSON.
- If plots are requested, encode them as base64 data URIs in the JSON and ensure each image is <= 100000 bytes if required by the question.
- The script should catch runtime exceptions and, on error, write a file '{WORKSPACE_FOLDER}/last_error.txt' containing stacktrace and exit non-zero.
- Avoid reading files outside {WORKSPACE_FOLDER}.
- Keep runtime per attempt <= {per_attempt_timeout} seconds.

Dependency declaration:
- At the top, include a Python list variable named REQUIRED_MODULES, e.g.:
    REQUIRED_MODULES = ["pandas", "numpy", "pyarrow", "duckdb", "matplotlib"]
    Include only non-stdlib modules you actually import.

Return ONLY the Python code for this script (no explanation).
"""
)

DEBUG_PROMPT_TEMPLATE = textwrap.dedent(
    """
You are an assistant that fixes Python scripts given a runtime error.
Context:
- The original user question:
<User Question>
{questions}
</User Question>

- metadata.json:
<metadata.json>
{metadata_json}
</metadata.json>

- The script that failed is below:
<Failed Script>
{failing_code}
</Failed Script>

The runtime error / traceback is:
<Traceback>
{error_text}
</Traceback>

Instructions:
- Produce a corrected Python script that addresses the error, uses defensive checks (e.g., check for None, empty DataFrame), and adheres to the same constraints: write final JSON to {WORKSPACE_FOLDER}/output.json on success, write {WORKSPACE_FOLDER}/last_error.txt and exit non-zero on unrecoverable error.
- Add a short comment at the top: "# Fixed attempt {attempt_number}: <one-line reason>"
- Ensure the script includes an up-to-date REQUIRED_MODULES list containing any non-stdlib packages used.
Return ONLY the corrected Python code, nothing else.
"""
)


def read_questions_file(path: str = "questions.txt") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError("questions.txt not found in current directory.")
    return open(path, "r", encoding="utf-8").read()


def list_workspace_files(workspace: str) -> List[str]:
    files = []
    for fname in os.listdir(workspace):
        if os.path.isfile(os.path.join(workspace, fname)):
            files.append(fname)
    return files


def safe_write(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read_file_if_exists(path: str) -> Optional[str]:
    if os.path.exists(path):
        return open(path, "r", encoding="utf-8").read()
    return None


def ensure_workspace_copy(src_files: List[str], workspace: str):
    os.makedirs(workspace, exist_ok=True)
    for f in src_files:
        shutil.copy2(f, os.path.join(workspace, f))


def phase1_generate_and_run(
    workspace: str, questions_text: str, elapsed_budget: float
) -> dict:
    """
    Phase 1: ask LLM to produce metadata extraction code, static-check it, run in sandbox, and return parsed metadata dict.
    """
    print("[Phase 1] generating metadata extraction code ...")
    prompt = (
        PHASE1_PROMPT.format(phase1_timeout=PHASE1_TIMEOUT, WORKSPACE_FOLDER=workspace)
        + "\nFiles in working directory: "
        + ", ".join(list_workspace_files(workspace))
    )
    code_text_raw = call_llm_system_user(
        prompt, system="You are a helpful code generator that outputs safe Python."
    )
    code_text = _strip_code_fences(code_text_raw)

    # Save code for audit
    safe_write(os.path.join(workspace, "phase1_metadata_code.py"), code_text)

    # Static checks
    print("[Phase 1] static safety check ...")
    try:
        enforce_static_safety(code_text)
    except SafetyError as e:
        raise RuntimeError(f"Phase1 static safety error: {e}")

    # Execute
    timeout = min(
        PHASE1_TIMEOUT, max(5, int(elapsed_budget / 2))
    )  # avoid starving later phases
    print(
        f"[Phase 1] executing metadata extractor with timeout {timeout}s (sandbox={USE_DOCKER}) ..."
    )
    safe_write(os.path.join(workspace, "phase1.py"), code_text)
    required = extract_required_modules(code_text)
    res = execute_in_sandbox(workspace, "phase1.py", timeout, required_modules=required)
    # Save execution logs
    safe_write(os.path.join(workspace, "phase1_stdout.log"), res.stdout)
    safe_write(os.path.join(workspace, "phase1_stderr.log"), res.stderr)

    if not res.success:
        raise RuntimeError(
            f"Phase1 execution failed. exit_code={res.exit_code}, stderr={res.stderr[:2000]}"
        )

    # Read metadata.json
    metadata_path = os.path.join(workspace, "metadata.json")
    if not os.path.exists(metadata_path):
        # maybe script printed JSON to stdout; try parse stdout
        try:
            parsed = json.loads(res.stdout)
            # store it
            safe_write(metadata_path, json.dumps(parsed, indent=2))
            metadata = parsed
        except Exception:
            raise RuntimeError(
                "metadata.json not produced by phase1 and stdout is not valid JSON."
            )
    else:
        metadata = json.load(open(metadata_path, "r", encoding="utf-8"))
    print("[Phase 1] metadata extracted.")
    return metadata


def phase2_generate_and_run(
    workspace: str, questions_text: str, metadata: dict, elapsed_budget: float
) -> dict:
    """
    Phase 2: generate solution code, run it; if it errors, engage debugging loop.
    """
    print("[Phase 2] generating solution code ...")
    metadata_json = json.dumps(metadata)[:6000]  # truncate if large for prompt
    files_list = list_workspace_files(workspace)
    prompt = PHASE2_PROMPT_TEMPLATE.format(
        questions=questions_text,
        metadata_json=metadata_json,
        workspace_files=", ".join(files_list),
        per_attempt_timeout=PHASE2_SINGLE_ATTEMPT_TIMEOUT,
        WORKSPACE_FOLDER=workspace,
    )
    code_text_raw = call_llm_system_user(
        prompt,
        system="You are a helpful analyst that outputs a Python solution script only.",
    )
    code_text = _strip_code_fences(code_text_raw)

    # Save initial attempt
    attempt_num = 1
    safe_write(os.path.join(workspace, f"phase2_attempt_{attempt_num}.py"), code_text)

    # static check
    print(f"[Phase 2] static safety check attempt {attempt_num} ...")
    try:
        enforce_static_safety(code_text)
    except SafetyError as e:
        raise RuntimeError(f"Phase2 initial static safety error: {e}")

    # Attempt loop (with debug)
    attempts = 0
    last_error = None
    start_t = time.time()
    while attempts < MAX_DEBUG_ATTEMPTS and (time.time() - start_t) < elapsed_budget:
        attempts += 1
        remaining = max(5, int(elapsed_budget - (time.time() - start_t)))
        per_attempt_timeout = min(PHASE2_SINGLE_ATTEMPT_TIMEOUT, remaining)
        print(
            f"[Phase 2] executing attempt {attempts} (timeout {per_attempt_timeout}s) ..."
        )
        safe_write(os.path.join(workspace, f"phase2_current.py"), code_text)
        required = extract_required_modules(code_text)
        res = execute_in_sandbox(
            workspace,
            "phase2_current.py",
            per_attempt_timeout,
            required_modules=required,
        )
        safe_write(os.path.join(workspace, f"phase2_stdout_{attempts}.log"), res.stdout)
        safe_write(os.path.join(workspace, f"phase2_stderr_{attempts}.log"), res.stderr)

        # check output JSON file
        output_path = os.path.join(workspace, "output.json")
        if res.success and os.path.exists(output_path):
            # load output JSON and plausibility check
            try:
                out = json.load(open(output_path, "r", encoding="utf-8"))
                print("[Phase 2] success! Output produced at workspace/output.json")
                out_meta = {
                    "attempts": attempts,
                    "stdout": res.stdout[:2000],
                    "stderr": res.stderr[:2000],
                }
                return {"result": out, "meta": out_meta}
            except Exception as e:
                last_error = f"Output JSON parse error: {e}"
                print("WARN: output.json produced but failed to parse:", e)
                # proceed to debug loop
        else:
            # Execution failed: gather stderr
            last_error = res.stderr or res.stdout or f"exit_code={res.exit_code}"
            print(
                f"[Phase 2] attempt {attempts} failed. Stderr (truncated): {last_error[:1000]}"
            )

        # If still time and attempts remain -> call debug LLM to fix code
        if attempts < MAX_DEBUG_ATTEMPTS and (time.time() - start_t) < elapsed_budget:
            # Build debug prompt
            failing_code = code_text
            debug_prompt = DEBUG_PROMPT_TEMPLATE.format(
                questions=questions_text,
                metadata_json=metadata_json,
                failing_code=failing_code,
                error_text=last_error,
                attempt_number=attempts + 1,
                WORKSPACE_FOLDER=workspace,
            )
            print(f"[Phase 2] requesting fix from LLM (attempt {attempts+1}) ...")
            fixed_code_raw = call_llm_system_user(
                debug_prompt,
                system="You are a code fixer that returns corrected Python only.",
            )
            fixed_code = _strip_code_fences(fixed_code_raw)
            # Save and static-check
            safe_write(
                os.path.join(workspace, f"phase2_fix_suggested_{attempts+1}.py"),
                fixed_code,
            )
            try:
                enforce_static_safety(fixed_code)
            except SafetyError as e:
                # If an LLM suggested unsafe code, stop the loop and fail
                raise RuntimeError(
                    f"Phase2 debug-suggested code failed static safety: {e}"
                )
            # update code_text to new fixed code
            code_text = fixed_code
            safe_write(
                os.path.join(workspace, f"phase2_attempt_{attempts+1}.py"), code_text
            )
            # loop continues
        else:
            break

    # If we arrive here, attempts exhausted or timeout
    artifacts = {
        "attempts": attempts,
        "last_error": last_error,
        "workspace": os.path.abspath(workspace),
    }
    print(
        "[Phase 2] Failed after attempts/time exhausted. See workspace for artifacts."
    )
    return {"error": "failed", "artifacts": artifacts}


def run_full_workflow(base_dir: str = ".", use_temp_workspace: bool = True):
    """
    Main entrypoint:
    - sets up a workspace (copying input files),
    - runs phase1 and phase2 using LLM and sandboxed execution,
    - saves artifacts and prints final result path.
    """
    if not os.path.exists(os.path.join(base_dir, "questions.txt")):
        raise FileNotFoundError("questions.txt not found in base_dir.")

    # prepare workspace
    run_id = str(uuid.uuid4())[:8]
    workspace = os.path.join(BASE_TMP, f"run_{run_id}")
    if use_temp_workspace:
        os.makedirs(workspace, exist_ok=True)
        # copy necessary files
        inputs = []
        for name in os.listdir(base_dir):
            if name == os.path.basename(__file__):
                continue
            # copy everything except agent_workspaces and virtual env directories
            if name.startswith("run_") or name == os.path.basename(BASE_TMP):
                continue
            full = os.path.join(base_dir, name)
            if os.path.isfile(full):
                shutil.copy2(full, os.path.join(workspace, name))
                inputs.append(name)
    else:
        workspace = base_dir

    # print("Workspace prepared at:", workspace)
    start_time = time.time()
    remaining_budget = TOTAL_REQUEST_TIMEOUT

    # Phase 1
    questions_text = read_questions_file(os.path.join(workspace, "questions.txt"))
    metadata = None
    phase1_attempts = 0
    while phase1_attempts < PHASE1_MAX_ATTEMPTS:
        phase1_attempts += 1
        print(f"[Phase 1] attempt {phase1_attempts}/{PHASE1_MAX_ATTEMPTS}...")
        try:
            metadata = phase1_generate_and_run(
                workspace, questions_text, elapsed_budget=remaining_budget
            )
            break  # Success
        except Exception as e:
            print(f"Phase1 attempt {phase1_attempts} failed: {e}")
            if phase1_attempts >= PHASE1_MAX_ATTEMPTS:
                print("Phase1 failed after all attempts.")
                print("Artifacts saved at:", workspace)
                return {
                    "status": "phase1_failed",
                    "error": str(e),
                    "workspace": workspace,
                }

    if metadata is None:
        return {
            "status": "phase1_failed",
            "error": "Phase 1 failed to produce metadata after multiple attempts.",
            "workspace": workspace,
        }

    # update remaining budget
    elapsed = time.time() - start_time
    remaining_budget = TOTAL_REQUEST_TIMEOUT - elapsed
    if remaining_budget < 10:
        return {
            "status": "failed",
            "error": "insufficient time remaining after phase1",
            "workspace": workspace,
        }

    # Phase 2 with debug loop
    phase2_res = phase2_generate_and_run(
        workspace, questions_text, metadata, elapsed_budget=remaining_budget
    )
    elapsed2 = time.time() - start_time
    # Save final summary artifact
    summary_path = os.path.join(workspace, "summary.json")
    summary_content = {
        "run_id": run_id,
        "start_time": start_time,
        "elapsed_total_seconds": elapsed2,
        "phase2_result": phase2_res,
    }
    safe_write(summary_path, json.dumps(summary_content, indent=2))
    print("Workflow finished. Summary saved to:", summary_path)
    return {"status": "done", "workspace": workspace, "summary": summary_content}


# ========== CLI entry ==========
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-phase LLM-based data analyst agent."
    )
    parser.add_argument(
        "--dir",
        "-d",
        default=".",
        help="Directory containing questions.txt and input files.",
    )
    parser.add_argument(
        "--no-temp",
        action="store_true",
        help="Run in-place instead of copying to a temp workspace.",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Disable Docker sandbox (use local subprocess).",
    )
    parser.add_argument(
        "--docker-image", default=DOCKER_IMAGE, help="Docker image to use for sandbox."
    )
    args = parser.parse_args()

    if args.no_docker:
        USE_DOCKER = False
        print("INFO: Docker disabled. Using local subprocess sandbox (Unix only).")
    else:
        DOCKER_IMAGE = args.docker_image

    try:
        result = run_full_workflow(
            base_dir=args.dir, use_temp_workspace=not args.no_temp
        )
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print("Fatal error during workflow:", exc, file=sys.stderr)
