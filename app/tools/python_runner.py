import os
from pathlib import Path
import sys
import sysconfig
import time
import time
import tempfile
import subprocess
from typing import Any, Dict, List, Callable, Optional

import importlib.util


def run_python_with_uv(
    code: str, start_time: float, time_limit: float, **kwargs
) -> dict:
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        raise Exception("Request timeout before code execution")

    remaining_time = time_limit - elapsed

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)

            cmd = ["uv", "run", "--python", "3.11", "--isolated"]

            for pkg in [
                "pandas",
                "numpy",
                "scikit-learn",
                "matplotlib",
                "lxml",
                "html5lib",
            ]:
                cmd.extend(["--with", pkg])

            cmd.append(script_path)

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir
            )

            try:
                stdout, stderr = proc.communicate(timeout=min(60, remaining_time))
            except subprocess.TimeoutExpired:
                proc.kill()
                raise Exception("Python code execution timed out")

            if proc.returncode != 0:
                raise Exception(f"Python execution failed: {stderr.decode()}")

            output = stdout.decode().strip()
            return {"output": output, "success": True}

    except Exception as e:
        return {"error": str(e), "success": False}


def run_python_with_packages(
    code: str, packages: list, start_time: float, time_limit: float, **kwargs
) -> dict:
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        raise Exception("Request timeout before code execution")

    remaining_time = time_limit - elapsed

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)

            cmd = ["uv", "run", "--python", "3.11", "--isolated"]

            if packages:
                cmd.extend(
                    [
                        flag
                        for pkg in classify_stdlib(packages)["non_stdlib"]
                        for flag in ["--with", pkg]
                    ]
                )

            cmd.append(script_path)

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tmpdir
            )

            try:
                stdout, stderr = proc.communicate(timeout=min(60, remaining_time))
            except subprocess.TimeoutExpired:
                proc.kill()
                raise Exception("Python code execution timed out")

            if proc.returncode != 0:
                raise Exception(f"Python execution failed: {stderr.decode()}")

            output = stdout.decode().strip()
            return {"output": output, "success": True}

    except Exception as e:
        return {"error": str(e), "success": False}


def _classify_uv_modules(
    modules: List[str],
    pkg_map: Optional[Dict[str, str]] = None,
    flag_template: str = "--with {pkg}",
    import_check: bool = True,
    custom_checks: Optional[Dict[str, Callable[[], bool]]] = None,
) -> Dict[str, Any]:
    """
    Classify modules into 'preinstalled' and 'install_required'.

    Args:
    modules: list of module names (strings) you want classified.
    pkg_map: optional mapping {module_name: package_name_for_flag}. If absent,
            the module name itself is used as the pkg in the flag.
    flag_template: string template for the configure flag, e.g. "--with {pkg}".
    import_check: if True, treat importable Python modules as preinstalled.
    custom_checks: optional dict mapping module_name -> function() -> bool
                    for performing custom (non-import) checks (e.g. system libs).

    Returns:
    {
        "preinstalled": [module, ...],
        "install_required": [
        {"module": ..., "pkg": ..., "flag": ...}, ...
        ]
    }
    """
    pkg_map = pkg_map or {}
    custom_checks = custom_checks or {}

    preinstalled = []
    install_required = []

    # helper to test importability / builtin / stdlib
    def is_importable(name: str) -> bool:
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except Exception:
            pass
        # builtin modules
        if name in getattr(sys, "builtin_module_names", ()):
            return True
        # Python 3.10+: stdlib_module_names set
        if hasattr(sys, "stdlib_module_names") and name in sys.stdlib_module_names:
            return True
        return False

    for mod in modules:
        ok = False

        # 1) import-based check (covers pip + stdlib)
        if import_check and is_importable(mod):
            ok = True

        # 2) custom check (overrides import_check result if provided and returns True)
        if not ok and mod in custom_checks:
            try:
                ok = bool(custom_checks[mod]())
            except Exception:
                ok = False

        if ok:
            preinstalled.append(mod)
        else:
            pkg = pkg_map.get(mod, mod)
            flag = flag_template.format(pkg=pkg)
            install_required.append({"module": mod, "pkg": pkg, "flag": flag})

    return {"preinstalled": preinstalled, "install_required": install_required}


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
    # built-in / frozen modules -> stdlib
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

    # If the module file is inside the stdlib directory (or subdirs), treat it as stdlib
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
