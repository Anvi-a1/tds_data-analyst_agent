from app.sandbox.sandbox_runner import SandboxRunner

sandbox = SandboxRunner()


def run_pandas_code(code: str, start_time: float, time_limit: float) -> dict:
    return sandbox.run_pandas_code(code, start_time, time_limit)


def run_duckdb_query(sql: str, start_time: float, time_limit: float) -> dict:
    return sandbox.run_duckdb_query(sql, start_time, time_limit)
