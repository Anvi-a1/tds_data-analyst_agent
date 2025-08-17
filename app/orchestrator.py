import re
import os
import time
import json
import inspect
import logging
import asyncio
import hashlib

from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

from app.providers import ProviderFactory

from app.tools.parser import fetch_and_parse_html
from app.sandbox.sandbox_runner import SandboxRunner
from app.tools.python_runner import run_python_with_uv, run_python_with_packages

from app.wrappers.timeout import timeout


class TimeoutException(Exception):
    pass


class Orchestrator:
    def __init__(self):
        logger.info("🔧 Initializing Orchestrator...")
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai")
        logger.info(f"🤖 LLM Provider: {self.llm_provider}")

        try:
            self.provider = ProviderFactory.create_provider(self.llm_provider)
            logger.info(
                f"✅ {self.provider.provider_name} provider initialized successfully"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM provider: {e}")
            raise

        self.use_sandbox = os.getenv("USE_SANDBOX", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        # Controls whether tool results are minimized when sent back to the LLM (compact stub instead of full payload)
        self.minimize_tool_output = os.getenv(
            "MINIMIZE_TOOL_OUTPUT", "true"
        ).lower() in (
            "true",
            "1",
            "yes",
        )
        self.sandbox_mode = os.getenv("SANDBOX_MODE", "docker")
        # Auto-store tool results if no result_key is provided by the model
        self.auto_store_results = os.getenv("AUTO_STORE_RESULTS", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        self.functions = [
            {
                "name": "fetch_and_parse_html",
                "description": (
                    "Fetch HTML content from a URL using Playwright and parse it with BeautifulSoup. "
                    "Supports optional CSS selector extraction and element limiting."
                    "Returns:"
                    "    ParseHtmlResult dict with keys:"
                    "        - results (List[ElementResult]): Each item has:"
                    "            - text (str): Text content of the element (may be truncated to MAX_ELEMENT_TEXT_SIZE)."
                    "            - html (str): HTML string of the element (may be truncated to MAX_ELEMENT_TEXT_SIZE)."
                    "        - total_elements (int): Total number of elements that matched the selector (before applying max_elements)."
                    "        - total_size (int): Approx total size of returned text + html across results (after truncation/limits)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "selector": {
                            "type": "string",
                            "description": "CSS selector to extract specific elements (optional)",
                        },
                        "max_elements": {
                            "type": "integer",
                            "description": "Optional limit on number of elements to include when selector is provided",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST"],
                            "default": "GET",
                        },
                        "headers": {
                            "type": "object",
                            "description": "Custom HTTP headers (optional)",
                        },
                        "timeout_seconds": {"type": "number", "default": 10},
                        "result_key": {
                            "type": "string",
                            "description": "Optional: Key to store the result in shared_results for subsequent tool calls.",
                        },
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "run_duckdb_query",
                "description": (
                    "Execute a SQL query using DuckDB in an in-memory database. "
                    "Ideal for querying CSV/Parquet/JSON files and performing analytics with SQL. "
                    "Returns an object: { output: string (JSON array of row objects), success: boolean }."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "DuckDB SQL to execute. You can reference files with read_csv/read_parquet, etc.",
                        },
                        "result_key": {
                            "type": "string",
                            "description": "Optional: Key to store the result in shared_results for subsequent tool calls.",
                        },
                    },
                    "required": ["sql"],
                },
            },
            {
                "name": "run_python_with_packages",
                "description": (
                    "Run Python code with specific packages using uv (Note: Always remember to print the final result/output). "
                    "Returns: { output: string, success: true } on success; { error: string, success: false } on failure."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        },
                        "packages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of package names to install (don't include any of the Python's standard library)",
                        },
                        "result_key": {
                            "type": "string",
                            "description": "Optional: Key to store the result in shared_results for subsequent tool calls.",
                        },
                    },
                    "required": ["code", "packages"],
                },
            },
        ]
        self.sandbox = SandboxRunner() if self.use_sandbox else None

        self.function_map = {
            "fetch_and_parse_html": fetch_and_parse_html,
            "run_duckdb_query": self.run_duckdb_query_handler,
            "run_python_with_uv": self.run_python_with_uv_handler,
            "run_python_with_packages": self.run_python_with_packages_handler,
        }
        logger.info("🔧 Function map initialized with async/sync function support")

    @timeout
    async def handle_request(
        self, question: str, start_time: float, time_limit: float
    ) -> Any:
        logger.info("🎯 Starting request handling...")
        logger.info(f"📝 Question length: {len(question)} characters")
        logger.debug(f"📝 Question preview: {question[:300]}...")

        # Preprocess user question: extract large inline HTML/CSV into shared_results
        question, initial_shared = self._preprocess_user_question(question)
        if initial_shared:
            logger.info(
                f"🧰 Preprocessed user input: moved {len(initial_shared)} large data block(s) into shared_results"
            )

        # Conciseness controls
        try:
            max_words = int(os.getenv("MAX_OUTPUT_WORDS", "200"))
        except Exception:
            max_words = 200

        system_message = {
            "role": "system",
            "content": f"""You are a Data Analyst Agent, an expert AI assistant specialized in data analysis, web scraping, and visualization. Your role is to help users analyze data from various sources including web scraping, file processing, and database queries.

CAPABILITIES:
- Combined web fetching and parsing with fetch_and_parse_html (ideal for single-step URL content extraction)
- HTML extraction via CSS selectors using fetch_and_parse_html
- Python code execution with run_python_with_packages

RESULT SHARING:
- Store tool results by passing 'result_key' in tool calls. Example: tool_call(name='fetch_and_parse_html', url='...', selector='...', result_key='key').
- To reference a previous tool result later, you MUST save it by setting 'result_key' in that tool call; only saved results are stored in the shared_results dict and can be reused.
- To reuse a stored result in later tool code, use placeholders that the orchestrator renders before execution:
    - {{shared_results.key.as_json}} inserts a JSON string literal of the result (wrap with json.loads(...) in Python to get an object).
    - {{shared_results.key.as_python}} inserts a Python literal (dict/list/str) that can be used directly in Python code.
    - For common fields in dict results, you can also use: {{shared_results.key.html.as_json}}, {{shared_results.key.results.as_json}} (and the .as_python variants).
    - For run_duckdb_query results: rows = json.loads({{shared_results.key.output.as_json}}); import pandas as pd; df = pd.DataFrame(rows)
- Avoid treating shared_results like a runtime variable (e.g., shared_results[...] or shared_results.key). Always use the {{shared_results...}} placeholders above inside the code you generate.
- The 'result_key' should be a simple string (e.g., 'data', 'plot_result', 'query_output'). Reusing the same key overwrites the previous value.

GUIDELINES (prefer DuckDB for data work and avoid hallucinations):
1. Always break complex tasks into logical steps
2. Use appropriate tools for each task (web scraping -> data processing -> analysis -> visualization)
3. For single-step URL content extraction, prefer fetch_and_parse_html over multi-step approaches
4. When scraping data, extract all relevant information and structure it properly
5. For data analysis, use pandas efficiently and handle edge cases
6. When creating plots, make them clear, well-labeled, and professionally formatted
7. Always validate and clean data before analysis
8. Provide accurate numerical answers and handle correlations/statistics correctly
9. For visualizations, ensure proper axis labels, titles, and formatting
10. Return base64-encoded images for plots as requested
11. Never invent facts, data values, sources, file paths, or tool names. Use only exposed tools and the data you actually fetched or were given.
12. Do not create mock/synthetic datasets when real data exists in shared_results; always use saved results. If necessary data is missing, call a tool to obtain it.
13. If you're UNABLE to fulfill a request, return a FAKE RESPONSE that adheres exactly to the requested format.
    - Use placeholders like "N/A", null, empty arrays, or 0 where appropriate.
    - Make sure to adhere to the requested response format.
14. Prefer DuckDB whenever possible for SQL-like analysis, Postgres-style queries, and local files (CSV/Parquet/JSON). Use the run_duckdb_query tool and functions like read_csv_auto, read_parquet. For date differences use date_diff('day', end, start) and parse strings with strptime(...); avoid sqlite-specific functions like julianday/str_to_date.
    - Example: tool_call(name='run_duckdb_query', sql="SELECT * FROM read_csv_auto('data.csv') LIMIT 5", result_key='rows')
    - To reuse in Python: data = json.loads({{shared_results.rows.as_json}})

CONCISENESS:
- Be brief by default. Unless the user asks for detailed code or long narratives, keep answers under {max_words} words.
- Prefer short bullet points over paragraphs when listing items.
- Do not add preambles, disclaimers, or extra commentary unless requested.

STRICT FORMATS:
- If the user requests JSON, output exactly a valid JSON array/object with no extra text before/after.
- If a specific schema or shape is requested, follow it exactly.

RESPONSE FORMAT:
- ALWAYS follow the exact format requested in the user's question
- If JSON array is requested, return a JSON array
- If JSON object is requested, return a JSON object
- For plots, return base64 data URIs in the specified format
- Ensure numerical precision for statistical calculations
- Any of the response without any function/tool call will be IMMEDIATELY considered as a final result. So, avoid making unnecessary tool calls.
- Make sure your final output is well-structured and adheres to the requested format.
- If returning a simulated response, make sure to stick to the requested format.

Remember: You have access to the internet for web scraping and can execute Python code safely. Use these capabilities to provide comprehensive and accurate data analysis.""",
        }

        logger.info("📋 Added system message to guide LLM behavior")
        logger.debug(
            f"📋 System message length: {len(system_message['content'])} characters"
        )

        messages = [system_message, {"role": "user", "content": question}]
        iteration_count = 0
        shared_results = dict(initial_shared)

        try:
            while True:
                iteration_count += 1
                elapsed = time.time() - start_time
                remaining_time = time_limit - elapsed

                logger.info(
                    f"🔄 Iteration {iteration_count} - Elapsed: {elapsed:.2f}s, Remaining: {remaining_time:.2f}s"
                )

                if elapsed > time_limit:
                    logger.error("⏰ Time limit exceeded!")
                    raise TimeoutException()

                logger.info(f"🤖 Calling {self.llm_provider} API...")

                message = self.provider.generate_response(messages, self.functions)
                logger.debug(f"🔍 Generated message: {message}")

                function_call_info = None
                if message.get("function_call"):
                    function_call_info = {
                        "name": message["function_call"]["name"],
                        "arguments": message["function_call"]["arguments"],
                    }
                elif message.get("tool_calls") and len(message["tool_calls"]) > 0:
                    tool_call = message["tool_calls"][0]
                    if tool_call.get("function"):
                        function_call_info = {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"],
                        }

                if function_call_info:
                    # Enforce exposed tool set to reduce turns and token use
                    allowed_function_names = {fn.get("name") for fn in self.functions}
                    func_name = function_call_info["name"]
                    func_args = json.loads(function_call_info["arguments"] or "{}")

                    logger.info(f"🔧 Executing function: {func_name}")
                    logger.debug(f"📋 Function arguments: {func_args}")

                    # If the model tries to call non-exposed legacy tools, nudge to use fetch_and_parse_html
                    if func_name not in allowed_function_names:
                        if func_name in {"fetch_url", "parse_html"}:
                            logger.warning(
                                "❌ Disallowed tool call detected; advising to use fetch_and_parse_html instead"
                            )
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": (
                                        "Please use the single-step tool 'fetch_and_parse_html' instead of separate 'fetch_url' or 'parse_html'. "
                                        "Example: tool_call(name='fetch_and_parse_html', url='<URL>', selector='<CSS selector>', result_key='html_data')."
                                    ),
                                }
                            )
                            continue
                        if func_name not in self.function_map:
                            logger.error(f"❌ Unknown function: {func_name}")
                            raise Exception(f"Unknown function: {func_name}")

                    try:
                        logger.info(f"▶️  Starting execution of {func_name}...")
                        result = await self.dispatch_function(
                            func_name, func_args, start_time, time_limit, shared_results
                        )
                        # if func_name in ["run_python_with_uv", "run_python_with_packages"]:
                        #     print("Result:", result)
                        logger.info(f"✅ Function {func_name} completed successfully")
                        logger.debug(f"📤 Function result type: {type(result)}")

                        # Determine effective result key (provided or auto-generated)
                        effective_key = None
                        if "result_key" in func_args and func_args["result_key"]:
                            effective_key = func_args["result_key"]
                            shared_results[effective_key] = result
                            logger.info(
                                f"Stored result of {func_name} in shared_results['{effective_key}']"
                            )
                        elif self.auto_store_results:
                            effective_key = f"{func_name}_{iteration_count}"
                            shared_results[effective_key] = result
                            logger.info(
                                f"Auto-stored result of {func_name} in shared_results['{effective_key}']"
                            )

                    except Exception as e:
                        error_msg = f"Error executing {func_name}: {e}"
                        logger.error(f"💥 {error_msg}")
                        logger.exception("Full traceback:")

                        assistant_msg = {
                            "role": "assistant",
                            "content": None,
                        }
                        if message.get("function_call"):
                            assistant_msg["function_call"] = message["function_call"]
                        elif message.get("tool_calls"):
                            assistant_msg["tool_calls"] = message["tool_calls"]

                        messages.append(assistant_msg)

                        if message.get("function_call"):
                            messages.append(
                                {
                                    "role": "function",
                                    "name": func_name,
                                    "content": json.dumps({"error": error_msg}),
                                }
                            )
                        elif (
                            message.get("tool_calls") and len(message["tool_calls"]) > 0
                        ):
                            for tool_call in message["tool_calls"]:
                                messages.append(
                                    {
                                        "role": "tool",
                                        "name": tool_call["function"]["name"],
                                        "tool_call_id": tool_call["id"],
                                        "content": json.dumps({"error": error_msg}),
                                    }
                                )
                        continue

                    logger.info(f"📨 Adding function result to message history")
                    messages.append(message)

                    if message.get("function_call"):
                        # If a result_key was provided or auto-generated, either send compact stub (default) or full result
                        key = (
                            func_args.get("result_key")
                            if func_args.get("result_key")
                            else (
                                effective_key if "effective_key" in locals() else None
                            )
                        )
                        if key and self.minimize_tool_output:
                            preview = None
                            try:
                                # Build a tiny preview to help the model without flooding context
                                if isinstance(result, str):
                                    preview = (
                                        (result[:500] + "...")
                                        if len(result) > 500
                                        else result
                                    )
                                elif isinstance(result, list):
                                    preview = result[:3]
                                elif isinstance(result, dict):
                                    # Keep at most top-level keys and sample of values
                                    preview = {
                                        k: (
                                            (v[:120] + "...")
                                            if isinstance(v, str) and len(v) > 120
                                            else v
                                        )
                                        for k, v in list(result.items())[:5]
                                    }
                                else:
                                    preview = str(result)[:200]
                            except Exception:
                                preview = None

                            compact_payload = {
                                "stored": True,
                                "result_key": key,
                                "note": f"Full result stored in shared_results['{key}']. Use {{shared_results.{key}.as_json}} (then json.loads(...)) or {{shared_results.{key}.as_python}} in subsequent tool code.",
                                "type": type(result).__name__,
                                "size_chars": (
                                    len(json.dumps(result))
                                    if not isinstance(result, str)
                                    else len(result)
                                ),
                                "preview": preview,
                            }
                            messages.append(
                                {
                                    "role": "function",
                                    "name": func_name,
                                    "content": json.dumps(compact_payload),
                                }
                            )
                        else:
                            messages.append(
                                {
                                    "role": "function",
                                    "name": func_name,
                                    "content": json.dumps(result),
                                }
                            )
                    elif message.get("tool_calls") and len(message["tool_calls"]) > 0:
                        for tool_call in message["tool_calls"]:
                            key = (
                                func_args.get("result_key")
                                if func_args.get("result_key")
                                else (
                                    effective_key
                                    if "effective_key" in locals()
                                    else None
                                )
                            )
                            if key and self.minimize_tool_output:
                                preview = None
                                try:
                                    if isinstance(result, str):
                                        preview = (
                                            (result[:500] + "...")
                                            if len(result) > 500
                                            else result
                                        )
                                    elif isinstance(result, list):
                                        preview = result[:3]
                                    elif isinstance(result, dict):
                                        preview = {
                                            k: (
                                                (v[:120] + "...")
                                                if isinstance(v, str) and len(v) > 120
                                                else v
                                            )
                                            for k, v in list(result.items())[:5]
                                        }
                                    else:
                                        preview = str(result)[:200]
                                except Exception:
                                    preview = None

                                compact_payload = {
                                    "stored": True,
                                    "result_key": key,
                                    "note": f"Full result stored in shared_results['{key}']. Use {{shared_results.{key}.as_json}} (then json.loads(...)) or {{shared_results.{key}.as_python}} in subsequent tool code.",
                                    "type": type(result).__name__,
                                    "size_chars": (
                                        len(json.dumps(result))
                                        if not isinstance(result, str)
                                        else len(result)
                                    ),
                                    "preview": preview,
                                }
                                messages.append(
                                    {
                                        "role": "tool",
                                        "name": tool_call["function"]["name"],
                                        "tool_call_id": tool_call["id"],
                                        "content": json.dumps(compact_payload),
                                    }
                                )
                            else:
                                messages.append(
                                    {
                                        "role": "tool",
                                        "name": tool_call["function"]["name"],
                                        "tool_call_id": tool_call["id"],
                                        "content": json.dumps(result),
                                    }
                                )
                else:
                    content = message.get("content", "")
                    logger.info(
                        f"📝 Received final response from LLM: {len(content)} characters"
                    )
                    logger.debug(f"📝 Response preview: {content[:200]}...")

                    try:
                        parsed = json.loads(content)
                        logger.info("✅ Successfully parsed JSON response")
                        return parsed
                    except json.JSONDecodeError:
                        data = self._extract_data_blocks(content)
                        if data:
                            try:
                                return json.loads(data)
                            except json.JSONDecodeError:
                                return data

                        logger.error(
                            f"❌ Failed to parse LLM response as JSON: {content}"
                        )
                        error_message = content.strip()
                        return {"error": error_message}
        except TimeoutException:
            logger.warning(
                "⏰ Request timed out, attempting to recover partial results..."
            )
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "function":
                    logger.info(f"🔄 Found partial result from previous function call")
                    return json.loads(messages[i]["content"])

            logger.error("💥 Timeout with no partial results available")
            raise TimeoutException(
                "Request timed out and no partial response could be generated."
            )

    def _extract_data_blocks(
        self, text: str, first_only: bool = True
    ) -> Union[Optional[str], List[str]]:
        """
        Extract content between triple backticks ```...``` from `text`.
        - If first_only is True (default) returns the first matched block as a string (or None if no match).
        - If first_only is False returns a list of all matched blocks (may be empty).
        The regex tolerates an optional language tag after the opening backticks, e.g. ```json.
        """
        pattern = r"```(?:[^\n`]*)\n?(.*?)```"
        matches = re.findall(pattern, text, flags=re.DOTALL)
        matches = [m.strip() for m in matches]
        if not matches:
            return None if first_only else []
        return matches[0] if first_only else matches

    def _preprocess_user_question(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Detect large inline HTML/CSV in the user prompt. If found, store the full
        content into shared_results and replace it in the prompt with a compact
        placeholder so the LLM can reference it via {{shared_results.key}} in tool calls.
        """
        try:
            MAX_INLINE_SIZE = int(os.getenv("MAX_INLINE_DATA_CHARS", "4000"))
        except Exception:
            MAX_INLINE_SIZE = 4000

        shared: Dict[str, Any] = {}
        modified = question

        # Heuristic: extract longest CSV block (>=30 lines, each with >=2 commas)
        def extract_csv_block(text: str):
            lines = text.splitlines()
            best_start = best_len = 0
            current_start = None
            current_len = 0
            for i, line in enumerate(lines):
                if (line.count(",") >= 2) and ("<" not in line and ">" not in line):
                    if current_start is None:
                        current_start = i
                        current_len = 1
                    else:
                        current_len += 1
                else:
                    if current_start is not None and current_len > best_len:
                        best_start, best_len = current_start, current_len
                    current_start = None
                    current_len = 0
            if current_start is not None and current_len > best_len:
                best_start, best_len = current_start, current_len
            if best_len >= 30:
                start = best_start
                end = best_start + best_len
                block = "\n".join(lines[start:end])
                return block, start, end
            return None, None, None

        # Heuristic: extract large HTML block
        def extract_html_block(text: str):
            # Prefer full <html>...</html> if present
            m = re.search(r"<html[\s\S]*?</html>", text, flags=re.IGNORECASE)
            if m and len(m.group(0)) >= MAX_INLINE_SIZE:
                return m.group(0), m.start(), m.end()
            # Otherwise, pick longest angle-bracket chunk
            if text.count("<") > 50 and text.count(">") > 50:
                first = text.find("<")
                last = text.rfind(">")
                if first != -1 and last != -1 and last - first >= MAX_INLINE_SIZE:
                    return text[first : last + 1], first, last + 1
            return None, None, None

        # Try HTML first
        html_block, h_start, h_end = extract_html_block(modified)
        if html_block:
            key = (
                f"user_html_{hashlib.sha1(html_block.encode('utf-8')).hexdigest()[:8]}"
            )
            shared[key] = html_block
            preview = html_block[:400] + ("..." if len(html_block) > 400 else "")
            placeholder = f"[Large HTML stored as {{shared_results.{key}}} | size={len(html_block)} chars]\n\nPreview:\n{preview}"
            modified = modified[:h_start] + placeholder + modified[h_end:]

        # Then CSV
        csv_block, c_start, c_end = extract_csv_block(modified)
        if csv_block and len(csv_block) >= MAX_INLINE_SIZE:
            key = f"user_csv_{hashlib.sha1(csv_block.encode('utf-8')).hexdigest()[:8]}"
            shared[key] = csv_block
            # Build a tiny preview (first 8 lines)
            csv_lines = csv_block.splitlines()
            preview_lines = csv_lines[:8]
            preview = "\n".join(preview_lines)
            placeholder = f"[Large CSV stored as {{shared_results.{key}}} | lines={len(csv_lines)} | size={len(csv_block)} chars]\n\nPreview (first 8 lines):\n{preview}"
            # Map line indices back to character indices approximately
            # Re-split to rebuild around the block
            lines = modified.splitlines()
            before = "\n".join(lines[:c_start]) if c_start is not None else modified
            after = "\n".join(lines[c_end:]) if c_end is not None else ""
            modified = (
                before
                + ("\n\n" if before and not before.endswith("\n\n") else "")
                + placeholder
                + ("\n\n" if after and not after.startswith("\n\n") else "")
                + after
            )

        return modified, shared

    async def dispatch_function(
        self,
        name: str,
        args: Dict,
        start_time: float,
        time_limit: float,
        shared_results: Dict,
    ):
        logger.info(f"🚀 Dispatching function: {name}")
        logger.debug(f"📋 Function args: {args}")

        func = self.function_map[name]
        sig = inspect.signature(func)
        func_params = sig.parameters
        logger.debug(f"🔧 Function signature: {list(func_params.keys())}")

        call_args = args.copy()
        if "start_time" in func_params:
            call_args["start_time"] = start_time
            logger.debug("⏰ Added start_time to function args")
        if "time_limit" in func_params:
            call_args["time_limit"] = time_limit
            logger.debug("⏰ Added time_limit to function args")
        if "shared_results" in func_params:
            call_args["shared_results"] = shared_results
            logger.debug("🤝 Added shared_results to function args")

        # Filter out any unexpected arguments not present in the function signature
        allowed_params = set(func_params.keys())
        clean_args = {k: v for k, v in call_args.items() if k in allowed_params}
        ignored = [k for k in call_args.keys() if k not in allowed_params]
        if ignored:
            logger.debug(f"🧹 Ignoring unexpected args for {name}: {ignored}")

        # print(allowed_params, clean_args)
        logger.info(f"▶️  Executing {name} with {len(clean_args)} arguments")

        if asyncio.iscoroutinefunction(func):
            logger.debug(f"🔄 {name} is async function")
            result = await func(**clean_args)
        else:
            logger.debug(f"🔄 {name} is sync function")
            result = func(**clean_args)

        logger.info(f"✅ Function {name} execution completed")

        # Manage result size to prevent token overflow
        if self.minimize_tool_output:
            result = self._manage_result_size(result, name)

        return result

    def _render_placeholders(self, text: str, shared_results: Dict[str, Any]) -> str:
        """
        Render placeholders of the form:
          - {{shared_results.KEY.as_json}} -> JSON-encoded string literal
          - {{shared_results.KEY.as_python}} -> Python literal representation
          - {{shared_results.KEY.subkey.as_json}} / .as_python for dict subkeys like 'html', 'results', 'content'
          - Nested and indexed paths are supported: e.g., {{shared_results.key.results.0.text.as_json}}

        This prevents the model from attempting runtime access like shared_results['KEY'] in the executed code.
        """
        if not text or (
            "{{shared_results" not in text and "{shared_results" not in text
        ):
            return text

        def to_python_literal(value: Any) -> str:
            try:
                # Prefer Python literal representation for direct embedding
                return repr(value)
            except Exception:
                return repr(value)

        # Pattern captures: path 'key(.subkey)*' and formatter (.as_json|.as_python)
        pattern = re.compile(
            r"\{\{?\s*shared_results\.([a-zA-Z0-9_\-]+(?:\.[a-zA-Z0-9_\-]+)*)\.(as_json|as_python)\s*\}\}?"
        )

        def _resolve_path(root: Any, segments: list[str]) -> Any:
            value = root
            for seg in segments:
                if value is None:
                    return None
                if isinstance(value, dict):
                    value = value.get(seg)
                elif isinstance(value, list) and seg.isdigit():
                    idx = int(seg)
                    if 0 <= idx < len(value):
                        value = value[idx]
                    else:
                        return None
                else:
                    return None
            return value

        def replacer(match: re.Match) -> str:
            full_path = match.group(1)
            fmt = match.group(2)

            segments = full_path.split(".")
            key, tail = segments[0], segments[1:]

            if key not in shared_results:
                return "null" if fmt == "as_json" else "None"

            value = _resolve_path(shared_results[key], tail)

            if fmt == "as_json":
                try:
                    return json.dumps(value)
                except Exception:
                    return json.dumps(str(value))
            else:  # as_python
                return to_python_literal(value)

        rendered = pattern.sub(replacer, text)

        # Backward-compatible simple forms defaulting to JSON: {{shared_results.key}} or nested {{shared_results.key.sub.path}}
        # Also accept single-brace variants like {shared_results.key}
        simple_pattern = re.compile(
            r"\{\{?\s*shared_results\.([a-zA-Z0-9_\-]+(?:\.[a-zA-Z0-9_\-]+)*)\s*\}\}?"
        )

        def simple_replacer(match: re.Match) -> str:
            full_path = match.group(1)
            segments = full_path.split(".")
            key, tail = segments[0], segments[1:]
            if key not in shared_results:
                return "null"
            value = _resolve_path(shared_results[key], tail)
            try:
                return json.dumps(value)
            except Exception:
                return json.dumps(str(value))

        return simple_pattern.sub(simple_replacer, rendered)

    def _manage_result_size(self, result: Any, function_name: str) -> Any:
        """
        Manage function result size to prevent token overflow in API calls.
        """
        if not result:
            return result

        result_str = json.dumps(result) if not isinstance(result, str) else result
        result_size = len(result_str)

        # Size limits based on typical token conversion, configurable via env
        try:
            MAX_RESULT_SIZE = int(os.getenv("MAX_FUNCTION_RESULT_CHARS", "20000"))
        except Exception:
            MAX_RESULT_SIZE = 20000  # ~6-7k tokens
        try:
            LARGE_RESULT_SIZE = int(os.getenv("LARGE_FUNCTION_RESULT_CHARS", "10000"))
        except Exception:
            LARGE_RESULT_SIZE = 10000

        if result_size > LARGE_RESULT_SIZE:
            logger.warning(f"⚠️  Large result from {function_name}: {result_size} chars")

        if result_size > MAX_RESULT_SIZE:
            logger.warning(
                f"🗜️  Compressing large result from {function_name}: {result_size} -> target: {MAX_RESULT_SIZE}"
            )

            if isinstance(result, dict):
                if "content" in result:
                    original_content = result["content"]
                    if len(original_content) > MAX_RESULT_SIZE:
                        result["content"] = (
                            original_content[:MAX_RESULT_SIZE]
                            + "\n\n[... Content truncated due to size constraints ...]"
                        )
                        result["truncated"] = True
                        result["original_size"] = len(original_content)

                elif "results" in result:
                    if (
                        isinstance(result["results"], list)
                        and len(result["results"]) > 10
                    ):
                        results = result["results"]
                        compressed_results = (
                            results[:5]
                            + [
                                {
                                    "text": f"[... {len(results) - 10} items omitted ...]",
                                    "html": "",
                                }
                            ]
                            + results[-5:]
                        )
                        result["results"] = compressed_results
                        result["truncated"] = True

                elif "tables" in result:
                    # Table data
                    tables = result["tables"]
                    for table in tables:
                        if "rows" in table and len(table["rows"]) > 100:
                            # Sample rows
                            rows = table["rows"]
                            table["rows"] = (
                                rows[:50]
                                + [[f"... {len(rows) - 100} rows omitted ..."]]
                                + rows[-50:]
                            )
                            table["truncated"] = True

            elif isinstance(result, list) and len(result) > 20:
                # Large list result
                result = (
                    result[:10]
                    + [f"... {len(result) - 20} items omitted ..."]
                    + result[-10:]
                )

            elif isinstance(result, str) and len(result) > MAX_RESULT_SIZE:
                # Large string result
                result = (
                    result[:MAX_RESULT_SIZE]
                    + "\n\n[... Text truncated due to size constraints ...]"
                )

            logger.info(f"🗜️  Result compressed for {function_name}")

        return result

    def run_pandas_code_sandboxed(
        self, code: str, start_time: float, time_limit: float
    ) -> dict:
        return self.sandbox.run_pandas_code(code, start_time, time_limit)

    def run_duckdb_query_sandboxed(
        self, sql: str, start_time: float, time_limit: float
    ) -> dict:
        return self.sandbox.run_duckdb_query(sql, start_time, time_limit)

    def run_duckdb_query_handler(
        self, sql: str, start_time: float, time_limit: float, shared_results: Dict
    ) -> dict:
        sql = sql.replace("julianday", "julian")

        if self.use_sandbox:
            return self.run_duckdb_query_sandboxed(sql, start_time, time_limit)
        else:
            # Render placeholders like {{shared_results.key.as_json}} / .as_python (and common subkeys)
            sql = self._render_placeholders(sql, shared_results)

            duckdb_code = f"""
import duckdb
import json

con = duckdb.connect(database=':memory:')
df = con.execute({json.dumps(sql)}).df()
records = df.to_dict(orient='records')
print(json.dumps(records, default=str))
"""
            return run_python_with_packages(
                duckdb_code, ["duckdb", "pandas"], start_time, time_limit
            )

    def run_python_with_uv_handler(
        self, code: str, start_time: float, time_limit: float, shared_results: Dict
    ) -> dict:
        # Render placeholders into code
        code = self._render_placeholders(code, shared_results)
        code = self._inject_shared_mapping_if_referenced(code, shared_results)

        with open(f"debug/{str(start_time).replace('.', '')}.py", "w") as f:
            f.write(code)

        if self.use_sandbox and self.sandbox_mode == "docker":
            return self.run_pandas_code_sandboxed(code, start_time, time_limit)
        else:
            return run_python_with_uv(code, start_time, time_limit)

    def run_python_with_packages_handler(
        self,
        code: str,
        packages: list,
        start_time: float,
        time_limit: float,
        shared_results: Dict,
    ) -> dict:
        # Render placeholders into code
        code = self._render_placeholders(code, shared_results)
        code = self._inject_shared_mapping_if_referenced(code, shared_results)

        with open(f"debug/{str(start_time).replace('.', '')}.py", "w") as f:
            f.write(code)

        if self.use_sandbox and self.sandbox_mode == "docker":
            sandboxed_code = f"""
{chr(10).join(f'import {pkg}' for pkg in packages)}

{code}
"""
            return self.run_pandas_code_sandboxed(
                sandboxed_code, start_time, time_limit
            )
        else:
            return run_python_with_packages(code, packages, start_time, time_limit)

    def _inject_shared_mapping_if_referenced(
        self, code: str, shared_results: Dict[str, Any]
    ) -> str:
        """
        Back-compat: If the generated code tries to access a runtime variable named
        `shared_results` or `shared_dict` (instead of using placeholders), inject a
        binding so the code can run without NameError.

        We avoid injection unless those identifiers appear as whole words.
        """
        try:
            if not code:
                return code

            # Only inject if references exist
            if not (
                re.search(r"\bshared_results\b", code)
                or re.search(r"\bshared_dict\b", code)
            ):
                return code

            # Don't clobber if user already defines these names explicitly
            if re.search(r"\bshared_results\s*=", code) and re.search(
                r"\bshared_dict\s*=", code
            ):
                return code

            payload = json.dumps(shared_results, default=str)
            preamble_lines = [
                "# Auto-injected shared mapping (back-compat)",
                "import json as __json  # safe re-import",
                f"__SHARED_RESULTS = __json.loads({json.dumps(payload)!r})",
            ]
            if not re.search(r"\bshared_results\s*=", code):
                preamble_lines.append("shared_results = __json.loads(__SHARED_RESULTS)")
            if not re.search(r"\bshared_dict\s*=", code):
                preamble_lines.append("shared_dict = __json.loads(__SHARED_RESULTS)")

            preamble = "\n".join(preamble_lines) + "\n\n"
            return preamble + code
        except Exception:
            # Fail-open: if anything goes wrong, return original code untouched
            return code
