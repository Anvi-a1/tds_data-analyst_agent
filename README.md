# 🚀 Data Analyst Agent

An API service that uses LLMs to fetch, parse, analyze, and visualize data with tool calls (web scraping, DuckDB, pandas, plotting) under a sandbox for safety. The server also starts concurrent backup and fake-response workflows to provide a fallback result if the primary path fails or times out.

---

## ✅ Features
- Web scraping via Playwright + BeautifulSoup (single-step fetch_and_parse_html)
- Data processing: Python (pandas), SQL (DuckDB)
- Plotting with matplotlib (returns base64 data URI)
- Function/tool calling with OpenAI, OpenRouter, and Gemini
- Shared-results store to keep large tool outputs out of the model context
- Sandbox execution (Docker or uv), timeouts, robust logging, plus backup and fake-response fallbacks

---

## � Getting Started

### Prerequisites
- Python 3.11+
- Windows/Linux/macOS
- API key: either `OPENAI_API_KEY` or `GEMINI_API_KEY` is required to start the server (OpenRouter is supported as a provider, but server startup currently checks for OpenAI/Gemini).
- Docker recommended for sandbox mode

### Quick Start

**Windows (cmd.exe)**
1. Create venv and install dependencies:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   playwright install
   REM Optional if you won't use Docker sandbox:
   pip install uv
   ```
2. Set provider and API key:
   ```cmd
   set LLM_PROVIDER=openai
   set OPENAI_API_KEY=YOUR_KEY
   ```
3. Start API:
   ```cmd
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

**Linux/macOS**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
# Optional if you won't use Docker sandbox:
pip install uv
export LLM_PROVIDER=openai
export OPENAI_API_KEY=YOUR_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Docker Sandbox Images (Optional)**
```bash
docker build -t data-agent-api .
docker build -f Dockerfile.sandbox -t data-agent-sandbox .
## If you build the sandbox with the tag above, set:
# export SANDBOX_DOCKER_IMAGE=data-agent-sandbox:latest
```

---

## ⚙️ Configuration

### Detailed Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | LLM provider to use | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENROUTER_API_KEY` | OpenRouter API key | - |
| `GEMINI_API_KEY` | Gemini API key | - |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4o-mini` |
| `OPENROUTER_MODEL` | OpenRouter model to use | `google/gemini-flash-1.5` |
| `GEMINI_MODEL` | Gemini model to use | `gemini-1.5-flash-latest` |
| `OPENAI_BASE_URL` | OpenAI base URL | `https://api.openai.com/v1` |
| `OPENROUTER_BASE_URL` | OpenRouter base URL | `https://openrouter.ai/api/v1` |
| `USE_SANDBOX` | Enable sandbox for code execution | `true` |
| `SANDBOX_MODE` | Sandbox mode (`docker` or `uv`) | `docker` |
| `SANDBOX_DOCKER_IMAGE` | Docker image for sandbox | `myorg/data-agent-sandbox:latest` |
| `REQUEST_TIMEOUT` | Request timeout in seconds | `170` |
| `LLM_MAX_OUTPUT_TOKENS` | Max output tokens for LLM | `8192` |
| `OPENAI_MAX_OUTPUT_TOKENS` | Max output tokens for OpenAI | `8192` |
| `OPENROUTER_MAX_OUTPUT_TOKENS`| Max output tokens for OpenRouter | `8192` |
| `GEMINI_MAX_OUTPUT_TOKENS` | Max output tokens for Gemini | `8192` |
| `MAX_FUNCTION_RESULT_CHARS` | Max characters for function result | `20000` |
| `LARGE_FUNCTION_RESULT_CHARS` | Threshold for large function result | `10000` |
| `MINIMIZE_TOOL_OUTPUT` | Minimize tool output in context | `true` |
| `AUTO_STORE_RESULTS` | Automatically store tool results | `true` |
| `MAX_INLINE_DATA_CHARS` | Max characters for inline data | `4000` |
| `MAX_OUTPUT_WORDS` | Max words for final answer | `200` |
| `BACKUP_RESPONSE_OPENAI_BASE_URL` | Base URL for backup OpenAI | `https://api.openai.com/v1` |
| `BACKUP_RESPONSE_OPENAI_API_KEY` | API key for backup OpenAI | - |
| `BACKUP_RESPONSE_OPENAI_MODEL` | Model for backup OpenAI | `openai/gpt-4.1-nano` |

### Example `.env` (snippet)
```
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
USE_SANDBOX=true
SANDBOX_MODE=docker
SANDBOX_DOCKER_IMAGE=data-agent-sandbox:latest
REQUEST_TIMEOUT=170
LLM_MAX_OUTPUT_TOKENS=800
MAX_FUNCTION_RESULT_CHARS=12000
MINIMIZE_TOOL_OUTPUT=true
AUTO_STORE_RESULTS=true
MAX_INLINE_DATA_CHARS=4000
MAX_OUTPUT_WORDS=200
```

---

## ▶️ Run locally

**Windows (cmd.exe)**
```cmd
set LLM_PROVIDER=openai
set OPENAI_API_KEY=...
set USE_SANDBOX=true
set SANDBOX_MODE=docker
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Linux/macOS**
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=...
export USE_SANDBOX=true
export SANDBOX_MODE=docker
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📄 API usage

**Endpoint:** `POST /api/`

This endpoint expects `multipart/form-data` with a required file field named `questions.txt`. Additional files are optional and will be saved to a per-request temp folder; their absolute paths are appended to the question text for tool code to access.

**`curl` (multipart)**
```bash
curl -X POST "http://localhost:8000/api/" \
	-F "questions.txt=@your_question.txt" \
	-F "data.csv=@data.csv" \
	-F "image.png=@image.png"
```

**Python `requests`**
```python
import requests
with open("your_question.txt","rb") as q:
		files = {"questions.txt": q}
		# optionally add other files
		resp = requests.post("http://localhost:8000/api/", files=files, timeout=200)
print(resp.json())
```

**Response**
- JSON object or array per your prompt format
- On error: `{"error": "..."}`

---

## 🧰 Tools and `shared_results`

Exposed tool calls in the current orchestrator:
- `fetch_and_parse_html(url, selector?, max_elements?, method?, headers?, timeout_seconds?, result_key?)`
- `run_duckdb_query(sql, result_key?)`
- `generate_plot(code, result_key?)` — should save a matplotlib figure to `output.png`; returns `{ data_uri }`
- `run_python_with_packages(code, packages, result_key?)` — executes with `uv`; print your final result

### Notes
- A non-advertised helper `run_python_with_uv(code)` is also supported internally; prefer `run_python_with_packages`.
- To avoid large token usage, pass `result_key` in tool calls. When `MINIMIZE_TOOL_OUTPUT=true`, full results are stored in `shared_results` and only a compact stub is added to the model context.
- If `AUTO_STORE_RESULTS=true` and `result_key` is omitted, the orchestrator will generate a key like `fetch_and_parse_html_1`.

### Referencing Saved Results in Later Tool Code
- Use placeholders rendered by the orchestrator before execution:
	- `{{shared_results.key.as_json}}` inserts a JSON string literal of the saved value (use `json.loads(...)` in Python)
	- `{{shared_results.key.as_python}}` inserts a Python literal (dict/list/str) you can use directly
	- Common subkeys for dicts are supported: e.g., `{{shared_results.key.results.as_json}}`
- Example with DuckDB into pandas:
	- `rows = json.loads({{shared_results.rows.output.as_json}})`
	- `import pandas as pd; df = pd.DataFrame(rows)`

---
## 🧪 Testing

**Quick Test**
```bash
python test_api.py
```

**Run Test Suite**
```bash
pytest -q
```

**Manual Test**
```bash
curl -X POST "http://localhost:8000/api/" -H "Content-Type: application/json" -d '{"question":"Scrape ... Return a JSON array ..."}'
```
**Note:** the API expects `multipart/form-data`; the JSON example above is only illustrative of a prompt and will not work against this server.

---

## 🛡️ Sandbox

### Modes
- `docker` (default): strong isolation, best for production
- `uv`: fast local isolation without Docker

### Env Vars
- `USE_SANDBOX`=`true`|`false`
- `SANDBOX_MODE`=`docker`|`uv`
- `SANDBOX_DOCKER_IMAGE`=`data-agent-sandbox:latest`

---

## 💸 Cost controls and token tips

- Set `LLM_MAX_OUTPUT_TOKENS=300–1000` for concise answers; use provider-specific caps as needed.
- Avoid pasting raw HTML/CSVs in prompts; let tools fetch/process.
- Keep data out of the chat: use `result_key` and `{{shared_results.key}}` to reference data instead of pasting it into prompts.
- Prefer `fetch_and_parse_html` over separate fetch/parse to reduce turns.
- Tune `MAX_FUNCTION_RESULT_CHARS` down (e.g., `8000–12000`) if tool outputs are still large.
- Choose economical models (e.g., `gpt-4o-mini`, `gemini-1.5-flash-latest`).

---

## 🧩 Extending & Security

### Extend Tools
1. Add schema in `Orchestrator.functions`
2. Implement in `app/tools/`
3. Map in `function_map` and handler

### Security
- Sandbox isolates execution; adjust CPU/memory/time as needed.
- Keep secrets in env vars. Avoid sending credentials to the model.

---

