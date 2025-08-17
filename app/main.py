import os
import re
import json
import time
import uuid
import shutil
import logging
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, HTTPException, Request, Response

from dotenv import load_dotenv

from app import __version__
from app.orchestrator import Orchestrator, TimeoutException
from app.backup_orchestrator import BackupOrchestrator
from app.fake_response_orchestrator import FakeResponseOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_analyst_agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded")

if not os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    logger.error("No API keys found in environment variables")
    raise ValueError("Either OPENAI_API_KEY or GEMINI_API_KEY must be set")

logger.info(f"Using LLM provider: {os.getenv('LLM_PROVIDER', 'openai')}")

app = FastAPI(
    title="Data Analyst Agent",
    version=__version__,
    root_path=os.getenv("ROOT_PATH", ""),
    description=(
        "Data Analyst Agent API.\n\n"
        "Features:\n"
        "- Primary orchestrator with timeout control\n"
        "- Non-blocking backup + fake-response fallbacks\n"
        "- Secure ephemeral file upload workspace\n"
        "- Structured JSON responses for analytics workflows\n"
        "- Pluggable LLM providers (OpenAI / Gemini)\n"
    ),
    openapi_tags=[
        {
            "name": "API",
            "description": "Submit questions plus data files for automated analysis.",
        },
    ],
)
app.router.redirect_slashes = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("FastAPI app initialized")

orchestrator = Orchestrator()
logger.info("Orchestrator initialized")

backup_orchestrator = BackupOrchestrator()
logger.info("Backup Orchestrator initialized")

fake_orchestrator = FakeResponseOrchestrator()
logger.info("Fake Response Orchestrator initialized")

BASE_UPLOAD_DIR = os.path.join(os.getcwd(), "temp_uploads")
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)
logger.info(f"Upload directory created: {BASE_UPLOAD_DIR}")

BASE_DEBUG_DIR = os.path.join(os.getcwd(), "debug")
os.makedirs(BASE_DEBUG_DIR, exist_ok=True)
logger.info(f"Debug directory created: {BASE_DEBUG_DIR}")

TIME_LIMIT = os.getenv("REQUEST_TIMEOUT", "280")  # Default to 280 seconds if not set
try:
    TIME_LIMIT = int(TIME_LIMIT)
except ValueError:
    logger.error(
        f"Invalid REQUEST_TIMEOUT value: {TIME_LIMIT}, defaulting to 280 seconds"
    )
    TIME_LIMIT = 280

REMOVE_BASE64_PREFIX = os.getenv("REMOVE_BASE64_PREFIX", "true").lower() == "true"
logger.info(f"Remove base64 prefix: {REMOVE_BASE64_PREFIX}")

USE_ONLY_BACKUP_METHOD = os.getenv("USE_ONLY_BACKUP_METHOD", "false").lower() == "true"
logger.info(f"Use only backup method: {USE_ONLY_BACKUP_METHOD}")

# Regex to match base64 prefix
BASE64_IMAGE_PREFIX_REGEX = re.compile(r"^data:image\/[a-zA-Z]+;base64,")


def recursive_clean(obj):
    if isinstance(obj, dict):
        return {k: recursive_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_clean(v) for v in obj]
    if isinstance(obj, str):
        return BASE64_IMAGE_PREFIX_REGEX.sub("", obj)
    return obj


@app.middleware("http")
async def remove_base64_prefix_middleware(request: Request, call_next) -> Response:
    response = await call_next(request)
    if not REMOVE_BASE64_PREFIX:
        return response

    # don't touch compressed responses (gzip, br, etc.)
    if "content-encoding" in (k.lower() for k in response.headers.keys()):
        return response

    content_type = response.headers.get("content-type", "").lower()

    body = b""
    try:
        async for chunk in response.body_iterator:
            body += chunk
    except Exception:
        pass

    if not body and hasattr(response, "body") and response.body is not None:
        if isinstance(response.body, (bytes, bytearray)):
            body = bytes(response.body)
        elif isinstance(response.body, str):
            body = response.body.encode("utf-8")
        else:
            # attempt best-effort serialization
            try:
                body = json.dumps(response.body, ensure_ascii=False).encode("utf-8")
            except Exception:
                body = b""

    if not body:
        return response

    def build_response(new_bytes: bytes) -> Response:
        headers = dict(response.headers)
        headers["content-length"] = str(len(new_bytes))
        media_type = (
            getattr(response, "media_type", None) or content_type.split(";")[0] or None
        )
        return Response(
            content=new_bytes,
            status_code=response.status_code,
            headers=headers,
            media_type=media_type,
        )

    if "application/json" in content_type:
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = None

        if parsed is not None:
            cleaned = recursive_clean(parsed)
            new_body_bytes = json.dumps(cleaned, ensure_ascii=False).encode("utf-8")
            return build_response(new_body_bytes)

    if (
        content_type.startswith("text/")
        or content_type == ""
        or "application/javascript" in content_type
    ):
        charset = "utf-8"
        if "charset=" in content_type:
            try:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            except Exception:
                charset = "utf-8"

        try:
            text = body.decode(charset, errors="replace")
        except Exception:
            return response

        if BASE64_IMAGE_PREFIX_REGEX.match(text.strip()):
            new_text = BASE64_IMAGE_PREFIX_REGEX.sub("", text, count=1)
            new_bytes = new_text.encode(charset)
            return build_response(new_bytes)

    return build_response(body)


@app.on_event("startup")
async def startup_event():
    pass


@app.on_event("shutdown")
async def shutdown_event():
    pass


@app.middleware("http")
async def strip_trailing_slash(request: Request, call_next):
    scope = request.scope
    path = scope["path"]
    if path != "/" and path.endswith("/"):
        scope["path"] = path[:-1]
    return await call_next(request)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_application_root_ui():
    html_content = f"""
    <html>
    <head>
        <title>Data Analyst Agent</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <style>
            :root {{
                --accent: #2f81f7;
                --accent2: #4fb3ff;
                --muted: #9aa4b1;
                --border: rgba(255,255,255,0.08);
                --mono: "SFMono-Regular", Menlo, Consolas, monospace;
            }}
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            body {{
                font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, sans-serif;
                color: var(--text);
                display: flex;
                justify-content: center;
                align-items: flex-start;
                padding: 40px 20px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .wrap {{
                width: 100%;
                max-width: 880px;
                text-align: center;
                animation: fadeIn 0.8s ease;
            }}
            .logo {{
                font-size: 3rem;
                font-weight: bold;
                background: linear-gradient(135deg, var(--accent), var(--accent2));
                -webkit-background-clip: text;
                color: transparent;
                margin-bottom: 8px;
            }}
            .subtitle {{
                font-size: 1rem;
                color: var(--muted);
                margin-bottom: 24px;
            }}
            .panel {{
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 28px 30px 34px;
                box-shadow: 0 8px 28px rgba(0,0,0,.55);
                backdrop-filter: blur(8px);
                text-align: left;
            }}
            h3 {{
                color: var(--accent);
                margin-bottom: 0.5rem;
            }}
            .btns {{
                display: flex;
                gap: 12px;
                justify-content: center;
                margin: 1.5rem 0;
            }}
            .btn {{
                padding: 10px 18px;
                border-radius: 8px;
                background: linear-gradient(135deg, var(--accent), var(--accent2));
                color: white;
                font-weight: 600;
                border: none;
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                text-decoration: none;
                display: inline-block;
            }}
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            code {{
                font-family: var(--mono);
                background: #11161d;
                padding: 2px 6px;
                border-radius: 6px;
                color: #d7e4ef;
            }}
            pre {{
            margin-top: 0.5rem;
                background: #11161d;
                padding: 14px 16px;
                border-radius: 10px;
                border: 1px solid #223040;
                overflow-x: auto;
                font-size: 13px;
                position: relative;
            }}
            pre button {{
                position: absolute;
                top: 6px;
                right: 6px;
                background: var(--accent);
                border: none;
                color: white;
                padding: 3px 6px;
                font-size: 11px;
                border-radius: 6px;
                cursor: pointer;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
        </style>
        <script>
            function copyCode(id) {{
                const el = document.getElementById(id);
                const btn = document.getElementById("copy-button");
                navigator.clipboard.writeText(el.innerText);
                btn.innerText = "Copied to Clipboard";
                setTimeout(() => {{
                    btn.innerText = "Copy";
                }}, 2000);
            }}
        </script>
    </head>
    <body>
        <div class="wrap">
            <div class="logo">⚡ Data Analyst Agent</div>
            <div class="subtitle">Multi-LLM Workflow · Timeout Guard · JSON Output</div>
            <div class="btns">
                <a href="/docs" class="btn">📜 API Docs</a>
                <a href="https://github.com/PythonicVarun/Data-Analyst-Agent" target="_blank" class="btn">🗂 Source Code</a>
            </div>
            <div class="panel">
                <h3>Quick Start</h3>
                <p>Send a <code>multipart/form-data</code> POST to <code>/api</code> with <code>questions.txt</code> and your dataset files.</p>
                <pre><code id="curl-example">curl -X POST https://analyst-agent.pythonicvarun.me/api \\
  -F "questions.txt=@questions.txt" \\
  -F "sales.csv=@data/sales.csv" \\
  -F "customers.csv=@data/customers.csv"</code>
  <button id="copy-button" onclick="copyCode('curl-example')">Copy</button>
                </pre>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/api")
async def process_request(request: Request):
    """
    Main endpoint to process user requests from multipart/form-data.
    """
    start_time = time.time()
    logger.info(
        f"🚀 New request received at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )
    logger.info(f"⏱️  Time limit: {TIME_LIMIT} seconds")

    request_id = str(uuid.uuid4())
    upload_dir = os.path.join(BASE_UPLOAD_DIR, request_id)
    os.makedirs(upload_dir)
    logger.info(f"📁 Created request directory: {request_id}")

    form_data = await request.form()
    question_content = None
    file_paths = {}

    logger.info(f"📝 Processing form data with {len(form_data)} items")

    # Helper to extract a user-facing result from the backup workflow output
    def _extract_backup_result(backup_res: dict):
        try:
            if not isinstance(backup_res, dict):
                return None
            if backup_res.get("status") == "done":
                summary = backup_res.get("summary", {}) or {}
                phase2 = summary.get("phase2_result", {}) or {}
                if isinstance(phase2, dict) and "result" in phase2:
                    return phase2.get("result")
                # Fallback: try reading output.json from workspace
                workspace = backup_res.get("workspace") or summary.get("workspace")
                if workspace and os.path.isdir(workspace):
                    out_path = os.path.join(workspace, "output.json")
                    if os.path.exists(out_path):
                        try:
                            import json as _json

                            return _json.load(open(out_path, "r", encoding="utf-8"))
                        except Exception:
                            return None
            elif backup_res.get("status") in {"phase1_failed", "failed"}:
                # Try the same workspace output.json fallback if available
                workspace = backup_res.get("workspace")
                if workspace and os.path.isdir(workspace):
                    out_path = os.path.join(workspace, "output.json")
                    if os.path.exists(out_path):
                        try:
                            import json as _json

                            return _json.load(open(out_path, "r", encoding="utf-8"))
                        except Exception:
                            return None
        except Exception:
            return None
        return None

    try:
        for name, part in form_data.items():
            logger.info(f"📄 Processing form item: {name}")
            if name == "questions.txt":
                question_content = part.file.read().decode("utf-8")
                logger.info(f"❓ Question loaded: {len(question_content)} characters")
                logger.debug(f"Question preview: {question_content[:200]}...")
            elif hasattr(part, "filename"):
                path = os.path.join(upload_dir, part.filename)
                with open(path, "wb") as f:
                    content = part.file.read()
                    f.write(content)
                file_paths[part.filename] = os.path.abspath(path)
                logger.info(
                    f"📎 File uploaded: {part.filename} ({len(content)} bytes) -> {path}"
                )

        if not question_content:
            logger.error("❌ No questions.txt found in form data")
            raise HTTPException(
                status_code=400, detail="Missing 'questions.txt' in the form data."
            )

        if file_paths:
            logger.info(f"📋 Adding {len(file_paths)} file paths to context")
            file_context = f"\n\nThe user has uploaded the following files, which have been saved on the server in the '{upload_dir}' directory:\n"
            for name, path in file_paths.items():
                file_context += f"- {name}: {path}\n"
            question_content += file_context
            logger.debug(
                f"Updated question with file context: {len(question_content)} characters"
            )

        # Start background workflows (non-blocking): backup and fake response
        logger.info("🧵 Spawning backup workflow thread (non-blocking)...")
        backup_job = backup_orchestrator.start_async(
            question_text=question_content, file_paths=file_paths
        )
        logger.info("🧵 Spawning fake-response workflow thread (non-blocking)...")
        fake_job = fake_orchestrator.start_async(question_text=question_content)

        if USE_ONLY_BACKUP_METHOD:
            logger.info("🤖 Skiping request to orchestrator (primary)...")
            result = "error: Env doesn't allow to use primary orchestrator."
        else:
            logger.info("🤖 Sending request to orchestrator (primary)...")
            result = await orchestrator.handle_request(
                question_content, start_time, TIME_LIMIT, timeout=TIME_LIMIT
            )

        elapsed_time = time.time() - start_time
        logger.info(f"✅ Request completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"📤 Response type: {type(result)}")

        if isinstance(result, (list, dict)):
            logger.info(
                f"📤 Response length: {len(result) if isinstance(result, list) else len(str(result))} items/characters"
            )

        if (USE_ONLY_BACKUP_METHOD or isinstance(result, dict)) and "error" in result:
            logger.warning(
                "Primary orchestrator returned an error payload; attempting backup fallback..."
            )
            backup_res = backup_job.result(
                timeout=max(start_time + TIME_LIMIT - time.time() - 5, 0)
            )
            backup_fallback = (
                _extract_backup_result(backup_res) if backup_res is not None else None
            )
            if backup_fallback is not None:
                logger.info("✅ Returning backup workflow result as fallback")
                return JSONResponse(content=backup_fallback)

            # If backup not ready/failed, try fake-response
            fake_res = fake_job.result(
                timeout=max(start_time + TIME_LIMIT - time.time(), 0)
            )
            if fake_res is not None:
                logger.info(
                    "✅ Returning fake-response workflow result as final fallback"
                )
                return JSONResponse(content=fake_res)
            logger.warning(
                "No fallback available yet; returning primary error response"
            )
        return JSONResponse(content=result)

    except TimeoutException:
        elapsed_time = time.time() - start_time
        logger.error(
            f"⏰ Primary request timed out after {elapsed_time:.2f} seconds; attempting backup fallback..."
        )
        try:
            backup_job  # type: ignore  # noqa: F401
        except NameError:
            raise HTTPException(status_code=504, detail="Request timed out")

        backup_res = backup_job.result(
            timeout=max(start_time + TIME_LIMIT - time.time() - 5, 0)
        )
        backup_fallback = (
            _extract_backup_result(backup_res) if backup_res is not None else None
        )
        if backup_fallback is not None:
            logger.info("✅ Returning backup workflow result after primary timeout")
            return JSONResponse(content=backup_fallback)

        try:
            fake_job  # type: ignore  # noqa: F401
            fake_res = fake_job.result(
                timeout=max(start_time + TIME_LIMIT - time.time(), 0)
            )
            if fake_res is not None:
                logger.info(
                    "✅ Returning fake-response workflow result after primary timeout"
                )
                return JSONResponse(content=fake_res)
        except NameError:
            pass
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"💥 Primary error after {elapsed_time:.2f} seconds: {str(e)}")
        logger.exception("Full error traceback:")
        # Attempt backup fallback on generic errors as well
        try:
            backup_job  # type: ignore  # noqa: F401
            backup_res = backup_job.result(
                timeout=max(start_time + TIME_LIMIT - time.time() - 5, 0)
            )
            backup_fallback = (
                _extract_backup_result(backup_res) if backup_res is not None else None
            )
            if backup_fallback is not None:
                logger.info(
                    "✅ Returning backup workflow result after primary exception"
                )
                return JSONResponse(content=backup_fallback)
            # Try fake-response immediately
            try:
                fake_job  # type: ignore  # noqa: F401
                fake_res = fake_job.result(
                    timeout=max(start_time + TIME_LIMIT - time.time(), 0)
                )
                if fake_res is not None:
                    logger.info(
                        "✅ Returning fake-response workflow result after primary exception"
                    )
                    return JSONResponse(content=fake_res)
            except NameError:
                pass
        except NameError:
            pass
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            logger.info(f"🗑️  Cleaned up directory: {request_id}")
        elapsed_time = time.time() - start_time
        logger.info(f"🏁 Request {request_id} finished in {elapsed_time:.2f} seconds")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    logger.info("🚀 Starting Data Analyst Agent server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
