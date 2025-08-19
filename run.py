import asyncio
import sys
import uvicorn
import argparse

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    parser = argparse.ArgumentParser(description="Run the ASGI app with uvicorn.")
    parser.add_argument("--app", default="app.main:app", help="ASGI app import string")
    parser.add_argument("--host", default="127.0.0.1", help="Bind socket to this host")
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind socket to this port"
    )
    parser.add_argument(
        "--reload", dest="reload", action="store_true", help="Enable auto-reload"
    )
    parser.add_argument(
        "--no-reload", dest="reload", action="store_false", help="Disable auto-reload"
    )
    parser.set_defaults(reload=False)
    args = parser.parse_args()

    # Thanks to https://stackoverflow.com/a/78846178 ;)
    # For windows keep the --no-reload
    if sys.platform == "win32":
        args.reload = False

    uvicorn.run(args.app, host=args.host, port=args.port, reload=args.reload)
