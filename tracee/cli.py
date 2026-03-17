"""Command line interface for tracee."""

import argparse
import importlib.util


def main():
    parser = argparse.ArgumentParser(
        prog="tracee",
        description="tracee - MAS tracing and visualization toolkit",
    )
    subcommands = parser.add_subparsers(dest="command")

    serve = subcommands.add_parser("serve", help="start the tracee server and UI")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="0.0.0.0")

    args = parser.parse_args()
    if args.command == "serve":
        if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None:
            parser.error("`tracee serve` requires server extras. Install with `pip install 'tracee[server]'`.")

        import uvicorn

        uvicorn.run("server.app:app", host=args.host, port=args.port)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
