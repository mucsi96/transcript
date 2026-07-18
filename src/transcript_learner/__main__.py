"""CLI entrypoint: launch the web app or list audio devices."""

from __future__ import annotations

import argparse
import sys

from .config import Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="transcript-learner",
        description="Live German transcript + vocabulary learning helper.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit.",
    )
    parser.add_argument("--host", default=None, help="Override bind host.")
    parser.add_argument("--port", type=int, default=None, help="Override bind port.")
    args = parser.parse_args(argv)

    if args.list_devices:
        from .audio import list_devices

        print(list_devices())
        return 0

    config = Config.load()
    host = args.host or config.host
    port = args.port or config.port

    if not config.speechmatics_api_key:
        print("WARNING: SPEECHMATICS_API_KEY is not set — recording will fail.", file=sys.stderr)
    if not config.openai_api_key:
        print("WARNING: OPENAI_API_KEY is not set — AI filtering will be skipped.", file=sys.stderr)

    import uvicorn

    from .server import create_app

    app = create_app(config)
    print(f"\n  German Transcript Learner running at http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
