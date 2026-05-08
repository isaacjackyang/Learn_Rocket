from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rocket_auto_research.dashboard_server import serve_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Auto Research dashboard with local control APIs.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    args = parser.parse_args()

    server = serve_dashboard(repo_root=ROOT, host=args.host, port=args.port)
    _safe_print(f"Dashboard server running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _safe_print(message: str) -> None:
    stream = getattr(sys, "stdout", None)
    if stream is None:
      return
    try:
      print(message)
    except OSError:
      return


if __name__ == "__main__":
    main()
