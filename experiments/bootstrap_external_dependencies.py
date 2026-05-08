from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rocket_auto_research.auto_research.external_paths import DEFAULT_CHALLENGE_REPO


BALLOON_CHALLENGE_REPO_URL = "https://github.com/ARRC-Rocket/BalloonPoppingChallenge.git"


def run_git(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch external repositories required by Rocket Auto Research.")
    parser.add_argument(
        "--challenge-root",
        default=str((ROOT / DEFAULT_CHALLENGE_REPO).resolve()),
        help="Target checkout path for BalloonPoppingChallenge.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Pull latest changes if the external repo already exists.",
    )
    args = parser.parse_args()

    challenge_root = Path(args.challenge_root).resolve()
    challenge_parent = challenge_root.parent
    challenge_parent.mkdir(parents=True, exist_ok=True)

    if not challenge_root.exists():
        run_git(["clone", "--depth", "1", BALLOON_CHALLENGE_REPO_URL, str(challenge_root)])
    elif args.refresh:
        run_git(["pull", "--ff-only"], cwd=challenge_root)

    run_git(["submodule", "update", "--init", "--depth", "1"], cwd=challenge_root)

    print(
        f"External dependencies are ready.\n"
        f"BalloonPoppingChallenge: {challenge_root}\n"
        f"ActiveRocketPy: {challenge_root / 'ActiveRocketPy'}"
    )


if __name__ == "__main__":
    main()
