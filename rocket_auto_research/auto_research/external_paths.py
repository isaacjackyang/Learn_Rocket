from __future__ import annotations

from pathlib import Path


DEFAULT_CHALLENGE_REPO = Path(".external/BalloonPoppingChallenge")
DEFAULT_CHALLENGE_ACTIVEROCKETPY = DEFAULT_CHALLENGE_REPO / "ActiveRocketPy"
DEFAULT_VENDOR_ACTIVEROCKETPY = Path("vendor/ActiveRocketPy")


def balloon_challenge_setup_hint() -> str:
    return (
        "Official challenge dependencies are missing. "
        "Run `python experiments/bootstrap_external_dependencies.py` "
        "or `bootstrap_external.cmd` first."
    )


def activerocketpy_setup_hint() -> str:
    return (
        "ActiveRocketPy dependency is missing. "
        "Run `python experiments/bootstrap_external_dependencies.py` "
        "or `bootstrap_external.cmd` first."
    )


def resolve_first_existing_path(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).resolve()
        if path.exists():
            return path
    return None
