from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ResearchStopRequested(RuntimeError):
    """Raised when the dashboard requests a cooperative research stop."""


class ResearchRuntimeControl:
    def __init__(
        self,
        control_dir: str | Path,
        poll_interval_s: float = 1.0,
        min_status_write_interval_s: float = 2.0,
    ) -> None:
        self.control_dir = Path(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.control_dir / "status.json"
        self.command_path = self.control_dir / "command.json"
        self.poll_interval_s = poll_interval_s
        self.min_status_write_interval_s = max(0.0, float(min_status_write_interval_s))
        self._cached_status: dict[str, Any] | None = None
        self._cached_status_mtime_ns: int | None = None
        self._last_status_write_monotonic = 0.0

    def begin_run(
        self,
        *,
        config_path: str,
        total_generations: int | None,
        population_size: int,
        configured_workers: int = 1,
        log_path: str | None = None,
    ) -> None:
        self.clear_command()
        self.update_status(
            status="running",
            message="Auto research started.",
            config_path=config_path,
            total_generations=total_generations,
            continuous=total_generations is None,
            population_size=population_size,
            configured_workers=configured_workers,
            active_workers=1,
            worker_mode="serial",
            current_generation=0,
            completed_experiments=0,
            log_path=log_path,
            started_at=self._timestamp(),
            force=True,
        )

    def update_status(self, **fields: Any) -> dict[str, Any]:
        force = bool(fields.pop("force", False))
        payload = dict(self._status_payload())
        payload.update(fields)
        payload["updated_at"] = self._timestamp()
        self._cached_status = payload
        now = time.monotonic()
        should_write = force or (now - self._last_status_write_monotonic >= self.min_status_write_interval_s)
        if should_write:
            self._write_json(self.status_path, payload)
            self._last_status_write_monotonic = now
        return payload

    def read_status(self) -> dict[str, Any]:
        return dict(self._status_payload())

    def request_pause(self) -> None:
        self._write_command("pause")

    def request_stop(self) -> None:
        self._write_command("stop")

    def clear_command(self) -> None:
        if self.command_path.exists():
            self.command_path.unlink()

    def current_action(self) -> str | None:
        payload = self._read_json(self.command_path)
        if not isinstance(payload, dict):
            return None
        action = str(payload.get("action") or "").strip().lower()
        return action or None

    def wait_if_paused(self, *, allow_stop: bool = True) -> None:
        emitted_paused = False
        while True:
            action = self.current_action()
            if allow_stop and action == "stop":
                self.update_status(status="stopped", message="Stop requested by dashboard.", force=True)
                raise ResearchStopRequested("Stop requested by dashboard.")
            if action != "pause":
                if emitted_paused:
                    self.update_status(status="running", message="Auto research resumed.", force=True)
                return
            if not emitted_paused:
                self.update_status(status="paused", message="Paused by dashboard.", force=True)
                emitted_paused = True
            time.sleep(self.poll_interval_s)

    def check_stop_requested(self) -> None:
        if self.current_action() == "stop":
            self.update_status(status="stopped", message="Stop requested by dashboard.", force=True)
            raise ResearchStopRequested("Stop requested by dashboard.")

    def mark_completed(self, *, best_experiment_id: str | None = None, strategy_name: str | None = None) -> None:
        self.clear_command()
        self.update_status(
            status="completed",
            message="Auto research completed.",
            best_experiment_id=best_experiment_id,
            best_strategy_name=strategy_name,
            completed_at=self._timestamp(),
            force=True,
        )

    def mark_stopped(self) -> None:
        self.clear_command()
        self.update_status(
            status="stopped",
            message="Auto research stopped.",
            completed_at=self._timestamp(),
            force=True,
        )

    def mark_error(self, message: str) -> None:
        self.clear_command()
        self.update_status(
            status="error",
            message=message,
            completed_at=self._timestamp(),
            force=True,
        )

    def _write_command(self, action: str) -> None:
        self._write_json(
            self.command_path,
            {
                "action": action,
                "updated_at": self._timestamp(),
            },
        )

    def flush_status(self) -> dict[str, Any]:
        payload = dict(self._status_payload())
        self._write_json(self.status_path, payload)
        self._last_status_write_monotonic = time.monotonic()
        return payload

    def _status_payload(self) -> dict[str, Any]:
        try:
            current_mtime_ns = self.status_path.stat().st_mtime_ns if self.status_path.exists() else None
        except OSError:
            current_mtime_ns = None
        if self._cached_status is not None and self._cached_status_mtime_ns == current_mtime_ns:
            return self._cached_status
        payload = self._read_json(self.status_path)
        self._cached_status = payload if isinstance(payload, dict) else {}
        self._cached_status_mtime_ns = current_mtime_ns
        return self._cached_status

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp_path.replace(path)
        if path == self.status_path:
            self._cached_status = dict(payload)
            try:
                self._cached_status_mtime_ns = path.stat().st_mtime_ns
            except OSError:
                self._cached_status_mtime_ns = None

    def _read_json(self, path: Path, attempts: int = 3) -> dict[str, Any] | None:
        for attempt in range(attempts):
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                if attempt == attempts - 1:
                    return None
                time.sleep(0.01)
        return None

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
