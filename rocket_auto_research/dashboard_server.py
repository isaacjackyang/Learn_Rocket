from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from rocket_auto_research.auto_research.problem_definition import default_problem_stages
from rocket_auto_research.auto_research.research_memory import ResearchMemory
from rocket_auto_research.auto_research.runtime_control import ResearchRuntimeControl
from rocket_auto_research.dashboard_builder import build_dashboard


def describe_config_entry(path: Path, repo_root: Path) -> dict[str, str]:
    relative_value = str(path.relative_to(repo_root)).replace("\\", "/")
    mode = "auto_research_loop" if path.name == "auto_research.yaml" else "single_strategy_run"
    group = "Auto Research Loops" if mode == "auto_research_loop" else "Single Strategy Runs"
    description_map = {
        "auto_research.yaml": {
            "summary": "Single staged Auto Research master config.",
            "recommended_use": "Use this for normal Auto Research runs. The loop will switch stage, adapter, and scenario internally.",
        },
    }
    metadata = description_map.get(
        path.name,
        {
            "summary": "Single strategy or auxiliary config.",
            "recommended_use": "Use this when you want a fixed strategy benchmark instead of a full evolution loop.",
        },
    )
    return {
        "label": path.name,
        "value": relative_value,
        "mode": mode,
        "group": group,
        "summary": metadata["summary"],
        "recommended_use": metadata["recommended_use"],
    }


class ResearchDashboardManager:
    def __init__(
        self,
        repo_root: str | Path,
        results_dir: str | Path = "results",
        dashboard_dir: str | Path = "results/dashboard",
        control_dir: str | Path = "results/dashboard_control",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.results_dir = (self.repo_root / results_dir).resolve()
        self.dashboard_dir = (self.repo_root / dashboard_dir).resolve()
        self.control_dir = (self.repo_root / control_dir).resolve()
        self.control = ResearchRuntimeControl(self.control_dir)
        self.process: subprocess.Popen[str] | None = None
        self._log_handle = None
        self._lock = threading.RLock()
        self.build_dashboard()

    def list_configs(self) -> list[dict[str, str]]:
        configs_dir = self.repo_root / "configs"
        configs: list[dict[str, str]] = []
        for path in sorted(configs_dir.glob("*.yaml")):
            configs.append(describe_config_entry(path, self.repo_root))
        return configs

    @staticmethod
    def worker_limits() -> dict[str, int]:
        available_cpu = max(1, os.cpu_count() or 1)
        return {
            "min": 1,
            "max": 32,
            "available_cpu": available_cpu,
            "recommended": max(1, min(32, available_cpu - 1)),
        }

    def build_dashboard(self) -> Path:
        return build_dashboard(results_dir=self.results_dir, output_dir=self.dashboard_dir)

    def status(self) -> dict[str, object]:
        with self._lock:
            self._refresh_process_state()
            payload = self.control.read_status()
            if not payload:
                payload = {
                    "status": "idle",
                    "message": "Auto research is idle.",
                }
            payload.setdefault("status", "idle")
            payload.setdefault("message", "Auto research is idle.")
            payload.setdefault("configured_workers", 1)
            payload.setdefault("active_workers", 1)
            payload.setdefault("worker_mode", "serial")
            payload["worker_limits"] = self.worker_limits()
            payload["stage_context"] = self._stage_context(payload)
            payload["running"] = self._is_research_running(payload)
            payload["available_configs"] = self.list_configs()
            payload["log_tail"] = self._read_log_tail()
            return payload

    def start(
        self,
        config_value: str,
        parallel_workers: int | None = None,
        population_size: int | None = None,
    ) -> dict[str, object]:
        with self._lock:
            self._refresh_process_state()
            if self._is_research_running(self.control.read_status()):
                raise RuntimeError("Auto research is already running.")
            config_path = (self.repo_root / config_value).resolve()
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_value}")
            worker_count = self._normalize_parallel_workers(parallel_workers)
            population_count = self._normalize_population_size(population_size)
            config_entry = describe_config_entry(config_path, self.repo_root)
            self.control_dir.mkdir(parents=True, exist_ok=True)
            self.control.clear_command()
            log_path = self.control_dir / "research.log"
            if config_entry["mode"] == "auto_research_loop":
                launch_cmd = [
                    sys.executable,
                    str((self.repo_root / "experiments" / "run_research_loop.py").resolve()),
                    "--config",
                    str(config_path),
                    "--parallel-workers",
                    str(worker_count),
                    "--population-size",
                    str(population_count),
                    "--control-dir",
                    str(self.control_dir),
                ]
                start_message = (
                    f"Starting auto research with {config_path.name} "
                    f"using {worker_count} workers and population size {population_count}."
                )
            else:
                launch_cmd = [
                    sys.executable,
                    str((self.repo_root / "experiments" / "run_batch.py").resolve()),
                    "--config",
                    str(config_path),
                    "--parallel-workers",
                    str(worker_count),
                    "--seeds",
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "--control-dir",
                    str(self.control_dir),
                ]
                start_message = f"Starting single strategy batch with {config_path.name} using {worker_count} workers."
            log_handle = log_path.open("a", encoding="utf-8")
            self._log_handle = log_handle
            self.process = subprocess.Popen(
                launch_cmd,
                cwd=self.repo_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.control.update_status(
                status="starting",
                message=start_message,
                pid=self.process.pid,
                config_path=str(config_path),
                config_mode=config_entry["mode"],
                configured_workers=worker_count,
                population_size=population_count,
                active_workers=1,
                worker_mode="serial",
                log_path=str(log_path),
            )
            return self.status()

    def _normalize_parallel_workers(self, raw_value: int | None) -> int:
        if raw_value is None:
            return 1
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError("parallel_workers must be an integer between 1 and 32.")
        if not 1 <= value <= 32:
            raise ValueError("parallel_workers must be between 1 and 32.")
        return value

    def _normalize_population_size(self, raw_value: int | None) -> int:
        if raw_value is None:
            return 6
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError("population_size must be an integer between 4 and 64.")
        if not 4 <= value <= 64:
            raise ValueError("population_size must be between 4 and 64.")
        return value

    def pause(self) -> dict[str, object]:
        with self._lock:
            self._refresh_process_state()
            if not self._is_research_running(self.control.read_status()):
                raise RuntimeError("No running auto research process to pause.")
            self.control.request_pause()
            self.control.update_status(status="pausing", message="Pause requested by dashboard.")
            return self.status()

    def resume(self) -> dict[str, object]:
        with self._lock:
            self._refresh_process_state()
            if not self._is_research_running(self.control.read_status()):
                raise RuntimeError("No running auto research process to resume.")
            self.control.clear_command()
            self.control.update_status(status="running", message="Resume requested by dashboard.")
            return self.status()

    def stop(self) -> dict[str, object]:
        with self._lock:
            self._refresh_process_state()
            if not self._is_research_running(self.control.read_status()):
                raise RuntimeError("No running auto research process to stop.")
            self.control.request_stop()
            self.control.update_status(status="stopping", message="Stop requested by dashboard.")
            return self.status()

    def _refresh_process_state(self) -> None:
        if self.process is None:
            return
        exit_code = self.process.poll()
        if exit_code is None:
            return
        current_status = self.control.read_status().get("status")
        if current_status not in {"completed", "stopped", "error"}:
            next_status = "completed" if exit_code == 0 else "stopped" if exit_code == 130 else "error"
            message = {
                "completed": "Auto research completed.",
                "stopped": "Auto research stopped.",
                "error": f"Auto research exited with code {exit_code}.",
            }[next_status]
            self.control.update_status(status=next_status, message=message, exit_code=exit_code)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self.process = None
        self.build_dashboard()

    def _is_research_running(self, payload: dict[str, object]) -> bool:
        if self.process and self.process.poll() is None:
            return True
        pid = payload.get("pid")
        if not isinstance(pid, int):
            return False
        phase = str(payload.get("status") or "").strip().lower()
        if phase not in {"starting", "running", "pausing", "paused", "stopping"}:
            return False
        return self._pid_exists(pid)

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return False
        output = (result.stdout or "").strip()
        if not output:
            return False
        return "No tasks are running" not in output and "資訊: 沒有執行中的工作" not in output

    def _read_log_tail(self, max_lines: int = 20) -> str:
        log_path = self.control_dir / "research.log"
        if not log_path.exists():
            return ""
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])

    def _stage_context(self, payload: dict[str, object]) -> dict[str, object]:
        stages = default_problem_stages()
        records = ResearchMemory(self.results_dir / "research_memory").load_records()
        current_stage_id = str(payload.get("current_stage") or "").strip()
        latest_plan = self._read_latest_plan()
        plan_stage_id = str(latest_plan.get("stage_id") or "").strip() if latest_plan else ""
        stage_id = current_stage_id or plan_stage_id
        if not stage_id:
            return {}
        stage_index = next((index for index, stage in enumerate(stages) if stage.stage_id == stage_id), None)
        if stage_index is None:
            return {}
        stage = stages[stage_index]
        next_stage = stages[stage_index + 1] if stage_index + 1 < len(stages) else None
        recent_success_rate = latest_plan.get("recent_stage_success_rate") if latest_plan else None
        stage_stats = ResearchMemory.stage_stats(records, stage.stage_id)
        promotion_decision = "unknown"
        promotion_reason = "No recent stage statistics available yet."
        recent_success_value = float(recent_success_rate) if isinstance(recent_success_rate, (int, float)) else float(
            stage_stats.get("mean_stage_success_rate", 0.0)
        )
        if isinstance(recent_success_value, float):
            if recent_success_value >= stage.success_threshold:
                if next_stage is None:
                    promotion_decision = "final_stage"
                    promotion_reason = (
                        f"Current stage success {recent_success_value:.2f} meets threshold "
                        f"{stage.success_threshold:.2f}; this is the terminal stage."
                    )
                else:
                    promotion_decision = "advance"
                    promotion_reason = (
                        f"Current stage success {recent_success_value:.2f} meets threshold "
                        f"{stage.success_threshold:.2f}; planner can advance to {next_stage.stage_id}."
                    )
            else:
                promotion_decision = "stay"
                promotion_reason = (
                    f"Current stage success {recent_success_value:.2f} is below threshold "
                    f"{stage.success_threshold:.2f}; keep optimizing this stage."
                )
        pipeline = []
        for index, candidate in enumerate(stages):
            candidate_stats = ResearchMemory.stage_stats(records, candidate.stage_id)
            best_success = float(candidate_stats.get("best_stage_success_rate", 0.0))
            state = "upcoming"
            if best_success >= candidate.success_threshold:
                state = "passed"
            if index == stage_index:
                state = "current"
            pipeline.append(
                {
                    "stage_id": candidate.stage_id,
                    "title": candidate.title,
                    "threshold": candidate.success_threshold,
                    "best_success_rate": best_success,
                    "state": state,
                }
            )
        failure_breakdowns = self._failure_breakdowns(payload, records, stage.stage_id)
        return {
            "stage_id": stage.stage_id,
            "stage_title": stage.title,
            "stage_description": stage.description,
            "success_threshold": stage.success_threshold,
            "recent_stage_success_rate": recent_success_value,
            "planner_rationale": latest_plan.get("rationale") if latest_plan else None,
            "planner_notes": latest_plan.get("planner_notes") if latest_plan else [],
            "next_stage_id": next_stage.stage_id if next_stage else None,
            "next_stage_title": next_stage.title if next_stage else None,
            "promotion_decision": promotion_decision,
            "promotion_reason": promotion_reason,
            "pipeline": pipeline,
            "failure_breakdowns": failure_breakdowns,
        }

    def _read_latest_plan(self) -> dict[str, object]:
        path = self.results_dir / "research_memory" / "latest_plan.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _failure_breakdowns(
        self,
        payload: dict[str, object],
        records,
        stage_id: str,
    ) -> dict[str, list[dict[str, object]]]:
        return {
            "current_experiment": self._failure_breakdown_for_experiment(
                str(payload.get("current_experiment_id") or "").strip()
                or str(payload.get("last_finished_experiment_id") or "").strip()
            ),
            "latest_best": self._failure_breakdown_for_experiment(
                str(payload.get("best_experiment_id") or "").strip()
            ),
            "recent_stage_aggregate": self._failure_breakdown_for_stage(records, stage_id),
        }

    def _failure_breakdown_for_experiment(self, experiment_id: str) -> list[dict[str, object]]:
        if not experiment_id:
            return []
        failure_path = self.results_dir / "runs" / experiment_id / "failure_report.json"
        if not failure_path.exists():
            return []
        try:
            report = json.loads(failure_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return self._normalize_failure_report(report)

    def _failure_breakdown_for_stage(self, records, stage_id: str, limit: int = 5) -> list[dict[str, object]]:
        stage_records = [record for record in records if record.stage_id == stage_id and record.rank == 1]
        if not stage_records:
            return []
        recent_records = stage_records[-3:]
        aggregate_counts: dict[str, int] = {}
        aggregate_rates: dict[str, list[float]] = {}
        for record in recent_records:
            failure_path = self.results_dir / "runs" / record.experiment_id / "failure_report.json"
            if not failure_path.exists():
                continue
            try:
                report = json.loads(failure_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            counts = report.get("counts", {})
            rates = report.get("rates", {})
            if not isinstance(counts, dict) or not isinstance(rates, dict):
                continue
            for name, count in counts.items():
                key = str(name)
                aggregate_counts[key] = aggregate_counts.get(key, 0) + int(count)
                aggregate_rates.setdefault(key, []).append(float(rates.get(name, 0.0)))
        entries = [
            {
                "name": name,
                "count": count,
                "rate": round(sum(aggregate_rates.get(name, [0.0])) / max(1, len(aggregate_rates.get(name, []))), 4),
            }
            for name, count in aggregate_counts.items()
        ]
        entries.sort(key=lambda item: (item["rate"], item["count"]), reverse=True)
        return entries[:limit]

    @staticmethod
    def _normalize_failure_report(report: dict[str, object], limit: int = 5) -> list[dict[str, object]]:
        counts = report.get("counts", {})
        rates = report.get("rates", {})
        if not isinstance(counts, dict) or not isinstance(rates, dict):
            return []
        entries = [
            {
                "name": str(name),
                "count": int(counts.get(name, 0)),
                "rate": float(rates.get(name, 0.0)),
            }
            for name in counts
        ]
        entries.sort(key=lambda item: (item["rate"], item["count"]), reverse=True)
        return entries[:limit]


class DashboardRequestHandler(BaseHTTPRequestHandler):
    manager: ResearchDashboardManager

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_json(self.manager.status())
            return
        if parsed.path == "/api/configs":
            self._send_json({"configs": self.manager.list_configs()})
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        body = self._read_json_body()
        try:
            if parsed.path == "/api/start":
                parallel_workers = body.get("parallel_workers")
                population_size = body.get("population_size")
                self._send_json(
                    self.manager.start(
                        str(body.get("config") or ""),
                        int(parallel_workers) if parallel_workers is not None else None,
                        int(population_size) if population_size is not None else None,
                    )
                )
                return
            if parsed.path == "/api/pause":
                self._send_json(self.manager.pause())
                return
            if parsed.path == "/api/resume":
                self._send_json(self.manager.resume())
                return
            if parsed.path == "/api/stop":
                self._send_json(self.manager.stop())
                return
            if parsed.path == "/api/rebuild-dashboard":
                self.manager.build_dashboard()
                self._send_json({"status": "ok"})
                return
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json_body(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length") or 0)
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length).decode("utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(serialized)))
        self.end_headers()
        self.wfile.write(serialized)

    def _serve_static(self, raw_path: str) -> None:
        relative_path = "index.html" if raw_path in {"", "/"} else raw_path.lstrip("/")
        resolved_path = (self.manager.dashboard_dir / relative_path).resolve()
        dashboard_root = self.manager.dashboard_dir.resolve()
        if dashboard_root not in resolved_path.parents and resolved_path != dashboard_root:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not resolved_path.exists() or resolved_path.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
        }.get(resolved_path.suffix.lower(), "application/octet-stream")
        data = resolved_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve_dashboard(
    *,
    repo_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    manager = ResearchDashboardManager(repo_root=repo_root)
    handler = type("DashboardRequestHandlerInstance", (DashboardRequestHandler,), {"manager": manager})
    server = ThreadingHTTPServer((host, port), handler)
    return server
