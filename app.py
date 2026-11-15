#!/usr/bin/env python3
"""Standard-library control plane. Training/inference always run in worker tasks.

No PyTorch is imported by this web process. Slurm mode submits three compute
nodes and exchanges artifacts over their shared filesystem.
"""
import argparse
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from faultdemo.io import atomic_json, read_json

ROOT = Path(__file__).resolve().parent
RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-f0-9]{6}$")
TERMINAL = {"completed", "failed", "cancelled"}


class ControlPlane:
    def __init__(self, runtime, mode="slurm", threads=2):
        self.runtime = Path(runtime).resolve()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.threads = threads
        self.lock = threading.RLock()
        self.children = {}
        self.poll_cache = {}

    def directory(self, run_id):
        if not RUN_ID.fullmatch(run_id or ""):
            raise ValueError("无效的任务编号")
        path = self.runtime / run_id
        if not (path / "config.json").is_file():
            raise FileNotFoundError("任务不存在")
        return path

    def latest(self):
        paths = sorted(self.runtime.glob("*/config.json"), reverse=True)
        return paths[0].parent if paths else None

    def info(self):
        manifest = read_json(ROOT / "data/manifest.json")
        return {"mode": self.mode, "threads": self.threads, "dataset": manifest,
                "slurm_available": shutil.which("sbatch") is not None,
                "samples": read_json(ROOT / "data/samples.json", [])}

    def _refresh(self, run):
        state = read_json(run / "state.json", {})
        if state.get("status") in TERMINAL:
            return state
        errors = list(run.glob("error-rank-*.json"))
        if errors:
            error = read_json(errors[0])
            state.update(status="failed", message=f"节点 {error['rank']} 失败：{error['error']}")
            atomic_json(run / "state.json", state)
            return state
        processes = self.children.get(run.name, [])
        if any(p.poll() is not None and p.returncode != 0 for p in processes):
            state.update(status="failed", message="本地 worker 异常退出，请下载节点日志。")
            (run / "STOP").touch()
            atomic_json(run / "state.json", state)
        if self.mode == "slurm":
            meta = read_json(run / "submission.json", {})
            job_id = meta.get("job_id")
            if job_id and time.monotonic() - self.poll_cache.get(run.name, 0) > 5:
                self.poll_cache[run.name] = time.monotonic()
                try:
                    result = subprocess.run(["squeue", "--noheader", "--jobs", job_id,
                                             "--format=%T|%R"], capture_output=True, text=True, timeout=10)
                    if result.returncode:
                        state["scheduler_message"] = result.stderr.strip()[:500]
                    elif result.stdout.strip():
                        phase, _, reason = result.stdout.strip().splitlines()[0].partition("|")
                        state.update(scheduler_state=phase, scheduler_message=reason)
                    else:
                        # A finished job can vanish from squeue before its files are read.
                        state = read_json(run / "state.json", state)
                        if state.get("status") not in TERMINAL:
                            meta["missing_polls"] = meta.get("missing_polls", 0) + 1
                            atomic_json(run / "submission.json", meta)
                            if meta["missing_polls"] >= 3:
                                state.update(status="failed", message="Slurm 作业已退出但未完成。请下载日志检查环境、超时或内存限制。")
                                atomic_json(run / "state.json", state)
                except (OSError, subprocess.TimeoutExpired) as error:
                    state["scheduler_message"] = f"调度器查询暂时失败：{error}"
        return state

    def snapshot(self, run_id=None, offset=0):
        with self.lock:
            run = self.directory(run_id) if run_id else self.latest()
            if run is None:
                return {"run_id": None, "state": {"status": "idle"}, "events": [], "offset": 0, "predictions": []}
            state = self._refresh(run)
            events_path = run / "events.jsonl"
            events = []
            next_offset = 0
            if events_path.exists():
                with events_path.open("rb") as stream:
                    stream.seek(min(max(offset, 0), events_path.stat().st_size))
                    while True:
                        position = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith(b"\n"):
                            stream.seek(position)
                            break
                        events.append(json.loads(line))
                    next_offset = stream.tell()
            return {"run_id": run.name, "state": state, "config": read_json(run / "config.json"),
                    "submission": read_json(run / "submission.json", {}), "events": events,
                    "offset": next_offset, "predictions": read_json(run / "predictions.json", [])}

    def start(self, options):
        with self.lock:
            # Refresh all unfinished runs, including jobs left after a web restart.
            for path in self.runtime.glob("*/config.json"):
                if self._refresh(path.parent).get("status") not in TERMINAL:
                    raise RuntimeError("已有任务正在排队或运行，请先停止或等待完成。")
            if self.mode == "slurm" and not shutil.which("sbatch"):
                raise ValueError("未找到 sbatch。请在可提交 Slurm 作业的主机启动，或用 --mode local 本地验证。")
            limits = {"rounds": (1, 60, 20), "epochs": (1, 3, 1)}
            config = {"mode": self.mode}
            for key, (low, high, default) in limits.items():
                value = options.get(key, default)
                if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                    raise ValueError(f"{key} 必须为 {low}–{high} 的整数")
                config[key] = value
            for key, default, low, high in [("lr", 0.03, 0.0001, 0.2), ("tau", 3.0, 0.1, 10.0),
                                             ("history_lambda", 0.6, 0.0, 0.99)]:
                value = options.get(key, default)
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                    raise ValueError(f"{key} 超出范围 {low}–{high}")
                config[key] = float(value)
            run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
            run = self.runtime / run_id
            run.mkdir()
            atomic_json(run / "config.json", config)
            atomic_json(run / "state.json", {"status": "queued", "round": 0, "rounds": config["rounds"],
                                             "metrics": [], "nodes": [], "predictions": 0})
            try:
                if self.mode == "local":
                    processes = []
                    self.children[run_id] = processes
                    env = {**os.environ, "OMP_NUM_THREADS": str(self.threads), "MKL_NUM_THREADS": str(self.threads),
                           "OPENBLAS_NUM_THREADS": str(self.threads)}
                    for rank in range(3):
                        with (run / f"rank-{rank}.log").open("wb") as log:
                            processes.append(subprocess.Popen(
                                [sys.executable, "-u", "-m", "faultdemo.worker", "--run-dir", str(run),
                                 "--rank", str(rank), "--threads", str(self.threads)], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, env=env))
                    atomic_json(run / "submission.json", {"mode": "local", "pids": [p.pid for p in processes]})
                else:
                    command = ["sbatch", "--parsable", "--chdir", str(ROOT),
                               "--cpus-per-task", str(self.threads), "--output", str(run / "slurm-%j.log")]
                    for variable, flag in [("SLURM_PARTITION", "--partition"), ("SLURM_ACCOUNT", "--account"),
                                           ("SLURM_QOS", "--qos")]:
                        if os.environ.get(variable):
                            command += [flag, os.environ[variable]]
                    command += [str(ROOT / "slurm/train.sbatch"), str(run), sys.executable]
                    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
                    if result.returncode:
                        raise RuntimeError(f"sbatch 提交失败：{result.stderr.strip()}")
                    job_id = result.stdout.strip().split(";")[0]
                    if not job_id.isdecimal():
                        raise RuntimeError(f"sbatch 返回了无法识别的作业号：{result.stdout.strip()[:200]}")
                    atomic_json(run / "submission.json", {"mode": "slurm", "job_id": job_id})
            except Exception as error:
                (run / "STOP").touch()
                atomic_json(run / "state.json", {"status": "failed", "message": str(error), "round": 0})
                raise
            return {"run_id": run_id}

    def stop(self, run_id):
        with self.lock:
            run = self.directory(run_id)
            state = self._refresh(run)
            if state.get("status") in TERMINAL:
                return {"status": state["status"]}
            (run / "STOP").touch()
            meta = read_json(run / "submission.json", {})
            if meta.get("job_id"):
                result = subprocess.run(["scancel", meta["job_id"]], capture_output=True, text=True, timeout=15)
                if result.returncode:
                    raise RuntimeError(f"已请求 worker 停止，但 scancel 失败：{result.stderr.strip()}")
            state.update(status="cancelled", message="已停止任务")
            atomic_json(run / "state.json", state)
            return {"status": "cancelled"}

    def logs(self, run_id):
        run = self.directory(run_id)
        sections = []
        for path in sorted(run.glob("*.log")):
            sections.append(f"===== {path.name} =====\n{path.read_text(encoding='utf-8', errors='replace')[-200000:]}")
        event_path = run / "events.jsonl"
        if event_path.exists():
            sections.append("===== events.jsonl =====\n" + event_path.read_text(encoding="utf-8"))
        return "\n\n".join(sections).encode("utf-8")


def make_handler(control):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200):
            self.send_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                            "application/json; charset=utf-8", status)

        def send_bytes(self, data, content_type, status=200, download=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{download}"')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/info":
                    return self.reply(control.info())
                if parsed.path == "/api/state":
                    return self.reply(control.snapshot(query.get("run", [None])[0], int(query.get("offset", [0])[0])))
                match = re.fullmatch(r"/api/runs/([^/]+)/(predictions\.csv|model\.pt|logs\.txt)", parsed.path)
                if match:
                    run_id, name = match.groups()
                    run = control.directory(run_id)
                    data = control.logs(run_id) if name == "logs.txt" else (run / name).read_bytes()
                    return self.send_bytes(data, "application/octet-stream", download=name)
                static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
                if parsed.path in static:
                    path = ROOT / "web" / static[parsed.path]
                    return self.send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "text/plain")
                self.reply({"error": "未找到资源"}, 404)
            except FileNotFoundError:
                self.reply({"error": "资源尚未生成或不存在"}, 404)
            except (ValueError, TypeError) as error:
                self.reply({"error": str(error)}, 400)

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    return self.reply({"error": "不允许跨站提交"}, 403)
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    return self.reply({"error": "请求必须使用 application/json"}, 415)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    return self.reply({"error": "无效的请求长度"}, 400)
                options = json.loads(self.rfile.read(length))
                if not isinstance(options, dict):
                    raise ValueError("请求必须为 JSON 对象")
                if self.path == "/api/runs":
                    return self.reply(control.start(options), 201)
                match = re.fullmatch(r"/api/runs/([^/]+)/stop", self.path)
                if match:
                    return self.reply(control.stop(match.group(1)))
                self.reply({"error": "未找到接口"}, 404)
            except (ValueError, TypeError) as error:
                self.reply({"error": str(error)}, 400)
            except FileNotFoundError as error:
                self.reply({"error": str(error)}, 404)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                self.reply({"error": str(error)}, 409)

        def log_message(self, fmt, *args):
            if "/api/state" not in str(args):
                super().log_message(fmt, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description="三节点 CPU 联邦故障诊断演示")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=["slurm", "local"], default="slurm")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime")
    args = parser.parse_args()
    control = ControlPlane(args.runtime, args.mode, args.threads)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(control))
    print(f"FaultDiag CPU demo: http://{args.host}:{args.port} · mode={args.mode} · 3 clients", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
