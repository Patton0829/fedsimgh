"""Small shared-filesystem protocol, with atomic publication of every artifact."""
import json
import os
import time
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def check_stop(run):
    if (run / "STOP").exists():
        raise InterruptedError("任务已停止")
    failures = list(run.glob("error-rank-*.json"))
    if failures:
        error = read_json(failures[0])
        raise RuntimeError(f"客户端 {error['rank']} 失败：{error['error']}")


def wait_for(path, run, timeout=300):
    deadline = time.monotonic() + timeout
    while True:
        check_stop(run)
        if path.exists():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"等待超时：{path.name}。请检查 Slurm 节点日志和共享目录。")
        time.sleep(0.1)
