#!/usr/bin/env python3
"""Run the three-client demo from a terminal, without starting an HTTP server."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from app import ControlPlane, ROOT


def print_events(snapshot):
    for event in snapshot["events"]:
        timestamp = time.strftime("%H:%M:%S", time.localtime(event["time"]))
        print(f"[{timestamp}] {event['message']}", flush=True)


def finish_local_workers(control, run_id):
    # Reap only processes created by this command. A failed rank must not leave
    # the other two ranks waiting for a checkpoint indefinitely.
    for process in control.children.get(run_id, []):
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description="终端运行三客户端 CPU 联邦训练，并输出每条信号的故障类别")
    parser.add_argument("--mode", choices=["local", "slurm"], default="local")
    parser.add_argument("--rounds", type=int, default=100, help="全局训练轮次，1–1000，默认 100")
    parser.add_argument("--epochs", type=int, default=1, help="每客户端每轮本地 epoch，1–3")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=1, help="每客户端 CPU 线程数")
    parser.add_argument("--lr", type=float, default=.03)
    parser.add_argument("--tau", type=float, default=3.)
    parser.add_argument("--history-lambda", type=float, default=.6)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime" / "terminal",
                        help="输出目录；默认与网页任务分开存储")
    args = parser.parse_args(argv)
    control = ControlPlane(args.runtime, args.mode, args.threads)
    run_id = None
    try:
        options = {key: getattr(args, key) for key in ["rounds", "epochs", "lr", "tau", "history_lambda"]}
        run_id = control.start(options)["run_id"]
        run = control.directory(run_id)
        print("FedSimGHHistNoSize · 三客户端 · CPU · 终端模式", flush=True)
        print(f"全局轮次={args.rounds}，本地 epochs={args.epochs}，每客户端线程={args.threads}", flush=True)
        print("数据：288 条训练信号，80 条测试信号；类别：正常、内圈、外圈、滚动体故障。", flush=True)
        print(f"输出目录：{run}", flush=True)
        if args.mode == "slurm":
            submission = control.snapshot(run_id)["submission"]
            print(f"Slurm 作业号：{submission['job_id']}；等待三个计算节点。", flush=True)
        else:
            print("正在启动三个本地 CPU 客户端，无需网页、端口或 Slurm。", flush=True)
        offset = 0
        scheduler = None
        while True:
            snapshot = control.snapshot(run_id, offset)
            print_events(snapshot)
            offset = snapshot["offset"]
            state = snapshot["state"]
            if args.mode == "slurm":
                current = (state.get("scheduler_state"), state.get("scheduler_message"))
                if current != scheduler and any(current):
                    print(f"Slurm：{current[0] or '查询中'} · {current[1] or ''}", flush=True)
                    scheduler = current
            if state["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(.1)
        if state["status"] != "completed":
            print(f"任务{state['status']}：{state.get('message', '请查看节点日志')}", file=sys.stderr, flush=True)
            if state["status"] == "failed":
                control.stop(run_id)
                # _refresh can already mark failure before stop() is called.
                (run / "STOP").touch()
                print(control.logs(run_id).decode("utf-8"), file=sys.stderr, flush=True)
            return 130 if state["status"] == "cancelled" else 1
        finish_local_workers(control, run_id)
        # Completion state may appear just before the final event is flushed.
        snapshot = control.snapshot(run_id, offset)
        print_events(snapshot)
        metrics = snapshot["state"]["final_metrics"]
        print(f"\n最终准确率：{metrics['accuracy']:.1%} · Macro-F1：{metrics['macro_f1']:.3f}", flush=True)
        print(f"逐条预测 CSV：{run / 'predictions.csv'}", flush=True)
        print(f"模型权重：{run / 'model.pt'}", flush=True)
        return 0
    except KeyboardInterrupt:
        if run_id:
            control.stop(run_id)
        print("\n已停止训练。", flush=True)
        return 130
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"运行失败：{error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if run_id:
            finish_local_workers(control, run_id)


if __name__ == "__main__":
    raise SystemExit(main())
