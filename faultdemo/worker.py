"""One Slurm task per client; rank 0 also coordinates all three client updates."""
import argparse
import csv
import json
import os
import socket
import time
import traceback
from pathlib import Path

import torch

from .core import HistoricalCoordinator, LABELS, TinySignalCNN, apply_delta, evaluate, load_arrays, local_update
from .io import atomic_json, check_stop, read_json
from .transport import FileTransport, TCPTransport

ROOT = Path(__file__).resolve().parents[1]


def save_torch(path, value):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def emit(run, kind, message, **fields):
    event = {"time": time.time(), "kind": kind, "message": message, **fields}
    with (run / "events.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    print(message, flush=True)


def execute(run, rank, threads, supplied_config=None):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    run.mkdir(parents=True, exist_ok=True)
    config = supplied_config if supplied_config is not None else read_json(run / "config.json")
    if supplied_config is not None:
        atomic_json(run / "config.json", config)
        (run / "config.json").chmod(0o600)
    check_stop(run)
    started = time.monotonic()
    model = TinySignalCNN().cpu()
    x, y = load_arrays(ROOT / "data" / f"client_{rank}.npz")
    node = {"rank": rank, "hostname": socket.gethostname(), "threads": threads,
            "samples": len(y), "pid": os.getpid()}
    atomic_json(run / f"node-{rank}.json", node)
    print(f"rank={rank} host={socket.gethostname()} CPU threads={threads} train={len(y)}", flush=True)
    kind = config.get("transport", "file")
    if kind not in ("file", "tcp"):
        raise ValueError("Unknown transport")
    transport = (TCPTransport if kind == "tcp" else FileTransport)(run, rank, config, model, node, save_torch)
    try:
        train(run, rank, config, model, x, y, started, transport)
    finally:
        transport.close()


def train(run, rank, config, model, x, y, started, transport):
    rounds = config["rounds"]
    if rank == 0:
        nodes = transport.nodes
        for info in nodes:
            atomic_json(run / f"node-{info['rank']}.json", info)
        if config["mode"] == "slurm" and len({n["hostname"] for n in nodes}) != 3:
            raise RuntimeError("Slurm 模式必须启动在三个不同节点，请检查 --ntasks-per-node=1")
        coordinator = HistoricalCoordinator(config["tau"], config["history_lambda"])
        test_x, test_y = load_arrays(ROOT / "data" / "test.npz")
        trajectory = []
        state = {"status": "running", "round": 0, "rounds": rounds, "nodes": nodes,
                 "metrics": [], "elapsed": 0, "predictions": 0}
        atomic_json(run / "state.json", state)
        emit(run, "start", "三客户端已就绪，CPU 联邦训练开始；节点 0 同时负责聚合。")
        emit(run, "transport", f"通信方式：{config.get('transport', 'file').upper()}；每轮按节点 0、1、2 顺序聚合。")
    for step in range(1, rounds + 1):
        transport.load_global(model, step)
        baseline = {key: value.clone() for key, value in model.state_dict().items()} if rank == 0 else None
        delta, loss = local_update(model, x, y, config["epochs"], config["lr"], 16,
                                   42 + step * 3 + rank, lambda: check_stop(run))
        print(f"round={step} rank={rank} loss={loss:.5f}", flush=True)
        uploads = transport.exchange(step, delta, loss)
        if rank != 0:
            continue
        for node, upload in enumerate(uploads):
            emit(run, "client", f"第 {step:02d} 轮 · 节点 {node} 本地训练完成，loss={upload['loss']:.4f}",
                 round=step, client=node, loss=upload["loss"])
        # Reset to the pre-round global model before applying weighted client deltas.
        model.load_state_dict(baseline)
        update, detail = coordinator.aggregate([u["delta"] for u in uploads])
        apply_delta(model, update)
        metrics, _ = evaluate(model, test_x, test_y)
        record = {"round": step, "loss": sum(u["loss"] for u in uploads) / 3,
                  **metrics, **detail}
        trajectory.append(record)
        weights = ", ".join(f"节点{i}={w:.3f}" for i, w in enumerate(detail["weights"]))
        emit(run, "aggregate", f"第 {step:02d} 轮 · 冲突投影 {detail['conflicts']} 对 · 历史相似度权重：{weights}",
             round=step, **detail)
        emit(run, "metric", f"第 {step:02d} 轮 · 演示测试集准确率 {metrics['accuracy']:.1%} · Macro-F1 {metrics['macro_f1']:.3f}",
             **record)
        # Save the last round, rather than selecting against the demonstration test set.
        transport.publish(model, step)
        state.update(round=step, metrics=trajectory, elapsed=round(time.monotonic() - started, 2))
        atomic_json(run / "state.json", state)
        # Existing system curve endpoints expect zero-based epoch JSON objects.
        atomic_json(run / "acc_data.json", {str(i): row["accuracy"] for i, row in enumerate(trajectory)})
        atomic_json(run / "loss_data.json", {str(i): row["loss"] for i, row in enumerate(trajectory)})
    transport.close(success=True)
    if rank != 0:
        return
    check_stop(run)
    save_torch(run / "model.pt", {"state_dict": model.state_dict(), "labels": LABELS,
                                 "window_size": 2048, "algorithm": "FedSimGHHistNoSize",
                                 "rounds": rounds, "config": {k: v for k, v in config.items() if k != "tcp_token"}})
    state["status"] = "predicting"
    atomic_json(run / "state.json", state)
    emit(run, "predict_start", f"训练完成，使用最后一轮模型逐条诊断 {len(test_y)} 条独立测试样本。")
    final_metrics, probabilities = evaluate(model, test_x, test_y)
    predictions = []
    for i, (p, actual) in enumerate(zip(probabilities, test_y)):
        check_stop(run)
        pred = int(p.argmax())
        row = {"id": i, "true_label": int(actual), "true_name": LABELS[int(actual)],
               "predicted_label": pred, "predicted_name": LABELS[pred], "confidence": float(p[pred]),
               "probabilities": p.tolist(), "correct": pred == int(actual)}
        predictions.append(row)
        emit(run, "prediction", f"样本 {i:03d} → {LABELS[pred]} · 预测概率 {float(p[pred]):.1%} · 真实类别 {LABELS[int(actual)]} · {'正确' if row['correct'] else '错误'}", **row)
    atomic_json(run / "predictions.json", predictions)
    with (run / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "true_name", "predicted_name", "confidence", "correct"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in predictions)
    state.update(status="completed", predictions=len(predictions), final_metrics=final_metrics,
                 elapsed=round(time.monotonic() - started, 2))
    atomic_json(run / "state.json", state)
    emit(run, "complete", f"演示完成 · {len(predictions)} 条诊断已输出 · CPU 作业耗时 {state['elapsed']:.1f} 秒。")
    # All ranks have finished using these protocol files by this point.
    for pattern in ("delta-*.pt", "global-*.pt"):
        for path in run.glob(pattern):
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    parser.add_argument("--config-env", action="store_true",
                        help="Read FEDSIMGH_WORKER_CONFIG; each node can use a private output directory")
    args = parser.parse_args()
    if args.rank not in range(3):
        parser.error("Expected exactly three ranks: 0, 1, 2")
    run = args.run_dir.resolve()
    try:
        supplied = json.loads(os.environ["FEDSIMGH_WORKER_CONFIG"]) if args.config_env else None
        execute(run, args.rank, args.threads, supplied)
    except InterruptedError:
        if args.rank == 0:
            state = read_json(run / "state.json", {})
            state.update(status="cancelled", message="任务已停止")
            atomic_json(run / "state.json", state)
        return
    except Exception as error:
        atomic_json(run / f"error-rank-{args.rank}.json", {"rank": args.rank, "error": str(error)})
        if args.rank == 0:
            state = read_json(run / "state.json", {})
            state.update(status="failed", message=str(error))
            atomic_json(run / "state.json", state)
            emit(run, "error", f"任务失败：{error}")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
