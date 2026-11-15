#!/usr/bin/env python3
"""Extract a reproducible small subset; never mix the upstream train/test splits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
QUOTAS = [[48, 24, 12, 12], [12, 48, 24, 12], [12, 12, 24, 48]]


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    origins = {}
    pools = {}
    for split in ("train", "test"):
        xs, ys, refs = [], [], []
        for path in sorted((source / split).glob("*.npz"), key=lambda p: int(p.stem)):
            # Only this offline conversion reads trusted upstream object arrays.
            with np.load(path, allow_pickle=True) as archive:
                data = archive["data"].item()
            xs.append(data["x"].reshape(-1, 2048).astype(np.float32))
            ys.append(data["y"].astype(np.int64))
            refs.extend(f"{split}/{path.name}:{i}" for i in range(len(data["y"])))
        if not xs:
            raise ValueError(f"No source data in {source / split}")
        pools[split] = (np.concatenate(xs), np.concatenate(ys), refs)
    tx, ty, tr = pools["train"]
    indices = {c: rng.permutation(np.flatnonzero(ty == c)) for c in range(4)}
    cursors = [0] * 4
    train_fingerprints = set()
    for client, quotas in enumerate(QUOTAS):
        selected = []
        for label, count in enumerate(quotas):
            start = cursors[label]
            part = indices[label][start:start + count]
            if len(part) != count:
                raise ValueError("Insufficient training samples")
            selected.extend(part)
            cursors[label] += count
        selected = np.asarray(selected)[rng.permutation(sum(quotas))]
        np.savez_compressed(output / f"client_{client}.npz", x=tx[selected], y=ty[selected])
        origins[f"client_{client}"] = [tr[i] for i in selected]
        train_fingerprints.update(hashlib.sha256(tx[i].tobytes()).hexdigest() for i in selected)
    vx, vy, vr = pools["test"]
    selected = []
    for label in range(4):
        candidates = rng.permutation(np.flatnonzero(vy == label))
        candidates = [i for i in candidates if hashlib.sha256(vx[i].tobytes()).hexdigest() not in train_fingerprints]
        if len(candidates) < 20:
            raise ValueError("Insufficient independent test samples")
        selected.extend(candidates[:20])
    selected = np.asarray(selected)[rng.permutation(80)]
    np.savez_compressed(output / "test.npz", x=vx[selected], y=vy[selected])
    origins["test"] = [vr[i] for i in selected]
    samples = [{"id": i, "true_label": int(vy[j]), "true_name": LABELS[int(vy[j])],
                "waveform": np.round(vx[j][::16], 4).tolist()} for i, j in enumerate(selected)]
    (output / "samples.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    config = json.loads((source / "config.json").read_text())
    manifest = {
        "dataset": "HUST small subset", "seed": 42, "labels": LABELS,
        "window_size": 2048, "channels": 1, "signal_channel": config["signal_channel"],
        "clients": 3, "train_counts": QUOTAS, "train_samples": 288, "test_samples": 80,
        "preprocessing": "Per-window (x - mean) / max(std, 1e-6); inherited from PFLlib HUST.",
        "split": "Subset upstream train and test independently, then repartition train into 3 logical non-IID clients.",
        "source_split_strategy": config["split_strategy"],
        "source_chunk_length": config["source_chunk_length"],
        "source_repository": "https://github.com/Patton0829/pfllib",
        "origins": origins,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.npz")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train": 288, "test": 80, "bytes": sum(p.stat().st_size for p in output.iterdir())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    prepare(args.source, args.output)
