import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from app import ControlPlane, ROOT, make_handler
from faultdemo.core import HistoricalCoordinator, TinySignalCNN, evaluate, load_arrays
from faultdemo.io import read_json


class AggregationTests(unittest.TestCase):
    def test_two_conflicting_updates_remove_negative_components(self):
        # [1, 0] and [-1, 1] have analytically projected updates [.5, .5], [0, 1].
        coordinator = HistoricalCoordinator(tau=1)
        update, detail = coordinator.aggregate([torch.tensor([1., 0.]), torch.tensor([-1., 1.])], [0, 1])
        expected = detail["weights"][0] * torch.tensor([.5, .5]) + detail["weights"][1] * torch.tensor([0., 1.])
        torch.testing.assert_close(update, expected)
        self.assertEqual(detail["conflicts"], 1)
        self.assertGreaterEqual(float(update.min()), 0)

    def test_aligned_updates_have_uniform_weights_and_mean_magnitude(self):
        update, detail = HistoricalCoordinator().aggregate([torch.tensor([1., 0.]), torch.tensor([3., 0.]), torch.tensor([5., 0.])])
        torch.testing.assert_close(update, torch.tensor([3., 0.]))
        self.assertEqual(detail["conflicts"], 0)
        np.testing.assert_allclose(detail["weights"], [1 / 3] * 3)

    def test_history_is_carried_by_client_identity(self):
        coordinator = HistoricalCoordinator(history_lambda=.6)
        coordinator.aggregate([torch.tensor([1., 0.]), torch.tensor([0., 1.])], [8, 9])
        old = dict(coordinator.history)
        _, detail = coordinator.aggregate([torch.tensor([1., 1.]), torch.tensor([1., 0.])], [9, 8])
        for i, client_id in enumerate([9, 8]):
            self.assertAlmostEqual(detail["history"][i], .6 * old[client_id] + .4 * detail["similarities"][i])
        self.assertAlmostEqual(sum(detail["weights"]), 1.0)

    def test_zero_updates_are_finite(self):
        update, detail = HistoricalCoordinator().aggregate([torch.zeros(4) for _ in range(3)])
        self.assertTrue(torch.isfinite(update).all())
        np.testing.assert_allclose(detail["weights"], [1 / 3] * 3)


class DatasetTests(unittest.TestCase):
    def test_small_dataset_hashes_labels_and_train_test_separation(self):
        manifest = read_json(ROOT / "data/manifest.json")
        train_hashes, train_origins = set(), set()
        for i in range(3):
            path = ROOT / "data" / f"client_{i}.npz"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), manifest["sha256"][path.name])
            x, y = load_arrays(path)
            self.assertEqual(tuple(x.shape), (96, 2048))
            self.assertEqual(np.bincount(y.numpy(), minlength=4).tolist(), manifest["train_counts"][i])
            self.assertTrue(torch.isfinite(x).all())
            train_hashes.update(hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in x)
            train_origins.update(manifest["origins"][f"client_{i}"])
        tx, ty = load_arrays(ROOT / "data/test.npz")
        self.assertEqual(np.bincount(ty.numpy(), minlength=4).tolist(), [20] * 4)
        self.assertFalse(train_origins.intersection(manifest["origins"]["test"]))
        self.assertFalse(train_hashes.intersection(hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in tx))
        self.assertLess(sum(p.stat().st_size for p in (ROOT / "data").iterdir()), 4 * 1024 ** 2)


class SlurmControlTests(unittest.TestCase):
    def test_slurm_submission_queue_and_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            control = ControlPlane(temp, "slurm", threads=2)
            with patch("app.shutil.which", return_value="/usr/bin/sbatch"), patch("app.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], 0, "31415;cluster\n", "")
                result = control.start({"rounds": 2})
                command = run.call_args.args[0]
                self.assertEqual(command[0], "sbatch")
                self.assertIn(str(ROOT / "slurm/train.sbatch"), command)
                self.assertIn("--cpus-per-task", command)
                self.assertIn(sys.executable, command)
                run.return_value = subprocess.CompletedProcess([], 0, "PENDING|Resources\n", "")
                state = control.snapshot(result["run_id"])
                self.assertEqual(state["state"]["scheduler_state"], "PENDING")
                self.assertEqual(state["submission"]["job_id"], "31415")
                run.return_value = subprocess.CompletedProcess([], 0, "", "")
                self.assertEqual(control.stop(result["run_id"])["status"], "cancelled")
                self.assertEqual(run.call_args.args[0], ["scancel", "31415"])
            script = (ROOT / "slurm/train.sbatch").read_text()
            for directive in ["--nodes=3", "--ntasks=3", "--ntasks-per-node=1"]:
                self.assertIn(directive, script)
            self.assertIn("--kill-on-bad-exit=1", script)

    def test_submission_errors_do_not_leave_an_active_run(self):
        with tempfile.TemporaryDirectory() as temp:
            control = ControlPlane(temp, "slurm")
            with patch("app.shutil.which", return_value="sbatch"), patch("app.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], 1, "", "Invalid partition")
                with self.assertRaisesRegex(RuntimeError, "Invalid partition"):
                    control.start({})
                self.assertEqual(control.snapshot()["state"]["status"], "failed")

    def test_invalid_options_and_run_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            control = ControlPlane(temp, "local")
            for options in [{"rounds": 0}, {"rounds": 1001}, {"lr": float("nan")}, {"rounds": True}, {"rounds": 1.5}]:
                with self.assertRaises(ValueError):
                    control.start(options)
            with self.assertRaises(ValueError):
                control.directory("../../pfllib")


class HTTPTests(unittest.TestCase):
    def test_static_api_validation_and_cross_site_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            control = ControlPlane(temp, "local")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(control))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urllib.request.urlopen(base + "/") as response:
                    self.assertIn("FaultDiag", response.read().decode())
                with urllib.request.urlopen(base + "/api/info") as response:
                    self.assertEqual(len(json.load(response)["samples"]), 80)
                with urllib.request.urlopen(base + "/api/state") as response:
                    self.assertEqual(json.load(response)["state"]["status"], "idle")
                request = urllib.request.Request(base + "/api/runs", data=b'{"rounds": 0}',
                                                 headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request)
                self.assertEqual(error.exception.code, 400)
                request = urllib.request.Request(base + "/api/runs", data=b'{}',
                    headers={"Content-Type": "application/json", "Origin": "https://unrelated.example"})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request)
                self.assertEqual(error.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


class EndToEndTests(unittest.TestCase):
    def test_terminal_entry_prints_all_predictions_without_web_server(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, str(ROOT / "train.py"), "--rounds", "2", "--threads", "1", "--runtime", temp],
                cwd=ROOT, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            predictions = [line for line in result.stdout.splitlines() if "预测概率" in line and "样本 " in line]
            self.assertEqual(len(predictions), 80)
            self.assertIn("最终准确率", result.stdout)
            self.assertIn("predictions.csv", result.stdout)
            paths = list(Path(temp).glob("*/predictions.json"))
            self.assertEqual(len(paths), 1)
            self.assertEqual(len(read_json(paths[0])), 80)

    def test_three_real_processes_train_export_and_predict_every_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            control = ControlPlane(temp, "local", threads=1)
            run_id = control.start({"rounds": 2})["run_id"]
            with self.assertRaises(RuntimeError):
                control.start({"rounds": 1})
            deadline = time.monotonic() + 90
            try:
                while time.monotonic() < deadline:
                    snapshot = control.snapshot(run_id)
                    if snapshot["state"]["status"] in {"completed", "failed", "cancelled"}:
                        break
                    time.sleep(.1)
                self.assertEqual(snapshot["state"]["status"], "completed", control.logs(run_id).decode())
                self.assertEqual(len(snapshot["state"]["nodes"]), 3)
                self.assertEqual(len(snapshot["predictions"]), 80)
                self.assertEqual(sum(e["kind"] == "prediction" for e in snapshot["events"]), 80)
                self.assertEqual(control.snapshot(run_id, snapshot["offset"])["events"], [])
                model = TinySignalCNN()
                checkpoint = torch.load(Path(temp) / run_id / "model.pt", weights_only=True)
                model.load_state_dict(checkpoint["state_dict"])
                x, y = load_arrays(ROOT / "data/test.npz")
                _, probabilities = evaluate(model, x, y)
                np.testing.assert_allclose(probabilities.numpy(), [p["probabilities"] for p in snapshot["predictions"]], atol=1e-6)
                self.assertTrue(all(0 <= p["predicted_label"] < 4 for p in snapshot["predictions"]))
                self.assertEqual(len((Path(temp) / run_id / "predictions.csv").read_text(encoding="utf-8-sig").splitlines()), 81)
            finally:
                for process in control.children[run_id]:
                    process.wait(timeout=15)

    def test_local_cancellation(self):
        with tempfile.TemporaryDirectory() as temp:
            control = ControlPlane(temp, "local", threads=1)
            run_id = control.start({"rounds": 60})["run_id"]
            control.stop(run_id)
            for process in control.children[run_id]:
                process.wait(timeout=30)
            self.assertEqual(control.snapshot(run_id)["state"]["status"], "cancelled")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
