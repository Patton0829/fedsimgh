import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from app import ControlPlane, ROOT
from faultdemo.core import TinySignalCNN, vector
from faultdemo.io import read_json
from faultdemo.transport import MAX_FRAME, TCPTransport, config_digest, recv_frame, send_frame


class ProtocolTests(unittest.TestCase):
    def pair(self):
        left, right = socket.socketpair()
        for peer in (left, right):
            peer.settimeout(.02)
            self.addCleanup(peer.close)
        return left, right

    def test_fragmented_header_payload_and_consecutive_messages(self):
        sender, receiver = self.pair()
        header = json.dumps({"kind": "test"}).encode()
        payload = torch.arange(32, dtype=torch.float32).numpy().astype("<f4").tobytes()
        body = struct.pack("!I", len(header)) + header + payload
        frame = struct.pack("!I", len(body)) + body

        def fragmented_send():
            for index in range(0, len(frame), 3):
                sender.sendall(frame[index:index + 3])
            send_frame(sender, {"kind": "next"})

        thread = threading.Thread(target=fragmented_send)
        thread.start()
        try:
            metadata, result = recv_frame(receiver, timeout=2)
            self.assertEqual(metadata, {"kind": "test"})
            self.assertEqual(result, payload)
            self.assertEqual(recv_frame(receiver, timeout=2), ({"kind": "next"}, b""))
        finally:
            thread.join(timeout=2)

    def test_oversized_and_truncated_frames_fail(self):
        sender, receiver = self.pair()
        sender.sendall(struct.pack("!I", MAX_FRAME + 1))
        with self.assertRaisesRegex(ValueError, "长度"):
            recv_frame(receiver)
        sender.sendall(struct.pack("!I", 100) + b"short")
        sender.close()
        with self.assertRaises(ConnectionError):
            recv_frame(receiver)

    def test_deadline_and_cancellation_interrupt_receiving(self):
        _, receiver = self.pair()
        with self.assertRaises(TimeoutError):
            recv_frame(receiver, timeout=.04)

        def stop():
            raise InterruptedError("stop")

        with self.assertRaises(InterruptedError):
            recv_frame(receiver, stop=stop)

    def test_old_round_and_nonfinite_or_wrong_length_updates_rejected(self):
        sender, receiver = self.pair()
        transport = TCPTransport.__new__(TCPTransport)
        transport.run_id, transport.timeout, transport.count = "run-a", 1, 2
        transport._stop = lambda: None
        send_frame(sender, {"version": 1, "run": "run-a", "kind": "update", "round": 1})
        with self.assertRaisesRegex(ValueError, "轮次"):
            transport._receive(receiver, "update", 2)
        with self.assertRaisesRegex(ValueError, "长度"):
            transport._tensor(b"1234")
        with self.assertRaisesRegex(ValueError, "NaN"):
            transport._tensor(torch.tensor([float("nan"), 0.]).numpy().tobytes())

    def handshake_failure(self, duplicate=False):
        with tempfile.TemporaryDirectory() as temp:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            config = {"tcp_port": port, "tcp_bind": "127.0.0.1", "tcp_run_id": "test",
                      "tcp_token": "test-token", "tcp_timeout": 2,
                      "rounds": 1, "epochs": 1, "lr": .03, "tau": 3., "history_lambda": .6}
            model = TinySignalCNN()
            errors = []

            def coordinator():
                try:
                    transport = TCPTransport(Path(temp), 0, config, model,
                                             {"rank": 0, "hostname": "local"}, None)
                    transport.close()
                except Exception as error:
                    errors.append(error)

            thread = threading.Thread(target=coordinator)
            thread.start()
            peers = []
            try:
                for index in range(2 if duplicate else 1):
                    deadline = time.monotonic() + 2
                    while True:
                        try:
                            peer = socket.create_connection(("127.0.0.1", port), timeout=.1)
                            break
                        except ConnectionRefusedError:
                            if time.monotonic() > deadline:
                                raise
                            time.sleep(.01)
                    peer.settimeout(.05)
                    peers.append(peer)
                    send_frame(peer, {"version": 1, "kind": "hello", "run": "test", "rank": 1,
                                      "token": "test-token" if duplicate else "wrong-token",
                                      "config": config_digest(config), "count": vector(model).numel(),
                                      "node": {"rank": 1, "hostname": "local"}})
                    if duplicate and index == 0:
                        self.assertEqual(recv_frame(peer, timeout=2)[0]["kind"], "ready")
            finally:
                thread.join(timeout=5)
                for peer in peers:
                    peer.close()
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIn("重复" if duplicate else "认证", str(errors[0]))

    def test_wrong_token_rejected(self):
        self.handshake_failure()

    def test_duplicate_client_identity_rejected(self):
        self.handshake_failure(duplicate=True)


class TCPTrainingTests(unittest.TestCase):
    def wait(self, control, run_id, terminal=True):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            snapshot = control.snapshot(run_id)
            if snapshot["state"]["status"] in {"completed", "failed", "cancelled"}:
                return snapshot
            if not terminal and snapshot["state"].get("round", 0) >= 1:
                return snapshot
            time.sleep(.03)
        self.fail("Workers timed out: " + control.logs(run_id).decode())

    def reap(self, control, run_id):
        for process in control.children[run_id]:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)

    def test_file_and_tcp_have_identical_models_curves_and_80_predictions(self):
        results = []
        with tempfile.TemporaryDirectory() as temp:
            for kind in ("file", "tcp"):
                control = ControlPlane(Path(temp) / kind, "local", 1, kind)
                run_id = control.start({"rounds": 4})["run_id"]
                try:
                    snapshot = self.wait(control, run_id)
                    self.assertEqual(snapshot["state"]["status"], "completed", control.logs(run_id).decode())
                    for process in control.children[run_id]:
                        process.wait(timeout=10)
                        self.assertEqual(process.returncode, 0)
                    run = control.directory(run_id)
                    checkpoint = torch.load(run / "model.pt", weights_only=True)
                    results.append((snapshot, checkpoint, read_json(run / "acc_data.json"),
                                    read_json(run / "loss_data.json")))
                    self.assertNotIn("tcp_token", snapshot["config"])
                    self.assertNotIn("tcp_token", checkpoint["config"])
                    if kind == "tcp":
                        self.assertFalse(list(run.rglob("delta-*.pt")))
                        self.assertFalse(list(run.rglob("global-*.pt")))
                        for rank in (1, 2):
                            self.assertTrue((run / f"client-{rank}" / f"node-{rank}.json").exists())
                            self.assertEqual(list((run / f"client-{rank}").glob("node-*.json")),
                                             [run / f"client-{rank}" / f"node-{rank}.json"])
                finally:
                    self.reap(control, run_id)
            for key in results[0][1]["state_dict"]:
                self.assertTrue(torch.equal(results[0][1]["state_dict"][key], results[1][1]["state_dict"][key]), key)
            self.assertEqual(results[0][0]["predictions"], results[1][0]["predictions"])
            self.assertEqual(len(results[1][0]["predictions"]), 80)
            self.assertEqual(results[0][0]["state"]["metrics"], results[1][0]["state"]["metrics"])
            self.assertEqual(results[0][2:], results[1][2:])

    def test_client_crash_fails_run_and_stops_other_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            control = ControlPlane(temp, "local", 1, "tcp", tcp_timeout=3)
            run_id = control.start({"rounds": 1000})["run_id"]
            try:
                self.assertEqual(self.wait(control, run_id, terminal=False)["state"]["status"], "running")
                victim = control.children[run_id][2]
                victim.kill()
                victim.wait(timeout=5)
                snapshot = self.wait(control, run_id)
                self.assertEqual(snapshot["state"]["status"], "failed")
                for process in control.children[run_id]:
                    process.wait(timeout=10)
                self.assertFalse((control.directory(run_id) / "model.pt").exists())
            finally:
                self.reap(control, run_id)

    def test_running_tcp_job_can_be_cancelled(self):
        with tempfile.TemporaryDirectory() as temp:
            control = ControlPlane(temp, "local", 1, "tcp")
            run_id = control.start({"rounds": 1000})["run_id"]
            try:
                self.assertEqual(self.wait(control, run_id, terminal=False)["state"]["status"], "running")
                self.assertEqual(control.stop(run_id)["status"], "cancelled")
                for process in control.children[run_id]:
                    process.wait(timeout=10)
                self.assertEqual(control.snapshot(run_id)["state"]["status"], "cancelled")
            finally:
                self.reap(control, run_id)

    def test_missing_client_times_out_instead_of_waiting_for_files(self):
        with tempfile.TemporaryDirectory() as temp:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            config = {"mode": "local", "transport": "tcp", "tcp_port": port,
                      "tcp_bind": "127.0.0.1", "tcp_run_id": "timeout-test",
                      "tcp_token": "token", "tcp_timeout": 1,
                      "rounds": 1, "epochs": 1, "lr": .03, "tau": 3., "history_lambda": .6}
            result = subprocess.run([sys.executable, "-m", "faultdemo.worker", "--run-dir", temp,
                                     "--rank", "0", "--threads", "1", "--config-env"], cwd=ROOT,
                                    env={**os.environ, "FEDSIMGH_WORKER_CONFIG": json.dumps(config)},
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 1)
            self.assertIn("客户端就绪超时", result.stdout + result.stderr)
            self.assertEqual(read_json(Path(temp) / "state.json")["status"], "failed")

    def test_tcp_slurm_submission_exports_config_without_command_line_token(self):
        with tempfile.TemporaryDirectory() as temp:
            control = ControlPlane(temp, "slurm", 1, "tcp")
            with patch("app.shutil.which", return_value="sbatch"), patch("app.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], 0, "123\n", "")
                control.start({"rounds": 2})
                config = json.loads(run.call_args.kwargs["env"]["FEDSIMGH_WORKER_CONFIG"])
                self.assertEqual(config["transport"], "tcp")
                self.assertNotIn(config["tcp_token"], " ".join(run.call_args.args[0]))

    def test_allocated_tcp_launcher_passes_config_without_shared_client_files(self):
        with tempfile.TemporaryDirectory(prefix="fedsimgh tcp allocation ") as temp:
            folder = Path(temp)
            scontrol = folder / "scontrol"
            scontrol.write_text("#!/usr/bin/env bash\nprintf 'node169\\nnode170\\nnode171\\n'\n")
            scontrol.chmod(0o755)
            srun = folder / "srun"
            srun.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
if os.environ.get("FEDSIMGH_TEST_FAIL"):
    raise SystemExit(7)
config = json.loads(os.environ["FEDSIMGH_WORKER_CONFIG"])
assert config["transport"] == "tcp" and config["tcp_host"] == "node169"
assert config["tcp_port"] == 32345 and config["tcp_timeout"] == 120
assert len(config["tcp_token"]) == 64
assert "--ntasks-per-node=1" in sys.argv and "--kill-on-bad-exit=1" in sys.argv
command = sys.argv[-1]
assert "--config-env" in command and "SLURM_TMPDIR" in command
assert "mktemp -d" in command
run = pathlib.Path(os.environ["DEMO_RUN_DIR"])
(run / "state.json").write_text(json.dumps({"status": "completed", "final_metrics": {"accuracy": .7, "macro_f1": .7}}))
print("Mock TCP workers finished")
''')
            srun.chmod(0o755)
            env = {**os.environ, "PATH": str(folder) + os.pathsep + os.environ["PATH"],
                   "FEDSIMGH_PROJECT": str(ROOT), "FEDSIMGH_OUTPUT": str(folder / "results"),
                   "FEDSIMGH_PYTHON": sys.executable, "FEDSIMGH_TRANSPORT": "tcp",
                   "FEDSIMGH_ROUNDS": "300", "SLURM_NTASKS": "3", "SLURM_JOB_NUM_NODES": "3",
                   "SLURM_CPUS_PER_TASK": "1", "SLURM_JOB_ID": "12345", "SLURM_JOB_NODELIST": "node[169-171]"}
            for key in ("FEDSIMGH_TCP_PORT", "FEDSIMGH_TCP_TIMEOUT"):
                env.pop(key, None)
            result = subprocess.run(["bash", str(ROOT / "slurm/system.sbatch")],
                                    env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Final accuracy: 70.0%", result.stdout)
            config = read_json(next((folder / "results").glob("*/config.json")))
            self.assertNotIn(config["tcp_token"], result.stdout + result.stderr)
            result = subprocess.run(["bash", str(ROOT / "slurm/system.sbatch")],
                                    env={**env, "FEDSIMGH_TEST_FAIL": "1"},
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 7)


if __name__ == "__main__":
    unittest.main()
