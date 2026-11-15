"""Round-synchronous transports. TCP uses bounded JSON + float32, never pickle.

Only rank 0 listens. Clients keep one authenticated connection for the run;
updates are collected in rank order, not arrival order. A broken connection
fails the run rather than silently dropping a client or reusing an old update.
"""
import hashlib
import hmac
import json
import math
import socket
import struct
import time

import numpy as np
import torch

from .core import vector
from .io import check_stop, read_json, wait_for

MAX_FRAME = 4 * 1024 * 1024
VERSION = 1


def _exact(sock, size, deadline, stop):
    chunks = bytearray()
    while len(chunks) < size:
        stop()
        if time.monotonic() >= deadline:
            raise TimeoutError("TCP 接收超时：节点未在规定时间内响应")
        try:
            chunk = sock.recv(min(size - len(chunks), 65536))
        except socket.timeout:
            continue
        if not chunk:
            raise ConnectionError("TCP 对端已断开，停止本次训练以避免聚合不完整的更新")
        chunks.extend(chunk)
    return bytes(chunks)


def send_frame(sock, metadata, tensor=None, *, timeout=120, stop=lambda: None):
    header = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
    payload = b"" if tensor is None else tensor.detach().cpu().numpy().astype("<f4").tobytes()
    body = struct.pack("!I", len(header)) + header + payload
    if len(header) > 16384 or len(body) > MAX_FRAME:
        raise ValueError("TCP 消息超出长度限制")
    pending = memoryview(struct.pack("!I", len(body)) + body)
    deadline = time.monotonic() + timeout
    while pending:
        stop()
        if time.monotonic() >= deadline:
            raise TimeoutError("TCP 发送超时")
        try:
            sent = sock.send(pending[:65536])
        except socket.timeout:
            continue
        if not sent:
            raise ConnectionError("TCP 发送连接已关闭")
        pending = pending[sent:]


def recv_frame(sock, *, timeout=120, stop=lambda: None):
    deadline = time.monotonic() + timeout
    size = struct.unpack("!I", _exact(sock, 4, deadline, stop))[0]
    if not 4 <= size <= MAX_FRAME:
        raise ValueError("TCP 消息长度无效")
    body = _exact(sock, size, deadline, stop)
    header_size = struct.unpack("!I", body[:4])[0]
    if not 2 <= header_size <= min(16384, size - 4):
        raise ValueError("TCP 消息头长度无效")
    metadata = json.loads(body[4:4 + header_size])
    if not isinstance(metadata, dict):
        raise ValueError("TCP 消息头必须为对象")
    return metadata, body[4 + header_size:]


def config_digest(config):
    # Address/output paths and per-run token do not affect training semantics.
    keys = ("rounds", "epochs", "lr", "tau", "history_lambda")
    return hashlib.sha256(json.dumps({k: config[k] for k in keys}, sort_keys=True).encode()).hexdigest()


class FileTransport:
    def __init__(self, run, rank, config, model, node, save):
        self.run, self.rank, self.save = run, rank, save
        self.nodes = None
        if rank == 0:
            for index in range(3):
                wait_for(run / f"node-{index}.json", run)
            self.nodes = [read_json(run / f"node-{i}.json") for i in range(3)]
            save(run / "global-000.pt", model.state_dict())

    def load_global(self, model, step):
        path = self.run / f"global-{step - 1:03d}.pt"
        wait_for(path, self.run)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    def exchange(self, step, delta, loss):
        self.save(self.run / f"delta-{step:03d}-{self.rank}.pt", {"delta": delta, "loss": loss})
        if self.rank != 0:
            return None
        uploads = []
        for rank in range(3):
            path = self.run / f"delta-{step:03d}-{rank}.pt"
            wait_for(path, self.run)
            uploads.append(torch.load(path, map_location="cpu", weights_only=True))
        return uploads

    def publish(self, model, step):
        self.save(self.run / f"global-{step:03d}.pt", model.state_dict())

    def close(self, success=False):
        pass


class TCPTransport:
    def __init__(self, run, rank, config, model, node, save):
        self.run, self.rank, self.config = run, rank, config
        self.peers = {}
        self.listener = None
        self.nodes = None
        # This wire format is deliberately scoped to the current GroupNorm CNN.
        # Refuse future architectures needing buffers/different dtypes instead
        # of silently discarding state or down-casting parameters.
        if list(model.buffers()) or any(p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("当前 TCP 协议仅支持无运行缓冲区的 float32 模型")
        self.count = vector(model).numel()
        self.token = config["tcp_token"]
        self.run_id = config["tcp_run_id"]
        self.digest = config_digest(config)
        self.timeout = float(config.get("tcp_timeout", 120))
        try:
            if rank == 0:
                self._listen(node)
            else:
                self._connect(node)
        except BaseException:
            self.close()
            raise

    def _stop(self):
        check_stop(self.run)

    def _meta(self, kind, step=0):
        return {"version": VERSION, "run": self.run_id, "kind": kind, "round": step}

    def _send(self, peer, kind, step=0, tensor=None, **fields):
        send_frame(peer, {**self._meta(kind, step), **fields}, tensor,
                   timeout=self.timeout, stop=self._stop)

    def _receive(self, peer, kind, step=0):
        meta, payload = recv_frame(peer, timeout=self.timeout, stop=self._stop)
        expected = self._meta(kind, step)
        if (type(meta.get("version")) is not int or type(meta.get("round")) is not int or
                any(meta.get(k) != v for k, v in expected.items())):
            raise ValueError(f"TCP 协议/任务/轮次不匹配，期望 {kind} round={step}")
        return meta, payload

    def _socket_options(self, peer):
        peer.settimeout(.25)
        peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        peer.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    def _listen(self, node):
        listener = self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.config.get("tcp_bind", "0.0.0.0"), int(self.config["tcp_port"])))
        listener.listen(4)
        listener.settimeout(.25)
        print(f"TCP 聚合节点监听端口 {self.config['tcp_port']}，等待另外两个客户端", flush=True)
        nodes = {0: node}
        deadline = time.monotonic() + self.timeout
        while len(nodes) < 3:
            self._stop()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("TCP 客户端就绪超时，需要 rank 1 和 rank 2")
            try:
                peer, _ = listener.accept()
            except socket.timeout:
                continue
            self._socket_options(peer)
            try:
                # Bound the entire startup, including peers that never send a hello.
                meta, payload = recv_frame(peer, timeout=min(remaining, 10), stop=self._stop)
                if (meta.get("kind") != "hello" or meta.get("version") != VERSION or
                        meta.get("run") != self.run_id or
                        not hmac.compare_digest(str(meta.get("token", "")), self.token)):
                    raise ValueError("TCP 握手认证失败")
                rank = meta.get("rank")
                if type(rank) is not int or rank not in (1, 2) or rank in nodes:
                    raise ValueError("TCP 客户端身份重复或无效")
                if payload or meta.get("config") != self.digest or meta.get("count") != self.count:
                    raise ValueError("TCP 客户端模型或训练配置不一致")
                info = meta.get("node")
                if (not isinstance(info, dict) or info.get("rank") != rank or
                        not isinstance(info.get("hostname"), str)):
                    raise ValueError("TCP 客户端节点信息无效")
                self.peers[rank] = peer
                nodes[rank] = info
                self._send(peer, "ready")
                print(f"TCP 客户端 {rank} 已连接：{info['hostname']}", flush=True)
            except BaseException:
                peer.close()
                raise
        self.nodes = [nodes[i] for i in range(3)]
        listener.close()
        self.listener = None

    def _connect(self, node):
        deadline = time.monotonic() + self.timeout
        peer = None
        while peer is None:
            self._stop()
            if time.monotonic() >= deadline:
                raise TimeoutError("无法连接 TCP 聚合节点，请检查主机、端口和计算节点防火墙")
            attempt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            attempt.settimeout(min(1, max(.01, deadline - time.monotonic())))
            try:
                attempt.connect((self.config["tcp_host"], int(self.config["tcp_port"])))
                peer = attempt
            except (ConnectionError, OSError):
                attempt.close()
                time.sleep(.1)
        self.peers[0] = peer
        self._socket_options(peer)
        self._send(peer, "hello", token=self.token, rank=self.rank, node=node,
                   config=self.digest, count=self.count)
        _, payload = self._receive(peer, "ready")
        if payload:
            raise ValueError("TCP ready 不允许携带参数")

    def _tensor(self, payload):
        if len(payload) != self.count * 4:
            raise ValueError("TCP 参数向量长度不匹配")
        array = np.frombuffer(payload, dtype="<f4").copy()
        if not np.isfinite(array).all():
            raise ValueError("TCP 参数包含 NaN 或无穷值")
        return torch.from_numpy(array)

    def load_global(self, model, step):
        if self.rank == 0:
            weights = vector(model)
            for peer in self.peers.values():
                self._send(peer, "global", step, weights)
        else:
            _, payload = self._receive(self.peers[0], "global", step)
            weights = self._tensor(payload)
            # Copy exactly, rather than adding a difference (which rounds twice).
            offset = 0
            with torch.no_grad():
                for parameter in model.parameters():
                    count = parameter.numel()
                    parameter.copy_(weights[offset:offset + count].view_as(parameter))
                    offset += count

    def exchange(self, step, delta, loss):
        if self.rank != 0:
            self._send(self.peers[0], "update", step, delta, rank=self.rank, loss=loss)
            return None
        uploads = [{"delta": delta, "loss": loss}]
        for rank in (1, 2):
            meta, payload = self._receive(self.peers[rank], "update", step)
            value = meta.get("loss")
            if (meta.get("rank") != rank or isinstance(value, bool) or
                    not isinstance(value, (float, int)) or not math.isfinite(value)):
                raise ValueError("TCP 客户端更新身份或 loss 无效")
            uploads.append({"delta": self._tensor(payload), "loss": float(value)})
        return uploads

    def publish(self, model, step):
        # The next global model is broadcast at the beginning of the next round.
        pass

    def close(self, success=False):
        try:
            if success:
                if self.rank == 0:
                    for peer in self.peers.values():
                        self._send(peer, "done", self.config["rounds"])
                else:
                    _, payload = self._receive(self.peers[0], "done", self.config["rounds"])
                    if payload:
                        raise ValueError("TCP done 不允许携带参数")
        finally:
            for peer in self.peers.values():
                peer.close()
            self.peers.clear()
            if self.listener is not None:
                self.listener.close()
                self.listener = None
