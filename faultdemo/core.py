"""FedSimGHHistNoSize aggregation, adapted from PFLlib (Apache-2.0).

Projection order, projected-direction consensus, EMA and no-size weighting follow
system/flcore/servers/serversimghhistnosize.py. The backbone is smaller for CPUs.
"""
import numpy as np
import torch
from torch import nn

LABELS = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
EPS = 1e-12


class TinySignalCNN(nn.Module):
    """Small 1D CNN; GroupNorm avoids unsynchronised BatchNorm running buffers."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, 15, stride=4, padding=7), nn.GroupNorm(2, 8), nn.ReLU(),
            nn.Conv1d(8, 16, 9, stride=4, padding=4), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv1d(16, 24, 7, stride=2, padding=3), nn.GroupNorm(4, 24), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(192, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x):
        return self.head(self.features(x.unsqueeze(1)))


def vector(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def apply_delta(model, delta):
    cursor = 0
    with torch.no_grad():
        for p in model.parameters():
            p.add_(delta[cursor:cursor + p.numel()].reshape_as(p))
            cursor += p.numel()


class HistoricalCoordinator:
    def __init__(self, tau=3.0, history_lambda=0.6, threshold=0.0):
        self.tau = tau
        self.history_lambda = history_lambda
        self.threshold = threshold
        self.history = {}

    def aggregate(self, deltas, client_ids=(0, 1, 2)):
        projected = [d.clone() for d in deltas]
        conflicts, cosines = 0, []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                a, b = deltas[i], deltas[j]
                na, nb = torch.norm(a), torch.norm(b)
                if na <= EPS or nb <= EPS:
                    continue
                cosine = torch.dot(a, b) / (na * nb + EPS)
                cosines.append(float(cosine))
                if float(cosine) >= self.threshold:
                    continue
                ai = torch.dot(projected[i], b) / (torch.dot(b, b) + EPS)
                bj = torch.dot(projected[j], a) / (torch.dot(a, a) + EPS)
                projected[i] -= ai * b
                projected[j] -= bj * a
                conflicts += 1
        units = [d / (torch.norm(d) + EPS) for d in projected]
        consensus = sum(units)
        consensus /= torch.norm(consensus) + EPS
        scores, histories = [], []
        for client_id, unit in zip(client_ids, units):
            score = float(torch.dot(unit, consensus))
            old = self.history.get(client_id, score)
            history = self.history_lambda * old + (1 - self.history_lambda) * score
            self.history[client_id] = history
            scores.append(score)
            histories.append(history)
        # Stable softmax is mathematically identical to exp(tau*s) / sum(exp(tau*s)).
        weights = torch.softmax(torch.tensor(histories, dtype=torch.float64) * self.tau, dim=0).tolist()
        update = sum(w * d for w, d in zip(weights, projected))
        return update, {"conflicts": conflicts, "weights": weights, "similarities": scores,
                        "history": histories, "mean_cosine": sum(cosines) / max(len(cosines), 1)}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        x = torch.from_numpy(data["x"].astype(np.float32))
        y = torch.from_numpy(data["y"].astype(np.int64))
    return x, y


def local_update(model, x, y, epochs, lr, batch_size, seed, check_stop=lambda: None):
    baseline = vector(model)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    loss_total, count = 0.0, 0
    for _ in range(epochs):
        indices = torch.randperm(len(y), generator=generator)
        for batch in indices.split(batch_size):
            check_stop()
            logits = model(x[batch])
            loss = nn.functional.cross_entropy(logits, y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(batch)
            count += len(batch)
    delta = vector(model) - baseline
    if not torch.isfinite(delta).all():
        raise ValueError("Non-finite client update")
    return delta, loss_total / count


@torch.inference_mode()
def evaluate(model, x, y):
    model.eval()
    probabilities = torch.cat([model(b).softmax(dim=1) for b in x.split(32)])
    predictions = probabilities.argmax(dim=1)
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    for actual, pred in zip(y, predictions):
        confusion[int(actual), int(pred)] += 1
    f1s = []
    for c in range(4):
        tp = int(confusion[c, c])
        denominator = int(confusion[c].sum() + confusion[:, c].sum())
        f1s.append(2 * tp / denominator if denominator else 0.0)
    metrics = {"accuracy": float((predictions == y).float().mean()),
               "macro_f1": sum(f1s) / 4, "confusion_matrix": confusion.tolist()}
    return metrics, probabilities
