
"""One config -> one results JSON.

Usage:
    python main.py                                   # uses configs/default.yaml
    python main.py --selector power_of_choice --alpha 0.5 --seed 1
    python main.py --rounds 5                        # quick smoke test

Later, move the loop into fl/train.py and keep this as a thin wrapper.
"""
import argparse, copy, json, os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import yaml
from torchvision import datasets

from baselines import REGISTRY
from data.partition import get_partition
from selection import REGISTRY as TEAM

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- data (stub; swap for Person A's get_partition) ----------------
def load_partition(cfg):
    """Suresh's index partition -> per-client (X, y) tensors."""
    ds = datasets.CIFAR10("data/raw", train=True, download=True)
    X = (torch.tensor(ds.data).permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5
    y = torch.tensor(ds.targets)
    part = get_partition(cfg["dataset"], cfg["alpha"], cfg["n_clients"], cfg["seed"],
                         labels=y.numpy())
    train = [(X[torch.from_numpy(i)], y[torch.from_numpy(i)]) for i in part["train"]]
    test = [(X[torch.from_numpy(i)], y[torch.from_numpy(i)]) for i in part["test"]]
    return train, test


# ---------------- model + helpers ----------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(3, 32, 3), nn.Conv2d(32, 64, 3)
        self.f1, self.f2 = nn.Linear(64 * 6 * 6, 128), nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))


def train_local(model, X, y, cfg):
    model.to(DEV).train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
    bs = cfg["batch_size"]
    for _ in range(cfg["local_epochs"]):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(X[b].to(DEV)), y[b].to(DEV)).backward()
            opt.step()


@torch.no_grad()
def evaluate(model, X, y):
    """returns (mean loss, accuracy)"""
    if len(X) == 0:
        return 0.0, 0.0
    model.to(DEV).eval()
    loss = correct = 0
    for i in range(0, len(X), 512):
        xb, yb = X[i:i + 512].to(DEV), y[i:i + 512].to(DEV)
        out = model(xb)
        loss += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return loss / len(X), correct / len(X)


def flat(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()]).cpu()


class Ctx:
    """What a selector may ask the server for. Random ignores it entirely."""
    def __init__(self):
        self.loss_fn = None      # ids -> losses of current global model (Power-of-Choice)
        self.update_fn = None    # ids -> local update vectors (DivFL)
        self.updates = {}        # {client_id: update vector} for clients trained this round


# ---------------- main ----------------
def run(cfg):
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    train, test = load_partition(cfg)
    n, k = cfg["n_clients"], cfg["k"]
    sizes = [len(t[0]) for t in train]

    registry = {**REGISTRY, **TEAM}
    if cfg["selector"] not in registry:
        raise KeyError(f"unknown selector {cfg['selector']}; have {list(registry)}")
    selector = registry[cfg["selector"]](n, k, sizes, cfg)

    glob = CNN().to(DEV)
    n_test = [len(t[0]) for t in test]
    counts, sel_log, acc_log, client_log, query_log = [0] * n, [], [], [], []

    def loss_fn(ids):
        return [evaluate(glob, *train[i])[0] for i in ids]

    def update_fn(ids):
        out = []
        for i in ids:
            m = copy.deepcopy(glob); train_local(m, *train[i], cfg)
            out.append((flat(m) - flat(glob)).numpy())
        return out

    for r in range(cfg["rounds"]):
        ctx = Ctx(); ctx.loss_fn, ctx.update_fn = loss_fn, update_fn
        chosen = [int(i) for i in selector.select(r, ctx)]
        assert len(set(chosen)) == k, f"selector returned {len(chosen)} ids, expected {k} distinct"
        states, ws = [], []
        g0 = flat(glob)
        for i in chosen:
            m = copy.deepcopy(glob); train_local(m, *train[i], cfg)
            ctx.updates[i] = (flat(m) - g0).numpy()
            states.append({kk: v.detach().clone() for kk, v in m.state_dict().items()})
            ws.append(sizes[i]); counts[i] += 1
        tot = sum(ws)                            # data-size-weighted FedAvg over selected clients
        glob.load_state_dict({kk: sum(s[kk] * (w / tot) for s, w in zip(states, ws))
                              for kk in states[0]})
        selector.update(r, chosen, ctx)
        sel_log.append(chosen)
        q = int(getattr(selector, "extra_queries", 0))
        query_log.append(q - sum(query_log))
        # global acc on the union of client test splits = test-size-weighted client acc
        accs = [evaluate(glob, *test[i])[1] for i in range(n)]
        client_log.append(accs)
        acc_log.append(sum(a * m for a, m in zip(accs, n_test)) / sum(n_test))
        print(f"round {r:3d}  global_acc {acc_log[-1]:.3f}", flush=True)

    out = {
        "cfg": {kk: v for kk, v in cfg.items()},
        "selected_ids": sel_log,
        "global_acc": acc_log,                   # on the union of client test splits
        "client_acc": client_log[-1],            # final per-client accuracy
        "client_acc_rounds": client_log,         # per-client accuracy after each round
        "selection_counts": counts,
        "data_sizes": sizes,
        "queried_per_round": query_log,          # extra client queries in each round
        "extra_queries": sum(query_log),
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/{cfg['selector']}_{cfg['dataset']}_{cfg['alpha']}_{cfg['good_def']}_{cfg['seed']}.json"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    for name, typ in [("selector", str), ("alpha", float), ("seed", int), ("rounds", int),
                      ("n_clients", int), ("k", int), ("lr", float), ("d", int),
                      ("good_def", str), ("local_epochs", int), ("T", int), ("filler", str)]:
        ap.add_argument(f"--{name}", type=typ, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for kk, v in vars(args).items():
        if kk != "config" and v is not None:
            cfg[kk] = v
    run(cfg)