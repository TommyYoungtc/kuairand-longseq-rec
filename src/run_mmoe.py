"""MMoE 多任务训练 + 每任务 AUC/GAUC 评估。

用法: python -m src.run_mmoe --config configs/1k.yaml
早停依据:主任务(第一个 task,即 is_click)的 val GAUC。
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.rank_dataset import RankDatasetSIM, TagHistory, UserHistory
from src.eval.metrics import auc, gauc, log_result
from src.models.mmoe import MMoE


@torch.no_grad()
def evaluate(model, ds, tasks, device, batch_size, workers):
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers)
    scores, labels, uids = [], [], []
    for batch in tqdm(dl, desc="eval", leave=False):
        *x, y = batch
        scores.append(model(*(t.to(device) for t in x)).cpu().numpy())
        labels.append(y.numpy())
        uids.append(x[0].numpy())
    s, l, u = np.concatenate(scores), np.concatenate(labels), np.concatenate(uids)
    out = {}
    for i, t in enumerate(tasks):
        out[f"AUC/{t}"] = auc(l[:, i], s[:, i])
        out[f"GAUC/{t}"] = gauc(l[:, i], s[:, i], u)
    return out


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    rk = cfg.rank
    tasks = rk["tasks"]
    exp_name = "mmoe" + rk.get("exp_suffix", "")
    meta = json.loads((cfg.out / "meta.json").read_text())

    sem = None
    if rk.get("use_sem", False):
        sem = torch.from_numpy(np.load(cfg.out / "sem_emb.npy")).to(device)

    inter = pd.read_parquet(cfg.out / "interactions.parquet",
                            columns=["uid", "iid_h", "vid", "tag1", "time_ms", cfg.main_label])
    clicks = inter[inter[cfg.main_label] == 1].drop(columns=[cfg.main_label])
    del inter
    history = UserHistory(clicks)
    tag_history = TagHistory(clicks)
    del clicks

    def load_split(name, sample=0):
        df = pd.read_parquet(cfg.out / f"rank_{name}.parquet")
        if sample and len(df) > sample:
            df = df.sample(sample, random_state=cfg.seed)
        return RankDatasetSIM(df, history, tag_history,
                              rk["hist_len"], rk["long_topk"], tasks)

    ds_train = load_split("train")
    ds_val = load_split("val", rk["val_sample"])
    print(f"exp={exp_name}  tasks={tasks}  device={device}  train={len(ds_train):,}")

    model = MMoE(meta["n_users"], meta["n_iid_h"], meta["n_author_h"], meta["n_tag"],
                 rk["emb_dim"], rk["mlp_dims"], sem=sem,
                 n_tasks=len(tasks), n_experts=rk.get("n_experts", 4),
                 expert_dim=rk.get("expert_dim", 64)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=rk["lr"])
    dl = DataLoader(ds_train, batch_size=rk["batch_size"], shuffle=True,
                    num_workers=rk["num_workers"], drop_last=True)

    main_key = f"GAUC/{tasks[0]}"
    best, best_state, patience = -1.0, None, 0
    for ep in range(rk["epochs"]):
        model.train()
        losses = []
        for batch in tqdm(dl, desc=f"epoch {ep}"):
            *x, y = batch
            logits = model(*(t.to(device) for t in x))
            loss = F.binary_cross_entropy_with_logits(logits, y.to(device))  # 各任务等权
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {ep}  loss={np.mean(losses):.4f}")
        m = evaluate(model, ds_val, tasks, device, rk["batch_size"], rk["num_workers"])
        log_result(cfg.reports, f"{exp_name}/{cfg.dataset}/val/ep{ep}", cfg._config_path, m)
        if m[main_key] > best:
            best, patience = m[main_key], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                print(f"early stop at epoch {ep} (best val {main_key}={best:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    m = evaluate(model, load_split("test"), tasks, device, rk["batch_size"], rk["num_workers"])
    log_result(cfg.reports, f"{exp_name}/{cfg.dataset}/test", cfg._config_path, m)
    torch.save(model.state_dict(), cfg.out / f"{exp_name}.pt")
    print("model saved ->", cfg.out / f"{exp_name}.pt")


if __name__ == "__main__":
    main(load_config())
