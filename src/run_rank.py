"""排序模型训练 + AUC/GAUC 评估。支持 DIN(短序列)与 SIM(短+长序列双支路)。

用法:
  python -m src.run_rank --config configs/1k.yaml                     # 用配置里的 model
  python -m src.run_rank --config configs/1k.yaml --set rank.model=din   # 消融对照
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.rank_dataset import RankDataset, RankDatasetSIM, TagHistory, UserHistory
from src.eval.metrics import auc, gauc, log_result
from src.models.din import DIN
from src.models.sim import SIM


@torch.no_grad()
def evaluate(model, ds, device, batch_size, workers):
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers)
    scores, labels, uids = [], [], []
    for batch in tqdm(dl, desc="eval", leave=False):
        *x, y = batch
        logit = model(*(t.to(device) for t in x))
        scores.append(logit.cpu().numpy())
        labels.append(y.numpy())
        uids.append(x[0].numpy())
    s, l, u = map(np.concatenate, [scores, labels, uids])
    return {"AUC": auc(l, s), "GAUC": gauc(l, s, u)}


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    rk = cfg.rank
    model_name = rk.get("model", "din")
    exp_name = model_name + rk.get("exp_suffix", "")
    meta = json.loads((cfg.out / "meta.json").read_text())

    # 语义表(冻结)
    sem = None
    if rk.get("use_sem", False):
        sem_np = np.load(cfg.out / "sem_emb.npy")
        sem = torch.from_numpy(sem_np).to(device)
        print(f"semantic table loaded: {tuple(sem.shape)} ({sem_np.nbytes / 1e6:.0f}MB)")

    # 历史点击流(严格 time < t 截取,含 val/test 期样本时刻前的点击,无泄漏)
    inter = pd.read_parquet(cfg.out / "interactions.parquet",
                            columns=["uid", "iid_h", "vid", "tag1", "time_ms", cfg.main_label])
    clicks = inter[inter[cfg.main_label] == 1].drop(columns=[cfg.main_label])
    del inter
    history = UserHistory(clicks)
    tag_history = TagHistory(clicks) if model_name == "sim" else None
    del clicks

    def load_split(name, sample=0):
        df = pd.read_parquet(cfg.out / f"rank_{name}.parquet")
        if sample and len(df) > sample:
            df = df.sample(sample, random_state=cfg.seed)
        if model_name == "sim":
            return RankDatasetSIM(df, history, tag_history,
                                  rk["hist_len"], rk["long_topk"], rk["label"])
        return RankDataset(df, history, rk["hist_len"], rk["label"])

    ds_train = load_split("train")
    ds_val = load_split("val", rk["val_sample"])
    print(f"model={model_name}  device={device}  train={len(ds_train):,}  val={len(ds_val):,}")

    cls = {"din": DIN, "sim": SIM}[model_name]
    model = cls(meta["n_users"], meta["n_iid_h"], meta["n_author_h"], meta["n_tag"],
                rk["emb_dim"], rk["mlp_dims"], sem=sem).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=rk["lr"])
    dl = DataLoader(ds_train, batch_size=rk["batch_size"], shuffle=True,
                    num_workers=rk["num_workers"], drop_last=True)

    best_gauc, best_state, patience = -1.0, None, 0
    for ep in range(rk["epochs"]):
        model.train()
        losses = []
        for batch in tqdm(dl, desc=f"epoch {ep}"):
            *x, y = batch
            logit = model(*(t.to(device) for t in x))
            loss = F.binary_cross_entropy_with_logits(logit, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {ep}  loss={np.mean(losses):.4f}")
        m = evaluate(model, ds_val, device, rk["batch_size"], rk["num_workers"])
        log_result(cfg.reports, f"{exp_name}/{cfg.dataset}/val/ep{ep}", cfg._config_path, m)
        if m["GAUC"] > best_gauc:
            best_gauc, patience = m["GAUC"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                print(f"early stop at epoch {ep} (best val GAUC={best_gauc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    m = evaluate(model, load_split("test"), device, rk["batch_size"], rk["num_workers"])
    log_result(cfg.reports, f"{exp_name}/{cfg.dataset}/test", cfg._config_path, m)
    torch.save(model.state_dict(), cfg.out / f"{exp_name}.pt")
    print("model saved ->", cfg.out / f"{exp_name}.pt")


if __name__ == "__main__":
    main(load_config())
