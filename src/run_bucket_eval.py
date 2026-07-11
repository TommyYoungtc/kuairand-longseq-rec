"""按 item 热度分桶评估已训练模型:验证语义特征对冷启动/长尾 item 的增益。

用法(评估某个已保存的 checkpoint):
  python -m src.run_bucket_eval --config configs/1k.yaml --set rank.exp_suffix=-sem
  python -m src.run_bucket_eval --config configs/1k.yaml --set rank.use_sem=false rank.exp_suffix=-nosem

分桶依据:item(vid)在训练期的曝光次数。
"""
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.rank_dataset import RankDataset, RankDatasetSIM, TagHistory, UserHistory
from src.eval.metrics import auc, log_result
from src.models.din import DIN
from src.models.sim import SIM

BUCKETS = [(0, 0, "0(纯新)"), (1, 9, "1-9"), (10, 29, "10-29"),
           (30, 99, "30-99"), (100, 10**9, "100+")]


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rk = cfg.rank
    model_name = rk.get("model", "din")
    exp_name = model_name + rk.get("exp_suffix", "")
    meta = json.loads((cfg.out / "meta.json").read_text())

    sem = None
    if rk.get("use_sem", False):
        sem = torch.from_numpy(np.load(cfg.out / "sem_emb.npy")).to(device)

    inter = pd.read_parquet(cfg.out / "interactions.parquet",
                            columns=["uid", "iid_h", "vid", "tag1", "time_ms", cfg.main_label])
    clicks = inter[inter[cfg.main_label] == 1].drop(columns=[cfg.main_label])
    del inter
    history = UserHistory(clicks)
    tag_history = TagHistory(clicks) if model_name == "sim" else None
    del clicks

    test = pd.read_parquet(cfg.out / "rank_test.parquet")
    train_cnt = pd.read_parquet(cfg.out / "rank_train.parquet", columns=["vid"])["vid"].value_counts()
    cnt = test["vid"].map(train_cnt).fillna(0).astype(int).to_numpy()

    if model_name == "sim":
        ds = RankDatasetSIM(test, history, tag_history, rk["hist_len"], rk["long_topk"], rk["label"])
    else:
        ds = RankDataset(test, history, rk["hist_len"], rk["label"])

    cls = {"din": DIN, "sim": SIM}[model_name]
    model = cls(meta["n_users"], meta["n_iid_h"], meta["n_author_h"], meta["n_tag"],
                rk["emb_dim"], rk["mlp_dims"], sem=sem).to(device)
    ckpt = cfg.out / f"{exp_name}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"loaded {ckpt}")

    scores, labels = [], []
    with torch.no_grad():
        for batch in tqdm(DataLoader(ds, batch_size=rk["batch_size"], num_workers=rk["num_workers"]),
                          desc="score"):
            *x, y = batch
            scores.append(model(*(t.to(device) for t in x)).cpu().numpy())
            labels.append(y.numpy())
    s, l = np.concatenate(scores), np.concatenate(labels)

    print(f"\n=== {exp_name} @ {cfg.dataset} test,按 item 训练期曝光量分桶 ===")
    out = {}
    for lo, hi, name in BUCKETS:
        m = (cnt >= lo) & (cnt <= hi)
        if m.sum() < 100:
            continue
        a = auc(l[m], s[m])
        out[f"AUC[{name}]"] = a
        print(f"  {name:>8}: n={m.sum():>9,}  pos_rate={l[m].mean():.3f}  AUC={a:.4f}")
    log_result(cfg.reports, f"bucket/{exp_name}/{cfg.dataset}", cfg._config_path, out)


if __name__ == "__main__":
    main(load_config())
