"""无偏评估:在随机曝光日志(random policy)上评估已训练排序模型。

标准日志的曝光由线上推荐策略选出,存在曝光偏差;随机曝光日志中每条视频
被等概率替换展示,评估结果不受旧策略影响 —— 这是 off-policy evaluation 的基础。

用法:
  python -m src.run_unbiased_eval --config configs/1k.yaml --set rank.exp_suffix=-sem
仅使用 test 期(date > val_end_date)的随机曝光,避免与训练期重叠。
"""
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.rank_dataset import RankDataset, RankDatasetSIM, TagHistory, UserHistory
from src.eval.metrics import auc, gauc, log_result
from src.models.din import DIN
from src.models.sim import SIM


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rk = cfg.rank
    model_name = rk.get("model", "din")
    exp_name = model_name + rk.get("exp_suffix", "")
    meta = json.loads((cfg.out / "meta.json").read_text())

    sem = None
    if rk.get("use_sem", False):
        sem = torch.from_numpy(np.load(cfg.out / "sem_emb.npy")).to(device)

    gsu_key = rk.get("gsu_key", "tag1")
    inter = pd.read_parquet(cfg.out / "interactions.parquet",
                            columns=["uid", "iid_h", "vid", "tag1", "cat2", "time_ms", cfg.main_label])
    clicks = inter[inter[cfg.main_label] == 1].drop(columns=[cfg.main_label])
    del inter
    history = UserHistory(clicks)
    tag_history = TagHistory(clicks, gsu_key) if model_name == "sim" else None
    del clicks

    rand = pd.read_parquet(cfg.out / "random_log.parquet")
    rand = rand[rand["date"] > cfg.val_end_date].reset_index(drop=True)
    print(f"random-exposure eval rows (test period): {len(rand):,}  "
          f"pos_rate={rand[rk['label']].mean():.3f}")

    if model_name == "sim":
        ds = RankDatasetSIM(rand, history, tag_history, rk["hist_len"], rk["long_topk"],
                            rk["label"], gsu_key)
    else:
        ds = RankDataset(rand, history, rk["hist_len"], rk["label"])

    cls = {"din": DIN, "sim": SIM}[model_name]
    model = cls(meta["n_users"], meta["n_iid_h"], meta["n_author_h"], meta["n_tag"],
                rk["emb_dim"], rk["mlp_dims"], sem=sem).to(device)
    model.load_state_dict(torch.load(cfg.out / f"{exp_name}.pt", map_location=device))
    model.eval()

    scores, labels, uids = [], [], []
    with torch.no_grad():
        for batch in tqdm(DataLoader(ds, batch_size=rk["batch_size"],
                                     num_workers=rk["num_workers"]), desc="score"):
            *x, y = batch
            scores.append(model(*(t.to(device) for t in x)).cpu().numpy())
            labels.append(y.numpy())
            uids.append(x[0].numpy())
    s, l, u = map(np.concatenate, [scores, labels, uids])
    m = {"AUC": auc(l, s), "GAUC": gauc(l, s, u)}
    log_result(cfg.reports, f"unbiased/{exp_name}/{cfg.dataset}", cfg._config_path, m)
    print("对比:标准 test 集上的同模型指标见 results.csv;两者差值反映曝光偏差的影响。")


if __name__ == "__main__":
    main(load_config())
