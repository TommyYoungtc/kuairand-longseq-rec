"""多目标融合权衡分析:MMoE 三头(click/long_view/like)融合权重扫描。

对每个 test 用户,把其曝光按融合分 score = (1-α)·p_click + α·p_long_view 重排,
计算 top-10 的点击命中率、长观命中率、类目多样性 —— 输出 α 从 0 到 1 的
tradeoff 曲线(csv + png + markdown 摘要)。"权重是产品决策":这条曲线就是决策依据。

用法: python -m src.run_tradeoff --config configs/1k.yaml --set rank.exp_suffix=-sem
需要已有 mmoe-sem.pt。
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.rank_dataset import RankDatasetSIM, TagHistory, UserHistory
from src.models.mmoe import MMoE

TOPK = 10
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rk = cfg.rank
    tasks = rk["tasks"]
    exp_name = "mmoe" + rk.get("exp_suffix", "")
    meta = json.loads((cfg.out / "meta.json").read_text())
    gsu_key = rk.get("gsu_key", "tag1")

    sem = None
    if rk.get("use_sem", False):
        sem = torch.from_numpy(np.load(cfg.out / "sem_emb.npy")).to(device)

    inter = pd.read_parquet(cfg.out / "interactions.parquet",
                            columns=["uid", "iid_h", "vid", "tag1", "cat2", "time_ms", cfg.main_label])
    clicks = inter[inter[cfg.main_label] == 1].drop(columns=[cfg.main_label])
    del inter
    history, tag_history = UserHistory(clicks), TagHistory(clicks, gsu_key)
    del clicks

    test = pd.read_parquet(cfg.out / "rank_test.parquet").reset_index(drop=True)
    ds = RankDatasetSIM(test, history, tag_history, rk["hist_len"], rk["long_topk"],
                        tasks, gsu_key)

    model = MMoE(meta["n_users"], meta["n_iid_h"], meta["n_author_h"], meta["n_tag"],
                 rk["emb_dim"], rk["mlp_dims"], sem=sem,
                 n_tasks=len(tasks), n_experts=rk.get("n_experts", 4),
                 expert_dim=rk.get("expert_dim", 64)).to(device)
    model.load_state_dict(torch.load(cfg.out / f"{exp_name}.pt", map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(ds, batch_size=rk["batch_size"],
                                     num_workers=rk["num_workers"]), desc="score"):
            *x, y = batch
            probs.append(torch.sigmoid(model(*(t.to(device) for t in x))).cpu().numpy())
    P = np.concatenate(probs)                       # (N, 3): click, long_view, like

    df = test[["uid", "cat2", "is_click", "long_view"]].copy()
    df["p_click"], df["p_lv"] = P[:, 0], P[:, 1]

    rows = []
    for a in ALPHAS:
        df["s"] = (1 - a) * df["p_click"] + a * df["p_lv"]
        hit_c, hit_lv, divs = [], [], []
        for _, g in df.groupby("uid"):
            if len(g) < TOPK * 2:
                continue
            top = g.nlargest(TOPK, "s")
            hit_c.append(top["is_click"].mean())
            hit_lv.append(top["long_view"].mean())
            divs.append(top["cat2"].nunique())
        rows.append({"alpha": a, "click@10": np.mean(hit_c),
                     "long_view@10": np.mean(hit_lv), "diversity@10": np.mean(divs)})
        print(rows[-1])

    res = pd.DataFrame(rows)
    res.to_csv(cfg.reports / "tradeoff.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(res["alpha"], res["click@10"], "o-", label="click@10")
    ax1.plot(res["alpha"], res["long_view@10"], "s-", label="long_view@10")
    ax1.set_xlabel("alpha (long_view weight)")
    ax1.set_ylabel("precision@10")
    ax1.legend(loc="center left")
    ax2 = ax1.twinx()
    ax2.plot(res["alpha"], res["diversity@10"], "^--", color="gray", label="diversity@10")
    ax2.set_ylabel("distinct cat2 in top10")
    ax2.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(cfg.reports / "tradeoff.png", dpi=130)
    print("->", cfg.reports / "tradeoff.csv", "and tradeoff.png")


if __name__ == "__main__":
    main(load_config())
