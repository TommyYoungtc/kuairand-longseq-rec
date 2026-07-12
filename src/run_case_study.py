"""Case study:语义近邻示例 + "语义救回"的点击案例,输出 reports/case_study.md。

两部分:
  A. 语义近邻:随机抽候选视频,按语义表余弦相似度取 Top-5 邻居,展示中文标题
     —— 直观验证语义空间质量。
  B. 救回案例:对 test 抽样打分,找出「实际被点击、+语义模型排名靠前、
     无语义模型排名靠后」的样本,展示用户近期点击标题 vs 目标视频标题
     —— 展示世界知识如何补足协同信号。

用法: python -m src.run_case_study --config configs/1k.yaml
需要已有 sim-sem.pt 与 sim-nosem.pt。
"""
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.rank_dataset import RankDatasetSIM, TagHistory, UserHistory
from src.models.sim import SIM

CAP_FILE = "data/raw/kuairand_video_captions.csv"
N_NEIGHBOR_DEMO = 8
N_RESCUE_DEMO = 12
SCORE_SAMPLE = 200_000


def load_captions(need_vids, vmap):
    """分块扫描 caption 文件,仅缓存需要的 vid → 标题。"""
    vid_of = dict(zip(vmap["video_id"], vmap["vid"]))
    need = set(int(v) for v in need_vids)
    caps = {}
    for chunk in tqdm(pd.read_csv(CAP_FILE, chunksize=500_000,
                                  usecols=["final_video_id", "caption"]),
                      desc="scan captions"):
        chunk["vid"] = chunk["final_video_id"].map(vid_of)
        hit = chunk.dropna(subset=["vid"])
        hit = hit[hit["vid"].astype(int).isin(need)]
        for v, c in zip(hit["vid"].astype(int), hit["caption"].fillna("")):
            caps[v] = str(c)[:80]
    return caps


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rk = cfg.rank
    meta = json.loads((cfg.out / "meta.json").read_text())
    rng = np.random.default_rng(cfg.seed)
    lines = ["# Case Study\n"]

    sem_np = np.load(cfg.out / "sem_emb.npy").astype(np.float32)
    vmap = pd.read_parquet(cfg.out / "video_map.parquet")
    imap = pd.read_parquet(cfg.out / "item_map.parquet")
    cand_vids = vmap.merge(imap, on="video_id")["vid"].to_numpy()  # 候选集的 vid

    # ---- A. 语义近邻 ----
    M = sem_np[cand_vids]
    norm = np.linalg.norm(M, axis=1)
    ok = norm > 1e-6
    M, cv = M[ok] / norm[ok, None], cand_vids[ok]
    queries = rng.choice(len(cv), N_NEIGHBOR_DEMO, replace=False)
    sims = M[queries] @ M.T
    nb_idx = np.argsort(-sims, axis=1)[:, :6]  # 含自身
    need_vids = set(cv[nb_idx].ravel().tolist())

    # ---- B. 救回案例:双模型打分 ----
    gsu_key = rk.get("gsu_key", "tag1")
    inter = pd.read_parquet(cfg.out / "interactions.parquet",
                            columns=["uid", "iid_h", "vid", "tag1", "cat2", "time_ms", cfg.main_label])
    clicks = inter[inter[cfg.main_label] == 1].drop(columns=[cfg.main_label])
    del inter
    history, tag_history = UserHistory(clicks), TagHistory(clicks, gsu_key)

    test = pd.read_parquet(cfg.out / "rank_test.parquet")
    test = test.sample(min(SCORE_SAMPLE, len(test)), random_state=cfg.seed).reset_index(drop=True)
    ds = RankDatasetSIM(test, history, tag_history, rk["hist_len"], rk["long_topk"],
                        rk["label"], gsu_key)

    def score_with(ckpt, sem):
        model = SIM(meta["n_users"], meta["n_iid_h"], meta["n_author_h"], meta["n_tag"],
                    rk["emb_dim"], rk["mlp_dims"], sem=sem).to(device)
        model.load_state_dict(torch.load(cfg.out / ckpt, map_location=device))
        model.eval()
        out = []
        with torch.no_grad():
            for batch in tqdm(DataLoader(ds, batch_size=rk["batch_size"],
                                         num_workers=rk["num_workers"]), desc=ckpt):
                *x, y = batch
                out.append(model(*(t.to(device) for t in x)).cpu().numpy())
        return np.concatenate(out)

    sem_t = torch.from_numpy(np.load(cfg.out / "sem_emb.npy")).to(device)
    s_sem = score_with("sim-sem.pt", sem_t)
    s_nosem = score_with("sim-nosem.pt", None)

    # 分位排名差:被点击 & sem 排名高 & nosem 排名低
    r_sem = pd.Series(s_sem).rank(pct=True).to_numpy()
    r_nosem = pd.Series(s_nosem).rank(pct=True).to_numpy()
    clicked = test[rk["label"]].to_numpy() == 1
    gain = np.where(clicked, r_sem - r_nosem, -1)
    top = np.argsort(-gain)[:N_RESCUE_DEMO]

    # 收集需要标题的 vid:目标 + 每个 case 用户最近 5 次点击
    recent = {}
    for k in top:
        uid, t = int(test["uid"][k]), int(test["time_ms"][k])
        _, hv, _ = history.window(uid, t, 5)
        recent[k] = hv.tolist()
        need_vids.update(hv.tolist())
        need_vids.add(int(test["vid"][k]))
    need_vids.discard(0)

    caps = load_captions(need_vids, vmap)
    cap = lambda v: caps.get(int(v), f"<无标题 vid={int(v)}>")

    lines.append("## A. 语义近邻示例(候选集内,余弦 Top-5)\n")
    for qpos, (qi, row) in enumerate(zip(queries, nb_idx)):
        lines.append(f"**Query**: {cap(cv[qi])}\n")
        for j in row[1:]:
            lines.append(f"- (cos={sims[qpos][j]:.2f}) {cap(cv[j])}")
        lines.append("")

    lines.append("\n## B. 语义救回案例(实际点击,+语义排名显著提升)\n")
    for k in top:
        lines.append(f"**目标视频**(点击,+语义分位 {r_sem[k]:.2f} vs 无语义 {r_nosem[k]:.2f}):"
                     f" {cap(test['vid'][k])}\n")
        lines.append("用户最近点击:")
        for v in recent[k]:
            lines.append(f"- {cap(v)}")
        lines.append("")

    out_md = cfg.reports / "case_study.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("case study ->", out_md)


if __name__ == "__main__":
    main(load_config())
