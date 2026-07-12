"""推荐系统交互 Demo:选用户 → 全候选集打分 → Top-N 推荐 + 中文标题 + 推荐理由。

启动:
  pip install streamlit
  python -m src.data.build_caption_cache --config configs/1k.yaml   # 先跑一次
  streamlit run demo/app.py
浏览器打开 http://localhost:8501
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import load_config
from src.data.rank_dataset import TagHistory, UserHistory, _pad
from src.models.sim import SIM

CONFIG = "configs/1k.yaml"
CKPT = "sim-sem-cat2.pt"     # 最优模型
GSU_KEY = "cat2"
TOPN = 10


@st.cache_resource(show_spinner="加载数据与模型(首次约2分钟)...")
def load_all():
    cfg = load_config(CONFIG)
    rk = cfg.rank
    meta = json.loads((cfg.out / "meta.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inter = pd.read_parquet(cfg.out / "interactions.parquet",
                            columns=["uid", "iid", "iid_h", "vid", "author_h",
                                     "tag1", "cat2", "time_ms", cfg.main_label])
    clicks = inter[inter[cfg.main_label] == 1]
    history = UserHistory(clicks)
    tag_history = TagHistory(clicks, GSU_KEY)
    user_now = inter.groupby("uid")["time_ms"].max().to_dict()
    # 用户长期类目偏好(展示理由用)
    user_cats = clicks.groupby(["uid", "cat2"]).size().rename("n").reset_index()

    # 候选 item 特征表(从交互中取每个候选的首行)
    cand = (inter[inter["iid"] > 0]
            .drop_duplicates("iid")[["iid", "iid_h", "vid", "author_h", "tag1", "cat2"]]
            .reset_index(drop=True))
    del inter, clicks

    caps = pd.read_parquet(cfg.out / "caption_cache.parquet")
    cap_of = dict(zip(caps["vid"], caps["caption"]))
    try:
        cn = pd.read_parquet(cfg.out / "cat2_names.parquet")
        cat_names = dict(zip(cn["cat2"], cn["name"]))
    except FileNotFoundError:
        cat_names = {}

    sem_np = np.load(cfg.out / "sem_emb.npy")
    sem = torch.from_numpy(sem_np).to(device)
    model = SIM(meta["n_users"], meta["n_iid_h"], meta["n_author_h"], meta["n_tag"],
                rk["emb_dim"], rk["mlp_dims"], sem=sem).to(device)
    model.load_state_dict(torch.load(cfg.out / CKPT, map_location=device))
    model.eval()
    return (cfg, rk, device, history, tag_history, user_now, user_cats, cand,
            cap_of, cat_names, sem_np, model)


(cfg, rk, device, history, tag_history, user_now, user_cats, cand,
 cap_of, cat_names, sem_np, model) = load_all()


def cap(v):
    c = str(cap_of.get(int(v), "")).strip()
    return c if c else f"(无标题视频 #{int(v)})"

st.title("短视频推荐 Demo")
st.caption("SIM 长序列 + caption 语义 + 细粒度 GSU · KuaiRand-1K · 测试期用户")

uid = st.selectbox("选择用户", sorted(user_now.keys())[:200])
t_now = user_now[uid]

# ---- 用户画像:近期点击 ----
recent_iids, recent_vids, _ = history.window(uid, t_now + 1, 10)
st.subheader("该用户最近点击")
for v in reversed(recent_vids.tolist()):
    st.markdown(f"- {cap(v)}")

# ---- 全候选打分 ----
with st.spinner("为 31,699 个候选打分..."):
    L, K = rk["hist_len"], rk["long_topk"]
    h, hv, t_start = history.window(uid, t_now + 1, L)
    hist_t = torch.from_numpy(_pad(h, L)).unsqueeze(0).to(device)
    histv_t = torch.from_numpy(_pad(hv, L)).unsqueeze(0).to(device)
    hlen_t = torch.tensor([len(h)]).to(device)

    scores = []
    B = 2048
    with torch.no_grad():
        for s in range(0, len(cand), B):
            c = cand.iloc[s:s + B]
            n = len(c)
            lh_list, lv_list, ll_list = [], [], []
            for key in c[GSU_KEY]:
                lh, lv = tag_history.search(uid, key, t_start, K)
                lh_list.append(_pad(lh, K)); lv_list.append(_pad(lv, K)); ll_list.append(len(lh))
            logit = model(
                torch.full((n,), uid, dtype=torch.long, device=device),
                torch.from_numpy(c["iid_h"].to_numpy()).to(device),
                torch.from_numpy(c["vid"].to_numpy()).to(device),
                torch.from_numpy(c["author_h"].to_numpy()).to(device),
                torch.from_numpy(c["tag1"].to_numpy()).to(device),
                hist_t.expand(n, -1), histv_t.expand(n, -1), hlen_t.expand(n),
                torch.from_numpy(np.stack(lh_list)).to(device),
                torch.from_numpy(np.stack(lv_list)).to(device),
                torch.tensor(ll_list).to(device),
            )
            scores.append(logit.cpu().numpy())
    scores = np.concatenate(scores)

# ---- 重排层:类目打散(每类目最多 MAX_PER_CAT 条)----
# 纯 pointwise 精排分会被最大兴趣类目霸榜,线上系统由重排层保证多样性,这里同理。
MAX_PER_CAT = 2
order = np.argsort(-scores)
picked, cat_cnt = [], {}
for i in order:
    c2 = int(cand["cat2"].iloc[i])
    if cat_cnt.get(c2, 0) >= MAX_PER_CAT:
        continue
    cat_cnt[c2] = cat_cnt.get(c2, 0) + 1
    picked.append(i)
    if len(picked) == TOPN:
        break
top = cand.iloc[picked].copy()
top["score"] = scores[picked]

# ---- 推荐理由 ----
ucats = user_cats[user_cats["uid"] == uid].set_index("cat2")["n"]
rv = [v for v in recent_vids.tolist() if v > 0]
rsem = sem_np[rv].astype(np.float32) if rv else None
if rsem is not None:
    rnorm = np.linalg.norm(rsem, axis=1, keepdims=True) + 1e-8

st.subheader(f"Top-{TOPN} 推荐")
for _, r in top.iterrows():
    reasons = []
    n_cat = int(ucats.get(r["cat2"], 0))
    if n_cat >= 5:
        cname = cat_names.get(int(r["cat2"]), "该类目")
        reasons.append(f"长期兴趣「{cname}」(历史点击 {n_cat} 次)")
    if rsem is not None:
        q = sem_np[int(r["vid"])].astype(np.float32)
        qn = np.linalg.norm(q) + 1e-8
        cos = (rsem @ q) / (rnorm.ravel() * qn)
        j = int(np.argmax(cos))
        if cos[j] > 0.55:
            reasons.append(f"与你最近看的「{cap(rv[j])[:25]}…」内容相近 (cos={cos[j]:.2f})")
    if not reasons:
        reasons.append("近期行为综合匹配")
    st.markdown(f"**{cap(r['vid'])}**")
    st.caption(f"score={r['score']:.2f} · " + " · ".join(reasons))
