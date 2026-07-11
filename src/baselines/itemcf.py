"""ItemCF 基线:点击序列滑窗共现 → 余弦归一 → Top-K 相似截断 → 求和打分。"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm


def build_sim(train_clicks: pd.DataFrame, n_items: int, max_seq: int,
              window: int, topk_sim: int) -> sp.csr_matrix:
    rows, cols = [], []
    for _, s in tqdm(train_clicks.groupby("uid")["iid"], desc="co-occurrence"):
        seq = s.to_numpy()[-max_seq:]
        for d in range(1, min(window, len(seq) - 1) + 1):
            rows.append(seq[:-d])
            cols.append(seq[d:])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    data = np.ones(len(r), dtype=np.float32)
    # 对称化
    cooc = sp.coo_matrix((data, (r, c)), shape=(n_items, n_items))
    cooc = (cooc + cooc.T).tocsr()

    # 余弦归一:cooc(i,j) / sqrt(cnt_i * cnt_j)
    cnt = train_clicks["iid"].value_counts().reindex(range(n_items), fill_value=0).to_numpy()
    norm = np.sqrt(np.maximum(cnt, 1)).astype(np.float32)
    d_inv = sp.diags(1.0 / norm)
    sim = d_inv @ cooc @ d_inv

    # 每行仅保留 Top-K 相似
    sim = sim.tocsr()
    for i in range(n_items):
        s, e = sim.indptr[i], sim.indptr[i + 1]
        if e - s > topk_sim:
            row_data = sim.data[s:e]
            thresh = np.partition(row_data, -topk_sim)[-topk_sim]
            row_data[row_data < thresh] = 0.0
    sim.eliminate_zeros()
    return sim


def recommend(sim: sp.csr_matrix, train_clicks: pd.DataFrame, users: list,
              recent: int, k_max: int) -> dict:
    seqs = train_clicks.groupby("uid")["iid"].agg(list).to_dict()
    recs = {}
    for uid in tqdm(users, desc="itemcf infer"):
        seq = seqs.get(uid)
        if not seq:
            recs[uid] = np.array([], dtype=np.int64)
            continue
        recent_items = list(dict.fromkeys(reversed(seq)))[:recent]  # 去重保序
        scores = np.asarray(sim[recent_items].sum(axis=0)).ravel()
        scores[list(set(seq))] = -np.inf  # 排除已点击
        scores[0] = -np.inf               # 排除 OOV 桶
        k = min(k_max, (scores > -np.inf).sum())
        top = np.argpartition(-scores, k - 1)[:k]
        recs[uid] = top[np.argsort(-scores[top])]
    return recs
