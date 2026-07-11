"""评估指标:Recall@K / NDCG@K(召回)、AUC / GAUC(排序)。"""
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def recall_ndcg_at_k(recs: dict, gt: dict, ks: list[int]) -> dict:
    """recs: {uid: np.ndarray 有序推荐列表}, gt: {uid: set 正例}.
    返回 {"Recall@k": v, "NDCG@k": v},对有 ground truth 的用户求平均。"""
    out = {}
    for k in ks:
        recalls, ndcgs = [], []
        idcg_cache = np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))
        for uid, truth in gt.items():
            if uid not in recs or not truth:
                continue
            rec_k = recs[uid][:k]
            hits = np.isin(rec_k, list(truth))
            recalls.append(hits.sum() / len(truth))
            dcg = (hits / np.log2(np.arange(2, len(rec_k) + 2))).sum()
            idcg = idcg_cache[min(len(truth), k) - 1]
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        out[f"Recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        out[f"NDCG@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    return out


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def gauc(labels: np.ndarray, scores: np.ndarray, uids: np.ndarray) -> float:
    """按用户分组的 AUC,以该用户曝光数加权;跳过全正/全负用户。"""
    by_user = defaultdict(list)
    for u, l, s in zip(uids, labels, scores):
        by_user[u].append((l, s))
    num, den = 0.0, 0.0
    for pairs in by_user.values():
        ls = np.array([p[0] for p in pairs])
        ss = np.array([p[1] for p in pairs])
        if ls.min() == ls.max():
            continue
        num += len(pairs) * roc_auc_score(ls, ss)
        den += len(pairs)
    return float(num / den) if den else float("nan")


def log_result(reports_dir, exp_name: str, config_path: str, metrics: dict):
    """结果追加到 reports/results.csv。"""
    import csv
    import datetime
    path = reports_dir / "results.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "exp_name", "config", "metric", "value"])
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for k, v in metrics.items():
            w.writerow([now, exp_name, config_path, k, f"{v:.6f}"])
    print(f"[{exp_name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
