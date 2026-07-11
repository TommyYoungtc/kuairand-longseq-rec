"""双塔召回训练集:每条样本 = (uid, 近期历史, 正例 iid, 作者)。

从 train_clicks 构建:对每个点击事件 t,历史 = 该用户 t 之前最近 H 次点击。
用 numpy 预展开(clicks 已按 uid,time 排序),训练时 O(1) 取样本。
"""
import numpy as np
import pandas as pd

try:  # torch 延迟依赖:构造与索引逻辑纯 numpy,可脱离 torch 测试
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch, Dataset = None, object


class RecallDataset(Dataset):
    def __init__(self, clicks: pd.DataFrame, item_author: np.ndarray,
                 hist_len: int = 50, min_hist: int = 3):
        self.H = hist_len
        uid = clicks["uid"].to_numpy()
        iid = clicks["iid"].to_numpy()
        # 每个事件在其用户序列内的位置
        starts = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
        seq_pos = np.arange(len(uid)) - np.repeat(starts, np.diff(np.r_[starts, len(uid)]))
        keep = seq_pos >= min_hist  # 历史太短的事件不作训练目标
        self.uid, self.iid, self.pos = uid[keep], iid[keep], seq_pos[keep]
        self.all_iid = iid
        self.event_idx = np.flatnonzero(keep)
        self.item_author = item_author

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, k):
        e = self.event_idx[k]
        n = min(self.pos[k], self.H)
        hist = np.zeros(self.H, dtype=np.int64)
        hist[:n] = self.all_iid[e - n:e]
        return (
            torch.tensor(self.uid[k]),
            torch.from_numpy(hist),
            torch.tensor(n, dtype=torch.float32),
            torch.tensor(self.iid[k]),
            torch.tensor(self.item_author[self.iid[k]]),
        )


def build_user_eval_inputs(clicks: pd.DataFrame, users: list, hist_len: int):
    """评估用:每个用户取训练期末尾 H 次点击作为历史。"""
    seqs = clicks.groupby("uid")["iid"].agg(list).to_dict()
    uid_t, hist_t, len_t = [], [], []
    for u in users:
        seq = seqs.get(u, [])[-hist_len:]
        hist = np.zeros(hist_len, dtype=np.int64)
        hist[:len(seq)] = seq
        uid_t.append(u)
        hist_t.append(hist)
        len_t.append(max(len(seq), 1))
    return (torch.tensor(uid_t), torch.from_numpy(np.stack(hist_t)),
            torch.tensor(len_t, dtype=torch.float32))
