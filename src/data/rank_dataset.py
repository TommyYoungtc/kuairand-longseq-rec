"""排序样本集:曝光级样本 + 严格按时间戳截取的用户历史。

序列同时携带 iid_h(哈希 ID,可训练 embedding)与 vid(稠密视频索引,
冻结语义表查表用)。均严格 time < t,无标签泄漏。

  UserHistory  近期点击流(短序列支路)
  TagHistory   按类目索引的全生命周期点击流(SIM/GSU 硬检索),
               配合 window() 起点做不相交检索:长支路只取短窗口之前的行为
"""
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch, Dataset = None, object


GAP_BOUNDS_MS = np.array([3600_000, 6 * 3600_000, 24 * 3600_000, 3 * 86400_000,
                          7 * 86400_000, 14 * 86400_000, 30 * 86400_000], dtype=np.int64)
N_GAP_BUCKETS = len(GAP_BOUNDS_MS) + 2   # +1 溢出桶, +1 padding(0)


def gap_bucket(t_sample, click_times):
    """时间间隔 → 桶 id(1..8;0 留给 padding)。桶界:1h/6h/1d/3d/7d/14d/30d。"""
    gaps = t_sample - click_times
    return np.searchsorted(GAP_BOUNDS_MS, gaps, side="left").astype(np.int64) + 1


class UserGroupBatchSampler:
    """按用户分组的 batch 采样器:每 batch = M 个用户组 × 每组 G 条该用户样本。

    保证组内样本同用户 → pairwise(BPR)损失可在组内向量化计算。
    每个用户的样本随机打乱后切成大小 G 的整组(尾部不足 G 的丢弃),
    所有组打乱后每 M 组拼一个 batch。每个 epoch 重新洗牌。
    """

    def __init__(self, uids: np.ndarray, group_size: int = 8,
                 groups_per_batch: int = 128, seed: int = 42):
        self.G, self.M = group_size, groups_per_batch
        self.rng = np.random.default_rng(seed)
        self.user_idx = {}
        order = np.argsort(uids, kind="stable")
        bounds = np.flatnonzero(np.r_[True, uids[order][1:] != uids[order][:-1]])
        for s, e in zip(bounds, np.r_[bounds[1:], len(order)]):
            self.user_idx[uids[order[s]]] = order[s:e]
        self.n_batches = sum(len(v) // self.G for v in self.user_idx.values()) // self.M

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        groups = []
        for idx in self.user_idx.values():
            idx = idx.copy()
            self.rng.shuffle(idx)
            n_full = (len(idx) // self.G) * self.G
            groups.extend(idx[:n_full].reshape(-1, self.G))
        self.rng.shuffle(groups)
        for b in range(len(groups) // self.M):
            yield np.concatenate(groups[b * self.M:(b + 1) * self.M]).tolist()


class UserHistory:
    def __init__(self, clicks: pd.DataFrame):
        clicks = clicks.sort_values(["uid", "time_ms"], kind="stable")
        self.times, self.iids, self.vids, self.auths = {}, {}, {}, {}
        has_auth = "author_h" in clicks.columns
        for uid, g in clicks.groupby("uid"):
            self.times[uid] = g["time_ms"].to_numpy()
            self.iids[uid] = g["iid_h"].to_numpy()
            self.vids[uid] = g["vid"].to_numpy()
            if has_auth:
                self.auths[uid] = g["author_h"].to_numpy()

    def before(self, uid, t, L):
        ts = self.times.get(uid)
        if ts is None:
            return np.zeros(0, dtype=np.int64)
        e = np.searchsorted(ts, t, side="left")   # 严格 < t
        return self.iids[uid][max(0, e - L):e]

    def window(self, uid, t, L):
        """→ (iid_h 序列, vid 序列, 窗口起点时间)。"""
        ts = self.times.get(uid)
        if ts is None:
            z = np.zeros(0, dtype=np.int64)
            return z, z, t
        e = np.searchsorted(ts, t, side="left")
        s = max(0, e - L)
        t_start = int(ts[s]) if e > s else t
        return self.iids[uid][s:e], self.vids[uid][s:e], t_start

    def window_ext(self, uid, t, L):
        """v4:→ (iid_h, vid, author_h, gap桶, 窗口起点时间)。"""
        ts = self.times.get(uid)
        if ts is None:
            z = np.zeros(0, dtype=np.int64)
            return z, z, z, z, t
        e = np.searchsorted(ts, t, side="left")
        s = max(0, e - L)
        t_start = int(ts[s]) if e > s else t
        return (self.iids[uid][s:e], self.vids[uid][s:e], self.auths[uid][s:e],
                gap_bucket(t, ts[s:e]), t_start)


class TagHistory:
    """GSU 硬检索,检索键可配置:tag1(66个一级标签)或 cat2(数百个二级类目)。"""

    def __init__(self, clicks: pd.DataFrame, key: str = "tag1"):
        clicks = clicks.sort_values(["uid", "time_ms"], kind="stable")
        self.times, self.iids, self.vids, self.auths = {}, {}, {}, {}
        has_auth = "author_h" in clicks.columns
        for (uid, tag), g in clicks.groupby(["uid", key]):
            self.times[(uid, tag)] = g["time_ms"].to_numpy()
            self.iids[(uid, tag)] = g["iid_h"].to_numpy()
            self.vids[(uid, tag)] = g["vid"].to_numpy()
            if has_auth:
                self.auths[(uid, tag)] = g["author_h"].to_numpy()

    def search(self, uid, tag, t, K):
        """→ (iid_h 序列, vid 序列):tag 类目下严格 t 之前的最近 K 次点击。"""
        ts = self.times.get((uid, tag))
        if ts is None:
            z = np.zeros(0, dtype=np.int64)
            return z, z
        e = np.searchsorted(ts, t, side="left")
        s = max(0, e - K)
        return self.iids[(uid, tag)][s:e], self.vids[(uid, tag)][s:e]

    def search_ext(self, uid, tag, t, K, t_ref):
        """v4:→ (iid_h, vid, author_h, gap桶)。gap 相对样本时刻 t_ref 计算。"""
        ts = self.times.get((uid, tag))
        if ts is None:
            z = np.zeros(0, dtype=np.int64)
            return z, z, z, z
        e = np.searchsorted(ts, t, side="left")
        s = max(0, e - K)
        return (self.iids[(uid, tag)][s:e], self.vids[(uid, tag)][s:e],
                self.auths[(uid, tag)][s:e], gap_bucket(t_ref, ts[s:e]))


def _pad(a, L):
    out = np.zeros(L, dtype=np.int64)
    out[:len(a)] = a
    return out


class RankDataset(Dataset):
    """DIN 用:短序列支路。返回 (uid, iid_h, vid, author, tag, hist, hist_vid, n, y)。"""

    def __init__(self, samples: pd.DataFrame, history: UserHistory,
                 hist_len: int, label: str):
        self.uid = samples["uid"].to_numpy()
        self.iid_h = samples["iid_h"].to_numpy()
        self.vid = samples["vid"].to_numpy()
        self.author = samples["author_h"].to_numpy()
        self.tag = samples["tag1"].to_numpy()
        self.t = samples["time_ms"].to_numpy()
        # label 可为单任务(str)或多任务(list) → y 为标量或 (T,) 向量
        self.y = samples[label].to_numpy().astype(np.float32)
        self.history = history
        self.L = hist_len

    def __len__(self):
        return len(self.uid)

    def _short(self, k):
        h, hv, t_start = self.history.window(self.uid[k], self.t[k], self.L)
        return _pad(h, self.L), _pad(hv, self.L), len(h), t_start

    def _base(self, k):
        return (torch.tensor(self.uid[k]), torch.tensor(self.iid_h[k]),
                torch.tensor(self.vid[k]), torch.tensor(self.author[k]),
                torch.tensor(self.tag[k]))

    def __getitem__(self, k):
        hist, hist_vid, n, _ = self._short(k)
        return (*self._base(k), torch.from_numpy(hist), torch.from_numpy(hist_vid),
                torch.tensor(n, dtype=torch.long), torch.tensor(self.y[k]))


class RankDatasetSIM(RankDataset):
    """SIM 用:短序列 + 不相交 GSU 长期检索双支路。"""

    def __init__(self, samples, history, tag_history, hist_len, long_topk, label,
                 gsu_key: str = "tag1"):
        super().__init__(samples, history, hist_len, label)
        self.tag_history = tag_history
        self.K = long_topk
        self.gsu = samples[gsu_key].to_numpy()   # 检索键取值(target 的 tag1/cat2)

    def __getitem__(self, k):
        hist, hist_vid, n, t_start = self._short(k)
        lh, lv = self.tag_history.search(self.uid[k], self.gsu[k], t_start, self.K)
        return (*self._base(k), torch.from_numpy(hist), torch.from_numpy(hist_vid),
                torch.tensor(n, dtype=torch.long),
                torch.from_numpy(_pad(lh, self.K)), torch.from_numpy(_pad(lv, self.K)),
                torch.tensor(len(lh), dtype=torch.long), torch.tensor(self.y[k]))


class RankDatasetV4(RankDataset):
    """v4:序列 token 携带 (iid_h, vid, author_h, 时间间隔桶) 四路特征,长短双支路。"""

    def __init__(self, samples, history, tag_history, hist_len, long_topk, label,
                 gsu_key: str = "tag1"):
        super().__init__(samples, history, hist_len, label)
        self.tag_history = tag_history
        self.K = long_topk
        self.gsu = samples[gsu_key].to_numpy()

    def __getitem__(self, k):
        uid, t = self.uid[k], self.t[k]
        h, hv, ha, hg, t_start = self.history.window_ext(uid, t, self.L)
        lh, lv, la, lg = self.tag_history.search_ext(uid, self.gsu[k], t_start, self.K, t)
        L, K = self.L, self.K
        return (*self._base(k),
                torch.from_numpy(_pad(h, L)), torch.from_numpy(_pad(hv, L)),
                torch.from_numpy(_pad(ha, L)), torch.from_numpy(_pad(hg, L)),
                torch.tensor(len(h), dtype=torch.long),
                torch.from_numpy(_pad(lh, K)), torch.from_numpy(_pad(lv, K)),
                torch.from_numpy(_pad(la, K)), torch.from_numpy(_pad(lg, K)),
                torch.tensor(len(lh), dtype=torch.long),
                torch.tensor(self.y[k]))
