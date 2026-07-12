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


class UserHistory:
    def __init__(self, clicks: pd.DataFrame):
        clicks = clicks.sort_values(["uid", "time_ms"], kind="stable")
        self.times, self.iids, self.vids = {}, {}, {}
        for uid, g in clicks.groupby("uid"):
            self.times[uid] = g["time_ms"].to_numpy()
            self.iids[uid] = g["iid_h"].to_numpy()
            self.vids[uid] = g["vid"].to_numpy()

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


class TagHistory:
    """GSU 硬检索,检索键可配置:tag1(66个一级标签)或 cat2(数百个二级类目)。"""

    def __init__(self, clicks: pd.DataFrame, key: str = "tag1"):
        clicks = clicks.sort_values(["uid", "time_ms"], kind="stable")
        self.times, self.iids, self.vids = {}, {}, {}
        for (uid, tag), g in clicks.groupby(["uid", key]):
            self.times[(uid, tag)] = g["time_ms"].to_numpy()
            self.iids[(uid, tag)] = g["iid_h"].to_numpy()
            self.vids[(uid, tag)] = g["vid"].to_numpy()

    def search(self, uid, tag, t, K):
        """→ (iid_h 序列, vid 序列):tag 类目下严格 t 之前的最近 K 次点击。"""
        ts = self.times.get((uid, tag))
        if ts is None:
            z = np.zeros(0, dtype=np.int64)
            return z, z
        e = np.searchsorted(ts, t, side="left")
        s = max(0, e - K)
        return self.iids[(uid, tag)][s:e], self.vids[(uid, tag)][s:e]


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
