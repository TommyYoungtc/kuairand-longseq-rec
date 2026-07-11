"""RecallDataset 正确性测试。

无 torch 环境:仅测索引构建不变量;有 torch:完整测 __getitem__。
用法: python tests/test_recall_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.recall_dataset import RecallDataset

# 构造两个用户的点击流
clicks = pd.DataFrame({
    "uid":     [0] * 6 + [1] * 4,
    "iid":     [3, 5, 7, 2, 9, 4, 8, 6, 1, 5],
    "time_ms": list(range(6)) + list(range(4)),
})
item_author = np.arange(10)
MIN_HIST = 2
ds = RecallDataset(clicks, item_author, hist_len=4, min_hist=MIN_HIST)

# 不变量 1:样本数 = 每用户 (len - min_hist) 之和
assert len(ds) == (6 - MIN_HIST) + (4 - MIN_HIST), len(ds)

# 不变量 2:每个事件的 pos >= min_hist,且历史全部属于同一用户、严格在事件之前
uid_all = clicks["uid"].to_numpy()
iid_all = clicks["iid"].to_numpy()
for k in range(len(ds)):
    e = ds.event_idx[k]
    n = min(ds.pos[k], ds.H)
    assert ds.pos[k] >= MIN_HIST
    assert (uid_all[e - n:e] == ds.uid[k]).all(), "历史跨越了用户边界!"
    assert ds.iid[k] == iid_all[e], "正例 iid 不匹配"

print("index invariants ok")

try:
    import torch  # noqa
except ImportError:
    print("torch 不可用,跳过 __getitem__ 测试")
else:
    uid, hist, hlen, iid, au = ds[0]
    # 用户0 第一个可训练事件: e=2 (iid=7), 历史=[3,5]
    assert iid.item() == 7 and hlen.item() == 2
    assert hist.tolist() == [3, 5, 0, 0]
    assert au.item() == item_author[7]
    # 用户1 第一个可训练事件: e=8 (iid=1), 历史=[8,6]
    k = (6 - MIN_HIST)
    uid, hist, hlen, iid, au = ds[k]
    assert uid.item() == 1 and iid.item() == 1
    assert hist.tolist() == [8, 6, 0, 0]
    print("__getitem__ ok")

print("all tests passed")
