"""UserHistory / RankDataset 正确性测试(重点:时间截取无泄漏)。

用法: python tests/test_rank_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.rank_dataset import UserHistory

clicks = pd.DataFrame({
    "uid":     [0] * 4 + [1] * 2,
    "iid_h":   [3, 5, 7, 9, 4, 6],
    "vid":     [103, 105, 107, 109, 104, 106],
    "time_ms": [10, 20, 30, 40, 15, 25],
})
h = UserHistory(clicks)

# 严格 < t
assert h.before(0, 30, L=10).tolist() == [3, 5]
assert h.before(0, 10, L=10).tolist() == []
assert h.before(0, 41, L=2).tolist() == [7, 9]
assert h.before(99, 100, L=5).tolist() == []
assert h.before(1, 100, L=5).tolist() == [4, 6]

# window:iid 与 vid 平行,且返回起点时间
hi, hv, t0 = h.window(0, 40, 2)
assert hi.tolist() == [5, 7] and hv.tolist() == [105, 107] and t0 == 20
hi, hv, t0 = h.window(9, 40, 2)
assert hi.tolist() == [] and t0 == 40

print("UserHistory ok")

try:
    import torch  # noqa
except ImportError:
    print("torch 不可用,跳过 RankDataset __getitem__ 测试")
else:
    from src.data.rank_dataset import RankDataset
    samples = pd.DataFrame({
        "uid": [0], "iid_h": [9], "vid": [109], "author_h": [2], "tag1": [5],
        "time_ms": [40], "is_click": [1],
    })
    ds = RankDataset(samples, h, hist_len=3, label="is_click")
    uid, iid_h, vid, author, tag, hist, hist_vid, hlen, y = ds[0]
    assert iid_h.item() == 9 and vid.item() == 109 and y.item() == 1.0
    assert hist.tolist() == [3, 5, 7] and hist_vid.tolist() == [103, 105, 107]
    assert hlen.item() == 3
    print("RankDataset ok")

print("all tests passed")
