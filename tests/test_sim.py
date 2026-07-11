"""TagHistory(GSU 硬检索)与 RankDatasetSIM 正确性测试。

用法: python tests/test_sim.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.rank_dataset import TagHistory, UserHistory

# 用户0:类目1 点击在 t=10,30,50,70;类目2 在 t=20,40;用户1:类目1 在 t=15
clicks = pd.DataFrame({
    "uid":     [0,  0,  0,  0,  0,  0,  1],
    "iid_h":   [11, 21, 12, 22, 13, 14, 31],
    "vid":     [111, 121, 112, 122, 113, 114, 131],
    "tag1":    [1,  2,  1,  2,  1,  1,  1],
    "time_ms": [10, 20, 30, 40, 50, 70, 15],
})
th = TagHistory(clicks)

# 类目隔离 + 严格时间 + K 截断 + 用户隔离
assert th.search(0, 1, 60, 10)[0].tolist() == [11, 12, 13]
assert th.search(0, 2, 60, 10)[0].tolist() == [21, 22]
assert th.search(0, 1, 50, 10)[0].tolist() == [11, 12]
ii, vv = th.search(0, 1, 100, 2)
assert ii.tolist() == [13, 14] and vv.tolist() == [113, 114]
assert th.search(1, 1, 100, 5)[0].tolist() == [31]
assert th.search(0, 7, 100, 5)[0].tolist() == []
assert th.search(9, 1, 100, 5)[0].tolist() == []

print("TagHistory ok")

try:
    import torch  # noqa
except ImportError:
    print("torch 不可用,跳过 RankDatasetSIM 测试")
else:
    from src.data.rank_dataset import RankDatasetSIM
    uh = UserHistory(clicks)
    samples = pd.DataFrame({
        "uid": [0], "iid_h": [99], "vid": [199], "author_h": [3], "tag1": [1],
        "time_ms": [60], "is_click": [1],
    })
    ds = RankDatasetSIM(samples, uh, th, hist_len=3, long_topk=4, label="is_click")
    (uid, iid_h, vid, author, tag, hist, hist_vid, hlen,
     long_hist, long_vid, long_len, y) = ds[0]
    # 短支路:t=60 前最近3次 = [12, 22, 13](时间 30,40,50)
    assert hist.tolist() == [12, 22, 13] and hist_vid.tolist() == [112, 122, 113]
    # 长支路(不相交):短窗口起点 t=30 之前的类目1点击 = 仅 [11]
    assert long_hist.tolist() == [11, 0, 0, 0] and long_vid.tolist() == [111, 0, 0, 0]
    assert long_len.item() == 1
    print("RankDatasetSIM ok (disjoint + vid)")

print("all tests passed")
