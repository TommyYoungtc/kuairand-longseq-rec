"""v4 组件测试:时间间隔分桶 / 扩展历史(作者+gap 平行)/ 用户分组采样器。

用法: python tests/test_v4.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.rank_dataset import (GAP_BOUNDS_MS, N_GAP_BUCKETS, TagHistory,
                                   UserGroupBatchSampler, UserHistory, gap_bucket)

H = 3600_000
D = 86400_000

# ---- gap_bucket:桶界 1h/6h/1d/3d/7d/14d/30d → 桶 1..8 ----
t = 100 * D
gaps_and_expect = [(30 * 60_000, 1), (2 * H, 2), (12 * H, 3), (2 * D, 4),
                   (5 * D, 5), (10 * D, 6), (20 * D, 7), (60 * D, 8)]
clicks_t = np.array([t - g for g, _ in gaps_and_expect])
buckets = gap_bucket(t, clicks_t)
assert buckets.tolist() == [e for _, e in gaps_and_expect], buckets
assert buckets.max() < N_GAP_BUCKETS
print("gap_bucket ok")

# ---- window_ext / search_ext:四路平行 + 不相交语义不变 ----
clicks = pd.DataFrame({
    "uid":      [0,   0,   0,   0],
    "iid_h":    [1,   2,   3,   4],
    "vid":      [11,  12,  13,  14],
    "author_h": [21,  22,  23,  24],
    "tag1":     [1,   1,   2,   1],
    "time_ms":  [t - 20 * D, t - 5 * D, t - 2 * D, t - 1 * H],
})
uh = UserHistory(clicks)
h, hv, ha, hg, t_start = uh.window_ext(0, t, 2)
assert h.tolist() == [3, 4] and hv.tolist() == [13, 14] and ha.tolist() == [23, 24]
assert hg.tolist() == [4, 1]              # 2天→桶4, 1小时→桶1
assert t_start == t - 2 * D
th = TagHistory(clicks, "tag1")
lh, lv, la, lg = th.search_ext(0, 1, t_start, 5, t)
assert lh.tolist() == [1, 2] and la.tolist() == [21, 22]   # 不相交:只取 t_start 之前的 tag1 点击
assert lg.tolist() == [7, 5]              # 20天→桶7, 5天→桶5
print("ext history ok (disjoint kept)")

# ---- 采样器:组内同用户、batch 尺寸恒定、覆盖率 ----
rng = np.random.default_rng(0)
uids = np.repeat(np.arange(20), rng.integers(5, 40, 20))   # 20 个用户,不等量样本
G, M = 4, 8
samp = UserGroupBatchSampler(uids, G, M, seed=1)
seen = 0
for batch in samp:
    assert len(batch) == G * M
    arr = uids[np.array(batch)].reshape(M, G)
    assert (arr == arr[:, :1]).all(), "组内出现了不同用户!"
    seen += len(batch)
assert seen == len(samp) * G * M and len(samp) > 0
# 两个 epoch 的顺序应不同(重新洗牌)
b1 = [tuple(b) for b in samp][0]
b2 = [tuple(b) for b in samp][0]
print(f"sampler ok ({len(samp)} batches/epoch, reshuffles: {b1 != b2})")

print("all tests passed")
