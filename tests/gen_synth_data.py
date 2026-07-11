"""生成复刻 KuaiRand schema 的合成数据,用于流水线冒烟测试(不代表真实分布)。

用法: python tests/gen_synth_data.py [out_dir]  (默认 data/raw/KuaiRand-Synth/data)
之后可用 configs/synth.yaml 跑全流程。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N_USERS, N_ITEMS, N_ROWS = 200, 800, 120_000
DATES = pd.date_range("2022-04-08", "2022-05-08").strftime("%Y%m%d").astype(int)


def gen_log(n, users, dates):
    uid = rng.choice(users, n)
    # 幂律 item 流行度
    pop = rng.zipf(1.3, size=n) % N_ITEMS
    date = rng.choice(dates, n)
    base = pd.to_datetime(date.astype(str)).astype("int64") // 10**6
    time_ms = base + rng.integers(0, 86_400_000, n)
    duration = rng.integers(3_000, 120_000, n)
    click = rng.random(n) < 0.35
    play = np.where(click, (duration * rng.random(n) * 1.5).astype(int), rng.integers(0, 3000, n))
    df = pd.DataFrame({
        "user_id": uid, "video_id": pop, "date": date,
        "hourmin": rng.integers(0, 2360, n), "time_ms": time_ms,
        "is_click": click.astype(int),
        "is_like": (click & (rng.random(n) < 0.05)).astype(int),
        "is_follow": (click & (rng.random(n) < 0.01)).astype(int),
        "is_comment": 0, "is_forward": 0, "is_hate": 0,
        "long_view": (click & (rng.random(n) < 0.5)).astype(int),
        "play_time_ms": play, "duration_ms": duration,
        "profile_stay_time": 0, "comment_stay_time": 0,
        "is_profile_enter": 0, "is_rand": 0, "tab": rng.integers(0, 15, n),
    })
    return df.sort_values("time_ms").reset_index(drop=True)


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    users = np.arange(N_USERS)

    d1 = DATES[DATES <= 20220421]
    d2 = DATES[DATES > 20220421]
    gen_log(int(N_ROWS * 0.45), users, d1).to_csv(out / "log_standard_4_08_to_4_21_synth.csv", index=False)
    gen_log(int(N_ROWS * 0.45), users, d2).to_csv(out / "log_standard_4_22_to_5_08_synth.csv", index=False)
    rand_log = gen_log(int(N_ROWS * 0.1), users, d2)
    rand_log["is_rand"] = 1
    rand_log.to_csv(out / "log_random_4_22_to_5_08_synth.csv", index=False)

    pd.DataFrame({
        "user_id": users,
        "user_active_degree": rng.choice(["high_active", "full_active", "middle_active"], N_USERS),
        "follow_user_num": rng.integers(0, 500, N_USERS),
        "register_days": rng.integers(30, 3000, N_USERS),
        "onehot_feat0": rng.integers(0, 2, N_USERS),
    }).to_csv(out / "user_features_synth.csv", index=False)

    pd.DataFrame({
        "video_id": np.arange(N_ITEMS),
        "author_id": rng.integers(0, 300, N_ITEMS),
        "video_type": rng.choice(["NORMAL", "AD"], N_ITEMS, p=[0.95, 0.05]),
        "upload_dt": "2022-01-01",
        "video_duration": rng.integers(3_000, 120_000, N_ITEMS),
        "music_id": rng.integers(0, 100, N_ITEMS),
        "tag": [",".join(map(str, rng.choice(66, rng.integers(1, 4), replace=False))) for _ in range(N_ITEMS)],
    }).to_csv(out / "video_features_basic_synth.csv", index=False)
    print("synthetic data ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/KuaiRand-Synth/data")
