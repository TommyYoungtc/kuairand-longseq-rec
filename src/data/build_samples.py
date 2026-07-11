"""样本构建:时间切分 → 排序样本 / 召回 ground truth / 用户点击序列。

输出(out_dir 下):
  rank_{train,val,test}.parquet  曝光级排序样本(uid, iid, 各标签, time_ms)
  train_clicks.parquet           训练期点击序列(uid, iid, time_ms 有序;ItemCF/双塔/序列模型共用)
  recall_gt_{val,test}.parquet   召回 ground truth:切分期内每用户点击过的候选 item(去掉训练期已点击的)
"""
import pandas as pd

from src.config import load_config


def main(cfg):
    df = pd.read_parquet(cfg.out / "interactions.parquet")

    train = df[df["date"] <= cfg.train_end_date]
    val = df[(df["date"] > cfg.train_end_date) & (df["date"] <= cfg.val_end_date)]
    test = df[df["date"] > cfg.val_end_date]
    print(f"split rows  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    keep_cols = ["uid", "iid", "iid_h", "vid", "author_h", "tag1", "time_ms", "date", "tab",
                 "play_time_ms", "duration_ms"] + list(cfg.labels)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[keep_cols].to_parquet(cfg.out / f"rank_{name}.parquet", index=False)

    # 训练期点击序列(召回用,仅候选集内 item;iid_h 供排序序列使用)
    clicks = train[(train[cfg.main_label] == 1) & (train["iid"] > 0)]
    clicks = clicks[["uid", "iid", "iid_h", "vid", "time_ms"]].sort_values(["uid", "time_ms"], kind="stable")
    clicks.to_parquet(cfg.out / "train_clicks.parquet", index=False)
    print(f"train clicks: {len(clicks):,}  (users={clicks['uid'].nunique():,})")

    # 召回 ground truth:该期点击的候选 item,排除训练期已点击(评估“发现新兴趣”)
    seen = clicks.groupby("uid")["iid"].agg(set).to_dict()
    for name, part in [("val", val), ("test", test)]:
        pos = part[(part[cfg.main_label] == 1) & (part["iid"] > 0)][["uid", "iid"]].drop_duplicates()
        mask = [row.iid not in seen.get(row.uid, ()) for row in pos.itertuples()]
        gt = pos[pd.Series(mask, index=pos.index)]
        gt.to_parquet(cfg.out / f"recall_gt_{name}.parquet", index=False)
        print(f"recall gt [{name}]: {len(gt):,} pairs, {gt['uid'].nunique():,} users")

    print("build_samples done ->", cfg.out)


if __name__ == "__main__":
    main(load_config())
