"""预处理:合并日志 → 候选集裁剪 → ID 重编号 + 长尾 hash 分桶 → parquet。

ID 体系(应对 1K 版 437 万 item、85% 长尾的稀疏建模方案):
  iid    召回用:0=OOV,1..N=候选集(高频 item)。召回只在候选集内检索。
  iid_h  排序/序列用:1..N=候选集,N+1..N+B=长尾 item 哈希桶(hash 冲突可接受),
         0 保留为 padding。行为序列与排序目标 item 都用 iid_h,长尾不再折叠成单一桶。
  author_h  作者 ID 哈希桶(1..A,0=未知)。
  tag1      视频第一个类目标签(1..67,0=未知)。

输出(out_dir 下):
  interactions.parquet / random_log.parquet   含 uid,iid,iid_h,author_h,tag1 与反馈信号
  item_map.parquet / user_map.parquet         ID 映射
  item_features.parquet / user_features.parquet
  meta.json                                   词表大小(n_candidates/item_buckets/...)
"""
import json

import pandas as pd

from src.config import load_config

LOG_USECOLS = [
    "user_id", "video_id", "date", "time_ms",
    "is_click", "is_like", "is_follow", "long_view",
    "play_time_ms", "duration_ms", "tab",
]


def read_logs(cfg, files):
    dfs = []
    for name in files:
        path = cfg.raw / name
        print(f"reading {path} ...")
        df = pd.read_csv(path, usecols=lambda c: c in LOG_USECOLS)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def attach_ids(df, item_map, user_map, item_side, n_cand, hash_cfg, video_map):
    """挂载 iid / iid_h / vid / author_h / tag1。"""
    df = df.merge(item_map, on="video_id", how="left")
    df["iid"] = df["iid"].fillna(0).astype("int64")
    df = df.merge(video_map, on="video_id", how="left")   # 稠密视频索引(语义查表用)
    df["vid"] = df["vid"].fillna(0).astype("int64")
    df = df.merge(user_map, on="user_id", how="left")
    # 长尾 hash:候选集内用真实 iid,否则 N+1+hash(video_id)
    B = hash_cfg["item_buckets"]
    df["iid_h"] = df["iid"].where(df["iid"] > 0, n_cand + 1 + df["video_id"] % B)
    # item 侧特征(全量 item)
    df = df.merge(item_side, on="video_id", how="left")
    A = hash_cfg["author_buckets"]
    df["author_h"] = (df["author_id"].fillna(-1).astype("int64") % A + 1).where(
        df["author_id"].notna(), 0).astype("int64")
    df["tag1"] = df["tag1"].fillna(0).clip(0, 67).astype("int64")
    df["cat2"] = df["cat2"].fillna(0).astype("int64")
    return df.drop(columns=["author_id"])


def main(cfg):
    logs = read_logs(cfg, cfg.log_files)
    print(f"standard log rows: {len(logs):,}")

    # ---- 候选集裁剪:曝光次数 >= min_item_count ----
    cnt = logs["video_id"].value_counts()
    cand = cnt[cnt >= cfg.min_item_count].index
    print(f"items total={len(cnt):,}  candidates(>= {cfg.min_item_count} views)={len(cand):,}"
          f"  coverage={cnt[cand].sum() / len(logs):.2%} of interactions")

    item_map = pd.DataFrame({"video_id": sorted(cand)})
    item_map["iid"] = range(1, len(item_map) + 1)
    n_cand = len(item_map)
    user_map = pd.DataFrame({"user_id": sorted(logs["user_id"].unique())})
    user_map["uid"] = range(len(user_map))
    # 稠密视频索引:日志中出现过的全部视频(0 保留为 padding/未知)
    video_map = pd.DataFrame({"video_id": sorted(cnt.index)})
    video_map["vid"] = range(1, len(video_map) + 1)

    # ---- 全量 item 侧特征(author/tag),供 hash 与排序模型使用 ----
    print("reading full video features ...")
    vf_all = pd.read_csv(cfg.raw / cfg.video_features,
                         usecols=["video_id", "author_id", "tag"])
    vf_all["tag1"] = pd.to_numeric(
        vf_all["tag"].astype(str).str.split(",").str[0], errors="coerce") + 1
    item_side = vf_all[["video_id", "author_id", "tag1"]]

    # 二级类目(细粒度 GSU 检索键;未配置/缺失时优雅降级为 0)
    cat_path = cfg.get("categories_file", "")
    try:
        if not cat_path:
            raise FileNotFoundError
        cats = []
        for ch in pd.read_csv(cat_path, chunksize=1_000_000,
                              usecols=["final_video_id", "second_level_category_id"]):
            ch = ch[ch["final_video_id"].isin(set(logs["video_id"].unique()))]
            cats.append(ch)
        cats = pd.concat(cats).rename(columns={"final_video_id": "video_id"})
        cats["cat2"] = cats["second_level_category_id"].fillna(-124)
        cats.loc[cats["cat2"] < 0, "cat2"] = 0
        cats["cat2"] = cats["cat2"].astype("int64")
        item_side = item_side.merge(cats[["video_id", "cat2"]], on="video_id", how="left")
        print(f"cat2 loaded: {len(cats):,} rows, {cats['cat2'].nunique()} distinct")
    except FileNotFoundError:
        print("!! categories 文件不存在,cat2 置 0(细粒度 GSU 不可用)")
        item_side = item_side.copy()
        item_side["cat2"] = 0

    logs = attach_ids(logs, item_map, user_map, item_side, n_cand, cfg.hash, video_map)
    logs = logs.sort_values(["uid", "time_ms"], kind="stable").reset_index(drop=True)
    logs.to_parquet(cfg.out / "interactions.parquet", index=False)
    item_map.to_parquet(cfg.out / "item_map.parquet", index=False)
    user_map.to_parquet(cfg.out / "user_map.parquet", index=False)
    video_map.to_parquet(cfg.out / "video_map.parquet", index=False)

    # ---- 随机曝光日志(无偏评估用) ----
    rand = read_logs(cfg, [cfg.random_log])
    rand = attach_ids(rand, item_map, user_map, item_side, n_cand, cfg.hash, video_map)
    rand = rand.dropna(subset=["uid"])
    rand["uid"] = rand["uid"].astype("int64")
    rand.to_parquet(cfg.out / "random_log.parquet", index=False)
    print(f"random log rows: {len(rand):,}")

    # ---- 特征表 ----
    uf = pd.read_csv(cfg.raw / cfg.user_features)
    uf = uf.merge(user_map, on="user_id", how="inner")
    uf.to_parquet(cfg.out / "user_features.parquet", index=False)

    vf = pd.read_csv(cfg.raw / cfg.video_features)
    vf = vf.merge(item_map, on="video_id", how="inner")
    vf.to_parquet(cfg.out / "item_features.parquet", index=False)
    print(f"user_features: {len(uf):,} rows, item_features(candidates): {len(vf):,} rows")

    meta = {
        "n_candidates": n_cand,
        "item_buckets": cfg.hash["item_buckets"],
        "author_buckets": cfg.hash["author_buckets"],
        "n_iid_h": n_cand + cfg.hash["item_buckets"] + 1,
        "n_author_h": cfg.hash["author_buckets"] + 1,
        "n_tag": 68,
        "n_users": len(user_map),
        "n_vid": len(video_map) + 1,
    }
    (cfg.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("meta:", meta)
    print("preprocess done ->", cfg.out)


if __name__ == "__main__":
    main(load_config())
# EOF
