"""为 Demo 构建 caption 缓存:候选 item + 各用户近期点击的视频标题。

用法: python -m src.data.build_caption_cache --config configs/1k.yaml
输出: out_dir/caption_cache.parquet (vid, caption)
"""
import pandas as pd
from tqdm import tqdm

from src.config import load_config

CAP_FILE = "data/raw/kuairand_video_captions.csv"
RECENT_PER_USER = 200


def main(cfg):
    vmap = pd.read_parquet(cfg.out / "video_map.parquet")
    vid_of = dict(zip(vmap["video_id"], vmap["vid"]))

    need = set()
    # 候选 item 的 vid
    imap = pd.read_parquet(cfg.out / "item_map.parquet")
    need.update(vmap.merge(imap, on="video_id")["vid"].tolist())
    # 各用户最近点击的 vid
    inter = pd.read_parquet(cfg.out / "interactions.parquet",
                            columns=["uid", "vid", "time_ms", cfg.main_label])
    clicks = inter[inter[cfg.main_label] == 1]
    recent = clicks.sort_values("time_ms").groupby("uid").tail(RECENT_PER_USER)
    need.update(recent["vid"].tolist())
    need.discard(0)
    print(f"captions needed: {len(need):,}")

    rows = []
    for chunk in tqdm(pd.read_csv(CAP_FILE, chunksize=500_000,
                                  usecols=["final_video_id", "caption"]),
                      desc="scan"):
        chunk["vid"] = chunk["final_video_id"].map(vid_of)
        hit = chunk.dropna(subset=["vid"])
        hit = hit[hit["vid"].astype(int).isin(need)]
        rows.append(pd.DataFrame({"vid": hit["vid"].astype(int),
                                  "caption": hit["caption"].fillna("").astype(str).str[:100]}))
    cache = pd.concat(rows).drop_duplicates("vid")
    cache.to_parquet(cfg.out / "caption_cache.parquet", index=False)
    print(f"cached {len(cache):,} captions -> {cfg.out / 'caption_cache.parquet'}")

    # 二级类目 id → 中文名(推荐理由展示用)
    cat_file = cfg.get("categories_file", "")
    if cat_file:
        names = {}
        for chunk in tqdm(pd.read_csv(cat_file, chunksize=2_000_000,
                                      usecols=["second_level_category_id",
                                               "second_level_category_name"]),
                          desc="cat names"):
            sub = chunk.dropna().drop_duplicates("second_level_category_id")
            for i, n in zip(sub["second_level_category_id"], sub["second_level_category_name"]):
                if i >= 0:
                    names.setdefault(int(i), n)
        pd.DataFrame({"cat2": list(names), "name": list(names.values())}) \
            .to_parquet(cfg.out / "cat2_names.parquet", index=False)
        print(f"cat2 names: {len(names)} -> {cfg.out / 'cat2_names.parquet'}")


if __name__ == "__main__":
    main(load_config())
