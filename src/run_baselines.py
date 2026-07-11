"""第 1 周基线:热门 + ItemCF,输出 Recall@K / NDCG@K。

用法: python -m src.run_baselines --config configs/pure.yaml
"""
import pandas as pd

from src.baselines import itemcf, popularity
from src.config import load_config
from src.eval.metrics import log_result, recall_ndcg_at_k


def load_gt(cfg, split):
    gt_df = pd.read_parquet(cfg.out / f"recall_gt_{split}.parquet")
    return gt_df.groupby("uid")["iid"].agg(set).to_dict()


def main(cfg):
    clicks = pd.read_parquet(cfg.out / "train_clicks.parquet")
    n_items = int(pd.read_parquet(cfg.out / "item_map.parquet")["iid"].max()) + 1
    k_max = max(cfg.eval_topk)

    gts = {split: load_gt(cfg, split) for split in ["val", "test"]}
    users = sorted(set().union(*[g.keys() for g in gts.values()]))

    # ---- 热门 ----
    recs = popularity.recommend(clicks, users, k_max)
    for split, gt in gts.items():
        m = recall_ndcg_at_k(recs, gt, cfg.eval_topk)
        log_result(cfg.reports, f"popularity/{cfg.dataset}/{split}", cfg._config_path, m)

    # ---- ItemCF ----
    sim = itemcf.build_sim(clicks, n_items, cfg.itemcf_max_seq,
                           cfg.itemcf_window, cfg.itemcf_topk_sim)
    print(f"sim matrix nnz: {sim.nnz:,}")
    recs = itemcf.recommend(sim, clicks, users, cfg.itemcf_user_recent, k_max)
    for split, gt in gts.items():
        m = recall_ndcg_at_k(recs, gt, cfg.eval_topk)
        log_result(cfg.reports, f"itemcf/{cfg.dataset}/{split}", cfg._config_path, m)


if __name__ == "__main__":
    main(load_config())
