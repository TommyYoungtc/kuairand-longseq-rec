"""双塔召回:训练 + 全量检索评估。

用法: python -m src.run_two_tower --config configs/pure.yaml
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.recall_dataset import RecallDataset, build_user_eval_inputs
from src.eval.metrics import log_result, recall_ndcg_at_k
from src.models.two_tower import TwoTower


def build_item_author(cfg, n_items):
    vf = pd.read_parquet(cfg.out / "item_features.parquet")
    authors = vf["author_id"].astype("category").cat.codes.to_numpy() + 1  # 0 = unknown
    item_author = np.zeros(n_items, dtype=np.int64)
    item_author[vf["iid"].to_numpy()] = authors
    return item_author, int(authors.max())


@torch.no_grad()
def evaluate(model, cfg, clicks, device, item_author, split):
    model.eval()
    gt_df = pd.read_parquet(cfg.out / f"recall_gt_{split}.parquet")
    gt = gt_df.groupby("uid")["iid"].agg(set).to_dict()
    users = sorted(gt.keys())
    tt = cfg.two_tower
    k_max = max(cfg.eval_topk)

    # 全量 item 向量
    n_items = len(item_author)
    ivecs = []
    for s in range(0, n_items, 8192):
        iid = torch.arange(s, min(s + 8192, n_items), device=device)
        au = torch.from_numpy(item_author[s:s + 8192]).to(device)
        ivecs.append(model.item_tower(iid, au))
    V = torch.cat(ivecs)                                   # (N, D)

    uid_t, hist_t, len_t = build_user_eval_inputs(clicks, users, tt["hist_len"])
    seen = clicks.groupby("uid")["iid"].agg(set).to_dict()
    recs = {}
    for s in range(0, len(users), 1024):
        u = model.user_tower(uid_t[s:s + 1024].to(device),
                             hist_t[s:s + 1024].to(device),
                             len_t[s:s + 1024].to(device))
        scores = u @ V.T                                   # (b, N)
        scores[:, 0] = -1e9                                # OOV 桶
        for j, uu in enumerate(users[s:s + 1024]):
            sc = scores[j]
            sn = list(seen.get(uu, ()))
            if sn:
                sc[torch.tensor(sn, device=device)] = -1e9
            top = torch.topk(sc, k_max).indices.cpu().numpy()
            recs[uu] = top
    return recall_ndcg_at_k(recs, gt, cfg.eval_topk)


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    tt = cfg.two_tower

    clicks = pd.read_parquet(cfg.out / "train_clicks.parquet")
    n_items = int(pd.read_parquet(cfg.out / "item_map.parquet")["iid"].max()) + 1
    n_users = int(pd.read_parquet(cfg.out / "user_map.parquet")["uid"].max()) + 1
    item_author, n_authors = build_item_author(cfg, n_items)

    ds = RecallDataset(clicks, item_author, tt["hist_len"], tt["min_hist"])
    dl = DataLoader(ds, batch_size=tt["batch_size"], shuffle=True,
                    num_workers=tt["num_workers"], drop_last=True)
    print(f"device={device}  train events={len(ds):,}  items={n_items:,}  users={n_users:,}")

    model = TwoTower(n_users, n_items, n_authors,
                     tt["emb_dim"], tt["out_dim"], tt["temperature"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tt["lr"])

    # logQ 修正:item 采样概率 = 训练点击占比
    cnt = clicks["iid"].value_counts().reindex(range(n_items), fill_value=0).to_numpy()
    logq_all = torch.log(torch.tensor((cnt + 1e-12) / cnt.sum(), dtype=torch.float32)).to(device)

    best_metric, best_state, patience = -1.0, None, 0
    for ep in range(tt["epochs"]):
        model.train()
        losses = []
        for uid, hist, hlen, iid, au in tqdm(dl, desc=f"epoch {ep}"):
            iid = iid.to(device)
            u = model.user_tower(uid.to(device), hist.to(device), hlen.to(device))
            v = model.item_tower(iid, au.to(device))
            loss = model.in_batch_loss(u, v, logq_all[iid])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {ep}  loss={np.mean(losses):.4f}")
        m = evaluate(model, cfg, clicks, device, item_author, "val")
        log_result(cfg.reports, f"two_tower/{cfg.dataset}/val/ep{ep}", cfg._config_path, m)
        # 以 val Recall@100 选最优 epoch,连续 2 轮不涨则早停
        key = m.get("Recall@100", max(m.values()))
        if key > best_metric:
            best_metric, patience = key, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                print(f"early stop at epoch {ep} (best val Recall@100={best_metric:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    m = evaluate(model, cfg, clicks, device, item_author, "test")
    log_result(cfg.reports, f"two_tower/{cfg.dataset}/test", cfg._config_path, m)
    ckpt = cfg.out / "two_tower.pt"
    torch.save(model.state_dict(), ckpt)
    print("model saved ->", ckpt)


if __name__ == "__main__":
    main(load_config())
