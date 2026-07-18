"""排序模型训练 + AUC/GAUC 评估。支持 DIN(短序列)与 SIM(短+长序列双支路)。

用法:
  python -m src.run_rank --config configs/1k.yaml                     # 用配置里的 model
  python -m src.run_rank --config configs/1k.yaml --set rank.model=din   # 消融对照
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.rank_dataset import (RankDataset, RankDatasetSIM, RankDatasetV4,
                                   TagHistory, UserGroupBatchSampler, UserHistory)
from src.eval.metrics import auc, gauc, log_result
from src.models.din import DIN
from src.models.sim import SIM
from src.models.sim_v4 import SIMv4, pairwise_bpr_loss


def build_user_feats(cfg, n_users):
    """用户画像:类别列 → one-hot 拼接矩阵(过滤高基数列),按 uid 对齐。"""
    uf = pd.read_parquet(cfg.out / "user_features.parquet")
    mats = []
    for c in uf.columns:
        if c in ("user_id", "uid"):
            continue
        codes, uniq = pd.factorize(uf[c])
        if len(uniq) > 60:          # 跳过高基数加密特征
            continue
        oh = np.zeros((len(uf), len(uniq) + 1), dtype=np.float32)
        oh[np.arange(len(uf)), codes + 1] = 1.0   # 缺失(-1)落到第 0 列
        mats.append(oh)
    M = np.concatenate(mats, axis=1)
    out = np.zeros((n_users, M.shape[1]), dtype=np.float32)
    out[uf["uid"].to_numpy()] = M
    print(f"user profile feats: {out.shape[1]} dims from {len(mats)} columns")
    return torch.from_numpy(out)


@torch.no_grad()
def evaluate(model, ds, device, batch_size, workers):
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers)
    scores, labels, uids = [], [], []
    for batch in tqdm(dl, desc="eval", leave=False):
        *x, y = batch
        logit = model(*(t.to(device) for t in x))
        scores.append(logit.cpu().numpy())
        labels.append(y.numpy())
        uids.append(x[0].numpy())
    s, l, u = map(np.concatenate, [scores, labels, uids])
    return {"AUC": auc(l, s), "GAUC": gauc(l, s, u)}


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    rk = cfg.rank
    model_name = rk.get("model", "din")
    exp_name = model_name + rk.get("exp_suffix", "")
    meta = json.loads((cfg.out / "meta.json").read_text())

    # 语义表(冻结)
    sem = None
    if rk.get("use_sem", False):
        sem_np = np.load(cfg.out / "sem_emb.npy")
        sem = torch.from_numpy(sem_np).to(device)
        print(f"semantic table loaded: {tuple(sem.shape)} ({sem_np.nbytes / 1e6:.0f}MB)")

    # 历史点击流(严格 time < t 截取,含 val/test 期样本时刻前的点击,无泄漏)
    gsu_key = rk.get("gsu_key", "tag1")
    cols = ["uid", "iid_h", "vid", "tag1", "cat2", "time_ms", cfg.main_label]
    if model_name == "simv4":
        cols.append("author_h")     # v4 序列 token 需要作者
    inter = pd.read_parquet(cfg.out / "interactions.parquet", columns=cols)
    clicks = inter[inter[cfg.main_label] == 1].drop(columns=[cfg.main_label])
    del inter
    history = UserHistory(clicks)
    tag_history = TagHistory(clicks, gsu_key) if model_name in ("sim", "simv4") else None
    del clicks

    def load_split(name, sample=0):
        df = pd.read_parquet(cfg.out / f"rank_{name}.parquet")
        if sample and len(df) > sample:
            df = df.sample(sample, random_state=cfg.seed)
        if model_name == "simv4":
            return RankDatasetV4(df, history, tag_history,
                                 rk["hist_len"], rk["long_topk"], rk["label"], gsu_key)
        if model_name == "sim":
            return RankDatasetSIM(df, history, tag_history,
                                  rk["hist_len"], rk["long_topk"], rk["label"], gsu_key)
        return RankDataset(df, history, rk["hist_len"], rk["label"])

    ds_train = load_split("train")
    ds_val = load_split("val", rk["val_sample"])
    print(f"model={model_name}  device={device}  train={len(ds_train):,}  val={len(ds_val):,}")

    if model_name == "simv4":
        user_feats = build_user_feats(cfg, meta["n_users"]) if rk.get("use_user_feats", True) else None
        model = SIMv4(meta["n_users"], meta["n_iid_h"], meta["n_author_h"], meta["n_tag"],
                      rk["emb_dim"], rk["mlp_dims"], sem=sem,
                      user_feats=user_feats.to(device) if user_feats is not None else None).to(device)
    else:
        cls = {"din": DIN, "sim": SIM}[model_name]
        model = cls(meta["n_users"], meta["n_iid_h"], meta["n_author_h"], meta["n_tag"],
                    rk["emb_dim"], rk["mlp_dims"], sem=sem).to(device)
    # 加载 next-item 预训练的 item embedding(可选)
    init_emb = rk.get("init_item_emb", "")
    if init_emb:
        w = torch.load(cfg.out / init_emb, map_location="cpu")
        assert w.shape == model.item_emb.weight.shape, f"{w.shape} != emb table"
        model.item_emb.weight.data.copy_(w)
        print(f"loaded pretrained item_emb from {init_emb} {tuple(w.shape)}")
    opt = torch.optim.Adam(model.parameters(), lr=rk["lr"])

    # v4:用户分组 batch(组内同用户)以支持 pairwise 损失;其余模型常规随机 batch
    pw_weight = rk.get("pairwise_weight", 0.5) if model_name == "simv4" else 0.0
    G = rk.get("group_size", 8)
    if pw_weight > 0:
        sampler = UserGroupBatchSampler(ds_train.uid, G,
                                        rk["batch_size"] // G, cfg.seed)
        dl = DataLoader(ds_train, batch_sampler=sampler, num_workers=rk["num_workers"])
        print(f"user-grouped batches: {len(sampler)}/epoch  (G={G}, pairwise_w={pw_weight})")
    else:
        dl = DataLoader(ds_train, batch_size=rk["batch_size"], shuffle=True,
                        num_workers=rk["num_workers"], drop_last=True)

    best_gauc, best_state, patience = -1.0, None, 0
    for ep in range(rk["epochs"]):
        model.train()
        losses = []
        for batch in tqdm(dl, desc=f"epoch {ep}"):
            *x, y = batch
            y = y.to(device)
            logit = model(*(t.to(device) for t in x))
            loss = F.binary_cross_entropy_with_logits(logit, y)
            if pw_weight > 0:
                pw, _ = pairwise_bpr_loss(logit, y, G)
                loss = loss + pw_weight * pw
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {ep}  loss={np.mean(losses):.4f}")
        m = evaluate(model, ds_val, device, rk["batch_size"], rk["num_workers"])
        log_result(cfg.reports, f"{exp_name}/{cfg.dataset}/val/ep{ep}", cfg._config_path, m)
        if m["GAUC"] > best_gauc:
            best_gauc, patience = m["GAUC"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                print(f"early stop at epoch {ep} (best val GAUC={best_gauc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    m = evaluate(model, load_split("test"), device, rk["batch_size"], rk["num_workers"])
    log_result(cfg.reports, f"{exp_name}/{cfg.dataset}/test", cfg._config_path, m)
    torch.save(model.state_dict(), cfg.out / f"{exp_name}.pt")
    print("model saved ->", cfg.out / f"{exp_name}.pt")


if __name__ == "__main__":
    main(load_config())
