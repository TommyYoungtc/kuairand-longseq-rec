"""自回归 next-item 预训练:Transformer 编码点击序列,预测下一个 item。

生成式/foundation-model 范式的落点:在全量点击流(iid_h 空间,含长尾哈希桶)上
自监督预训练 item embedding,再交给排序模型微调(run_rank --set rank.init_item_emb=...)。
训练目标 = in-batch softmax + logQ 修正(与双塔一致)。

用法:
  python -m src.run_pretrain --config configs/1k.yaml
  python -m src.run_rank --config configs/1k.yaml --set rank.init_item_emb=pretrained_item_emb.pt rank.exp_suffix=-sem-pt
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import load_config


class NextItemDataset(Dataset):
    """每个点击事件为一条样本:此前最近 L 次点击(iid_h)→ 预测当前 iid_h。"""

    def __init__(self, clicks: pd.DataFrame, seq_len: int, min_hist: int = 5):
        uid = clicks["uid"].to_numpy()
        iid = clicks["iid_h"].to_numpy()
        starts = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
        seq_pos = np.arange(len(uid)) - np.repeat(starts, np.diff(np.r_[starts, len(uid)]))
        keep = seq_pos >= min_hist
        self.pos = seq_pos[keep]
        self.event_idx = np.flatnonzero(keep)
        self.all_iid = iid
        self.L = seq_len

    def __len__(self):
        return len(self.event_idx)

    def __getitem__(self, k):
        e = self.event_idx[k]
        n = min(self.pos[k], self.L)
        hist = np.zeros(self.L, dtype=np.int64)
        hist[:n] = self.all_iid[e - n:e]
        return torch.from_numpy(hist), torch.tensor(self.all_iid[e])


class NextItemTransformer(nn.Module):
    def __init__(self, n_items, emb_dim=64, n_layers=2, n_heads=2, seq_len=50,
                 temperature=0.05):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_len, emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(emb_dim, n_heads, emb_dim * 4,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.temperature = temperature

    def forward(self, hist):
        """hist: (B, L),padding 在尾部 → 取最后一个非 padding 位置的状态。"""
        B, L = hist.shape
        x = self.item_emb(hist) + self.pos_emb.weight[:L]
        mask = hist == 0
        h = self.encoder(x, src_key_padding_mask=mask)          # (B, L, D)
        last = (hist > 0).sum(1).clamp(min=1) - 1               # (B,)
        state = h[torch.arange(B, device=hist.device), last]    # (B, D)
        return F.normalize(state, dim=-1)

    def loss(self, state, target, logq):
        v = F.normalize(self.item_emb(target), dim=-1)
        logits = state @ v.T / self.temperature - logq.unsqueeze(0)
        labels = torch.arange(len(state), device=state.device)
        return F.cross_entropy(logits, labels)


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    pt = cfg.get("pretrain", {})
    seq_len = pt.get("seq_len", 50)
    emb_dim = cfg.rank["emb_dim"]           # 必须与排序模型一致才能加载
    meta = json.loads((cfg.out / "meta.json").read_text())

    clicks = pd.read_parquet(cfg.out / "train_clicks.parquet",
                             columns=["uid", "iid_h", "time_ms"])
    clicks = clicks.sort_values(["uid", "time_ms"], kind="stable")
    ds = NextItemDataset(clicks, seq_len)
    dl = DataLoader(ds, batch_size=pt.get("batch_size", 512), shuffle=True,
                    num_workers=pt.get("num_workers", 4), drop_last=True)
    print(f"pretrain events={len(ds):,}  vocab={meta['n_iid_h']:,}  device={device}")

    cnt = clicks["iid_h"].value_counts().reindex(range(meta["n_iid_h"]), fill_value=0).to_numpy()
    logq = torch.log(torch.tensor((cnt + 1e-12) / cnt.sum(), dtype=torch.float32)).to(device)

    model = NextItemTransformer(meta["n_iid_h"], emb_dim, seq_len=seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=pt.get("lr", 0.001))

    for ep in range(pt.get("epochs", 2)):
        model.train()
        losses = []
        for hist, target in tqdm(dl, desc=f"pretrain ep{ep}"):
            hist, target = hist.to(device), target.to(device)
            loss = model.loss(model(hist), target, logq[target])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch {ep}  loss={np.mean(losses):.4f}")

    out = cfg.out / "pretrained_item_emb.pt"
    torch.save(model.item_emb.weight.detach().cpu(), out)
    print("pretrained item embedding ->", out)


if __name__ == "__main__":
    main(load_config())
