"""双塔召回模型。

用户塔: 近期点击序列 embedding mean-pooling ⊕ 用户 ID emb → MLP → L2 归一
物品塔: item ID emb ⊕ 作者 ID emb → MLP → L2 归一
训练:  in-batch softmax(带温度),等价 sampled softmax
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_authors: int,
                 emb_dim: int = 64, out_dim: int = 64, temperature: float = 0.05):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.author_emb = nn.Embedding(n_authors + 1, emb_dim, padding_idx=0)
        self.user_mlp = mlp([emb_dim * 2, emb_dim * 2, out_dim])
        self.item_mlp = mlp([emb_dim * 2, emb_dim * 2, out_dim])
        self.temperature = temperature
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.author_emb.weight, std=0.02)

    def user_tower(self, uid, hist, hist_len):
        """uid: (B,)  hist: (B, H) 近期点击 iid,0 为 padding  hist_len: (B,)"""
        h = self.item_emb(hist)                              # (B, H, D)
        mask = (hist > 0).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / hist_len.clamp(min=1).unsqueeze(-1)
        u = torch.cat([pooled, self.user_emb(uid)], dim=-1)
        return F.normalize(self.user_mlp(u), dim=-1)

    def item_tower(self, iid, author):
        v = torch.cat([self.item_emb(iid), self.author_emb(author)], dim=-1)
        return F.normalize(self.item_mlp(v), dim=-1)

    def in_batch_loss(self, u_vec, i_vec, logq=None):
        """in-batch softmax:对角线为正例,batch 内其余 item 为负例。

        logq: (B,) batch 内各 item 的采样概率对数。in-batch 负采样下热门 item
        更常被当作负例而被系统性打压,logQ 修正(logits - logq)消除该偏差
        (Sampled Softmax bias correction,参见 YouTube 双塔召回)。"""
        logits = u_vec @ i_vec.T / self.temperature        # (B, B)
        if logq is not None:
            logits = logits - logq.unsqueeze(0)            # 按列(负例 item)修正
        labels = torch.arange(len(u_vec), device=u_vec.device)
        return F.cross_entropy(logits, labels)
