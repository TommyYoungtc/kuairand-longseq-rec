"""DIN 风格排序模型:target attention 聚合用户历史(可选语义表增强)。

语义增强:冻结的 caption 语义表(vid → sem_dim)经可训练投影后,
加进 target 与历史 token 的表征——长尾/哈希冲突 item 由此获得
不依赖点击量的内容表征(世界知识注入点)。
"""
import torch
import torch.nn as nn


class TargetAttention(nn.Module):
    def __init__(self, emb_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hist_emb, target, mask):
        t = target.unsqueeze(1).expand_as(hist_emb)
        scores = self.mlp(torch.cat([hist_emb, t, hist_emb * t, hist_emb - t], -1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        w = torch.softmax(scores, dim=-1)
        w = w * mask.any(dim=-1, keepdim=True)
        return (w.unsqueeze(-1) * hist_emb).sum(1)


class SemMixin(nn.Module):
    """语义表挂载:sem 为 (n_vid, sem_dim) 冻结 fp16 tensor,可为 None。"""

    def init_sem(self, sem, emb_dim):
        if sem is not None:
            self.register_buffer("sem_table", sem, persistent=False)
            self.sem_proj = nn.Linear(sem.shape[1], emb_dim)
        else:
            self.sem_table = None

    def add_sem(self, base_emb, vid):
        """base_emb: (..., D)  vid: (...) → 表征加上投影后的语义向量。"""
        if self.sem_table is None:
            return base_emb
        return base_emb + self.sem_proj(self.sem_table[vid].float())


class DIN(SemMixin):
    def __init__(self, n_users, n_iid_h, n_author, n_tag,
                 emb_dim=64, mlp_dims=(256, 128, 64), att_hidden=64, sem=None):
        super().__init__()
        self.item_emb = nn.Embedding(n_iid_h, emb_dim, padding_idx=0)
        self.author_emb = nn.Embedding(n_author, emb_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(n_tag, emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        for e in [self.item_emb, self.author_emb, self.tag_emb, self.user_emb]:
            nn.init.normal_(e.weight, std=0.02)
        self.init_sem(sem, emb_dim)

        self.att = TargetAttention(emb_dim, att_hidden)
        dims = [emb_dim * 4] + list(mlp_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-1], 1)]
        self.out_mlp = nn.Sequential(*layers)

    def forward(self, uid, iid_h, vid, author, tag, hist, hist_vid, hist_len):
        target = self.add_sem(
            self.item_emb(iid_h) + self.author_emb(author) + self.tag_emb(tag), vid)
        hist_emb = self.add_sem(self.item_emb(hist), hist_vid)
        user_int = self.att(hist_emb, target, hist > 0)
        x = torch.cat([target, user_int, target * user_int, self.user_emb(uid)], dim=-1)
        return self.out_mlp(x).squeeze(-1)
