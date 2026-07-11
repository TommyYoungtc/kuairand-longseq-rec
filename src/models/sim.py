"""SIM 式长序列排序模型:短期序列 + 不相交 GSU 长期检索,双支路 target attention。

语义增强(可选):长支路检索出的远期行为多为长尾/哈希 ID,
冻结语义表为它们提供内容表征,是长支路信息质量的关键。
"""
import torch
import torch.nn as nn

from src.models.din import SemMixin, TargetAttention


class SIM(SemMixin):
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

        self.att_short = TargetAttention(emb_dim, att_hidden)
        self.att_long = TargetAttention(emb_dim, att_hidden)
        dims = [emb_dim * 6] + list(mlp_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-1], 1)]
        self.out_mlp = nn.Sequential(*layers)

    def forward(self, uid, iid_h, vid, author, tag, hist, hist_vid, hist_len,
                long_hist, long_vid, long_len):
        target = self.add_sem(
            self.item_emb(iid_h) + self.author_emb(author) + self.tag_emb(tag), vid)
        short_emb = self.add_sem(self.item_emb(hist), hist_vid)
        long_emb = self.add_sem(self.item_emb(long_hist), long_vid)
        short_int = self.att_short(short_emb, target, hist > 0)
        long_int = self.att_long(long_emb, target, long_hist > 0)
        x = torch.cat([target, short_int, long_int,
                       target * short_int, target * long_int,
                       self.user_emb(uid)], dim=-1)
        return self.out_mlp(x).squeeze(-1)
