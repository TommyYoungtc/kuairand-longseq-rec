"""SIM v4:在 SIM(长短双支路 + 语义)基础上的三项升级。

1. 序列 token 表征加料:token = item_emb + author_emb(共享目标侧作者表)
   + 时间间隔桶 emb(远期行为如何衰减交给模型学习)+ 语义投影
2. 用户画像特征:user_features 的类别特征 one-hot 矩阵(冻结 buffer)
   经可训练 Linear 投影,拼入输出层
3. 训练目标(在 run_rank 中):BCE + 用户内 pairwise(BPR)辅助损失,
   与 GAUC(用户内排序)指标对齐
"""
import torch
import torch.nn as nn

from src.data.rank_dataset import N_GAP_BUCKETS
from src.models.din import SemMixin, TargetAttention


class SIMv4(SemMixin):
    def __init__(self, n_users, n_iid_h, n_author, n_tag,
                 emb_dim=64, mlp_dims=(256, 128, 64), att_hidden=64, sem=None,
                 user_feats=None):
        super().__init__()
        self.item_emb = nn.Embedding(n_iid_h, emb_dim, padding_idx=0)
        self.author_emb = nn.Embedding(n_author, emb_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(n_tag, emb_dim, padding_idx=0)
        self.gap_emb = nn.Embedding(N_GAP_BUCKETS, emb_dim, padding_idx=0)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        for e in [self.item_emb, self.author_emb, self.tag_emb, self.gap_emb, self.user_emb]:
            nn.init.normal_(e.weight, std=0.02)
        self.init_sem(sem, emb_dim)

        # 用户画像:冻结 one-hot 矩阵 + 可训练投影
        if user_feats is not None:
            self.register_buffer("user_feats", user_feats, persistent=False)
            self.user_feat_proj = nn.Linear(user_feats.shape[1], emb_dim)
            extra = emb_dim
        else:
            self.user_feats = None
            extra = 0

        self.att_short = TargetAttention(emb_dim, att_hidden)
        self.att_long = TargetAttention(emb_dim, att_hidden)
        dims = [emb_dim * 6 + extra] + list(mlp_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-1], 1)]
        self.out_mlp = nn.Sequential(*layers)

    def _token(self, iid, vid, author, gap):
        return self.add_sem(
            self.item_emb(iid) + self.author_emb(author) + self.gap_emb(gap), vid)

    def forward(self, uid, iid_h, vid, author, tag,
                hist, hist_vid, hist_auth, hist_gap, hist_len,
                long_hist, long_vid, long_auth, long_gap, long_len):
        target = self.add_sem(
            self.item_emb(iid_h) + self.author_emb(author) + self.tag_emb(tag), vid)
        short_emb = self._token(hist, hist_vid, hist_auth, hist_gap)
        long_emb = self._token(long_hist, long_vid, long_auth, long_gap)
        short_int = self.att_short(short_emb, target, hist > 0)
        long_int = self.att_long(long_emb, target, long_hist > 0)
        feats = [target, short_int, long_int,
                 target * short_int, target * long_int, self.user_emb(uid)]
        if self.user_feats is not None:
            feats.append(self.user_feat_proj(self.user_feats[uid].float()))
        return self.out_mlp(torch.cat(feats, dim=-1)).squeeze(-1)


def pairwise_bpr_loss(logits, labels, group_size):
    """用户内 BPR:batch 由 M 个同用户组(大小 G)构成,组内 正例>负例。

    logits/labels: (B,) 且 B = M*G → 组内两两差,mask 出 (正,负) 对。
    返回 (loss, 有效对数量);无有效对时 loss 为 0。
    """
    G = group_size
    s = logits.view(-1, G)
    y = labels.view(-1, G)
    diff = s.unsqueeze(2) - s.unsqueeze(1)                 # (M, G, G): s_i - s_j
    mask = (y.unsqueeze(2) > 0.5) & (y.unsqueeze(1) < 0.5) # i 为正、j 为负
    if not mask.any():
        return logits.new_zeros(()), 0
    return -torch.nn.functional.logsigmoid(diff[mask]).mean(), int(mask.sum())
