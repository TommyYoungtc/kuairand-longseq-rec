"""MMoE 多任务排序模型:SIM 特征底座 + 多专家软路由 + 任务塔。

结构:
  共享底座 = SIM 的表征层(target/短序列/长序列双 attention + 语义表)→ 特征向量 x
  experts  = N 个共享专家 MLP
  gate_t   = 每任务一个 softmax 门控,对专家输出加权求和
  tower_t  = 每任务独立塔 → logit
MMoE 即软性 MoE;放大用户异质性场景时,把 gate 换成 Top-k 稀疏路由即是 sparse MoE。
"""
import torch
import torch.nn as nn

from src.models.din import SemMixin, TargetAttention


def mlp(dims, out_dim=None):
    layers = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    if out_dim is not None:
        layers += [nn.Linear(dims[-1], out_dim)]
    return nn.Sequential(*layers)


class MMoE(SemMixin):
    def __init__(self, n_users, n_iid_h, n_author, n_tag,
                 emb_dim=64, mlp_dims=(256, 128, 64), att_hidden=64, sem=None,
                 n_tasks=3, n_experts=4, expert_dim=64, tower_hidden=64):
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
        in_dim = emb_dim * 6

        self.experts = nn.ModuleList(
            [mlp([in_dim] + list(mlp_dims), expert_dim) for _ in range(n_experts)])
        self.gates = nn.ModuleList(
            [nn.Linear(in_dim, n_experts) for _ in range(n_tasks)])
        self.towers = nn.ModuleList(
            [mlp([expert_dim, tower_hidden], 1) for _ in range(n_tasks)])
        self.n_tasks = n_tasks

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

        expert_out = torch.stack([e(x) for e in self.experts], dim=1)   # (B, E, D)
        logits = []
        for gate, tower in zip(self.gates, self.towers):
            w = torch.softmax(gate(x), dim=-1)                          # (B, E)
            mixed = (w.unsqueeze(-1) * expert_out).sum(1)               # (B, D)
            logits.append(tower(mixed).squeeze(-1))
        return torch.stack(logits, dim=-1)                              # (B, T)
