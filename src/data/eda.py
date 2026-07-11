"""EDA:序列长度分布 / 各反馈信号正例率 / item 长尾分布。

输出 reports/eda_{dataset}.md 与配图。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config


def main(cfg):
    df = pd.read_parquet(cfg.out / "interactions.parquet")
    lines = [f"# EDA — KuaiRand-{cfg.dataset}\n"]
    lines.append(f"- 交互总数: {len(df):,}")
    lines.append(f"- 用户数: {df['uid'].nunique():,}")
    lines.append(f"- 候选 item 数(iid>0): {df.loc[df['iid'] > 0, 'iid'].nunique():,}")
    lines.append(f"- OOV 交互占比: {(df['iid'] == 0).mean():.2%}")
    lines.append(f"- 日期范围: {df['date'].min()} ~ {df['date'].max()}\n")

    # 各信号正例率
    lines.append("## 反馈信号正例率\n")
    lines.append("| 信号 | 正例率 |")
    lines.append("|---|---|")
    for sig in ["is_click", "long_view", "is_like", "is_follow"]:
        if sig in df.columns:
            lines.append(f"| {sig} | {df[sig].mean():.4f} |")

    # 用户序列长度分布
    seq_len = df.groupby("uid").size()
    lines.append("\n## 用户序列长度\n")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        lines.append(f"- P{int(q * 100)}: {int(seq_len.quantile(q)):,}")
    lines.append(f"- 均值: {seq_len.mean():,.0f}  最大: {seq_len.max():,}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(seq_len.clip(upper=seq_len.quantile(0.99)), bins=50)
    axes[0].set_title("user sequence length (clip P99)")
    item_cnt = df.loc[df["iid"] > 0, "iid"].value_counts().values
    axes[1].loglog(np.arange(1, len(item_cnt) + 1), item_cnt)
    axes[1].set_title("item popularity (log-log)")
    fig.tight_layout()
    fig_path = cfg.reports / f"eda_{cfg.dataset}.png"
    fig.savefig(fig_path, dpi=120)
    lines.append(f"\n![eda]({fig_path.name})")

    # 每日交互量(检查切分点两侧数据量)
    daily = df.groupby("date").size()
    lines.append("\n## 每日交互量\n")
    lines.append("| date | rows |")
    lines.append("|---|---|")
    for d, n in daily.items():
        mark = ""
        if d == cfg.train_end_date:
            mark = " ← train_end"
        elif d == cfg.val_end_date:
            mark = " ← val_end"
        lines.append(f"| {d} | {n:,}{mark} |")

    out_md = cfg.reports / f"eda_{cfg.dataset}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("EDA report ->", out_md)


if __name__ == "__main__":
    main(load_config())
