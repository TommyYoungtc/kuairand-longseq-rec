# 长序列大模型化推荐排序系统 (KuaiRand)

面向短视频场景的完整两级推荐系统:双塔召回 + SIM 式长序列精排 + caption 语义增强 + MMoE 多任务。
数据集:KuaiRand-1K(快手真实日志,1000 用户 / 437 万视频 / 1171 万曝光,人均 8300+ 条行为)。

**核心结果**(1K test,GAUC):

| 模型 | GAUC | 增量来源 |
|---|---|---|
| DIN(短序列 200,纯 ID) | 0.5769 | baseline |
| + caption 语义表(bge+PCA,冻结) | 0.5943 | **+1.7pt**,冷启动 item 上 +1.6pt AUC 且随冷度单调 |
| + 细粒度长序列检索(GSU@二级类目) | 0.5976 | +0.3pt,序列长度暴力加长在 200 饱和后仍有增量 |
| + 用户内 pairwise 损失(v4,目标-指标对齐) | **0.6054** | **+0.8pt**,消融确认几乎全部来自 BPR 辅助损失 |

亮点:随机曝光日志上的**无偏评估**确认语义增益非曝光偏差产物;MMoE 三任务无损扩展(long_view GAUC 0.615);
长序列模块经历"重叠检索→不相交检索→细粒度键"三次有据可查的问题定位迭代;全实验流水在 `reports/results.csv`。

📄 **[完整实验报告](reports/项目实验报告.md)** · 🔍 **[Case Study(语义近邻/救回案例)](reports/case_study.md)** · 📋 **[产品决策文档(指标体系/权衡/AB方案)](docs/产品决策文档.md)**

交互 Demo(选用户 → 全候选打分 → Top-10 推荐 + 中文标题 + 推荐理由):

```bash
pip install streamlit
python -m src.data.build_caption_cache --config configs/1k.yaml   # 一次性
streamlit run demo/app.py
```

## 环境

建议 WSL2 + conda/venv,Python ≥ 3.10。

```bash
pip install -r requirements.txt
# GPU 版 torch 请按 https://pytorch.org 选择对应 CUDA 版本安装
```

## 数据下载

```bash
bash scripts/download_data.sh pure   # KuaiRand-Pure, 194MB, 调试用
bash scripts/download_data.sh 1k    # KuaiRand-1K, 4.3GB, 正式实验
bash scripts/download_data.sh meta  # caption/类目补充文件(第3周语义增强用)
```

## 第 1 周流程

```bash
# 1. 预处理:合并日志、候选集裁剪、重编号、转 parquet
python -m src.data.preprocess --config configs/pure.yaml

# 2. EDA 报告(序列长度/正例率/长尾分布)→ reports/
python -m src.data.eda --config configs/pure.yaml

# 3. 构建样本:时间切分、召回 ground truth、排序样本
python -m src.data.build_samples --config configs/pure.yaml

# 4. 基线:热门 + ItemCF,输出 Recall@K / NDCG@K
python -m src.run_baselines --config configs/pure.yaml

# 5. 双塔召回:训练 + 全量检索评估(需要 torch;有 GPU 自动使用)
python tests/test_recall_dataset.py        # 先跑单元测试
python -m src.run_two_tower --config configs/pure.yaml
```

双塔冒烟:可先用合成数据快速验证 torch 链路
`python tests/gen_synth_data.py && python -m src.data.preprocess --config configs/synth.yaml && python -m src.data.build_samples --config configs/synth.yaml && python -m src.run_two_tower --config configs/synth.yaml`

## 第 2 周:排序模型(DIN)

预处理 schema 已升级(长尾 item/作者 hash 分桶,应对 1K 版 85% OOV),需重跑 1-3 步后再训练排序:

```bash
python -m src.data.preprocess --config configs/1k.yaml
python -m src.data.build_samples --config configs/1k.yaml
python tests/test_rank_dataset.py     # 单元测试(历史截取无泄漏)
python -m src.run_rank --config configs/1k.yaml   # 排序模型训练+AUC/GAUC
```

## 核心模块:SIM 长序列两阶段(GSU→ESU)

- GSU(硬检索):按 target 类目从用户全生命周期点击(不限窗口)中检索最近 K 条相关行为
- ESU:短期序列(hist_len)与长期检索序列(long_topk)双支路 target attention 融合

```bash
python tests/test_sim.py                            # GSU 检索正确性
python -m src.run_rank --config configs/1k.yaml     # 默认 model=sim
python -m src.run_rank --config configs/1k.yaml --set rank.model=din   # 消融对照
python -m src.run_rank --config configs/1k.yaml --set rank.hist_len=50 # 序列长度消融
```

实验结果统一追加到 `reports/results.csv`。

## 第 3 周:caption 语义增强

冻结的中文句向量表(bge-small-zh + PCA 64 维)注入 target 与历史序列表征,
为 85% 长尾/哈希 item 提供不依赖点击量的内容表征。

```bash
pip install sentence-transformers
export HF_ENDPOINT=https://hf-mirror.com    # 国内下载模型权重

# schema 又升级了(vid 索引),先重跑 1-2 步
python -m src.data.preprocess --config configs/1k.yaml
python -m src.data.build_samples --config configs/1k.yaml

# caption 编码(约 30-60 分钟,GPU)
python -m src.data.encode_captions --config configs/1k.yaml

# 主实验 + 消融
python -m src.run_rank --config configs/1k.yaml --set rank.exp_suffix=-sem          # SIM+语义
python -m src.run_rank --config configs/1k.yaml --set rank.use_sem=false rank.exp_suffix=-nosem  # 消融
python -m src.run_rank --config configs/1k.yaml --set rank.model=din rank.exp_suffix=-sem       # DIN+语义
```

先用 `pure.yaml` 把全流程跑通(几分钟),再切 `1k.yaml` 跑正式版。

## 目录结构

```
configs/          # 实验配置(pure=调试, 1k=正式)
scripts/          # 数据下载
src/data/         # 预处理 / EDA / 样本构建
src/baselines/    # 热门、ItemCF
src/eval/         # Recall/NDCG/AUC/GAUC
data/raw/         # 原始数据(git 忽略)
data/processed/   # parquet 中间产物(git 忽略)
reports/          # EDA 报告与实验结果
```

## 实验记录约定

每次实验结果追加写入 `reports/results.csv`(列:date, exp_name, config, metric, value),方便汇总对比。
