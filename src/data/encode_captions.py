"""caption 语义编码:bge-small-zh-v1.5 句向量 → PCA 降维 → 冻结语义表。

流程:
  1. 分块读取 kuairand_video_captions.csv,仅保留本数据集日志中出现过的视频
  2. GPU 批量编码 caption+封面文本(L2 归一,512 维)写入 fp16 memmap
  3. IncrementalPCA 降到 sem_dim(默认 64),保存 sem_emb.npy(按 vid 索引,行 0 为零向量)

用法: python -m src.data.encode_captions --config configs/1k.yaml
依赖: pip install sentence-transformers
国内下载模型: export HF_ENDPOINT=https://hf-mirror.com
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config

RAW_DIM = 512
MODEL = "BAAI/bge-small-zh-v1.5"


def main(cfg):
    import torch
    from sentence_transformers import SentenceTransformer

    sem_dim = cfg.rank.get("sem_dim", 64)
    vmap = pd.read_parquet(cfg.out / "video_map.parquet")
    vid_of = dict(zip(vmap["video_id"], vmap["vid"]))
    n_vid = len(vmap) + 1
    print(f"videos to cover: {n_vid - 1:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device)

    raw_path = cfg.out / "sem_raw_fp16.npy"
    emb = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16,
                                    shape=(n_vid, RAW_DIM))
    covered = 0
    cap_file = "data/raw/kuairand_video_captions.csv"

    for chunk in tqdm(pd.read_csv(cap_file, chunksize=200_000,
                                  usecols=["final_video_id", "caption", "show_cover_text"]),
                      desc="encode"):
        chunk["vid"] = chunk["final_video_id"].map(vid_of)
        chunk = chunk.dropna(subset=["vid"])
        if not len(chunk):
            continue
        texts = (chunk["caption"].fillna("") + " " + chunk["show_cover_text"].fillna("")).str.strip()
        keep = texts.str.len() > 0
        chunk, texts = chunk[keep], texts[keep]
        if not len(chunk):
            continue
        vecs = model.encode(texts.tolist(), batch_size=256, show_progress_bar=False,
                            normalize_embeddings=True)
        emb[chunk["vid"].astype(int).to_numpy()] = vecs.astype(np.float16)
        covered += len(chunk)
    emb.flush()
    print(f"encoded {covered:,} videos  coverage={covered / (n_vid - 1):.2%}")

    # ---- PCA 降维(在最多 100 万行非零样本上拟合) ----
    from sklearn.decomposition import IncrementalPCA
    rng = np.random.default_rng(cfg.seed)
    nz = np.flatnonzero(np.abs(emb[:, 0]) > 0)          # 已编码的行
    fit_rows = rng.choice(nz, min(1_000_000, len(nz)), replace=False)
    fit_rows.sort()
    ipca = IncrementalPCA(n_components=sem_dim, batch_size=8192)
    for s in tqdm(range(0, len(fit_rows), 100_000), desc="pca fit"):
        ipca.partial_fit(emb[fit_rows[s:s + 100_000]].astype(np.float32))
    print(f"PCA explained variance: {ipca.explained_variance_ratio_.sum():.3f}")

    out = np.zeros((n_vid, sem_dim), dtype=np.float16)
    for s in tqdm(range(0, n_vid, 200_000), desc="pca transform"):
        block = emb[s:s + 200_000].astype(np.float32)
        out[s:s + 200_000] = ipca.transform(block).astype(np.float16)
    out[0] = 0                                           # padding 行
    # 未编码的行置零
    mask = np.ones(n_vid, dtype=bool)
    mask[nz] = False
    out[mask] = 0
    np.save(cfg.out / "sem_emb.npy", out)
    print("semantic table ->", cfg.out / "sem_emb.npy", out.shape)


if __name__ == "__main__":
    main(load_config())
