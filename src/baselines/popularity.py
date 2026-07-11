"""热门基线:按训练期点击量排序,推荐时排除用户已点击 item。"""
import numpy as np
import pandas as pd


def recommend(train_clicks: pd.DataFrame, users: list, k_max: int) -> dict:
    pop = train_clicks["iid"].value_counts()
    top_global = pop.index.to_numpy()  # 按热度降序
    seen = train_clicks.groupby("uid")["iid"].agg(set).to_dict()
    recs = {}
    for uid in users:
        s = seen.get(uid, ())
        pool = top_global[: k_max + len(s)]
        recs[uid] = pool[~np.isin(pool, list(s))][:k_max] if s else pool[:k_max]
    return recs
