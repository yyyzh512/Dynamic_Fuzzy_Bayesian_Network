import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

F_IN  = Path("raw_duration_102.csv")                 # 原始 7 列文件
F_OUT = Path("raw_duration_102_with_zck.csv")         # 输出文件

# 1) 读取与基础清洗
df = pd.read_csv(F_IN)
df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
df["duration"] = df.groupby("participant_id")["duration"]\
                   .transform(lambda x: x.fillna(x.median()))
df["duration"].fillna(df["duration"].median(), inplace=True)

# 2) 个体层 —— p、z
df = df.rename(columns={"persentage_in_seq": "p"})           # 序列占比
def zscore(x):                                               # participant 内 Z
    std = x.std(ddof=0)
    return (x - x.mean()) / std if std else 0
df["z"] = df.groupby("participant_id")["duration"].apply(zscore)\
            .reset_index(level=0, drop=True)

# 3) 群体层 —— z_obj、r_obj
def z_obj(x):                                                # object 内 Z
    std = x.std(ddof=0)
    return (x - x.mean()) / std if std else 0
df["z_obj"] = df.groupby("object_id")["duration"].apply(z_obj)\
                .reset_index(level=0, drop=True)

def pct_rank(x):                                             # object 内百分位
    return rankdata(x, method="average") / len(x)
df["r_obj"] = df.groupby("object_id")["duration"].transform(pct_rank)

# 4) **归一化到 0–1**  (p 已在 0‑1，r_obj 亦在 0‑1)
for col in ["z", "z_obj"]:
    col_min, col_max = df[col].min(), df[col].max()
    df[col + "_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-12)

df = df.rename(columns={"p": "fsp", "r_obj": "fopr","z_norm":"fid","z_obj_norm":"fgd"})   # 统一带 _norm

# 5) 输出仅保留需要聚类的 4 维特征 + 关键标识列
cols_keep = ["position_id", "sample_id", "participant_id",
             "object_name", "object_id", "duration",
             "fsp", "fid", "fgd", "fopr"]
df[cols_keep].to_csv(F_OUT, index=False, encoding="utf-8-sig")
print(f"[Done] 归一化完毕并写入 {F_OUT}")
