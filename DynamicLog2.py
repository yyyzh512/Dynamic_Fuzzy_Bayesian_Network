###############################################################################
# dynamic_lr_predict.py
#   • 行级 one-hot（不再 pivot）  → 保留原始 1975 行
#   • 输出 obj_probs.csv 供后续动态贝叶斯累乘
###############################################################################
import random, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# ----------------------- 全局常量 -----------------------
SEQ_CSV = "sequence_new_aug2.csv"
# HT_CSV  = "FG_Ht.csv"                 # 或 FG_Ht.csv
HT_CSV  = "FG_Ht.csv"
OBJECTS = ["Kettle","cup","medicine","milk","coffee",
           "yoghurt","apple","banana","bowl","Alarm"]
USE_H, USE_PREV, USE_CUM = True, True, True


SEED = 2025
random.seed(SEED); np.random.seed(SEED)
rng = np.random.default_rng(SEED)
# -------------------------------------------------------

### 1) 读取并对齐序列 + Ht
seq = pd.read_csv(SEQ_CSV)
ht  = pd.read_csv(HT_CSV)
df  = (seq.merge(ht, on=["sample_id", "timestep"])
          .sort_values(["sample_id", "timestep"]))

### 2) 行级 one-hot ：为每个 OBJECTS 新增 0/1 列
for o in OBJECTS:
    df[o] = (df["object_name"] == o).astype(int)
# （可选）删除原 object_name 列
df = df.drop(columns=["object_name"])

### 3) 历史 prev_* / cum_*
def add_hist(d, objs):
    d = d.sort_values(["sample_id", "timestep"]).copy()
    for o in objs:
        d[f"prev_{o}"] = d.groupby("sample_id")[o].shift().fillna(0)
        d[f"cum_{o}"]  = d.groupby("sample_id")[o].cumsum().shift(fill_value=0)
    return d
df = add_hist(df, OBJECTS)

### 4) Intention one-hot
enc   = OneHotEncoder(sparse_output=False)
I_hot = enc.fit_transform(df[["label"]])
I_df  = pd.DataFrame(I_hot, columns=enc.get_feature_names_out(["label"]))
df    = pd.concat([df.reset_index(drop=True), I_df], axis=1)

### 5) 特征列组装
h_cols    = [c for c in df.columns if c.startswith("h")]     if USE_H    else []
prev_cols = [c for c in df.columns if c.startswith("prev_")] if USE_PREV else []
cum_cols  = [c for c in df.columns if c.startswith("cum_")]  if USE_CUM  else []
intent_cols = list(I_df.columns)

base_feats = h_cols + prev_cols + cum_cols + intent_cols
X_full     = df[base_feats].values
scaler     = StandardScaler().fit(X_full)
X_scaled   = scaler.transform(X_full)

y_table = {obj: df[obj].values for obj in OBJECTS}

### 6) Train / Test 划分（按 sample_id，80/20）
ids = df["sample_id"].unique()
rng.shuffle(ids)
train_ids = set(ids[: int(0.8 * len(ids))])
is_train  = df["sample_id"].isin(train_ids).values

### 7) 训练 10 个 LR & 评估
models={}
auc_tr={}
auc_te={}
acc_tr={}
acc_te={}

for obj in OBJECTS:
    keep_cols = [c for c in base_feats if c != obj]   # 去泄漏
    keep_idx  = [base_feats.index(c) for c in keep_cols]
    X_obj     = X_scaled[:, keep_idx]
    y         = y_table[obj]

    lr = LogisticRegression(max_iter=500, penalty="l2",
                            class_weight="balanced",
                            random_state=SEED)
    lr.fit(X_obj[is_train], y[is_train])
    models[obj] = (lr, keep_idx)

    p_tr = lr.predict_proba(X_obj[is_train])[:, 1]
    p_te = lr.predict_proba(X_obj[~is_train])[:, 1]

    auc_tr[obj] = roc_auc_score(y[is_train], p_tr)
    auc_te[obj] = roc_auc_score(y[~is_train], p_te)
    acc_tr[obj] = accuracy_score(y[is_train], (p_tr >= 0.5))
    acc_te[obj] = accuracy_score(y[~is_train], (p_te >= 0.5))

print("\n[RESULT]  AUC  (Train | Test)   &   Accuracy (Train | Test)")
for o in OBJECTS:
    print(f"{o:<9}: {auc_tr[o]:.3f} | {auc_te[o]:.3f}    "
          f"{acc_tr[o]:.3f} | {acc_te[o]:.3f}")

### 8) 打印 & 保存 LR 参数（Top-N）
param_rows, TOP_N = [], 10
print("\n[PARAMETERS] θ_j (intercept + top-N weights)")
for obj in OBJECTS:
    lr, keep_idx = models[obj]
    coef, intercept = lr.coef_[0], lr.intercept_[0]
    feat_names = [base_feats[i] for i in keep_idx]
    abs_rank   = np.argsort(-np.abs(coef))[:TOP_N]

    print(f"\n▶ {obj}\n  intercept = {intercept:+.4f}")
    for r in abs_rank:
        print(f"  {feat_names[r]:<18s} {coef[r]:+7.4f}")

    row = {"object": obj, "intercept": intercept}
    row.update({feat_names[i]: coef[i] for i in range(len(coef))})
    param_rows.append(row)

pd.DataFrame(param_rows).to_csv("lr_params.csv", index=False)
print("\n[INFO] 全参数已保存 → lr_params.csv")

### 9) 逐行预测示例
def predict_probs_rows(k=10, normalize=True):
    rows = df.head(k).copy()
    P = pd.DataFrame(index=rows.index)
    for obj in OBJECTS:
        lr, idx = models[obj]
        X = scaler.transform(rows[base_feats].values)[:, idx]
        P[obj] = lr.predict_proba(X)[:, 1]
    if normalize:
        P = P.div(P.sum(axis=1), axis=0)
    return P

print("\n前 10 行预测概率（行和=1）：")
print(predict_probs_rows(10, True).round(3))

### 10) 保存所有帧的归一化概率表
prob_df = pd.DataFrame({
    "sample_id": df["sample_id"],
    "timestep" : df["timestep"]
})

# ① 收集未归一化概率
for obj in OBJECTS:
    lr, idx   = models[obj]
    prob_df[obj] = lr.predict_proba(X_scaled[:, idx])[:, 1]

# ② 行归一化：∑_obj p(obj | row) = 1
row_sum = prob_df[OBJECTS].sum(axis=1)
# 若出现极端情况（行和为 0），可加一个极小值避免除零
row_sum = row_sum.replace(0, np.finfo(float).eps)
prob_df[OBJECTS] = prob_df[OBJECTS].div(row_sum, axis=0)

prob_df.to_csv("FG_Ht_obj_probs.csv", index=False)
print("[INFO] 归一化概率表已保存 → obj_probs.csv")

### 11) 可视化 Test AUC
plt.figure(figsize=(9,4))
plt.bar(range(len(OBJECTS)), [auc_te[o] for o in OBJECTS])
plt.xticks(range(len(OBJECTS)), OBJECTS, rotation=45)
plt.ylim(0.5, 1)
plt.ylabel("Test AUC")
plt.title("Dynamic LR – Test AUC per Object")
plt.tight_layout()
plt.show()
