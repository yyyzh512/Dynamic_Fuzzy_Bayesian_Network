"""
dyn_bayes_soft_cv.py (fixed + F1 + confusion matrix)
----------------------------------------------------
动态贝叶斯意图推理（无转移矩阵）。
  • Φ 由真实序列统计 (10×10)
  • LR 概率表作为软观测
  • 10-Fold CV 打印先验、Φ、折间准确率与 F1-score (mean ± std)
  • 额外输出整体混淆矩阵 (Times New Roman, 9 pt；蓝色底；7 cm × 5 cm)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------- (新增) 绘图依赖 ----------
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- 预设参数 ----------
OBJ_COLS = [
    "Kettle", "cup", "medicine", "milk", "coffee",
    "yoghurt", "apple", "banana", "bowl", "Alarm"
]
K = len(OBJ_COLS)

# ---------- 0) 构建 Φ ----------
def build_phi(sequence_csv: str) -> np.ndarray:
    seq = pd.read_csv(sequence_csv)
    obj2id = {name: i for i, name in enumerate(OBJ_COLS)}
    seq["obj_id"] = seq["object_name"].map(obj2id)

    phi = np.ones((K, K))
    for i in range(1, K + 1):
        obj_ids = seq.loc[seq["label"] == i, "obj_id"].astype(int)
        counts = np.bincount(obj_ids, minlength=K)
        phi[i - 1] += counts
    phi /= phi.sum(axis=1, keepdims=True)

    print("\n[Likelihood Φ = P(O=o | I=i)]")
    print(pd.DataFrame(phi,
                       index=[f"I{i}" for i in range(1, K + 1)],
                       columns=[f"O{o}" for o in range(1, K + 1)]).round(4))
    return phi

# ---------- 1) 推理器 ----------
class DynBayesSoft:
    def __init__(self, phi: np.ndarray, prior: np.ndarray):
        self.phi = phi
        self.log_phi = np.log(phi + 1e-12)
        self.prior = prior / prior.sum()
        self.log_prior = np.log(self.prior + 1e-12)
        self.classes = np.arange(1, K + 1)

    def predict_sequence(self, lr_seq_df: pd.DataFrame) -> int:
        log_post = self.log_prior.copy()
        for row in lr_seq_df[OBJ_COLS].to_numpy():
            log_post += row @ self.log_phi.T
        return int(self.classes[log_post.argmax()])

# ---------- 2) CV ----------
def cross_validate(obj_prob_csv: str, sequence_csv: str, n_splits: int = 10):
    df_prob = pd.read_csv(obj_prob_csv)
    phi = build_phi(sequence_csv)

    seq_labels = df_prob.groupby("sample_id")["label"].first()
    prior_full = seq_labels.value_counts().reindex(range(1, K + 1)).fillna(0).values + 1
    print("\n[Prior] P(I=i)")
    print(pd.Series(prior_full / prior_full.sum(), index=[f"I{i}" for i in range(1, K + 1)]).round(4))

    ids = seq_labels.index.to_numpy()
    y_all = seq_labels.values

    skf = StratifiedKFold(n_splits, shuffle=True, random_state=2025)
    accs, f1s = [], []

    # ---------- (新增) 聚合混淆矩阵 ----------
    conf_total = np.zeros((K, K), dtype=int)

    for fold, (tr, te) in enumerate(skf.split(ids, y_all), 1):
        train_ids = ids[tr]
        test_ids = ids[te]

        prior_tr = seq_labels.loc[list(train_ids)].value_counts().reindex(range(1, K + 1)).fillna(0).values + 1
        model = DynBayesSoft(phi, prior_tr)

        y_true, y_pred = [], []
        for sid in test_ids:
            seq_rows = df_prob[df_prob["sample_id"] == sid]
            pred = model.predict_sequence(seq_rows)
            y_pred.append(pred)
            y_true.append(seq_labels.loc[sid])

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        accs.append(acc)
        f1s.append(f1)

        # ---------- (新增) 更新混淆矩阵 ----------
        conf_total += confusion_matrix(y_true, y_pred, labels=range(1, K + 1))

        print(f"Fold {fold:2d}: ACC = {acc:.4f}, F1 = {f1:.4f}")

    accs = np.array(accs)
    f1s = np.array(f1s)
    print("\n10-Fold ACC : {:.4f} ± {:.4f}".format(accs.mean(), accs.std()))
    print("10-Fold F1  : {:.4f} ± {:.4f}".format(f1s.mean(), f1s.std()))

    # ---------- (新增) 绘制混淆矩阵 ----------
    row_pct = conf_total / conf_total.sum(axis=1, keepdims=True) * 100

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 6
    labels = [f'I{i}' for i in range(1, K+1)]

    plt.figure(figsize=(3.15, 2.5))
    sns.heatmap(row_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, alpha=0.7)
    plt.title('Confusion Matrix(%)')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.savefig('confmat.png', dpi=1200)

# ---------- CLI ----------
if __name__ == "__main__":
    cross_validate(
        r"FG_Ht_obj_probs.csv", #  Original sequence
        r"sequence_new_aug2.csv"   #  Original sequence
    )
