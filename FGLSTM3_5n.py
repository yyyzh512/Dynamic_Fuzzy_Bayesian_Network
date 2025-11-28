###############################################################################
# FG-LSTM gt=[gIA, gGA, gDA]
# 10倍交叉验证，单图单 CSV
###############################################################################
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# ============================ 全局超参 / 随机种子 ============================
CSV_PATH   = r"sequence_new_aug2.csv"   # ★ 数据集
N_SPLITS   = 10
N_EPOCHS   = 50
SEED       = 2025
HIDDEN_DIM = 8
LR         = 0.01

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# ============================ 基础函数 ======================================
def sigmoid(x): return 1/(1+np.exp(-x))
def dsigmoid(s): return s*(1-s)
def tanh(x):     return np.tanh(x)
def dtanh(t):    return 1-t**2
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)
def cross_entropy_loss(p, y): return -np.log(p[y] + 1e-12)

# ============================ FGLSTM Cell ====================================
class FGLSTMCellTrainable:
    def __init__(self, input_dim, hidden_dim, lr=LR):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.lr         = lr
        lim = 0.1

        # W 参数不变：concat([h_mod, x_main])→4H
        self.W = rng.uniform(-lim, lim, (input_dim - 3 + hidden_dim, 4 * hidden_dim))
        # U 由 (4,H) → (4,3,H)，对每个 gate 用 g_vec (3,) 投影到 H 维
        self.U = rng.uniform(-lim, lim, (4, 3, hidden_dim))
        self.b = np.zeros(4 * hidden_dim)
        # alpha 由标量 → 长度 3 的向量
        self.alpha = rng.uniform(size=3)

        self.zero_grads()

    def zero_grads(self):
        self.dW     = np.zeros_like(self.W)
        self.dU     = np.zeros_like(self.U)
        self.db     = np.zeros_like(self.b)
        self.dalpha = np.zeros_like(self.alpha)

    def forward_step(self, x, h_prev, C_prev):
        # 提取三维的 g_vec（最后 3 维）
        g_vec  = x[-3:]        # shape (3,)
        x_main = x[:-3]        # 其余输入
        # B2 乘性增益：beta = 1 + alpha^T g_vec
        beta   = 1.0 + np.dot(self.alpha, g_vec)
        h_mod  = h_prev * beta

        # 拼接后送入 W
        concat = np.concatenate([h_mod, x_main])  # shape (H + (D-3),)
        # 对于每个 gate，u_proj_k = g_vec^T U[k] → (H,)
        u_proj = np.concatenate([g_vec @ self.U[k] for k in range(4)])  # shape (4H,)

        z = concat @ self.W + u_proj + self.b
        H = self.hidden_dim
        f = sigmoid(z[:H])
        i = sigmoid(z[H:2*H])
        o = sigmoid(z[2*H:3*H])
        g = tanh(z[3*H:])

        C = f * C_prev + i * g
        h = o * tanh(C)

        # 缓存
        self.cache.append(dict(
            concat=concat, g_vec=g_vec, h_prev=h_prev, beta=beta,
            C_prev=C_prev, f=f, i=i, o=o, g=g, C=C
        ))
        return h, C

    def forward_sequence(self, X):
        self.cache = []
        h = np.zeros(self.hidden_dim)
        C = np.zeros(self.hidden_dim)
        Hs = []
        for x in X:
            h, C = self.forward_step(x, h, C)
            Hs.append(h)
        return np.stack(Hs)

    def backward(self, dH):
        dh_next = np.zeros(self.hidden_dim)
        dC_next = np.zeros(self.hidden_dim)

        for t in reversed(range(len(self.cache))):
            cache = self.cache[t]
            H     = self.hidden_dim

            # —— LSTM+B2 的原始反向 —— 
            dh_total = dh_next + dH[t]
            C = cache['C']
            do = dh_total * tanh(C)
            dC = dh_total * cache['o'] * dtanh(tanh(C)) + dC_next

            df = dC * cache['C_prev']
            di = dC * cache['g']
            dg = dC * cache['i']

            dz = np.concatenate([
                df * dsigmoid(cache['f']),
                di * dsigmoid(cache['i']),
                do * dsigmoid(cache['o']),
                dg * dtanh(cache['g']),
            ])

            self.dW += np.outer(cache['concat'], dz)
            dz_slices = [dz[k * H:(k + 1) * H] for k in range(4)]

            # 反向 self.U 和 self.alpha
            for k in range(4):
                # U[k] shape (3,H), g_vec shape (3,)
                self.dU[k] += np.outer(cache['g_vec'], dz_slices[k])

            self.db += dz

            dconcat = dz @ self.W.T
            dh_mod  = dconcat[:H]

            # B2 的反向：beta = 1 + alpha^T g_vec
            d_beta = np.dot(dh_mod, cache['h_prev'])   # scalar
            # alpha 梯度 ∂L/∂alpha = g_vec * d_beta
            self.dalpha += cache['g_vec'] * d_beta

            # 继续给上一步 h_prev 的梯度（暂不回传 g_vec）
            dh_prev = dh_mod * cache['beta']
            dh_next, dC_next = dh_prev, dC * cache['f']

    def step_update(self):
        # 更新 LSTM + B2 参数
        self.W     -= self.lr * self.dW
        self.U     -= self.lr * self.dU
        self.b     -= self.lr * self.db
        self.alpha -= self.lr * self.dalpha
        self.zero_grads()


# ============================ FGLSTM Classifier ==============================
class FGLSTMClassifierMultiStep:
    def __init__(self, input_dim, hidden_dim, n_obj, n_lbl, lr=LR):
        self.lstm = FGLSTMCellTrainable(input_dim, hidden_dim, lr)
        lim = 0.1
        self.Wo = rng.uniform(-lim, lim, (hidden_dim, n_obj))
        self.bo = np.zeros(n_obj)
        self.Wl = rng.uniform(-lim, lim, (hidden_dim, n_lbl))
        self.bl = np.zeros(n_lbl)
        self.dWo = np.zeros_like(self.Wo)
        self.dbo = np.zeros_like(self.bo)
        self.dWl = np.zeros_like(self.Wl)
        self.dbl = np.zeros_like(self.bl)
        self.lr  = lr
        self.hidden_dim = hidden_dim

    def forward(self, X, return_h=False):
        H = self.lstm.forward_sequence(X)
        po, pl = [], []
        for h in H:
            po.append(softmax(h @ self.Wo + self.bo))
            pl.append(softmax(h @ self.Wl + self.bl))
        self.cache = dict(H=H, po=po, pl=pl)
        if return_h:
            return po, pl, H
        return po, pl

    def backward(self, obj, lbl):
        T = len(obj)
        dH = np.zeros((T, self.hidden_dim))
        for t in range(T):
            go = self.cache['po'][t].copy()
            gl = self.cache['pl'][t].copy()
            go[obj[t]] -= 1
            gl[lbl[t]] -= 1
            h = self.cache['H'][t]
            self.dWo += np.outer(h, go)
            self.dbo += go
            self.dWl += np.outer(h, gl)
            self.dbl += gl
            dH[t] += self.Wo @ go + self.Wl @ gl
        self.lstm.zero_grads()
        self.lstm.backward(dH)

    def step_update(self):
        self.Wo -= self.lr * self.dWo
        self.bo -= self.lr * self.dbo
        self.Wl -= self.lr * self.dWl
        self.bl -= self.lr * self.dbl
        self.dWo.fill(0)
        self.dbo.fill(0)
        self.dWl.fill(0)
        self.dbl.fill(0)
        self.lstm.step_update()

    def inference_stepwise(self, fold, sid, X, obj, lbl):
        po, pl = self.forward(X)
        rec = []
        for t in range(len(obj)):
            rec.append(dict(
                fold=fold, sample_id=sid, time_step=t+1,
                object_id=obj[t]+1, object_id_predicted=int(np.argmax(po[t]) + 1),
                label=lbl[t]+1, label_predicted=int(np.argmax(pl[t]) + 1)
            ))
        return rec

    def hidden_seq(self, fold, sid, X):
        _, _, H = self.forward(X, return_h=True)
        rec = []
        for t, h in enumerate(H, 1):
            row = dict(fold=fold, sample_id=int(sid), timestep=t)
            row.update({f"h{i+1}": float(v) for i, v in enumerate(h)})
            rec.append(row)
        return rec

# ============================ 数据加载 ======================================
def load_samples(path, num_objects=10):
    """
    适配新的 CSV 列：
    sample_id, timestep, time, object_name, object_id, gIA, gGA, gDA, label
    仍然构造：X = [one-hot(object_id), gIA, gGA, gDA]
    """
    df = pd.read_csv(path)

    # 仍然假定 object_id 和 label 从 1 开始，转换为 0-based
    df['object_id_zero'] = df['object_id'] - 1
    df['label_zero']     = df['label'] - 1

    samples = []
    for sid, g in df.groupby('sample_id'):
        g = g.sort_values('timestep')
        
        obj = g['object_id_zero'].to_numpy(int)
        lbl = g['label_zero'].to_numpy(int)

        # one-hot 对象
        onehot = np.eye(num_objects)[obj]   # shape (T, num_objects)

        # 用新的三列 gIA / gGA / gDA 构造三维 g_vec
        gIA = g['gIA'].to_numpy(float)[:, None]
        gGA = g['gGA'].to_numpy(float)[:, None]
        gDA = g['gDA'].to_numpy(float)[:, None]
        g_vec = np.hstack([gIA, gGA, gDA])  # shape (T,3)

        # 整体输入：one-hot + 三维模糊特征
        X = np.hstack([onehot, g_vec])      # shape (T, num_objects+3)
        samples.append((sid, X, obj, lbl))

    return samples

# ============================ 训练 & 评估辅助 ===============================
def train_one_epoch(model, train_s):
    losso = lossl = steps = 0
    rng.shuffle(train_s)
    for _, X, obj, lbl in train_s:
        po, pl = model.forward(X)
        for t in range(len(obj)):
            losso += cross_entropy_loss(po[t], obj[t])
            lossl += cross_entropy_loss(pl[t], lbl[t])
            steps += 1
        model.backward(obj, lbl)
        model.step_update()
    return losso / steps, lossl / steps

def evaluate(model, fold, test_s):
    stepo = stepl = tot = 0
    finalo = finall = seqs = 0
    rec_step = []
    rec_ht   = []
    for sid, X, obj, lbl in test_s:
        po, pl = model.forward(X)
        T = len(obj)
        for t in range(T):
            stepo += (np.argmax(po[t]) == obj[t])
            stepl += (np.argmax(pl[t]) == lbl[t])
            tot   += 1
        finalo += (np.argmax(po[-1]) == obj[-1])
        finall += (np.argmax(pl[-1]) == lbl[-1])
        seqs   += 1
        rec_step.extend(model.inference_stepwise(fold, sid, X, obj, lbl))
    for sid, X, _, _ in test_s:
        rec_ht.extend(model.hidden_seq(fold, sid, X))
    return dict(
        step_obj = stepo / tot, step_lbl = stepl / tot,
        final_obj = finalo / seqs, final_lbl = finall / seqs,
        step_records = rec_step, ht_records = rec_ht
    )

# ============================ 主 10-折流程 ==================================
def main():
    # 1) 加载样本，注意 input_dim = num_objects + 3
    num_objects = 10
    input_dim   = num_objects + 3  # one-hot(10) + [gIA,gGA,gDA]
    samples     = load_samples(CSV_PATH, num_objects)
    y_final     = np.array([s[3][-1] for s in samples])  # 用最终标签做分层

    skf = StratifiedKFold(
        n_splits   = N_SPLITS,
        shuffle    = True,
        random_state = SEED
    )

    # 用于汇总
    loss_fold_o, loss_fold_l = [], []
    acc_fold_o , acc_fold_l  = [], []
    metrics = []
    all_steps = []
    all_ht    = []

    # 2) 10 折循环
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(samples)), y_final), 1):
        train_s = [samples[i] for i in tr_idx]
        test_s  = [samples[i] for i in te_idx]

        # 初始化模型
        model = FGLSTMClassifierMultiStep(input_dim, HIDDEN_DIM, num_objects, num_objects, lr=LR)

        # 训练记录
        lo_hist, ll_hist = [], []
        ao_hist, al_hist = [], []

        for epoch in range(N_EPOCHS):
            lo, ll = train_one_epoch(model, train_s)
            lo_hist.append(lo)
            ll_hist.append(ll)

            # 计算训练集 final-step accuracy
            co = cl = 0
            for _, X, obj, lbl in train_s:
                po, pl = model.forward(X)
                co += (np.argmax(po[-1]) == obj[-1])
                cl += (np.argmax(pl[-1]) == lbl[-1])
            ao_hist.append(co / len(train_s))
            al_hist.append(cl / len(train_s))

        loss_fold_o.append(lo_hist)
        loss_fold_l.append(ll_hist)
        acc_fold_o.append(ao_hist)
        acc_fold_l.append(al_hist)

        # 在测试集上评估
        m = evaluate(model, fold, test_s)
        metrics.append({k: v for k, v in m.items() if k.endswith('_obj') or k.endswith('_lbl')})
        all_steps.extend(m['step_records'])
        all_ht.extend(m['ht_records'])

        print(f"[Fold {fold}] "
              f"step_obj={m['step_obj']:.3%}, step_lbl={m['step_lbl']:.3%}, "
              f"final_obj={m['final_obj']:.3%}, final_lbl={m['final_lbl']:.3%}")

    # 3) 汇总训练曲线 (mean ± std)
    epochs = np.arange(1, N_EPOCHS + 1)
    def agg(arr): return np.mean(arr, 0), np.std(arr, 0)

    mlo, slo = agg(loss_fold_o)
    mll, sll = agg(loss_fold_l)
    mao, sao = agg(acc_fold_o)
    mal, sal = agg(acc_fold_l)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mlo, label="Obj Loss")
    plt.fill_between(epochs, mlo - slo, mlo + slo, alpha=.2)
    plt.plot(epochs, mll, label="Lbl Loss")
    plt.fill_between(epochs, mll - sll, mll + sll, alpha=.2)
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.title("10-Fold Mean Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, mao, label="Obj Acc")
    plt.fill_between(epochs, mao - sao, mao + sao, alpha=.2)
    plt.plot(epochs, mal, label="Lbl Acc")
    plt.fill_between(epochs, mal - sal, mal + sal, alpha=.2)
    plt.xlabel("Epoch")
    plt.ylabel("Final-step Acc")
    plt.ylim(0, 1)
    plt.title("10-Fold Mean Final-Acc")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4) 打印汇总指标
    dfm = pd.DataFrame(metrics)
    print("\n===== 10-Fold CV Summary (mean ± std) =====")
    for col in dfm.columns:
        print(f"{col:9s}: {dfm[col].mean():.4f} ± {dfm[col].std():.4f}")

    # 5) 保存所有记录到 CSV
    pd.DataFrame(all_steps).to_csv("FG_test_stepwise_outputs.csv", index=False)
    pd.DataFrame(all_ht   ).to_csv("FG_Ht.csv", index=False)
    print("\n[Info] 已生成 test_stepwise_outputs_all.csv 和 Ht_all.csv")

if __name__ == "__main__":
    main()
