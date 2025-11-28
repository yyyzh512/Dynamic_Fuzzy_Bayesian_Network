###############################################################################
# full_attention_pipeline.py
#  1) 无监督抽取 IA/GA 规则 + 拟合输出侧MF → 计算 IA/GA
#  2) IA/GA 作为输入，再无监督抽取 ATT 规则 + 拟合输出侧MF → 计算 ATT
###############################################################################
import numpy as np, pandas as pd
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
from sklearn.metrics import pairwise_distances
import matplotlib as mpl

# =============================== 工具: IO & 隶属度 ===============================
def _normalize_id_series(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors='coerce')
    out = pd.Series(index=s.index, dtype=object)
    mnum = s_num.notna()
    if mnum.any():
        s_round = s_num.round().astype('Int64')
        is_int_like = (s_num - s_round.astype(float)).abs() <= 1e-6
        out.loc[mnum & is_int_like]  = s_round[mnum & is_int_like].astype(str).values
        out.loc[mnum & ~is_int_like] = s[mnum & ~is_int_like].astype(str).str.strip().values
    mstr = ~mnum
    if mstr.any():
        out.loc[mstr] = s[mstr].astype(str).str.strip().values
    return out.astype(str).str.replace(r'\.0+$', '', regex=True)

def load_feature_values(csv_path, id_col='ID', feature_col=None):
    df = pd.read_csv(csv_path)
    ids = _normalize_id_series(df[id_col] if id_col in df.columns else df.iloc[:,0])
    if feature_col is None:
        cols = list(df.columns)
        if len(cols) < 2: raise ValueError(f"{csv_path} 需至少两列（ID,value）")
        feature_col = cols[1]
    x = pd.to_numeric(df[feature_col], errors='coerce').astype(float)
    keep = x.notna() & ids.notna()
    return pd.DataFrame({'ID': ids[keep].values, 'x': x[keep].values})

def trapezoidal_mu(x, a,b,c,d):
    x = np.asarray(x, float)
    y = np.zeros_like(x)
    if b > a: m = (x>a)&(x<b); y[m] = (x[m]-a)/(b-a)
    y[(x>=b)&(x<=c)] = 1.0
    if d > c: m = (x>c)&(x<d); y[m] = (d-x[m])/(d-c)
    return np.clip(y, 0, 1)

def load_mf_params(path):
    p = pd.read_csv(path)
    p.columns = [c.lower().strip() for c in p.columns]
    p['label'] = p['label'].str.lower().str.strip().replace({'med':'medium'})
    mf = {}
    for lab in ['low','medium','high']:
        row = p[p['label']==lab]
        if row.empty: raise ValueError(f'{path} 缺少 {lab}')
        a,b,c,d = row[['a','b','c','d']].values[0].astype(float)
        mf[lab] = tuple(np.sort([a,b,c,d]))
    return mf

def save_mfs_to_csv(mfs: dict, path: str):
    rows = [{'label': lab, 'a': a, 'b': b, 'c': c, 'd': d}
            for lab, (a,b,c,d) in mfs.items()]
    pd.DataFrame(rows).to_csv(path, index=False)

# =============================== 规则抽取（无监督） ===============================
def memberships_scalar(x, mf):
    return {lab: float(trapezoidal_mu([x], *mf[lab])[0]) for lab in ['low','medium','high']}

def extract_rules_unsupervised_from_csvs(
    norm_csv_1, params_csv_1, norm_csv_2, params_csv_2,
    id_col='ID', w1=0.5, w2=0.5, use_quantile_thresholds=True, tau1=0.30, tau2=0.70
):
    # 读取并按 ID 对齐
    df1 = load_feature_values(norm_csv_1, id_col=id_col).rename(columns={'x':'x1'})
    df2 = load_feature_values(norm_csv_2, id_col=id_col).rename(columns={'x':'x2'})
    df  = df1.merge(df2, on='ID', how='inner')
    if df.empty: raise RuntimeError("两表 ID 交集为空")
    mf1, mf2 = load_mf_params(params_csv_1), load_mf_params(params_csv_2)

    # 内部注意得分 s 与阈值
    s = (w1*df['x1'] + w2*df['x2']).to_numpy()
    if use_quantile_thresholds:
        t1, t2 = float(np.quantile(s, 0.33)), float(np.quantile(s, 0.66))
    else:
        t1, t2 = tau1, tau2
    def lab_by_s(v):
        if v < t1: return 'low'
        if v > t2: return 'high'
        return 'medium'
    ylab = np.array([lab_by_s(v) for v in s])

    # 模糊计数
    L = ['low','medium','high']
    W = {(l1,l2): {k:0.0 for k in L} for l1 in L for l2 in L}
    for (_, r), k in zip(df.iterrows(), ylab):
        mu1 = memberships_scalar(r['x1'], mf1)
        mu2 = memberships_scalar(r['x2'], mf2)
        for l1 in L:
            for l2 in L:
                W[(l1,l2)][k] += mu1[l1]*mu2[l2]

    # 规则与置信度
    rule = {}
    conf = {}
    for l1 in L:
        for l2 in L:
            d = W[(l1,l2)]
            tot = d['low']+d['medium']+d['high']
            if tot <= 1e-12:
                rule[(l1,l2)] = 'medium'; conf[(l1,l2)] = 0.0
            else:
                kstar = max(d, key=d.get)
                rule[(l1,l2)] = kstar
                conf[(l1,l2)] = d[kstar]/tot

    # 表格化
    df_rule = pd.DataFrame(index=L, columns=L, dtype=object)
    df_conf = pd.DataFrame(index=L, columns=L, dtype=float)
    for l1 in L:
        for l2 in L:
            df_rule.loc[l1,l2] = rule[(l1,l2)]
            df_conf.loc[l1,l2] = conf[(l1,l2)]
    meta = {'tau1': t1, 'tau2': t2, 'w1': w1, 'w2': w2, 'n': int(len(df))}
    return df_rule, df_conf, meta

def extract_rules_unsupervised_from_arrays(
    x1_vals, mf1, x2_vals, mf2, w1=0.5, w2=0.5, q1=0.33, q2=0.66
):
    x1_vals = np.asarray(x1_vals, float); x2_vals = np.asarray(x2_vals, float)
    s = w1*x1_vals + w2*x2_vals
    t1, t2 = float(np.quantile(s, q1)), float(np.quantile(s, q2))
    def lab_by_s(v):
        if v < t1: return 'low'
        if v > t2: return 'high'
        return 'medium'
    ylab = np.array([lab_by_s(v) for v in s])

    L = ['low','medium','high']
    W = {(l1,l2): {k:0.0 for k in L} for l1 in L for l2 in L}
    for x1,x2,k in zip(x1_vals, x2_vals, ylab):
        mu1 = memberships_scalar(x1, mf1)
        mu2 = memberships_scalar(x2, mf2)
        for l1 in L:
            for l2 in L:
                W[(l1,l2)][k] += mu1[l1]*mu2[l2]

    rule, conf = {}, {}
    for l1 in L:
        for l2 in L:
            d = W[(l1,l2)]
            tot = d['low']+d['medium']+d['high']
            if tot <= 1e-12:
                rule[(l1,l2)] = 'medium'; conf[(l1,l2)] = 0.0
            else:
                kstar = max(d, key=d.get)
                rule[(l1,l2)] = kstar
                conf[(l1,l2)] = d[kstar]/tot
    df_rule = pd.DataFrame(index=L, columns=L, dtype=object)
    df_conf = pd.DataFrame(index=L, columns=L, dtype=float)
    for l1 in L:
        for l2 in L:
            df_rule.loc[l1,l2] = rule[(l1,l2)]
            df_conf.loc[l1,l2] = conf[(l1,l2)]
    meta = {'tau1': t1, 'tau2': t2, 'w1': w1, 'w2': w2, 'n': int(len(x1_vals))}
    return df_rule, df_conf, meta

# =============================== 输出侧 MF（分位数驱动） ===============================
def fit_output_mfs_by_score_arrays(x1_vals, x2_vals, w1=0.5, w2=0.5, q1=0.33, q2=0.66,
                                   gamma=0.4, delta_max=0.15):
    s = (w1*np.asarray(x1_vals,float) + w2*np.asarray(x2_vals,float)).astype(float)
    q33, q66 = float(np.quantile(s, q1)), float(np.quantile(s, q2))
    m = 0.5*(q33+q66)
    delta = min(delta_max, gamma*max(q66-q33, 0.0))

    low    = (0.0, 0.0, q33, min(m, q33+delta))
    medium = (max(q33-delta, 0.0), m, m, min(q66+delta, 1.0))
    high   = (max(q66-delta, m), q66, 1.0, 1.0)

    def _sorted_clip(t):
        a,b,c,d = t
        v = np.clip(np.array([a,b,c,d], float), 0.0, 1.0); v.sort()
        return tuple(v.tolist())
    mfs = {'low':_sorted_clip(low), 'medium':_sorted_clip(medium), 'high':_sorted_clip(high)}
    meta= {'q33':q33,'q66':q66,'m':m,'delta':delta,'w1':w1,'w2':w2,'n':len(s)}
    return mfs, meta

# =============================== Mamdani + MAR 推理 ===============================
def infer_two_inputs(x1, x2, mf1, mf2, rule_df, out_mf, grid_n=1001):
    mu1 = {lab: trapezoidal_mu([x1], *mf1[lab])[0] for lab in mf1}
    mu2 = {lab: trapezoidal_mu([x2], *mf2[lab])[0] for lab in mf2}
    y = np.linspace(0, 1, grid_n)
    agg = {lab: np.zeros_like(y) for lab in out_mf}  # low/medium/high
    for l1 in ['low','medium','high']:
        for l2 in ['low','medium','high']:
            w = min(mu1[l1], mu2[l2])
            if w <= 0: continue
            out_lab = rule_df.loc[l1, l2]
            mu_out = trapezoidal_mu(y, *out_mf[out_lab])
            agg[out_lab] = np.maximum(agg[out_lab], np.minimum(w, mu_out))
    areas = {lab: np.trapz(agg[lab], y) for lab in agg}
    cents = {lab: (np.trapz(y*agg[lab], y)/areas[lab] if areas[lab]>1e-12 else 0.0) for lab in agg}
    denom = sum(areas.values())
    crisp = 0.0 if denom<=1e-12 else sum(areas[k]*cents[k] for k in areas)/denom
    return crisp


# =============================== 主流程 ===============================
def main(
    # 输入侧：四特征
    fsp_norm='DurationandInputMF\\normalized_fsp.csv',  fsp_params='DurationandInputMF\\params_fsp.csv',
    fid_norm='DurationandInputMF\\normalized_fid.csv',  fid_params='DurationandInputMF\\params_fid.csv',
    fgd_norm='DurationandInputMF\\normalized_fgd.csv',  fgd_params='DurationandInputMF\\params_fgd.csv',
    fopr_norm='DurationandInputMF\\normalized_fopr.csv',fopr_params='DurationandInputMF\\params_fopr.csv',
    id_col='ID',
    # 超参
    w_fsp=0.5, w_fid=0.5, w_fgd=0.5, w_fopr=0.5,   # IA/GA 阶段的内部权重（一般 0.5/0.5）
    w_IA=0.5,  w_GA=0.5,                            # ATT 融合的内部权重
    q1=0.33, q2=0.66, gamma=0.4, delta_max=0.15
):
    # ---------- 1) IA：规则 + 输出MF ----------
    df_rule_IA, df_conf_IA, meta_IA_rule = extract_rules_unsupervised_from_csvs(
        fsp_norm, fsp_params, fid_norm, fid_params,
        id_col=id_col, w1=w_fsp, w2=w_fid, use_quantile_thresholds=True
    )
    df_rule_IA.to_csv('Rules\\rules_IA.csv'); df_conf_IA.to_csv('Rules\\rules_IA_conf.csv')
    print('[IA rules] meta:', meta_IA_rule)

    # 输出侧 MF(IA)
    df_fsp = load_feature_values(fsp_norm, id_col=id_col)
    df_fid = load_feature_values(fid_norm, id_col=id_col)
    df_ia_xy = df_fsp.merge(df_fid, on='ID', how='inner')
    mfs_out_IA, meta_mf_IA = fit_output_mfs_by_score_arrays(
        df_ia_xy['x_x'], df_ia_xy['x_y'], w1=w_fsp, w2=w_fid, q1=q1, q2=q2,
        gamma=gamma, delta_max=delta_max
    )
    save_mfs_to_csv(mfs_out_IA, 'OutputMF\\params_out_IA.csv')
    print('[IA output MF] meta:', meta_mf_IA, 'mfs:', mfs_out_IA)

    # ---------- 2) GA：规则 + 输出MF ----------
    df_rule_GA, df_conf_GA, meta_GA_rule = extract_rules_unsupervised_from_csvs(
        fgd_norm, fgd_params, fopr_norm, fopr_params,
        id_col=id_col, w1=w_fgd, w2=w_fopr, use_quantile_thresholds=True
    )
    df_rule_GA.to_csv('Rules\\rules_GA.csv'); df_conf_GA.to_csv('Rules\\rules_GA_conf.csv')
    print('[GA rules] meta:', meta_GA_rule)

    df_fgd = load_feature_values(fgd_norm, id_col=id_col)
    df_opr = load_feature_values(fopr_norm, id_col=id_col)
    df_ga_xy = df_fgd.merge(df_opr, on='ID', how='inner')
    mfs_out_GA, meta_mf_GA = fit_output_mfs_by_score_arrays(
        df_ga_xy['x_x'], df_ga_xy['x_y'], w1=w_fgd, w2=w_fopr, q1=q1, q2=q2,
        gamma=gamma, delta_max=delta_max
    )
    save_mfs_to_csv(mfs_out_GA, 'OutputMF\\params_out_GA.csv')
    print('[GA output MF] meta:', meta_mf_GA, 'mfs:', mfs_out_GA)

    # ---------- 3) 计算 IA & GA 数值 ----------
    mf_fsp, mf_fid = load_mf_params(fsp_params), load_mf_params(fid_params)
    mf_fgd, mf_opr = load_mf_params(fgd_params), load_mf_params(fopr_params)
    # 作为输出侧 MF 的参数
    out_mf_IA, out_mf_GA = mfs_out_IA, mfs_out_GA

    df_all = df_fsp.rename(columns={'x':'fsp'})\
        .merge(df_fid.rename(columns={'x':'fid'}), on='ID', how='inner')\
        .merge(df_fgd.rename(columns={'x':'fgd'}), on='ID', how='inner')\
        .merge(df_opr.rename(columns={'x':'fopr'}), on='ID', how='inner')
    IA_vals, GA_vals = [], []
    for _, r in df_all.iterrows():
        ia = infer_two_inputs(r['fsp'], r['fid'], mf_fsp, mf_fid, df_rule_IA, out_mf_IA)
        ga = infer_two_inputs(r['fgd'], r['fopr'], mf_fgd, mf_opr, df_rule_GA, out_mf_GA)
        IA_vals.append(ia); GA_vals.append(ga)
    df_IA_GA = df_all[['ID']].copy()
    df_IA_GA['IA'] = IA_vals; df_IA_GA['GA'] = GA_vals
    df_IA_GA.to_csv('Results\\IA_GA.csv', index=False)
    print('[IA_GA] head:\n', df_IA_GA.head())

    # ---------- 4) 融合层（ATT）：规则 + 输出MF ----------
    # 输入侧 MF：采用 IA/GA 各自的输出侧 MF（语义一致）
    in_mf_IA_for_ATT = out_mf_IA
    in_mf_GA_for_ATT = out_mf_GA

    # 规则（基于 IA/GA 数值的软计数）
    df_rule_ATT, df_conf_ATT, meta_ATT_rule = extract_rules_unsupervised_from_arrays(
        df_IA_GA['IA'].to_numpy(), in_mf_IA_for_ATT,
        df_IA_GA['GA'].to_numpy(), in_mf_GA_for_ATT,
        w1=w_IA, w2=w_GA, q1=q1, q2=q2
    )
    df_rule_ATT.to_csv('Rules\\rules_ATT.csv'); df_conf_ATT.to_csv('Rules\\rules_ATT_conf.csv')
    print('[ATT rules] meta:', meta_ATT_rule)

    # 输出侧 MF（基于 IA/GA 分布）
    mfs_out_ATT, meta_mf_ATT = fit_output_mfs_by_score_arrays(
        df_IA_GA['IA'].to_numpy(), df_IA_GA['GA'].to_numpy(),
        w1=w_IA, w2=w_GA, q1=q1, q2=q2, gamma=gamma, delta_max=delta_max
    )
    save_mfs_to_csv(mfs_out_ATT, 'OutputMF\\params_out_ATT.csv')
    print('[ATT output MF] meta:', meta_mf_ATT, 'mfs:', mfs_out_ATT)

    # ---------- 5) 计算 ATT 数值 ----------
    ATT_vals = []
    for _, r in df_IA_GA.iterrows():
        a = infer_two_inputs(r['IA'], r['GA'],
                             in_mf_IA_for_ATT, in_mf_GA_for_ATT,
                             df_rule_ATT, mfs_out_ATT)
        ATT_vals.append(a)
    df_ATT = df_IA_GA[['ID']].copy()
    df_ATT['ATT'] = ATT_vals
    df_ATT.to_csv('Results\\ATTENTION.csv', index=False)
    print('[ATTENTION] saved to ATTENTION.csv. head:\n', df_ATT.head())

if __name__ == '__main__':
    # 如需调整权重/分位点/重叠，修改 main 的参数
    main()
