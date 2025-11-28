import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
from sklearn.metrics import pairwise_distances

# 定义梯形隶属度函数
def trapezoidal(x, a, b, c, d):
    if b == a:
        left_slope = np.where(x <= a, 0, 1)
    else:
        left_slope = (x - a) / (b - a)

    if d == c:
        right_slope = np.where(x >= d, 0, 1)
    else:
        right_slope = (d - x) / (d - c)

    return np.clip(np.maximum(0, np.minimum(left_slope, np.minimum(1, right_slope))), 0, 1)


# AFCM 聚类
class AFCM:
    def __init__(self, n_clusters, m=2, max_iter=150, error=1e-5):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error

    def initialize_membership(self, X):
        N = X.shape[0]
        self.u = np.random.rand(N, self.n_clusters)
        self.u = self.u / np.sum(self.u, axis=1, keepdims=True)

    def compute_entropy(self, X):
        return -np.sum(self.u * np.log(self.u + 1e-8), axis=1)

    def compute_affinity(self, E):
        aff = (1 - E) / (len(E) - np.sum(E))
        aff = aff / np.sum(aff)
        return aff

    def update_centers(self, X, aff):
        v = np.dot((self.u ** self.m * aff[:, np.newaxis]).T, X) / \
            np.sum(self.u ** self.m * aff[:, np.newaxis], axis=0)[:, np.newaxis]
        return v

    def update_membership(self, X, v, aff):
        dist = pairwise_distances(X, v, metric='euclidean')
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        temp = dist ** (-2 / (self.m - 1))
        self.u = temp / np.sum(temp, axis=1, keepdims=True)
        return self.u

    def fit(self, X):
        self.initialize_membership(X)
        for _ in range(self.max_iter):
            u_old = self.u.copy()
            E = self.compute_entropy(X)
            aff = self.compute_affinity(E)
            self.centers = self.update_centers(X, aff)
            self.u = self.update_membership(X, self.centers, aff)
            if np.linalg.norm(self.u - u_old) < self.error:
                break


# 处理但不绘图
def process_and_plot(file_path, feature_name, subplot_index, labels,
                     output_file, normalized_output_file,
                     data_colors=None, fit_colors=None, id_col='ID'):

    # 读取 CSV
    data = pd.read_csv(file_path)

    if id_col in data.columns:
        ids = data[id_col].to_numpy()
    else:
        ids = data.iloc[:, 0].to_numpy()

    # 读取特征列
    if feature_name not in data.columns:
        raise KeyError(f"列 {feature_name} 不在文件 {file_path} 中。已有列：{list(data.columns)}")
    feature_raw = data[feature_name].to_numpy().reshape(-1, 1)

    # 删除缺失值
    mask = ~np.isnan(feature_raw).ravel()
    ids = ids[mask]
    feature_raw = feature_raw[mask].reshape(-1, 1)

    # 归一化
    scaler = MinMaxScaler()
    feature_scaled = scaler.fit_transform(feature_raw)
    x = feature_scaled.ravel()

    # 保存归一化数据
    normalized_data = pd.DataFrame({id_col: ids, feature_name: x})
    normalized_data.to_csv(normalized_output_file, index=False)

    # AFCM
    c = len(labels)
    afcm = AFCM(n_clusters=c)
    afcm.fit(feature_scaled)

    # 隶属度排序
    u = afcm.u
    centers = afcm.centers
    sorted_indices = np.argsort(centers[:, 0])
    sorted_centers = centers[sorted_indices]
    sorted_u = u[:, sorted_indices]

    # 拟合梯形隶属度
    trapezoidal_params = []
    for i in range(c):
        y = sorted_u[:, i]
        popt, _ = curve_fit(
            trapezoidal, x, y,
            p0=[np.min(x),
                np.clip(np.mean(x) - 0.1, 0, 1),
                np.clip(np.mean(x) + 0.1, 0, 1),
                np.max(x)],
            bounds=([0, 0, 0, 0], [1, 1, 1, 1]),
            maxfev=20000
        )
        popt = np.clip(popt, 0, 1)

        # 平台约束
        if i == 0:
            popt[1] = popt[0]
        elif i == c - 1:
            popt[2] = popt[3]

        trapezoidal_params.append(popt)

    # 保存隶属度端点
    params_df = pd.DataFrame(trapezoidal_params, columns=['a', 'b', 'c', 'd'])
    params_df['label'] = labels
    params_df.to_csv(output_file, index=False)


# 三类颜色配置仍然保留（尽管不绘图）
data_colors = ['#E53528', '#F09739', '#193E8F']
fit_colors = ['#E53528', '#F09739', '#193E8F']

# 调用保持不变
process_and_plot('raw_duration_102_with_zck.csv', 'fsp', 1,
                 ['low', 'medium', 'high'],
                 'DurationandInputMF/params_fsp.csv',
                 'DurationandInputMF/normalized_fsp.csv',
                 data_colors=data_colors, fit_colors=fit_colors)

process_and_plot('raw_duration_102_with_zck.csv', 'fid', 2,
                 ['low', 'medium', 'high'],
                 'DurationandInputMF/params_fid.csv',
                 'DurationandInputMF/normalized_fid.csv',
                 data_colors=data_colors, fit_colors=fit_colors)

process_and_plot('raw_duration_102_with_zck.csv', 'fgd', 3,
                 ['low', 'medium', 'high'],
                 'DurationandInputMF/params_fgd.csv',
                 'DurationandInputMF/normalized_fgd.csv',
                 data_colors=data_colors, fit_colors=fit_colors)

process_and_plot('raw_duration_102_with_zck.csv', 'fopr', 4,
                 ['low', 'medium', 'high'],
                 'DurationandInputMF/params_fopr.csv',
                 'DurationandInputMF/normalized_fopr.csv',
                 data_colors=data_colors, fit_colors=fit_colors)
