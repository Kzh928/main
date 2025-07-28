#
#
#
#0001-3 でDBSCAN やってる
#それのデータ違い
#0203.21.11.13
#クラスタリング自体はできてる，結局データが微妙，反射強度，座標#

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# --- フィルタリング条件の設定 ---
filter_conditions = {
    "x_min": -5.0,  # x座標の最小値
    "x_max": 5.0,   # x座標の最大値
    "y_min": -2.0,  # y座標の最小値
    "y_max": 3.0,   # y座標の最大値
    "z_min": -5.0,  # z座標の最小値
    "z_max": 5.0    # z座標の最大値（Noneの場合制限なし）
}

# --- データ準備 ---
# CSVファイルの読み込み
filename = '0203.21.11.13.csv'  # CSVファイル名
with open(filename) as f:
    reader = csv.reader(f)
    data = [row for row in reader]

# NumPy 配列に変換
data = np.asarray(data, dtype=np.float32)
if data.size == 0:  # データが空の場合エラー表示
    raise ValueError("Data is empty.")

# --- データの範囲確認 ---
print("Original Data Range:")
print(f"x: {np.min(data[:, 0]):.2f} to {np.max(data[:, 0]):.2f}")
print(f"y: {np.min(data[:, 1]):.2f} to {np.max(data[:, 1]):.2f}")
print(f"z: {np.min(data[:, 2]):.2f} to {np.max(data[:, 2]):.2f}")
print(f"intensity: {np.min(data[:, 3]):.2f} to {np.max(data[:, 3]):.2f}")

# --- フィルタリング処理 ---
filtered_data = data
if filter_conditions["x_min"] is not None:
    filtered_data = filtered_data[filtered_data[:, 0] >= filter_conditions["x_min"]]
if filter_conditions["x_max"] is not None:
    filtered_data = filtered_data[filtered_data[:, 0] <= filter_conditions["x_max"]]
if filter_conditions["y_min"] is not None:
    filtered_data = filtered_data[filtered_data[:, 1] >= filter_conditions["y_min"]]
if filter_conditions["y_max"] is not None:
    filtered_data = filtered_data[filtered_data[:, 1] <= filter_conditions["y_max"]]
if filter_conditions["z_min"] is not None:
    filtered_data = filtered_data[filtered_data[:, 2] >= filter_conditions["z_min"]]
if filter_conditions["z_max"] is not None:
    filtered_data = filtered_data[filtered_data[:, 2] <= filter_conditions["z_max"]]

if filtered_data.size == 0:
    raise ValueError("Filtered data is empty. Check your filtering conditions.")

# --- サンプリング処理 ---
sample_size = 10000  # 使用するデータ点の最大数
if len(filtered_data) > sample_size:
    np.random.seed(42)  # ランダム性を固定
    sampled_indices = np.random.choice(len(filtered_data), sample_size, replace=False)
    filtered_data = filtered_data[sampled_indices]

# --- 特徴量の選択 ---
features = filtered_data[:, :3]  # x, y, z を特徴量として使用

# --- データの標準化 ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- 無効な値のチェック ---
scaled_features = scaled_features[~np.isnan(scaled_features).any(axis=1)]
scaled_features = scaled_features[~np.isinf(scaled_features).any(axis=1)]

# --- DBSCANの実行 ---
eps = 0.5  # 近傍距離
min_samples = 5  # クラスタを形成する最小データ点数

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(scaled_features)

# --- 結果の確認 ---
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
print(f"Number of clusters: {n_clusters}")
print(f"Noise points: {n_noise}")

# --- クラスタリング結果の可視化 ---
# 各クラスタに色を割り当てる
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = "k"  # ノイズは黒色

    class_member_mask = (labels == k)

    xyz = features[class_member_mask]
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=f"Cluster {k}" if k != -1 else "Noise", color=col)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend()
plt.title("DBSCAN Clustering of Point Cloud")
plt.show()
