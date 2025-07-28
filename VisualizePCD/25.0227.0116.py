#
#
#エルボー含めたK-Means
#
#ランダムサンプリング，フィルタリング，表示サイズを小さく
##



import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv


# --- フィルタリング条件の設定 ---
filter_conditions = {
    "x_min": 0.0,  # x座標の最小値
    "x_max": 5.0,   # x座標の最大値
    "y_min": -2.0,  # y座標の最小値
    "y_max": 3.0,   # y座標の最大値
    "z_min": -5.0,  # z座標の最小値
    "z_max": 5.0    # z座標の最大値（Noneの場合制限なし）
}

# --- データの読み込み ---
filename = '2024.12.18.02.13.38.csv'  # データファイル
with open(filename) as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data = np.asarray(data, dtype=np.float32)
if data.size == 0:
    raise ValueError("Data is empty.")

# --- サンプリング（例: 50% のデータをランダムサンプリング） ---
sampling_ratio = 0.1
np.random.seed(42)  # 再現性のために固定
sampled_indices = np.random.choice(data.shape[0], int(data.shape[0] * sampling_ratio), replace=False)
data = data[sampled_indices]


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



# # --- フィルタリング（例: 範囲条件によるフィルタリング） ---
# filter_conditions = (data[:, 0] > 0) & (data[:, 1] > 0) & (data[:, 2] > 0)  # 全ての値が正の場合のみ
# filtered_data = data[filter_conditions]
# if filtered_data.size == 0:
#     raise ValueError("Filtered data is empty.")

# --- 特徴量の選択とスケーリング ---
features = filtered_data[:, :3]  # x, y, z を特徴量として選択
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- エルボー法 ---
max_k = 10  # 最大クラスタ数
inertia_values = []

for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia_values.append(kmeans.inertia_)

# エルボー法のプロット
plt.figure()
plt.plot(range(1, max_k + 1), inertia_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid()
plt.show()

# --- 選択したクラスタ数で K-Means 実行 ---
optimal_k = 5  # エルボー法に基づいて適切な値を選択（ここでは例として 5 を使用）
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(scaled_features)
labels = kmeans.labels_  # 各データ点のクラスタラベル
centroids = kmeans.cluster_centers_  # クラスタの中心点

# --- クラスタリング結果の可視化 ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for cluster_id in range(optimal_k):
    cluster_points = scaled_features[labels == cluster_id]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], s=10, label=f'Cluster {cluster_id}')  # 点サイズを調整

# クラスタ中心をプロット
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=200, c='black', marker='X', label='Centroids')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend()
plt.title("K-Means Clustering with Optimal K")
plt.show()
