#
#重すぎたからサンプル数減らしてsvm
#異常値の割合示す，異常値のみの分布もだす
#


import csv
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # カラーマップを利用
from sklearn import svm
from sklearn.preprocessing import StandardScaler

# --- フィルタリング条件の設定 ---
filter_conditions = {
    "x_min": -5.0,  # x座標の最小値
    "x_max": 5.0,   # x座標の最大値
    "y_min": -2.0,  # y座標の最小値
    "y_max": 3.0,   # y座標の最大値
    "z_min": -5.0,  # z座標の最小値
    "z_max": None   # z座標の最大値（Noneの場合制限なし）
}

# --- データ読み込み ---
filename = '2024.12.18.02.10.21.csv'  # CSVファイル名
with open(filename) as f:
    reader = csv.reader(f)
    data = [row for row in reader]

# NumPy 配列に変換
data = np.asarray(data, dtype=np.float32)

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

# --- サンプリング処理 ---
sample_size = 10000  # 使用するデータ点の最大数
if len(filtered_data) > sample_size:
    np.random.seed(42)  # ランダム性を固定
    sampled_indices = np.random.choice(len(filtered_data), sample_size, replace=False)
    filtered_data = filtered_data[sampled_indices]

print(f"\nFiltered Data Size: {len(filtered_data)}")

# --- グラデーションの設定（反射強度による色分け） ---
min_value = np.min(filtered_data[:, 3])  # 反射強度の最小値
max_value = np.max(filtered_data[:, 3])  # 反射強度の最大値

# 反射強度を正規化（0～1にスケール変換）
normalized_intensity = (filtered_data[:, 3] - min_value) / (max_value - min_value)
normalized_intensity = np.clip(normalized_intensity, 0, 1)

# カラーマップを選択
cmap = cm.get_cmap("jet")
colors = cmap(normalized_intensity)[:, :3]

# --- 点群データ作成 ---
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_data[:, :3])
pcd.colors = o3d.utility.Vector3dVector(colors)

# --- 可視化 ---
o3d.visualization.draw_geometries([pcd])

# --- One-Class SVM ---
# 特徴量を選択（例: x, y, z）
features = filtered_data[:, :3]

# データを分割
split_idx = int(len(features) * 0.7)
x_train = features[:split_idx]
x_test = features[split_idx:]

# スケール変換
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# モデルの定義と学習
clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
clf.fit(x_train_scaled)

# 予測
y_pred_test = clf.predict(x_test_scaled)

# 結果の確認
print("\n=== One-Class SVM Results ===")
print("Parameters:", clf.get_params())
print("Decision Function (first 10):", clf.decision_function(x_test_scaled)[:10])
print("Predictions (first 10):", y_pred_test[:10])

# 異常値の数をカウント
num_anomalies = np.sum(y_pred_test == -1)
print(f"Number of anomalies: {num_anomalies}")

total_points = len(y_pred_test)
anomaly_ratio = num_anomalies / total_points
print(f"Anomaly ratio: {anomaly_ratio * 100:.2f}%")

# 異常値のみを抽出
anomalies = x_test_scaled[y_pred_test == -1]

# 点群の可視化（異常値のみ）
anomalies_pcd = o3d.geometry.PointCloud()
anomalies_pcd.points = o3d.utility.Vector3dVector(anomalies)
o3d.visualization.draw_geometries([anomalies_pcd])

#https://memo.onl.jp/?5LWPUq1dmN