#105の異常検知の分布をxyだけでなくxyz座標で表示させる
#
#ファイルによってフィルタリングで切り取る範囲が変わってくると思う
#特徴量に反射強度を含んでいない，座標のみで特徴量決めてるから中心から離れているものがいじょうとして選ばれやすい
#
##


import numpy as np
from sklearn.svm import OneClassSVM
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
filename = '2024.12.18.02.13.38.csv'  # CSVファイル名
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
sample_size = 1000  # 使用するデータ点の最大数
if len(filtered_data) > sample_size:
    np.random.seed(42)  # ランダム性を固定
    sampled_indices = np.random.choice(len(filtered_data), sample_size, replace=False)
    filtered_data = filtered_data[sampled_indices]

# --- 特徴量の選択 ---
# x, y, z のみを使用
features = filtered_data[:, :3]

# --- データの標準化 ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- サンプリング処理（追加部分） ---
max_samples = 10000  # SVM に使用する最大サンプル数
if scaled_features.shape[0] > max_samples:
    np.random.seed(42)
    indices = np.random.choice(scaled_features.shape[0], max_samples, replace=False)
    scaled_features = scaled_features[indices]
    features = features[indices]
print(f"Sampled data shape: {scaled_features.shape}")

# --- One-Class SVM モデルの定義 ---
clf = OneClassSVM(nu=0.001, kernel='rbf', gamma='scale')  # gamma='scale'は推奨
clf.fit(scaled_features)  # 正常データを学習

# --- 異常検知 ---
predictions = clf.predict(scaled_features)  # 1: 正常, -1: 異常
decision_scores = clf.decision_function(scaled_features)  # 決定関数のスコア

# --- 結果の表示 ---
print("Predictions:", predictions)  # 異常かどうかの判定
print("Decision Scores:", decision_scores)

# --- データの可視化 ---
# 正常データと異常データを色分けしてプロット
normal_data = features[predictions == 1]
anomalous_data = features[predictions == -1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(normal_data[:, 0], normal_data[:, 1], normal_data[:, 2], label="Normal", color="blue")
ax.scatter(anomalous_data[:, 0], anomalous_data[:, 1], anomalous_data[:, 2], label="Anomalous", color="red")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend()
plt.title("3D Detection of Rebar vs Non-Rebar")
plt.show()
