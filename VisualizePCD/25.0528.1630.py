
#Isolation Forestによる異常検知コード
#
#
#ためし
#４のやつのデータ違い
#0203.21.11.13
#
##
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# --- フィルタリング条件の設定 ---
filter_conditions = {
    "x_min": -5.0,
    "x_max": 5.0,
    "y_min": -2.0,
    "y_max": 3.0,
    "z_min": -5.0,
    "z_max": 5.0,
}

# --- データ準備 ---
filename = '0203.21.11.13.csv'
with open(filename) as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data = np.asarray(data, dtype=np.float32)
if data.size == 0:
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
sample_size = 10000
if len(filtered_data) > sample_size:
    np.random.seed(42)
    sampled_indices = np.random.choice(len(filtered_data), sample_size, replace=False)
    filtered_data = filtered_data[sampled_indices]

# --- 特徴量の選択 ---
# 重みを設定
coordinate_weight = 1.0
intensity_weight = 2.0

features = filtered_data[:, :3] * coordinate_weight  # x, y, z
intensity = filtered_data[:, 3].reshape(-1, 1) * intensity_weight  # 反射強度
combined_features = np.hstack((features, intensity))

# --- データの標準化 ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)

# --- Isolation Forest モデルの定義 ---
contamination = 0.1
isolation_forest = IsolationForest(contamination=contamination, random_state=42)
isolation_forest.fit(scaled_features)

# --- 異常検知 ---
predictions = isolation_forest.predict(scaled_features)
anomaly_scores = isolation_forest.decision_function(scaled_features)

# --- 結果の確認 ---
print(f"Number of normal points: {np.sum(predictions == 1)}")
print(f"Number of anomalous points: {np.sum(predictions == -1)}")

# --- データの可視化 ---
normal_data = filtered_data[predictions == 1]
anomalous_data = filtered_data[predictions == -1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(normal_data[:, 0], normal_data[:, 1], normal_data[:, 2], label="Normal", color="blue", s=1)
ax.scatter(anomalous_data[:, 0], anomalous_data[:, 1], anomalous_data[:, 2], label="Anomalous", color="red", s=1)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend()
plt.title("Isolation Forest Anomaly Detection (With Intensity)")
plt.show()
