#105の異常検知の分布をxyだけでなくxyz座標で表示させる
#
#ファイルによってフィルタリングで切り取る範囲が変わってくると思う
#特徴量の部分で反射強度だけで見る
#点のサイズも小さくする
#フィルタリング範囲も変更
#結果としてサンプル数減らすと表示範囲おかしい感じになった
#反射強度の差がそこまでないからおかしくなっている気がする
#svmをはんしゃきょうどという値だけで見てしまうとその枠がそもそも点群データの向こう側，端の方に大きい値が集まってしまうような
##



import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# --- フィルタリング条件の設定 ---
filter_conditions = {
    "x_min": -2.0,  # x座標の最小値
    "x_max": 5.0,   # x座標の最大値
    "y_min": -2.0,  # y座標の最小値
    "y_max": 3.0,   # y座標の最大値
    "z_min": -5.0,  # z座標の最小値
    "z_max": 5.0    # z座標の最大値
}

# --- データ準備 ---
filename = '2024.12.18.02.13.38.csv'  # CSVファイル名
with open(filename) as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data = np.asarray(data, dtype=np.float32)
if data.size == 0:
    raise ValueError("Data is empty.")

# フィルタリング処理
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

# --- 特徴量の選択（反射強度のみを使用） ---
features = filtered_data[:, 3].reshape(-1, 1)  # intensity のみ

# --- データの標準化 ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)



# --- サンプリング処理（追加部分） ---
max_samples = 30000  # SVM に使用する最大サンプル数
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

# --- 正常・異常データの分類 ---
normal_indices = np.where(predictions == 1)[0]  # 正常データのインデックス
anomalous_indices = np.where(predictions == -1)[0]  # 異常データのインデックス

normal_data = filtered_data[normal_indices, :3]  # 正常データの座標
anomalous_data = filtered_data[anomalous_indices, :3]  # 異常データの座標

# --- データ数の表示 ---
print(f"Number of points displayed:")
print(f"  Normal points: {len(normal_data)}")
print(f"  Anomalous points: {len(anomalous_data)}")

# --- データの可視化 ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(normal_data[:, 0], normal_data[:, 1], normal_data[:, 2], label="Normal", color="blue", s=5)
ax.scatter(anomalous_data[:, 0], anomalous_data[:, 1], anomalous_data[:, 2], label="Anomalous", color="red", s=5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend()
plt.title("3D Detection of Anomalies Using Intensity Only")
plt.show()
