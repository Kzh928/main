#0012からランダムサンプリングで動作軽く
#90%減らして動かしてる
#１２のデータ違い
#0203.21.11.13
#
##

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import csv
import matplotlib.colors as mcolors

# --- フィルタリング条件の設定 ---
filter_conditions = {
    "x": (-2.0, 5.0),
    "y": (-2.0, 3.0),
    "z": (-5.0, 5.0)
}

# --- データ準備 ---
filename = '0203.21.11.13.csv'
with open(filename) as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data = np.asarray(data, dtype=np.float32)
if data.size == 0:
    raise ValueError("Data is empty.")

# --- フィルタリング ---
filtered_data = data
for axis, (min_val, max_val) in filter_conditions.items():
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    if min_val is not None:
        filtered_data = filtered_data[filtered_data[:, axis_index] >= min_val]
    if max_val is not None:
        filtered_data = filtered_data[filtered_data[:, axis_index] <= max_val]

if filtered_data.size == 0:
    raise ValueError("Filtered data is empty.")

# --- ランダムサンプリング（10%）---
sampling_rate = 0.1  # 10%に削減（軽くしたいなら0.05や0.02に変更）
num_samples = int(len(filtered_data) * sampling_rate)
sampled_indices = np.random.choice(len(filtered_data), num_samples, replace=False)
filtered_data = filtered_data[sampled_indices]

# --- 特徴量の選択（反射強度） ---
features = filtered_data[:, 3]  # 反射強度 (Intensity)

# --- 標準化 ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features.reshape(-1, 1))

# --- One-Class SVM モデル ---
clf = OneClassSVM(nu=0.001, kernel='rbf', gamma='scale')
clf.fit(scaled_features)
predictions = clf.predict(scaled_features)

# --- 異常データの抽出 ---
normal_indices = np.where(predictions == 1)[0]
anomalous_indices = np.where(predictions == -1)[0]

normal_data = filtered_data[normal_indices, :3]
anomalous_data = filtered_data[anomalous_indices, :3]
normal_intensity = features[normal_indices]
anomalous_intensity = features[anomalous_indices]

# --- 反射強度の色分け（10区分） ---
num_bins = 10
intensity_bins = np.linspace(20, 100, num_bins + 1)

# はっきりした色
custom_colors = ["#FF0000", "#FF7F00", "#FFFF00", "#7FFF00", "#00FF00",
                 "#00FFFF", "#0000FF", "#8B00FF", "#FF1493", "#800000"]
cmap = mcolors.ListedColormap(custom_colors)

def get_intensity_color(intensity):
    bin_index = np.digitize(intensity, intensity_bins) - 1
    return custom_colors[min(bin_index, num_bins - 1)]

normal_colors = [get_intensity_color(i) for i in normal_intensity]
anomalous_colors = [get_intensity_color(i) for i in anomalous_intensity]

# --- 3Dプロット ---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(normal_data[:, 0], normal_data[:, 1], normal_data[:, 2], label="Normal", c=normal_colors, s=5)
ax.scatter(anomalous_data[:, 0], anomalous_data[:, 1], anomalous_data[:, 2], label="Anomalous", c=anomalous_colors, s=5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# --- YZ平面を正面から見る ---
ax.view_init(elev=0, azim=0)

plt.title("3D Anomaly Detection (Color by Intensity)")
plt.legend()

# --- カラーバーを表示 ---
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=256))
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.1)
cbar.set_label("Reflection Intensity")

plt.show()



# --- 反射強度ヒストグラム ---
plt.figure(figsize=(8, 5))
plt.hist(features, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("Reflection Intensity")
plt.ylabel("Frequency")
plt.xlim(0,256)
plt.title("Histogram of Reflection Intensity")
plt.show()


data = []
for i in range(len(features)):
    minimum = 20
    maximum = 100
    if minimum <= features[i] and features[i] <= maximum:
        data.append(True)
    else:
        data.append(False)


# --- 3Dプロット ---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')


normal_intensity = features[data]

def get_intensity_color(intensity):
    bin_index = np.digitize(intensity, intensity_bins) - 1
    return custom_colors[min(bin_index, num_bins - 1)]

normal_colors = [get_intensity_color(i) for i in normal_intensity]



ax.scatter(filtered_data[data, 0], filtered_data[data, 1], filtered_data[data, 2], label="data", c=normal_colors, s=5)


ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# --- YZ平面を正面から見る ---
ax.view_init(elev=0, azim=0)

plt.title("3D Anomaly Detection (Color by Intensity)")
plt.legend()

plt.show()

print(np.array(features).shape)
print(features)
data = [0 for i in range(256)]
for i in range(len(features)):
    data[int(features)] += 1
print(data)
