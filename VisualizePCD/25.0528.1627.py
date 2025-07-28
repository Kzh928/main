#
#ヒストグラムで色分けするんじゃなくて
#もう一度３Dプロットを表示させたい
#順序としては
#3Dプロット（反射強度による色分け）
#ヒストグラム（通常）
#３Dプロット（頻度による色分け）
#フィルタリングが座標じゃなくなってる!!!!!!  使用時変更必要　いったん放置

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import csv
import matplotlib.cm as cm

# --- データ準備 ---
filename = '2024.12.18.02.13.38.csv'
with open(filename) as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data = np.asarray(data, dtype=np.float32)

# --- フィルタリング（0 < 反射強度 < 100）---
filtered_data = data[(data[:, 3] > 0) & (data[:, 3] < 100)]
xyz = filtered_data[:, :3]
intensity = filtered_data[:, 3]

# --- 3Dプロット（反射強度による色分け）---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=intensity, cmap='jet', s=5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=0, azim=-180)  # 正面から見たYZ平面
plt.title("3D Scatter Plot (Colored by Intensity)")
plt.colorbar(sc, label="Reflection Intensity")
plt.show()

# --- ヒストグラム（通常）---
plt.figure(figsize=(8, 5))
plt.hist(intensity, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("Reflection Intensity")
plt.ylabel("Frequency")
plt.title("Histogram of Reflection Intensity")
plt.show()

# --- 反射強度の頻度に基づく色分け ---
unique, counts = np.unique(intensity, return_counts=True)
freq_dict = dict(zip(unique, counts))
freq_values = np.array([freq_dict[val] for val in intensity])

# --- 3Dプロット（頻度による色分け）---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=freq_values, cmap='plasma', s=5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.view_init(elev=0, azim=-180)  # 正面から見たYZ平面
plt.title("3D Scatter Plot (Colored by Frequency)")
plt.colorbar(sc, label="Frequency of Reflection Intensity")
plt.show()
