#
#
#壁面とりだして表示しただけ
#
#使いやすいように#

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# === 関数定義：ROIフィルタリング ===
def filter_roi(points, reflectance, x_range=None, y_range=None, z_range=None):
    mask = np.ones(len(points), dtype=bool)
    if x_range:
        mask &= (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
    if y_range:
        mask &= (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    if z_range:
        mask &= (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    return points[mask], reflectance[mask]

# === 1. CSV読み込みと初期回転 ===
file_path = "mid360_XYZ_Reflect_data_03-28-13-53-11.csv"
df = pd.read_csv(file_path, header=None, names=["x", "y", "z", "reflectance"], encoding="shift_jis")
print(df.head())
print(df.shape)

points = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
reflectance = df["reflectance"].to_numpy(dtype=np.float32)

# --- 初期回転（Y軸まわりに32度） ---
theta = np.radians(32.0)
RotationMat = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])
points_rotated = (RotationMat @ points.T).T

# === 2. フィルタリング ===
x_range = (3000, 4000)
y_range = (-500, 700)
z_range = (-1500, -500)

filtered_points, filtered_reflectance = filter_roi(points_rotated, reflectance, x_range, y_range, z_range)
print(f"points:{filtered_points.shape[0]}")

if filtered_points.shape[0] == 0:
    print("no point cloud")
    exit()

# === 3. フィルタリング後の点群をOpen3Dで表示 ===
# 反射強度を0～1に正規化して色に変換
reflectance_normalized = (filtered_reflectance - np.min(filtered_reflectance)) / (np.ptp(filtered_reflectance) + 1e-8)

colors = plt.cm.plasma(reflectance_normalized)[:, :3]  # RGBのみ（アルファ除く）

# Open3D PointCloud作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# ビューアで表示
o3d.visualization.draw_geometries([pcd], window_name="Filtered PointCloud with Reflectance")


# === 3.5 PCAを使った壁面抽出 ===
pca = PCA(n_components=3)
pca.fit(filtered_points)

normal_vector = pca.components_[2]
centered = filtered_points - pca.mean_
distances = np.abs(centered @ normal_vector)

# 壁面だけを抽出
threshold = 25.0  # 調整しやすく
mask_plane = distances < threshold

wall_points = filtered_points[mask_plane]
wall_reflectance = filtered_reflectance[mask_plane]

reflectance_normalized = (wall_reflectance - np.min(wall_reflectance)) / (np.ptp(wall_reflectance) + 1e-8)
colors = plt.cm.plasma(reflectance_normalized)[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(wall_points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd], window_name="Wall PointCloud after PCA Plane Filtering")
