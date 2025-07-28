#
#
#5層を分けて出力
#回転行列の取得の仕方が違う、（05302118）
#こっちはフィルタリング後の点群から得られた回転行列
#あっちは壁面の点だけに絞った点群から
#
##


import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# === ROIフィルタ関数 ===
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
points = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
reflectance = df["reflectance"].to_numpy(dtype=np.float32)

# Y軸回転（例：32度）
theta = np.radians(32.0)
RotationMat = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])
points_rotated = (RotationMat @ points.T).T

# === 2. ROIフィルタリング ===
x_range = (3000, 4000)
y_range = (-500, 700)
z_range = (-1500, -500)
filtered_points, filtered_reflectance = filter_roi(points_rotated, reflectance, x_range, y_range, z_range)

if filtered_points.shape[0] == 0:
    print("No point cloud after ROI filtering.")
    exit()

# === 3. PCAで壁面抽出 → 回転行列取得（※これは一時的な処理） ===
pca = PCA(n_components=3)
pca.fit(filtered_points)
normal_vector = pca.components_[2]
centered = filtered_points - pca.mean_
distances = np.abs(centered @ normal_vector)

# 壁面っぽい点を使って回転行列取得
threshold = 20.0
mask_wall = distances < threshold
wall_points = filtered_points[mask_wall]

if wall_points.shape[0] == 0:
    print("No wall-like points found for PCA.")
    exit()

R_pca = pca.components_

# === 4. ROIフィルタ済みの全点群を回転（傾き補正） ===
aligned_points = (filtered_points - pca.mean_) @ R_pca.T
aligned_reflectance = filtered_reflectance  # Reflectanceは回転不要

# === 5. Z軸方向にソート＆5層分割 ===
z_coords = aligned_points[:, 2]
sorted_indices = np.argsort(z_coords)
sorted_points = aligned_points[sorted_indices]
sorted_reflect = aligned_reflectance[sorted_indices]

N = len(sorted_points)
layer_size = N // 5
layers = []
reflect_layers = []

for i in range(5):
    start = i * layer_size
    end = (i + 1) * layer_size if i < 4 else N  # 最後の層に余りを含める
    layers.append(sorted_points[start:end])
    reflect_layers.append(sorted_reflect[start:end])

# === 6. 各層を表示＋ヒストグラム ===
for i, (layer_points, layer_reflect) in enumerate(zip(layers, reflect_layers)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(layer_points)

    norm_reflect = (layer_reflect - layer_reflect.min()) / (np.ptp(layer_reflect) + 1e-6)
    gray_colors = np.stack([norm_reflect] * 3, axis=1)
    pcd.colors = o3d.utility.Vector3dVector(gray_colors)

    print(f"Showing Layer {i+1}")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Layer {i+1}", width=1600, height=1200)
    vis.add_geometry(pcd)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=500))
    #vis.run()
    vis.destroy_window()

    plt.figure()
    plt.hist(layer_reflect, bins=50, color='gray')
    plt.title(f"Layer {i+1} Reflectance Histogram")
    plt.xlabel("Reflectance")
    plt.ylabel("Count")
    plt.grid(True)
    #plt.show()

# === 7. 全層を色分けして重ねて表示 ===
colors_per_layer = [
    [1, 0, 0],   # 赤
    [0, 1, 0],   # 緑
    [0, 0, 1],   # 青
    [1, 1, 0],   # 黄
    [1, 0, 1],   # マゼンタ
]

pcd_list = []
for i, layer_points in enumerate(layers):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(layer_points)
    color = np.tile(colors_per_layer[i], (layer_points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(color)
    pcd_list.append(pcd)

print("Showing all 5 layers together with different colors")
vis_all = o3d.visualization.Visualizer()
vis_all.create_window(window_name="All Layers", width=1600, height=1200)
for pcd in pcd_list:
    vis_all.add_geometry(pcd)
vis_all.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=500))
vis_all.run()
vis_all.destroy_window()
