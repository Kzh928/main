#
#
#鉄筋周りを回収
#PCAかけてから、さらにフィルタリングをかけることで鉄筋だけにする
#上側の鉄筋だけ
#
#手動でフィルタリングした
#
#点群尾反射強度確認　ヒストグラムからのカラーマップ
#
##

import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.decomposition import PCA

# === 関数：ROIフィルタリング ===
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

# --- 初期回転（Y軸まわりに32度） ---
theta = np.radians(32.0)
rot_matrix = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])
points_rotated = (rot_matrix @ points.T).T

# === 2. フィルタリング（壁面を切り出すための初期絞り込み）===
x_range = (3000, 4000)
y_range = (-500, 700)
z_range = (-1500, -500)
filtered_points, filtered_reflectance = filter_roi(points_rotated, reflectance, x_range, y_range, z_range)

if filtered_points.shape[0] == 0:
    print("No point cloud after initial filtering.")
    exit()

# === 3. PCAによる主成分整列 ===
pca = PCA(n_components=3)
pca.fit(filtered_points)
points_pca = pca.transform(filtered_points)  # この時点で壁面は整列された状態

# === 4. 回転後の点群を可視化 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_pca)

# 反射強度をグレースケールに変換
norm_reflect = (filtered_reflectance - filtered_reflectance.min()) / (np.ptp(filtered_reflectance) + 1e-6)
gray_colors = np.stack([norm_reflect] * 3, axis=1)
pcd.colors = o3d.utility.Vector3dVector(gray_colors)

print("Showing PCA-aligned point cloud")
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="PCA Aligned", width=1600, height=1200)
vis.add_geometry(pcd)
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
vis.add_geometry(axis)
# vis.run()
vis.destroy_window()

# === 5. PCA後の点群をフィルタリングできる準備 ===
# 以下のようにして数値を入れれば自由に切れる
# 例）x_range_pca = (-1000, 1000) などを指定
x_range_pca = (-500, 4000)
y_range_pca = (-350, 50)
z_range_pca = (-70, 500)

filtered_pca_points, filtered_pca_reflect = filter_roi(points_pca, filtered_reflectance,
                                                       x_range=x_range_pca,
                                                       y_range=y_range_pca,
                                                       z_range=z_range_pca)

# フィルタリング後の点群を表示（必要に応じてコメントアウト解除）
if filtered_pca_points.shape[0] > 0:
    pcd_cut = o3d.geometry.PointCloud()
    pcd_cut.points = o3d.utility.Vector3dVector(filtered_pca_points)

    norm_reflect_cut = (filtered_pca_reflect - filtered_pca_reflect.min()) / (np.ptp(filtered_pca_reflect) + 1e-6)
    gray_colors_cut = np.stack([norm_reflect_cut] * 3, axis=1)
    pcd_cut.colors = o3d.utility.Vector3dVector(gray_colors_cut)
    
    print("Showing filtered PCA-aligned point cloud")
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="Filtered PCA Points", width=1600, height=1200)
    vis2.add_geometry(pcd_cut)
    #vis2.add_geometry(axis)
    vis2.run()
    vis2.destroy_window()
else:
    print("No points after PCA-space filtering.")
import matplotlib.pyplot as plt

# === 6. 反射強度のヒストグラム表示 ===
if filtered_pca_reflect.size > 0:
    plt.figure(figsize=(8, 5))
    plt.hist(filtered_pca_reflect, bins=100, color='red', alpha=0.7, edgecolor='black')
    plt.title("Reflectance Histogram of Filtered PCA Points")
    plt.xlabel("Reflectance")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No reflectance data to plot.")


import matplotlib.cm as cm

# === 7. 反射強度をカラーマップで色分けして表示 ===
if filtered_pca_points.shape[0] > 0:
    # 反射強度を0～1に正規化
    reflect_norm = (filtered_pca_reflect - filtered_pca_reflect.min()) / (np.ptp(filtered_pca_reflect) + 1e-6)
    
    # カラーマップを適用（ここでは "jet" を使用，"viridis", "plasma", "inferno" なども可）
    colormap = cm.get_cmap("jet")
    colors_mapped = colormap(reflect_norm)[:, :3]  # RGBA → RGB（最後のAは不要）

    # 点群作成と色設定
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(filtered_pca_points)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors_mapped)

    # 表示
    print("Showing reflectance-based colored point cloud")
    vis3 = o3d.visualization.Visualizer()
    vis3.create_window(window_name="Reflectance Colored", width=1600, height=1200)
    vis3.add_geometry(pcd_colored)
    vis3.run()
    vis3.destroy_window()
else:
    print("No points available for reflectance-based coloring.")
