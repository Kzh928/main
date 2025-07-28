#CSV読み込みと初期回転
#フィルタリング（壁面を切り出すための初期絞り込み）
#PCAによる主成分整列
#回転後の点群を可視化（補正済反射強度を使用）
#PCA後の鉄筋領域を再フィルタリング
#フィルタリング後の点群を表示
#壁面除去処理：Z軸奥側の壁面を除く
#残った点群のみ表示（壁面除去後）
#除去された壁面と残った点群を色分けして同時に表示
#残った点群を反射強度で色分けして表示
import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

# === 距離補正（回転後の距離で補正） ===
distances = np.linalg.norm(points_rotated, axis=1)
reflectance_corrected = reflectance / (distances + 1e-6)  # 0除算防止の微小項

# === 2. フィルタリング（壁面を切り出すための初期絞り込み）===
x_range = (3000, 4000)
y_range = (-500, 700)
z_range = (-1500, -500)
filtered_points, filtered_reflectance = filter_roi(points_rotated, reflectance_corrected, x_range, y_range, z_range)

if filtered_points.shape[0] == 0:
    print("No point cloud after initial filtering.")
    exit()

# === 3. PCAによる主成分整列 ===
pca = PCA(n_components=3)
pca.fit(filtered_points)
points_pca = pca.transform(filtered_points)

# === 4. 回転後の点群を可視化（補正済反射強度を使用） ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_pca)
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

# === 5. PCA後の鉄筋領域を再フィルタリング ===
x_range_pca = (-500, 4000)
y_range_pca = (-350, 50)
z_range_pca = (-70, 500)
filtered_pca_points, filtered_pca_reflect = filter_roi(points_pca, filtered_reflectance,
                                                       x_range=x_range_pca,
                                                       y_range=y_range_pca,
                                                       z_range=z_range_pca)

# === 6. フィルタリング後の点群を表示 ===
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
    # vis2.add_geometry(axis)
    vis2.run()
    vis2.destroy_window()
else:
    print("No points after PCA-space filtering.")


## === 7. 壁面除去処理：Z軸奥側の壁面を除く（例：Z < 100 の点を除外） ===
# → 鉄筋を含む領域だけ残す（Z > 100）など条件を変更可
wall_threshold_z = -5
mask_remain = filtered_pca_points[:, 2] > wall_threshold_z  # 鉄筋＋背面のみ
mask_removed = ~mask_remain  # 除去された壁面（例：Zが小さい）

points_remain = filtered_pca_points[mask_remain]
reflect_remain = filtered_pca_reflect[mask_remain]
points_removed = filtered_pca_points[mask_removed]

# === 8. 残った点群のみ表示（壁面除去後） ===
pcd_remain = o3d.geometry.PointCloud()
pcd_remain.points = o3d.utility.Vector3dVector(points_remain)

norm_reflect_remain = (reflect_remain - reflect_remain.min()) / (np.ptp(reflect_remain) + 1e-6)
gray_colors_remain = np.stack([norm_reflect_remain] * 3, axis=1)
pcd_remain.colors = o3d.utility.Vector3dVector(gray_colors_remain)

print("Showing only remaining points (wall removed)")
vis3 = o3d.visualization.Visualizer()
vis3.create_window(window_name="After Wall Removal", width=1600, height=1200)
vis3.add_geometry(pcd_remain)
vis3.run()
vis3.destroy_window()

# === 9. 除去された壁面と残った点群を色分けして同時に表示 ===
pcd_removed = o3d.geometry.PointCloud()
pcd_removed.points = o3d.utility.Vector3dVector(points_removed)
removed_color = np.tile(np.array([[0.6, 0.6, 0.6]]), (points_removed.shape[0], 1))  # 灰色
pcd_removed.colors = o3d.utility.Vector3dVector(removed_color)

colored_remain = o3d.geometry.PointCloud()
colored_remain.points = o3d.utility.Vector3dVector(points_remain)
remain_color = np.tile(np.array([[1.0, 0.0, 0.0]]), (points_remain.shape[0], 1))  # 赤色
colored_remain.colors = o3d.utility.Vector3dVector(remain_color)

print("Showing both: gray (removed wall) + red (remaining)")
vis4 = o3d.visualization.Visualizer()
vis4.create_window(window_name="Wall Removed vs Remain", width=1600, height=1200)
vis4.add_geometry(pcd_removed)
vis4.add_geometry(colored_remain)
vis4.run()
vis4.destroy_window()

# === 10. 残った点群を反射強度で色分けして表示 ===
pcd_color_by_ref = o3d.geometry.PointCloud()
pcd_color_by_ref.points = o3d.utility.Vector3dVector(points_remain)
# カラーマップ変換：viridis（matplotlibのカラーマップを使う）
cmap = cm.get_cmap('viridis')
colors_cmap = cmap(norm_reflect_remain)[:, :3]  # RGBA→RGB
pcd_color_by_ref.colors = o3d.utility.Vector3dVector(colors_cmap)

print("Showing remaining points colored by corrected reflectance")
vis5 = o3d.visualization.Visualizer()
vis5.create_window(window_name="Reflectance Colormap", width=1600, height=1200)
vis5.add_geometry(pcd_color_by_ref)
vis5.run()
vis5.destroy_window()
