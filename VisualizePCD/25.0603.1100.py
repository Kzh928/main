#Y軸32度回転：センサの傾きを補正

#ROIフィルタリング：壁面部分だけ抽出（初期的な絞り込み）

#PCAによる主成分整列：壁面を座標軸に正対

#鉄筋領域の再フィルタリング：必要領域のみ抽出

#任意回転補正（微調整）→ 主にY軸方向

#再表示：補正後の点群をグレースケールで表示

#Z軸方向の壁面除去：Z < -5 を除き、鉄筋＋背面だけ残す

#壁除去後の表示（鉄筋を含む領域）

#除去された壁面 vs 残りの領域を色分け表示：灰色 vs 赤色
#
#反射強度で色わけ
#正規化とかなんとか
#
#6/3　発表で没に#

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
#vis.add_geometry(axis)
vis.run()
vis.destroy_window()

# === 5. PCA後の鉄筋領域を再フィルタリング ===
x_range_pca = (-500, 4000)
y_range_pca = (-350, 50)
z_range_pca = (-70, 500)
filtered_pca_points, filtered_pca_reflect = filter_roi(points_pca, filtered_reflectance,
                                                       x_range=x_range_pca,
                                                       y_range=y_range_pca,
                                                       z_range=z_range_pca)

# === 回転角度（度）を各軸ごとに指定 ===
angle_x = 0.0  # X軸まわりの回転角度
angle_y =-5.0  # Y軸まわりの回転角度
angle_z =0.0  # Z軸まわりの回転角度

# === ラジアンに変換 ===
theta_x = np.radians(angle_x)
theta_y = np.radians(angle_y)
theta_z = np.radians(angle_z)

# === 各軸の回転行列を作成 ===
Rx = np.array([
    [1, 0, 0],
    [0, np.cos(theta_x), -np.sin(theta_x)],
    [0, np.sin(theta_x),  np.cos(theta_x)]
])
Ry = np.array([
    [np.cos(theta_y), 0, np.sin(theta_y)],
    [0, 1, 0],
    [-np.sin(theta_y), 0, np.cos(theta_y)]
])
Rz = np.array([
    [np.cos(theta_z), -np.sin(theta_z), 0],
    [np.sin(theta_z),  np.cos(theta_z), 0],
    [0, 0, 1]
])

# === 合成回転行列（順番はZ→Y→X）===
# 回転順序は用途に応じて変更可能
R = Rz @ Ry @ Rx

# === 回転適用 ===
filtered_pca_points = (R @ filtered_pca_points.T).T


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


## === 7. 壁面除去処理：Z軸奥側の壁面を除く（例：Z < 100 の点を除外） === -11がちょうどいい
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

# # === 10. 残った点群を反射強度で色分けして表示 ===
# pcd_color_by_ref = o3d.geometry.PointCloud()
# pcd_color_by_ref.points = o3d.utility.Vector3dVector(points_remain)
# # カラーマップ変換：viridis（matplotlibのカラーマップを使う）
# cmap = cm.get_cmap('viridis')
# colors_cmap = cmap(norm_reflect_remain)[:, :3]  # RGBA→RGB
# pcd_color_by_ref.colors = o3d.utility.Vector3dVector(colors_cmap)

# print("Showing remaining points colored by corrected reflectance")
# vis5 = o3d.visualization.Visualizer()
# vis5.create_window(window_name="Reflectance Colormap", width=1600, height=1200)
# vis5.add_geometry(pcd_color_by_ref)
# vis5.run()
# vis5.destroy_window()

# #10a   5～95パーセンタイルで正規化（外れ値を除去してコントラスト強調）

# # === 10. 残った点群を反射強度で色分けして表示（パーセンタイル正規化） ===
# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm

# pcd_color_by_ref = o3d.geometry.PointCloud()
# pcd_color_by_ref.points = o3d.utility.Vector3dVector(points_remain)

# # 正規化（5〜95パーセンタイル）＋カラーマップ
# vmin, vmax = np.percentile(reflect_remain_corrected, [5, 95])
# norm = plt.Normalize(vmin=vmin, vmax=vmax)
# cmap = cm.get_cmap('viridis')
# colors_cmap = cmap(norm(reflect_remain_corrected))[:, :3]
# pcd_color_by_ref.colors = o3d.utility.Vector3dVector(colors_cmap)

# print("Showing remaining points colored by corrected reflectance (percentile normalization)")
# vis5 = o3d.visualization.Visualizer()
# vis5.create_window(window_name="Reflectance Colormap (Percentile)", width=1600, height=1200)
# vis5.add_geometry(pcd_color_by_ref)
# vis5.run()
# vis5.destroy_window()



# #10b カラーマップを 'turbo' に変更（
# # === 10. 残った点群を反射強度で色分けして表示（カラーマップ変更：turbo） ===
# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm

# pcd_color_by_ref = o3d.geometry.PointCloud()
# pcd_color_by_ref.points = o3d.utility.Vector3dVector(points_remain)

# # 正規化（0〜1）＋カラーマップ変更
# norm = plt.Normalize(vmin=np.min(reflect_remain_corrected), vmax=np.max(reflect_remain_corrected))
# cmap = cm.get_cmap('turbo')  # 'plasma' や 'inferno' でも可
# colors_cmap = cmap(norm(reflect_remain_corrected))[:, :3]
# pcd_color_by_ref.colors = o3d.utility.Vector3dVector(colors_cmap)

# print("Showing remaining points colored by corrected reflectance (turbo colormap)")
# vis5 = o3d.visualization.Visualizer()
# vis5.create_window(window_name="Reflectance Colormap (Turbo)", width=1600, height=1200)
# vis5.add_geometry(pcd_color_by_ref)
# vis5.run()
# vis5.destroy_window()

# #10c ヒストグラム均等化で色のばらつきを強調（擬似CLAHE）

# # === 10. 残った点群を反射強度で色分けして表示（ヒストグラム均等化） ===
# import open3d as o3d
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from sklearn.preprocessing import MinMaxScaler
# from skimage import exposure

# pcd_color_by_ref = o3d.geometry.PointCloud()
# pcd_color_by_ref.points = o3d.utility.Vector3dVector(points_remain)

# # 0〜1にスケーリング→ヒストグラム均等化
# scaled_ref = MinMaxScaler().fit_transform(reflect_remain_corrected.reshape(-1, 1)).flatten()
# eq_ref = exposure.equalize_hist(scaled_ref)

# # カラーマップ：turbo
# cmap = cm.get_cmap('turbo')
# colors_cmap = cmap(eq_ref)[:, :3]
# pcd_color_by_ref.colors = o3d.utility.Vector3dVector(colors_cmap)

# print("Showing remaining points colored by equalized reflectance (histogram equalization)")
# vis5 = o3d.visualization.Visualizer()
# vis5.create_window(window_name="Reflectance Colormap (Equalized)", width=1600, height=1200)
# vis5.add_geometry(pcd_color_by_ref)
# vis5.run()
# vis5.destroy_window()


#1-d
#
#10db
#
##

# === 10. 残った点群を反射強度で色分けして表示（パーセンタイル＋均等化＋カラーマップ） ===
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from skimage import exposure

pcd_color_by_ref = o3d.geometry.PointCloud()
pcd_color_by_ref.points = o3d.utility.Vector3dVector(points_remain)

# Step 1: パーセンタイルで外れ値除去（5～95%）→ クリップ
vmin, vmax = np.percentile(reflect_remain, [5, 95])
reflect_clipped = np.clip(reflect_remain, vmin, vmax)

# Step 2: 0～1 にスケーリング
scaled_ref = MinMaxScaler().fit_transform(reflect_clipped.reshape(-1, 1)).flatten()

# Step 3: ヒストグラム均等化
equalized_ref = exposure.equalize_hist(scaled_ref)

# Step 4: カラーマップ適用（'turbo'）
cmap = cm.get_cmap('turbo')
colors_cmap = cmap(equalized_ref)[:, :3]
pcd_color_by_ref.colors = o3d.utility.Vector3dVector(colors_cmap)

print("Showing remaining points colored by combined reflectance processing")
vis5 = o3d.visualization.Visualizer()
vis5.create_window(window_name="Reflectance Colormap (Combined)", width=1600, height=1200)
vis5.add_geometry(pcd_color_by_ref)
vis5.run()
vis5.destroy_window()


