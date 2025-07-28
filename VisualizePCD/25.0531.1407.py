#
#
#pca
#フィルタリング、回転
#Z軸で５分割（２０％）
#各層の点群とヒストグラム（座標）
#最後に５層を重ね合わせた点群
#
#表示窓の大きさ変更
#
#上位20％の反射強度のものを層ごとに回収
#全部のまとめを最後に
#
#
#各層の点群、各層のヒストグラム（20%反射強度）、全体の色付き、２０％だけの集合#



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

# === 3. PCAによる主成分回転 ===
pca = PCA(n_components=3)
pca.fit(filtered_points)
points_pca = pca.transform(filtered_points)
z_pca = points_pca[:, 2]

# === 4. PCA後Z軸に沿って5層分割 ===
sorted_indices = np.argsort(z_pca)
sorted_points = points_pca[sorted_indices]
sorted_reflect = filtered_reflectance[sorted_indices]

N = len(sorted_points)
layer_size = N // 6
layers = []
reflect_layers = []

for i in range(6):
    start = i * layer_size
    end = (i + 1) * layer_size if i < 5 else N
    layers.append(sorted_points[start:end])
    reflect_layers.append(sorted_reflect[start:end])

# # === 5. 各層の点群＋ヒストグラムを表示（Visualizer使用） ===
# for i, (layer_points, layer_reflect) in enumerate(zip(layers, reflect_layers)):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(layer_points)

#     norm_reflect = (layer_reflect - layer_reflect.min()) / (np.ptp(layer_reflect) + 1e-6)
#     gray_colors = np.stack([norm_reflect] * 3, axis=1)
#     pcd.colors = o3d.utility.Vector3dVector(gray_colors)

#     print(f"Showing Layer {i+1}")

#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=f"Layer {i+1}", width=1600, height=1200)
#     vis.add_geometry(pcd)

#     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
#     vis.add_geometry(axis)

#     #vis.run()
#     vis.destroy_window()

#     plt.figure() 
#     plt.hist(layer_reflect, bins=50, color='gray')
#     plt.title(f"Layer {i+1} Reflectance Histogram")
#     plt.xlabel("Reflectance")
#     plt.ylabel("Count")
#     plt.grid(True)
#     #plt.show()

# === 5. 各層の点群＋ヒストグラムを表示（高反射強度のみ抽出） ===
percentile = 80  # 上位20%を抽出
for i, (layer_points, layer_reflect) in enumerate(zip(layers, reflect_layers)):
    threshold = np.percentile(layer_reflect, percentile)
    mask = layer_reflect >= threshold

    high_reflect_points = layer_points[mask]
    high_reflect = layer_reflect[mask]

    if high_reflect_points.shape[0] == 0:
        print(f"Layer {i+1}: no high-reflectance points.")
        continue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(high_reflect_points)

    # 強度を正規化してグレースケールで色付け
    norm_reflect = (high_reflect - high_reflect.min()) / (np.ptp(high_reflect) + 1e-6)
    gray_colors = np.stack([norm_reflect] * 3, axis=1)
    pcd.colors = o3d.utility.Vector3dVector(gray_colors)

    print(f"Showing Layer {i+1} (high reflectance only)")
    # o3d.visualization.draw_geometries([pcd])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定
    # ポイントクラウドと座標軸を追加
    vis.add_geometry(pcd)
    
    # 描画
    #vis.run()
    vis.destroy_window()


    # ヒストグラムは層全体に対して表示（フィルタ前）
    plt.figure()
    plt.hist(layer_reflect, bins=50, color='gray')
    plt.axvline(threshold, color='red', linestyle='--', label=f"{percentile}th percentile")
    plt.title(f"Layer {i+1} Reflectance Histogram")
    plt.xlabel("Reflectance")
    plt.ylabel("Count")
    plt.grid(True)
    plt.legend()
    # plt.show()


# === 6. 全層を色分けして重ねて表示（Visualizer使用） ===
colors_per_layer = [
    [1, 0, 0],   # 赤
    [0, 1, 0],   # 緑
    [0, 0, 1],   # 青
    [1, 1, 0],   # 黄
    [1, 0, 1],   # マゼンタ
    [0, 0, 0],
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

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
vis_all.add_geometry(axis)

# vis_all.run()
vis_all.destroy_window()

# # --- 各層の高反射点（上位20％）をすべてまとめる ---
# all_high_reflect_points = []

# for i in range(6):
#     start = i * layer_size
#     end = (i + 1) * layer_size if i < 5 else N
#     layer_points = sorted_points[start:end]
#     layer_reflect = sorted_reflect[start:end]

#     threshold = np.percentile(layer_reflect, percentile)
#     mask = layer_reflect >= threshold
#     high_reflect_points = layer_points[mask]

#     print(f"Layer {i+1}: threshold={threshold:.2f}, selected={high_reflect_points.shape[0]}")
#     all_high_reflect_points.append(high_reflect_points)

# # --- 結合（非空チェック付き） ---
# non_empty_layers = [layer for layer in all_high_reflect_points if layer.shape[0] > 0]

# if not non_empty_layers:
#     print("No high reflectance points found in any layer.")
#     exit()

# all_high_reflect_points = np.vstack(non_empty_layers)
# print(f"Total high reflectance points: {all_high_reflect_points.shape[0]}")

# # --- 可視化 ---
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(all_high_reflect_points)
# colors = np.tile([0, 0, 0], (all_high_reflect_points.shape[0], 1))
# pcd.colors = o3d.utility.Vector3dVector(colors)

# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="High Reflectance Points (All Layers)", width=1600, height=1200)
# vis.add_geometry(pcd)
# axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
# vis.add_geometry(axis)
# vis.run()
# vis.destroy_window()


# --- 各層の高反射点（上位20％）をすべてまとめる（色付き） ---
all_high_reflect_points = []
all_high_reflect_colors = []

for i in range(6):
    start = i * layer_size
    end = (i + 1) * layer_size if i < 5 else N
    layer_points = sorted_points[start:end]
    layer_reflect = sorted_reflect[start:end]

    percentile = 80
    threshold = np.percentile(layer_reflect, percentile)
    mask = layer_reflect >= threshold

    high_reflect_points = layer_points[mask]

    if high_reflect_points.shape[0] > 0:
        all_high_reflect_points.append(high_reflect_points)

        # 層ごとの色を繰り返す
        color = np.tile(colors_per_layer[i], (high_reflect_points.shape[0], 1))
        all_high_reflect_colors.append(color)

# --- 点群と色を結合 ---
if all_high_reflect_points:
    all_high_reflect_points = np.vstack(all_high_reflect_points)
    all_high_reflect_colors = np.vstack(all_high_reflect_colors)

    # --- 可視化 ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_high_reflect_points)
    pcd.colors = o3d.utility.Vector3dVector(all_high_reflect_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="High Reflectance Points (All Layers with Color)", width=1600, height=1200)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
else:
    print("No high-reflectance points found in any layer.")



