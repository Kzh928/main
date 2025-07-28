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
#層の数変えるなら数値と色の数いじる#



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
layer_size = N // 5
layers = []
reflect_layers = []

for i in range(5):
    start = i * layer_size
    end = (i + 1) * layer_size if i < 4 else N
    layers.append(sorted_points[start:end])
    reflect_layers.append(sorted_reflect[start:end])

# === 5. 各層の点群＋ヒストグラムを表示（Visualizer使用） ===
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

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
    vis.add_geometry(axis)

    vis.run()
    vis.destroy_window()

    plt.figure()
    plt.hist(layer_reflect, bins=50, color='gray')
    plt.title(f"Layer {i+1} Reflectance Histogram")
    plt.xlabel("Reflectance")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

# === 6. 全層を色分けして重ねて表示（Visualizer使用） ===
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

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
vis_all.add_geometry(axis)

vis_all.run()
vis_all.destroy_window()
