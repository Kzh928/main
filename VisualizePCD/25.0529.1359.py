#
#距離補正
#正規化
#カラーマップ
#
#全体
#
#2回表示
#
#１回だけなら点群が色づく→vis,pcdを別のものでつくることで解決　名前変えて複数つくる
##

import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# CSV読み込み（ヘッダーあり）
df = pd.read_csv('mid360_XYZ_Reflect_data_03-28-13-53-11.csv')

# x, y, z, reflect列の抽出
data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
reflect_raw = df.iloc[:, 3].to_numpy(dtype=np.float32)

# 回転（センサー角度補正）
theta = np.radians(30 + 2.0)
RotationMat = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])
data_rotated = np.dot(RotationMat, data_xyz.T).T

# 点ごとの距離計算
distances = np.linalg.norm(data_rotated, axis=1)

# 距離補正（強度 ÷ 距離²）
reflect_corrected = reflect_raw / (distances**2 + 1e-6)  # 0除算防止

# 任意の範囲指定
x_min, x_max = 0, 10000
y_min, y_max = -1800, 2000
z_min, z_max = -1700, 4000

# マスク
mask = (
    (data_rotated[:, 0] >= x_min) & (data_rotated[:, 0] <= x_max) &
    (data_rotated[:, 1] >= y_min) & (data_rotated[:, 1] <= y_max) &
    (data_rotated[:, 2] >= z_min) & (data_rotated[:, 2] <= z_max)
)
filtered_points = data_rotated[mask]
filtered_reflect = reflect_corrected[mask]

# Min-Max正規化
r_min = filtered_reflect.min()
r_max = filtered_reflect.max()
reflect_norm = (filtered_reflect - r_min) / (r_max - r_min + 1e-6)

# Jetカラーマップ変換
colors = plt.cm.jet(reflect_norm)[:, :3]

# 点群と色
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(filtered_points)
pcd1.colors = o3d.utility.Vector3dVector(colors)

# 座標軸
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# 表示
vis1 = o3d.visualization.Visualizer()
vis1.create_window(window_name="Reflectance Colored (Distance Corrected)", width=1600, height=1200)
vis1.add_geometry(pcd1)
vis1.add_geometry(cs)

#Point size の設定（小さくする例：1.0など）
opt = vis1.get_render_option()
opt.point_size = 3.0  # デフォルトは5.0くらい、大きさを小さくしたければ数値を下げる

vis1.run()
vis1.destroy_window()


# 任意の範囲指定
x_min, x_max = 3000, 4000
y_min, y_max = -500, 700
z_min, z_max = -1500, -500

# マスク
mask = (
    (data_rotated[:, 0] >= x_min) & (data_rotated[:, 0] <= x_max) &
    (data_rotated[:, 1] >= y_min) & (data_rotated[:, 1] <= y_max) &
    (data_rotated[:, 2] >= z_min) & (data_rotated[:, 2] <= z_max)
)
filtered_points = data_rotated[mask]
filtered_reflect = reflect_corrected[mask]

# Min-Max正規化
r_min = filtered_reflect.min()
r_max = filtered_reflect.max()
reflect_norm = (filtered_reflect - r_min) / (r_max - r_min + 1e-6)

# Jetカラーマップ変換
colors = plt.cm.jet(reflect_norm)[:, :3]

# 点群と色
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 座標軸
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# 表示
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Reflectance Colored (Distance Corrected)", width=1600, height=1200)
vis.add_geometry(pcd)
vis.add_geometry(cs)

#Point size の設定（小さくする例：1.0など）
opt = vis.get_render_option()
opt.point_size = 3.0  # デフォルトは5.0くらい、大きさを小さくしたければ数値を下げる

vis.run()
vis.destroy_window()

