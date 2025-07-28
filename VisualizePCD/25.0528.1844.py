#
#正規化
#カラーマップ
#全体
#
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

# Min-Max正規化のみ
r_min = reflect_raw.min()
r_max = reflect_raw.max()
reflect_norm = (reflect_raw - r_min) / (r_max - r_min + 1e-6)

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
filtered_reflect = reflect_norm[mask]

# Jetカラーマップ変換（matplotlib使用）→ RGBA → RGB
colors = plt.cm.jet(filtered_reflect)[:, :3]  # RGBのみ取り出し

# 点群と色
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 座標軸
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# 表示
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Reflectance Colored (Min-Max)", width=1600, height=1200)
vis.add_geometry(pcd)
vis.add_geometry(cs)
vis.run()
vis.destroy_window()
