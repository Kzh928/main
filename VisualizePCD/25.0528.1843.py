#
#
#読み込んで開くだけ
#全体表示で範囲区切って変更して２回表示
#表示窓の大きさ変更
#
#1回目は全体
#２回目は鉄筋回り いったん取り出せた
#
#点群表示の大きさ変更　粒の大きさ#

import pandas as pd
import open3d as o3d
import numpy as np

# CSV読み込み（ヘッダーあり）
df = pd.read_csv('mid360_XYZ_Reflect_data_03-28-13-53-11.csv')

# x,y,zの3列だけ抽出
data01 = df.iloc[:, :3].to_numpy(dtype=np.float32)

# 回転（センサー角度補正）
theta = np.radians(30 + 2.0)
RotationMat = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])
data02 = np.dot(RotationMat, data01.T).T

# ========== 1回目の表示 ==========
x_min, x_max = 0, 10000
y_min, y_max = -1800, 2000
z_min, z_max = -1700, 4000

mask = (
    (data02[:, 0] >= x_min) & (data02[:, 0] <= x_max) &
    (data02[:, 1] >= y_min) & (data02[:, 1] <= y_max) &
    (data02[:, 2] >= z_min) & (data02[:, 2] <= z_max)
)
filtered = data02[mask]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered)
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="display 1", width=1600, height=1200)
vis.add_geometry(pcd)
vis.add_geometry(cs)
# vis.run()
vis.destroy_window()

# ========== 2回目の表示（範囲変更） ==========
x_min, x_max = 3000, 4000
y_min, y_max = -500, 700
z_min, z_max = -1500, -500

mask = (
    (data02[:, 0] >= x_min) & (data02[:, 0] <= x_max) &
    (data02[:, 1] >= y_min) & (data02[:, 1] <= y_max) &
    (data02[:, 2] >= z_min) & (data02[:, 2] <= z_max)
)
filtered = data02[mask]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered)
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="display 2", width=1600, height=1200)
vis.add_geometry(pcd)
vis.add_geometry(cs)

# Point size の設定（小さくする例：1.0など）
opt = vis.get_render_option()
opt.point_size = 3.0  # デフォルトは5.0くらい、大きさを小さくしたければ数値を下げる

vis.run()
vis.destroy_window()
