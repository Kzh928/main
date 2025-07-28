#
#全赤点
#面の頂点移動させてるけど頂点だけが動いてる
#頂点を選んでその面全体に適応させて全体で動かしたい
##


import csv
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA

# CSVからデータを読み込む
with open('mid360_XYZdata_06-25-10-03-54.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data01 = np.asarray(data, dtype=np.float32)

# センサーの設置角度は30度
theta = np.radians(30 + 2.0)
RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0], 
                        [-np.sin(theta), 0, np.cos(theta)]])
data02 = np.dot(RotationMat, data01.T).T
lendt2 = len(data02)
print(lendt2)

# フィルタリング
data03 = data02[data02[:, 0] > 0.0]
lendt3 = len(data03)
print('len3', lendt3)

data04 = data03[data03[:, 1] < 1600.0]
data05 = data04[data04[:, 1] > -2000.0]
data06 = data05[data05[:, 0] < 5000.0]
data07 = data06[data06[:, 2] > -850.0]

# ボックスの頂点を抽出
x_min, y_min, z_min = np.min(data07, axis=0)
x_max, y_max, z_max = np.max(data07, axis=0)

# 4つの頂点を設定
corner_points = np.array([
    [x_min, y_min, z_min],  # 左下
    [x_min, y_min, z_max],  # 左上
    [x_min, y_max, z_min],  # 右下
    [x_min, y_max, z_max],  # 右上
])

# 元のデータに4つの頂点を追加
data_with_corners = np.vstack((data07, corner_points))

# 選ばれた点と周囲の点を選択
selected_points = set(map(tuple, corner_points))
radius = 50  # 周囲の点を選ぶ半径
distances = np.linalg.norm(data_with_corners[:, None, :] - np.array(corner_points), axis=-1)
selected_indices = np.any(distances < radius, axis=1)
selected_indices[-4:] = True  # 確実に4つの頂点を含める

# 色の設定
colors = np.ones((len(data_with_corners), 3))  # 全ての点を白に設定
colors[~selected_indices] = [1, 0, 0]  # 選ばなかった点を赤に設定
colors[selected_indices] = [0, 0, 0]  # 選ばれた点を黒に設定

# ポイントクラウドの作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data_with_corners)
pcd.colors = o3d.utility.Vector3dVector(colors)
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# PCAによるYZ平面への整列
pca = PCA(n_components=2)
yz_data = data_with_corners[:, 1:3]  # YとZデータを抽出
yz_data_pca = pca.fit_transform(yz_data)

# PCA変換後のYとZデータを元のデータに戻す
data_with_corners_pca = data_with_corners.copy()
data_with_corners_pca[:, 1:3] = yz_data_pca

# 再度ポイントクラウドの作成と表示
pcd_pca = o3d.geometry.PointCloud()
pcd_pca.points = o3d.utility.Vector3dVector(data_with_corners_pca)
pcd_pca.colors = o3d.utility.Vector3dVector(colors)

# 表示
o3d.visualization.draw_geometries([pcd_pca, cs])
