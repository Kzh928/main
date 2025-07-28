#
#フィルタリングされたデータを微小区間に区切り、各区間でx軸方向に最も近い点を抽出して、
#PCA（主成分分析）を用いて統計的に面を作成するコードです。
#区間設定できるように
#yz平面に平行な面を作製
#create_planeメソッドがないので手動で作製
#面おかしい
#適当にpca掛けてみたけど意味わかってない
##
#破壊
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

# 回転行列の設定（センサーの設置角度は30度）
theta = np.radians(30 + 2.0)
RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0], 
                        [-np.sin(theta), 0, np.cos(theta)]])

# データの回転
data02 = np.dot(RotationMat, data01.T).T
lendt2 = len(data02)
print(f"Total data points: {lendt2}")

# データのフィルタリング
data03 = data02[data02[:, 0] > 0.0]
lendt3 = len(data03)
print(f"Filtered data points: {lendt3}")

data04 = data03[data03[:, 1] < 1600.0]
data05 = data04[data04[:, 1] > -2000.0]
data06 = data05[data05[:, 0] < 5000.0]
data07 = data06[data06[:, 2] > -850.0]

# ユーザーが設定できる微小区間の大きさ
interval = 50.0  # <- ここを変更すると区間の大きさを変えられる

x_min = np.min(data07[:, 0])
x_max = np.max(data07[:, 0])

# 微小区間の数を計算
num_intervals = int((x_max - x_min) / interval)
print(f"Number of intervals: {num_intervals}")

filtered_points = []

for x_start in np.arange(x_min, x_max, interval):
    x_end = x_start + interval
    subset = data07[(data07[:, 0] >= x_start) & (data07[:, 0] < x_end)]
    
    if len(subset) > 0:
        # x軸方向に最も近い点を選択
        closest_point = subset[np.argmin(np.abs(subset[:, 0] - x_start))]
        filtered_points.append(closest_point)

filtered_points = np.array(filtered_points)

# PCAによる統計的な面の作成
if len(filtered_points) >= 3:  # PCAには最低3点が必要
    pca = PCA(n_components=3)
    pca.fit(filtered_points)
    
    # PCAの第3主成分を取得（面の法線ベクトルとして使用）
    normal_vector = pca.components_[2]
    
    # 法線ベクトルをYZ平面に平行にする
    normal_vector[0] = 0  # X成分を0にしてYZ平面に平行にする

    print("Corrected Normal vector (PCA):", normal_vector)

    # 平面上の1点を取得
    plane_point = np.mean(filtered_points, axis=0)
    
    # 平面の頂点とインデックスを定義
    d = -np.dot(plane_point, normal_vector)
    width, height = 500, 500  # 平面のサイズ
    plane_vertices = np.array([
        [-width / 2, -height / 2, (-d - normal_vector[0] * (-width / 2) - normal_vector[1] * (-height / 2)) / normal_vector[2]],
        [width / 2, -height / 2, (-d - normal_vector[0] * (width / 2) - normal_vector[1] * (-height / 2)) / normal_vector[2]],
        [width / 2, height / 2, (-d - normal_vector[0] * (width / 2) - normal_vector[1] * (height / 2)) / normal_vector[2]],
        [-width / 2, height / 2, (-d - normal_vector[0] * (-width / 2) - normal_vector[1] * (height / 2)) / normal_vector[2]]
    ])
    
    plane_faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    # TriangleMeshの作成
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(plane_vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(plane_faces)
    
    # 法線ベクトルを使って平面を回転させる
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(np.cross([0, 0, 1], normal_vector))
    plane_mesh.rotate(rotation_matrix, center=plane_point)
    plane_mesh.translate(plane_point)
    
    # ポイントクラウドと座標系の表示
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)
    o3d.visualization.draw_geometries([pcd, cs, plane_mesh])
else:
    print("Not enough points in the filtered data for PCA.")
