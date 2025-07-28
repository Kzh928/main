#範囲で区切って一番手前の点を抽出
#pca掛けてあげて手前の綺麗な面をゲット
#その面とyz平面が平行になるように修正
#データ全体がyz平面に対して正対するようになった
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

# 初期データ数の表示
print(len(data01))

# 回転行列の設定（センサーの設置角度は30度）
theta = np.radians(30 + 2.0)
RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0], 
                        [-np.sin(theta), 0, np.cos(theta)]])

# データの回転
data02 = np.dot(RotationMat, data01.T).T

# z軸回りの回転行列の設定（-3度回転）
theta_z = np.radians(-3.0)
RotationMat_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                          [np.sin(theta_z), np.cos(theta_z), 0], 
                          [0, 0, 1]])

# データの回転（z軸回り）
data002 = np.dot(RotationMat_z, data02.T).T

# 条件に基づいてデータをフィルタリング
data03 = data002[data002[:, 0] > 0.0]
data04 = data03[data03[:, 1] < 1600.0]
data05 = data04[data04[:, 1] > -2000.0]
data06 = data05[data05[:, 0] < 5000.0]
data07 = data06[data06[:, 2] > -850.0]

# フィルタリング後のデータの数の表示
print(len(data07))

# CSVからデータを読み込んだ後、フィルタリングされたデータの表示
pcd_full = o3d.geometry.PointCloud()
pcd_full.points = o3d.utility.Vector3dVector(data07)


# コーディネートフレームの作成
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)


# Visualizerを使用して全体の点群を表示
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Full Point Cloud", width=1600, height=1200)

# 全体の点群と座標軸を追加
vis.add_geometry(pcd_full)
vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()


# ユーザーが指定する範囲の入力
y_min = -2000
y_max = 1600
z_min = -850
z_max = 1000

# ポイントクラウドの作成
pcd = o3d.geometry.PointCloud()

# YZ平面方向に区切る
interval = 200.0

# 色を設定するためのマップと区間のリストを作成
color_map = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1]
]

selected_points = []
selected_colors = []

# YZ平面の区間ごとに処理
for y_start in np.arange(y_min, y_max, interval):
    for z_start in np.arange(z_min, z_max, interval):
        mask = (data07[:, 1] >= y_start) & (data07[:, 1] < y_start + interval) & \
               (data07[:, 2] >= z_start) & (data07[:, 2] < z_start + interval)
        section_points = data07[mask]
        
        if len(section_points) > 0:
            nearest_point = section_points[np.argmin(section_points[:, 0])]
            selected_points.append(nearest_point)
            
            color = color_map[(int(y_start / interval) + int(z_start / interval)) % len(color_map)]
            selected_colors.append(color)

# 選択したデータの数の表示
print(len(selected_points))

# 選択した点をNumPy配列に変換
selected_points_np = np.array(selected_points)

# 選択した点と色をポイントクラウドに設定
pcd.points = o3d.utility.Vector3dVector(selected_points_np)
pcd.colors = o3d.utility.Vector3dVector(np.array(selected_colors))

# コーディネートフレームの作成
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# PCAインスタンスを作成し、データに適用
pca = PCA(n_components=3)
pca.fit(selected_points_np)

# 主成分 (固有ベクトル)
principal_components = pca.components_
print("Principal Components (Eigenvectors):")
print(principal_components)

# 各主成分に対応する固有値
explained_variance = pca.explained_variance_
print("Explained Variance (Eigenvalues):")
print(explained_variance)

# 平面の中心点を計算
mean_point = np.mean(selected_points_np, axis=0)

# 平面を定義するための頂点を計算
plane_size = 2000
plane_vertices = np.array([
    mean_point + plane_size * (principal_components[0] + principal_components[1]),
    mean_point + plane_size * (principal_components[0] - principal_components[1]),
    mean_point + plane_size * (-principal_components[0] - principal_components[1]),
    mean_point + plane_size * (-principal_components[0] + principal_components[1]),
])

# 平面のポリゴンメッシュを作成
plane_mesh = o3d.geometry.TriangleMesh()
plane_mesh.vertices = o3d.utility.Vector3dVector(plane_vertices)
plane_mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [2, 3, 0]]))

light_gray = [0.8, 0.8, 0.8]
plane_mesh.paint_uniform_color(light_gray)

# 裏面も含めるために、同じ平面の裏面も生成
plane_mesh_double = o3d.geometry.TriangleMesh()
plane_mesh_double.vertices = o3d.utility.Vector3dVector(plane_vertices)
plane_mesh_double.triangles = o3d.utility.Vector3iVector(np.array([[0, 2, 1], [0, 3, 2]]))
plane_mesh_double.paint_uniform_color(light_gray)

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window", width=1600, height=1200)

vis.add_geometry(pcd)
vis.add_geometry(plane_mesh)
vis.add_geometry(plane_mesh_double)
vis.add_geometry(cs)

vis.run()
vis.destroy_window()


# PCAの第1主成分をYZ平面に平行にするための回転行列を計算
target_direction = np.array([0, 1, 0])  # Y軸方向
current_direction = principal_components[0]

rotation_axis = np.cross(current_direction, target_direction)
rotation_axis /= np.linalg.norm(rotation_axis)  # 正規化

rotation_angle = np.arccos(np.dot(current_direction, target_direction) / 
                           (np.linalg.norm(current_direction) * np.linalg.norm(target_direction)))

K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
              [rotation_axis[2], 0, -rotation_axis[0]],
              [-rotation_axis[1], rotation_axis[0], 0]])
rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)

# フィルタリング前のデータ全体を回転
data_rotated = np.dot(data07, rotation_matrix.T)

# 回転後のデータをポイントクラウドとして表示
pcd_rotated = o3d.geometry.PointCloud()
pcd_rotated.points = o3d.utility.Vector3dVector(data_rotated)

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window Rotated", width=1600, height=1200)

# 回転後のポイントクラウドと座標軸を追加
vis.add_geometry(pcd_rotated)
#vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()
