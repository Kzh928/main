# -*- coding: utf-8 -*-

#範囲で区切って一番手前の点を抽出
#pca掛けてあげて手前の綺麗な面をゲット
#その面とyz平面が平行になるように修正
#データ全体がyz平面に対して正対するようになった
#
#
#DBSCAN使ってクラスタリング
#閾値とクラスタ（所属先）の数値いじることで区別の度合変えれる
#
#表示はクラスタリングされた後のやつだけにしてる
#クラスタリングによって鉄筋探せないかなのpart１
#DBSCANつかうなら例えば，手前側（綺麗な面）を消してからクラスタリングしてみるか
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

# # Visualizerを使用してウィンドウのサイズを設定
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Custom Window", width=1600, height=1200)

# vis.add_geometry(pcd)
# vis.add_geometry(plane_mesh)
# vis.add_geometry(plane_mesh_double)
# vis.add_geometry(cs)

# vis.run()
# vis.destroy_window()


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

# # Visualizerを使用してウィンドウのサイズを設定
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Custom Window Rotated", width=1600, height=1200)

# # 回転後のポイントクラウドと座標軸を追加
# vis.add_geometry(pcd_rotated)
# #vis.add_geometry(cs)

# # 描画
# vis.run()
# vis.destroy_window()

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

# クラスタリング用にデータを準備（回転後のデータを使用）
data_for_clustering = data_rotated

# DBSCANの設定（eps: 距離の閾値、min_samples: クラスタを形成するための最小点数）
db = DBSCAN(eps=20, min_samples=6).fit(data_for_clustering)

# 各点に対応するクラスタのラベルを取得（-1はノイズとして認識された点）
labels = db.labels_

# クラスタごとの色を設定する
unique_labels = set(labels)
colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(unique_labels)))

# クラスタごとに色をつけてポイントクラウドを表示するためのデータを準備
clustered_points = []
clustered_colors = []

for label in unique_labels:
    if label == -1:
        # ノイズは黒色にする
        color = [0, 0, 0]
    else:
        # クラスタに色を割り当てる
        color = colors[label % len(colors)][:3]
    
    # クラスタに属する点を取得
    cluster_points = data_for_clustering[labels == label]
    
    clustered_points.append(cluster_points)
    
    # クラスタに属する点に色を設定
    for _ in range(len(cluster_points)):
        clustered_colors.append(color)

# クラスタリング後の点群をNumPy配列に変換
clustered_points_np = np.vstack(clustered_points)

# クラスタリング後のポイントクラウドを設定
pcd_clustered = o3d.geometry.PointCloud()
pcd_clustered.points = o3d.utility.Vector3dVector(clustered_points_np)
pcd_clustered.colors = o3d.utility.Vector3dVector(np.array(clustered_colors))




# # ヒストグラム作製
# # 距離行列を計算
# distances = distance_matrix(data_for_clustering, data_for_clustering)

# # 近距離のデータだけに注目（対角成分を除外）
# non_zero_distances = distances[np.triu_indices_from(distances, k=1)]
# plt.hist(non_zero_distances, bins=50)
# plt.xlabel("Distance")
# plt.ylabel("Frequency")
# plt.title("Histogram of point distances")
# plt.show()




# Visualizerを使用してクラスタリング結果を表示
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Clustered Point Cloud", width=1600, height=1200)

# クラスタリング結果を追加して表示
vis.add_geometry(pcd_clustered)
#vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()

# 各クラスタの点数を表示（参考）
for label in unique_labels:
    if label != -1:  # ノイズ以外のクラスタを表示
        print(f"Cluster {label}: {sum(labels == label)} points")
