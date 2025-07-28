#
#抽出したなかからランダムで何点か選んでpcaの面作製
#何点（割合）は数値入力できるように
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

#初期データ数の表示
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

#フィルタリング後のデータの数の表示
print(len(data07))

# ユーザーが指定する範囲の入力
y_min = -2000  # 例: -2000
y_max = 1600   # 例: 1600
z_min = -850   # 例: -850
z_max = 1000   # 例: 1000

print(len(data07))

# ランダムに何パーセントのデータを使うか
percentage_to_use = 1.0  # 例: 50%のデータを使う
#percentage_to_use = float(input("Enter percentage: "))/100.0  #入力した数値の100分の1のデータ
num_points = len(data07)
num_to_select = int(num_points * percentage_to_use)

# ランダムにデータを選択
indices = np.random.choice(num_points, num_to_select, replace=False)
data07 = data07[indices, :]

#ランダムにデータ選択後のデータの数の表示
print(len(data07))

# ポイントクラウドの作成
pcd = o3d.geometry.PointCloud()

# YZ平面方向に区切る
interval = 200.0  # 微小区間のサイズ

# 色を設定するためのマップと区間のリストを作成
color_map = [
    [1, 0, 0],  # 赤
    [1, 1, 0],  # 黄
    [0, 1, 0],  # 緑
    [0, 0, 1]   # 青
]

selected_points = []
selected_colors = []

# YZ平面の区間ごとに処理
for y_start in np.arange(y_min, y_max, interval):
    for z_start in np.arange(z_min, z_max, interval):
        # 現在の区間に属する点を抽出
        mask = (data07[:, 1] >= y_start) & (data07[:, 1] < y_start + interval) & \
               (data07[:, 2] >= z_start) & (data07[:, 2] < z_start + interval)
        section_points = data07[mask]
        
        if len(section_points) > 0:
            # x軸方向で最も手前の点を選択
            nearest_point = section_points[np.argmin(section_points[:, 0])]
            selected_points.append(nearest_point)
            
            # 色を設定
            color = color_map[(int(y_start / interval) + int(z_start / interval)) % len(color_map)]
            selected_colors.append(color)

# 選択した点をNumPy配列に変換
selected_points_np = np.array(selected_points)

# 選択した点と色をポイントクラウドに設定
pcd.points = o3d.utility.Vector3dVector(selected_points_np)
pcd.colors = o3d.utility.Vector3dVector(np.array(selected_colors))

# コーディネートフレームの作成
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# PCAインスタンスを作成し、データに適用
pca = PCA(n_components=3)  # 3次元データの場合は3を指定
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
mean_point = np.mean(selected_points_np, axis=0)  # 点群の平均

# 平面を定義するための頂点を計算
plane_size = 2000  # 平面のサイズ
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

# 表面の色を設定
light_gray = [0.8, 0.8, 0.8]  # 薄い灰色
plane_mesh.paint_uniform_color(light_gray)

# 裏面も含めるために、同じ平面の裏面も生成
plane_mesh_double = o3d.geometry.TriangleMesh()
plane_mesh_double.vertices = o3d.utility.Vector3dVector(plane_vertices)
plane_mesh_double.triangles = o3d.utility.Vector3iVector(np.array([[0, 2, 1], [0, 3, 2]]))
plane_mesh_double.paint_uniform_color(light_gray)

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウド、平面、座標軸を追加
vis.add_geometry(pcd)
vis.add_geometry(plane_mesh)
vis.add_geometry(plane_mesh_double)
vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()
