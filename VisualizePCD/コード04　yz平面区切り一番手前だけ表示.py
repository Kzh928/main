#
#洗濯した範囲の点のみをみる
#ｙｚ平面の区間ごとに処理
#それぞれの区間で一番手前だけ抽出
#区切った後に一番手前だけを抽出してる
##


import csv
import open3d as o3d
import numpy as np

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

# # ユーザーが指定する範囲の入力
# y_min = float(input("Enter y_min: "))  # 例: -2000
# y_max = float(input("Enter y_max: "))  # 例: 1600
# z_min = float(input("Enter z_min: "))  # 例: -850
# z_max = float(input("Enter z_max: "))  # 例: 1000

# ユーザーが指定する範囲の入力
y_min = -2000  # 例: -2000
y_max = 1600   # 例: 1600
z_min = -850   # 例: -850
z_max = 1000   # 例: 1000
min_area = 50.0  # 最小面積の指定 (例: 50.0)


# ポイントクラウドの作成
pcd = o3d.geometry.PointCloud()

# YZ平面方向に区切る
interval = 50.0  # 微小区間のサイズ

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

# 選択した点と色をポイントクラウドに設定
pcd.points = o3d.utility.Vector3dVector(np.array(selected_points))
pcd.colors = o3d.utility.Vector3dVector(np.array(selected_colors))

# コーディネートフレームの作成
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# 表示
o3d.visualization.draw_geometries([pcd, cs])
