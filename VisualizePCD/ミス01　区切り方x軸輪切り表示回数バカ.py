#
#微小区間の区切りがx軸方向だった
#x軸方向で100ずつ区切って4回表示された
#区切り方おかしい
#yz平面に正対してくぎりたかったのにｘ軸を輪切りしてる#

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

# データのフィルタリング
data03 = data002[data002[:, 0] > 0.0]
data04 = data03[data03[:, 1] < 1600.0]
data05 = data04[data04[:, 1] > -2000.0]
data06 = data05[data05[:, 0] < 5000.0]
data07 = data06[data06[:, 2] > -850.0]

# ユーザーが設定できる微小区間の大きさ
interval = 100.0  # <- ここで区間の大きさを設定

# `x`の範囲を取得
x_min = np.min(data07[:, 0])
x_max = np.max(data07[:, 0])

# 微小区間の数を計算
num_intervals = int((x_max - x_min) / interval)
print(f"Number of intervals: {num_intervals}")

# 区間ごとにデータを分割
intervals = []
for x_start in np.arange(x_min, x_max, interval):
    x_end = x_start + interval
    subset = data07[(data07[:, 0] >= x_start) & (data07[:, 0] < x_end)]
    intervals.append(subset)

# 各区間のデータを表示する
for i, subset in enumerate(intervals):
    # ポイントクラウドの作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(subset)
    
    # 座標系の作成
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)
    
    # 表示
    print(f"Interval {i + 1} with {len(subset)} points")
    o3d.visualization.draw_geometries([pcd, cs])
