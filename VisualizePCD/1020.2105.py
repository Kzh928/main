#
#reflectってきたら数値放り込んでそれより大きいか小さいかで色分け
#1020.2010のやつのいろつけただけ#

import csv
import open3d as o3d
import numpy as np

# CSVファイルの読み込み
with open('2024.12.18.02.10.21.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

# NumPy 配列に変換
data01 = np.asarray(data, dtype=np.float32)

# # 配列の形状を表示
# print("データの形状:", data01.shape)

# 反射強度のしきい値をユーザーから入力
threshold = float(input("reflect "))

# 色の設定
colors = np.zeros((data01.shape[0], 3))  # RGBのための配列を初期化
for i in range(data01.shape[0]):
    intensity = data01[i][3]  # 4列目をリフレクションの強度とする
    # ユーザーが指定したしきい値に基づいて色を設定
    if intensity > threshold:
        colors[i] = [1, 0, 0]  # 赤色
    else:
        colors[i] = [0, 0, 1]  # 青色

# 点群データを作成
data002 = data01[:, :3]  # 最初の3列を使用
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data002)
pcd.colors = o3d.utility.Vector3dVector(colors)  # 点群に色を追加

# 座標軸の作成
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウドと座標軸を追加
vis.add_geometry(pcd)
vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()
