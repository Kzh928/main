#鉄筋奥行きで検出
#めっちゃ細かい間隔で区切ってる
#鉄筋これは見えんくね？
#センサの精度的に厳しそう
#鉄筋の検出は一点に当ててその横の点に当てることで
#　鉄筋の丸みを検出できるかどうかくらいしかない
#　
#　これpca掛けてない#

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
lendt2 = len(data02)
print(lendt2)

# z軸回りの回転行列の設定（-3度回転）
theta_z = np.radians(-3.0)
RotationMat_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                          [np.sin(theta_z), np.cos(theta_z), 0], 
                          [0, 0, 1]])

# データの回転（z軸回り）
data002 = np.dot(RotationMat_z, data02.T).T
# 条件に基づいてデータをフィルタリング
data03 = data002[data002[:, 0] > 0.0]
lendt3 = len(data03)
print('len3', lendt3)

data04 = data03[data03[:, 1] < 1600.0]
data05 = data04[data04[:, 1] > -2000.0]
data06 = data05[data05[:, 0] < 5000.0]
data07 = data06[data06[:, 2] > -850.0]

# ポイントクラウドの作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data07)
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# x軸方向で区切るための間隔設定
x_min = np.min(data07[:, 0])
x_max = np.max(data07[:, 0])
interval_size = 0.05
num_intervals = 10  # 分割数

# カラーマップの設定
color_map = [
    [1, 0, 0],   # 赤
    [1, 0.5, 0], # オレンジ
    [1, 1, 0],   # 黄色
    [0.5, 1, 0], # 黄緑
    [0, 1, 0],   # 緑
    [0, 1, 0.5], # エメラルドグリーン
    [0, 1, 1],   # シアン
    [0, 0.5, 1], # 青
    [0, 0, 1],   # ダークブルー
    [0.5, 0, 1]  # 紫
]

colors = []

# 各ポイントに対して色を割り当て
for point in data07:
    x_value = point[0]
    # x軸方向に位置する区間を計算
    interval_index = int((x_value - x_min) / ((x_max - x_min) / num_intervals))
    interval_index = min(interval_index, num_intervals - 1)  # 最大インデックスを超えないように調整
    colors.append(color_map[interval_index])

# 色をポイントクラウドに追加
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウドと座標軸を追加
vis.add_geometry(pcd)
vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()
