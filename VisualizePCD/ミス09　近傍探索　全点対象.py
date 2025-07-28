#近傍探索上手くいってない
#なんか全部表示されてる気する#
#
#手前からの点を黒色にする処理を、各点
# #近傍探索を行う
# 全点を対象にループする形で選ばれたらそのあとは選ばれないようなループ
# どんな点を残すかというと一点選んだ時にその点からみてyz平面がおなじ（ｘ軸方向の奥行きが同じ）で縦方向に点が存在するものだけが選ばれるようにしたい
# 縦と横の選ばれる範囲には多少の余裕を持たせることができるコードで
# 点えらんだらそいつから一直線にたてにのばして当たったやつをまとめてtekkinn_pointsに入れる
# でも縦横の閾値は数値入力してできる感じで
# いろわけもしてほしくて
# どういう色分けかいいますとｘ軸方向の奥行きが同じやつを同じ色に#
# 全てのてんでループじゃなくてyz平面を微小区間で区切って
# それの一番手前のてんあつめたやつでループ
# x軸ほうこうでいろわけしたい
# このコードに手前からなんぼかは点を表示しないというコードをついか
# スキップするんじゃなくて黒色にする

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

# PCAの第1主成分をYZ平面に平行にするための回転行列を計算
pca = PCA(n_components=3)
pca.fit(data07)

principal_components = pca.components_

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

# カラーマップの設定（複数色を使用）
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

# x軸方向で区切るための間隔設定
x_min = np.min(data_rotated[:, 0])
x_max = np.max(data_rotated[:, 0])
interval_size = 0.05
num_intervals = len(color_map)

colors = []

# 各ポイントに対して色を割り当て
skip_threshold = 4000  # 例えば、x軸の値が500以下の点は黒色にする
for point in data_rotated:
    x_value = point[0]
    if x_value < skip_threshold:
        colors.append([0, 0, 0])  # 黒色
    else:
        interval_index = int((x_value - x_min) / ((x_max - x_min) / num_intervals))
        interval_index = min(interval_index, num_intervals - 1)  # 最大インデックスを超えないように調整
        colors.append(color_map[interval_index])

# 回転後のデータをポイントクラウドとして表示
pcd_rotated = o3d.geometry.PointCloud()
pcd_rotated.points = o3d.utility.Vector3dVector(data_rotated)
pcd_rotated.colors = o3d.utility.Vector3dVector(colors)

# 鉄筋の検出: yz平面を微小区間で区切り、一番手前の点を集める
yz_interval = 10.0  # YZ平面の区切り幅
tekkinn_points = []

# YZ平面ごとに最もx値が小さい点を集める
y_min, y_max = np.min(data_rotated[:, 1]), np.max(data_rotated[:, 1])
z_min, z_max = np.min(data_rotated[:, 2]), np.max(data_rotated[:, 2])

for y in np.arange(y_min, y_max, yz_interval):
    for z in np.arange(z_min, z_max, yz_interval):
        # 各yz平面区間内の点を取得
        mask = (data_rotated[:, 1] >= y) & (data_rotated[:, 1] < y + yz_interval) & \
               (data_rotated[:, 2] >= z) & (data_rotated[:, 2] < z + yz_interval)
        points_in_section = data_rotated[mask]
        
        if len(points_in_section) > 0:
            # 一番手前の点（x値が最も小さい点）を選択
            nearest_point = points_in_section[np.argmin(points_in_section[:, 0])]
            tekkinn_points.append(nearest_point)

print("tetu", len(tekkinn_points))

# 鉄筋として選ばれた点を可視化
tekkinn_points_np = np.array(tekkinn_points)
pcd_tekkinn = o3d.geometry.PointCloud()
pcd_tekkinn.points = o3d.utility.Vector3dVector(tekkinn_points_np)

# コーディネートフレームの作成
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# 可視化処理
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Colored by Depth & Tekkinn Points", width=1600, height=1200)
vis.add_geometry(pcd_rotated)
vis.add_geometry(pcd_tekkinn)
vis.add_geometry(cs)

vis.run()
vis.destroy_window()
