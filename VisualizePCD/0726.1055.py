# 角度y軸30度変更のみ　
#切りとり方座標指定　やり方わかった　どこを切り出していいかわからん
#
#中心をどこに合わせる
#壁を正面に
#ずらした後にきりとると前後逆だと変わる？
#中心座標動かせる，
#真ん中に持ってきて周り消去
#
#赤緑青xyz
#範囲絞っていく段階
#
#座標移動なし　
#
#Visuallizer 場所移動
#
#
#bounding_box open3d依存やからあんま良くない
#numpyのほうで
#
#
#



import csv
import open3d as o3d
import numpy as np
import math

def Rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    theta_x = math.radians(theta_x)
    theta_y = math.radians(theta_y)
    theta_z = math.radians(theta_z)
    rot_x = np.array([[ 1,                 0,                  0],
                      [ 0, math.cos(theta_x), -math.sin(theta_x)],
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])

    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                      [                 0, 1,                  0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])

    rot_z = np.array([[ math.cos(theta_z), -math.sin(theta_z), 0],
                      [ math.sin(theta_z),  math.cos(theta_z), 0],
                      [                 0,                  0, 1]])

    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    rot_pointcloud = rot_matrix.dot(pointcloud.T).T
    return rot_pointcloud, rot_matrix

# ビジュアライザを作成
visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=600, height=500, left=1250, top=250)
print("Window created")

# 原点にサイズ指定のx,y,z軸を描写
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
print("Coordinate frame created")

# ファイルの読み込み
with open('mid360_XYZdata_06-25-10-03-54.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]
print("Reading file")

# データを適切な形に，いらないところを切捨て
data_np = np.array(data, dtype=np.float64)
lend = len(data_np)
filtered_data = [row for row in data_np if row[1] > 0]
filtered_data = np.array(filtered_data)
print("Data conversion")

# 行の長さを表示
print(len(filtered_data))

# 点群データを回転
rotated_data, rot_matrix = Rotation_xyz(filtered_data, 0, 30, 0)
print("Data rotated")

# 点群データのフィルタリング（例としてX, Y, Z軸の範囲を指定）
min_bound = np.array([0, -1000, -720])  # 最小値
max_bound = np.array([7000, 1500, 600])  # 最大値
#××〇××〇

# 点群オブジェクトの作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(rotated_data)

# Bounding Boxの作成
bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

# 点群のクリッピング
pcd_crop = pcd.crop(bounding_box)

# フィルタリングによるデータの変更量確認
print(f"Number of points before filtering: {len(rotated_data)}")
print(f"Number of points after filtering: {len(np.asarray(pcd_crop.points))}")

# ビジュアライザに点群と座標フレームを追加
visualizer.add_geometry(pcd_crop)
visualizer.add_geometry(coordinate_frame)

# ビジュアライザを実行
visualizer.run()
print("Visualizer running")

# ビジュアライザを破棄
visualizer.destroy_window()
print("Visualizer window destroyed")
