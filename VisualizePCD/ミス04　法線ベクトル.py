#
#真っ白出力
#法線ベクトル，回転行列計算できず破損
#フィルタリング破損
##


import csv
import open3d as o3d
import numpy as np

# CSVからデータを読み込む（適切なエンコーディングを指定）
with open('mid360_XYZdata_06-25-10-03-54.csv', encoding='latin1') as f:

  # 'utf-8-sig' などを試す
    reader = csv.reader(f)
    data = [row for row in reader]

data01 = np.asarray(data, dtype=np.float32)

# センサーの設置角度は30度
theta = np.radians(30 + 2.0)
RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0], 
                        [-np.sin(theta), 0, np.cos(theta)]])
data02 = np.dot(RotationMat, data01.T).T

# フィルタリング
data03 = data02[data02[:, 0] > 0.0]
data04 = data03[(data03[:, 1] < 1600.0) & (data03[:, 1] > -2000.0)]
data05 = data04[data04[:, 0] < 5000.0]
data06 = data05[data05[:, 2] > -850.0]

# ボックスの頂点を抽出
x_min, y_min, z_min = np.min(data06, axis=0)
x_max, y_max, z_max = np.max(data06, axis=0)

# 4つの頂点を設定
corner_points = np.array([
    [x_min, y_min, z_min],  # 左下
    [x_min, y_min, z_max],  # 左上
    [x_min, y_max, z_min],  # 右下
    [x_min, y_max, z_max],  # 右上
])

# 面の法線ベクトルを計算
v1 = corner_points[1] - corner_points[0]  # 左下から左上へのベクトル
v2 = corner_points[2] - corner_points[0]  # 左下から右下へのベクトル
normal = np.cross(v1, v2)  # 法線ベクトルを計算

# 法線ベクトルがゼロでないか確認
if np.linalg.norm(normal) == 0:
    print("法線ベクトルがゼロです。回転行列を生成できません。")
    RotationMat = np.eye(3)  # 回転行列を単位行列に設定
else:
    # 法線ベクトルをYZ平面に整列させる回転行列を計算
    z_axis = np.array([1, 0, 0])  # X軸方向のベクトル
    axis = np.cross(normal, z_axis)
    
    if np.linalg.norm(axis) != 0:  # 軸ベクトルがゼロでない場合
        angle = np.arccos(np.dot(normal, z_axis) / (np.linalg.norm(normal) * np.linalg.norm(z_axis)))
        axis = axis / np.linalg.norm(axis)
        
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        RotationMat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    else:
        # 法線ベクトルがX軸と平行な場合
        RotationMat = np.eye(3)

# 全体のデータを回転させる
data_rotated = np.dot(RotationMat, data06.T).T

# ポイントクラウドの作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data_rotated)

# 色を設定（全ての点を青に設定）
colors = np.array([[0, 0, 1] for _ in range(len(data_rotated))])
pcd.colors = o3d.utility.Vector3dVector(colors)

# 原点を示す座標軸
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# デバッグ用出力
print("Filtered data size:", len(data06))  # フィルタリング後のデータサイズ
print("Min values:", data_rotated.min(axis=0))  # 回転後の最小値
print("Max values:", data_rotated.max(axis=0))  # 回転後の最大値
print("Normal vector:", normal)  # 計算された法線ベクトル
print("Rotation matrix:\n", RotationMat)  # 回転行列

# カメラの視点を手動で設定
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.add_geometry(cs)
ctr = vis.get_view_control()
ctr.set_lookat([0, 0, 0])  # カメラが原点を向くように設定
ctr.set_front([1, 0, 0])   # カメラの前方向を設定
ctr.set_up([0, 0, 1])      # カメラの上方向を設定
ctr.set_zoom(0.5)          # ズームレベルを設定
vis.run()
vis.destroy_window()
