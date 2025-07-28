#
#任意の間隔で色変更
#間隔毎に色変更のところループ巻数になっている
#
#
#
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
lendt2 = len(data02)
print(lendt2)

# 条件に基づいてデータをフィルタリング
data03 = data02[data02[:, 0] > 0.0]
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

# アルファシェイプ法によるメッシュの作成
alpha = 30
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
print(f"alpha={alpha:.3f}")
print(len(mesh.triangles))

# x軸方向に任意の間隔ごとに色分けする
colors = []
x_min = np.min(data07[:, 0])
intervals = [50.0, 100.0, 150.0, 200.0]
color_map = [
    [1, 0, 0],  # 赤
    [1, 1, 0],  # 黄
    [0, 1, 0],  # 緑
    [0, 0, 1]   # 青
]

for point in data07:
    x_value = point[0]
    # 間隔ごとに色を適用
    index = 0
    for i, interval in enumerate(intervals):
        if x_value < x_min + interval * (i + 1):
            index = i
            break
    else:
        index = len(intervals) - 1

    colors.append(color_map[index % len(color_map)])

# 色をポイントクラウドに追加
pcd.colors = o3d.utility.Vector3dVector(colors)

# 表示
o3d.visualization.draw_geometries([pcd, cs])
o3d.visualization.draw_geometries([mesh, cs])
