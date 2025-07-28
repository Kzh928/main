##
#もらったやつのVisualizerの表示の大きさだけ変更
#chatgpt
#壁の面取り出す
#小区間取り出して分散の計算で取り出す
#きれいな壁面とえぐれているとこでの差
#横から見てばらつき検出
#基本的にブラックボックスにならないように
#open3dではなくnumpyのほうでする
#
#
##


import csv
import pprint
import open3d as o3d
import numpy as np
import sklearn
from sklearn.decomposition import PCA

# CSVファイルの読み込み
with open('mid360_XYZdata_06-25-10-03-54.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data01 = np.asarray(data, dtype=np.float32)
theta = np.radians(30 + 2.0)
RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0], 
                        [-np.sin(theta), 0, np.cos(theta)]])
# センサーの設置角度は30度
data02 = np.dot(RotationMat, data01.T).T
lendt2 = len(data02)
print(lendt2)

data03 = data02[data02[:, 0] > 0.0]
lendt3 = len(data03)
print('len3', lendt3)

data04 = data03[data03[:, 1] < 1600.0]
data05 = data04[data04[:, 1] > -2000.0]
data06 = data05[data05[:, 0] < 5000.0]
data07 = data06[data06[:, 2] > -850.0]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data07)
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# メッシュの生成
alpha = 30
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
print(f"alpha={alpha:.3f}")
print(len(mesh.triangles))

# Visualizerクラスを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Point Cloud Visualization', width=1000, height=800)
vis.add_geometry(pcd)
vis.add_geometry(cs)
vis.run()
vis.destroy_window()

# 別のウィンドウでメッシュを表示
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Mesh Visualization', width=1000, height=800)
vis.add_geometry(mesh)
#vis.add_geometry(cs)
vis.run()
vis.destroy_window()
