#
#
#4行目無視して点群データ表示
#範囲選択何もせず
#回転もしてない
#4行目のリフレくてぃんぐで色変更されるように　
#　強度高い　よく反射してるやつは赤く？
#
##

import csv
import pprint
import open3d as o3d
import numpy as np
import sklearn
from sklearn.decomposition import PCA

with open('2024.10.08.19.57.57.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data01 = np.asarray(data,dtype=np.float32) 
theta = np.radians(30 + 2.0);
RotationMat = np.array([[np.cos(theta),0, np.sin(theta)],
                        [0,1,0], 
                        [-np.sin(theta),0,np.cos(theta)]])
#センサーの設置角度は30度
data002 = data01[:,:3]
data02 = np.dot(RotationMat,data002.T).T
lendt2 = len(data02)
print(lendt2)

data03 = data02[data02[:,0]>0.0];
lendt3 = len(data03)
print('len3',lendt3)

# data04 = data03[data03[:,1] < 1600.0];
# data05 = data04[data04[:,1] > -2000.0];
# data06 = data05[data05[:,0] < 5000.0];
# data07 = data06[data06[:,2] > -850.0];

pcd  = o3d.geometry.PointCloud();
pcd.points = o3d.utility.Vector3dVector( data002 );
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0);


# #method 1 alpha
# alpha = 30
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# print(f"alpha={alpha:.3f}")
# print(len(mesh.triangles))

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウドと座標軸を追加
vis.add_geometry(pcd)
vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()