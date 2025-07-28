#open3dひらいてデータ羅列，データ長さ，点群オブジェクトと座標フレーム

import csv
import pprint
import open3d as o3d
import numpy as np

# ビジュアライザを作成
visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=600, height=500, left=450, top=250)
print("Window created")

#原点にサイズ10000のx,y,z軸を描写
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10000, origin=[0, 0, 0])
print("Coordinate frame created")

#ファイルの読み込み
with open('mid360_XYZdata_06-25-10-03-54.csv') as f:
  reader = csv.reader(f)
  data = [row for row in reader]
print("Reading file")  

#データを適切な形に，いらないところを切捨て
data_np = np.array(data, dtype=np.float64) 
lend = len(data_np)
for i in range(lend):
  if data_np[i,1] > 0:
    print( data_np[i,:]);
print("Data conversion")

#行の長さを表示
print(len(data_np))

#pcdという名前の点群オブジェクトを作成
pcd  = o3d.geometry.PointCloud();
pcd.points = o3d.utility.Vector3dVector(data_np);



#o3d.visualization.draw_geometries([pcd, mesh])


#ビジュアライザに点群と座標フレームを追加
visualizer.add_geometry(pcd)
#visualizer.add_geometry(coordinate_frame)

# ビジュアライザを実行
visualizer.run()
print("Visualizer running")

# ビジュアライザを破棄
visualizer.destroy_window()
print("Visualizer window destroyed")
