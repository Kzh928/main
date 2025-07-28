#ランダム点群，ボックス，座標フレーム


import numpy as np
import open3d as o3d
import os
print(os.getcwd())

# ビジュアライザを作成
visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=1080, height=720)
print("Window created")

# ランダムなポイントを生成
points = np.random.normal(0, 0.5, (1000, 3))
print("Points generated")

#tri_geometry = o3d.geometry.TriangleMesh()
#print("TriangleMesh object created")
#box = tri_geometry.create_box(width=1000, height=1000,depth=1000)


# PointCloudオブジェクトを作成
pc_geometry = o3d.geometry.PointCloud()
print("PointCloud object created")

# ポイントをPointCloudオブジェクトに追加
pc_geometry.points = o3d.utility.Vector3dVector(points)
print("Points added to PointCloud")

# PointCloudオブジェクトをビジュアライザに追加
visualizer.add_geometry(pc_geometry)
print("PointCloud added to visualizer")


tri_geometry = o3d.geometry.TriangleMesh()
print("TriangleMesh object created")
box = tri_geometry.create_box(width=1, height=1,depth=1)
axis = tri_geometry.create_coordinate_frame(size=1.0)
print("Coordinate frame created")
visualizer.add_geometry(box)
visualizer.add_geometry(axis)
print("Coordinate frame added to visualizer")

# ビジュアライザを実行
try:
    visualizer.run()
    print("Visualizer running")
except Exception as e:
    print(f"Failed to run visualizer: {e}")

# ビジュアライザを破棄
visualizer.destroy_window()
print("Visualizer window destroyed")