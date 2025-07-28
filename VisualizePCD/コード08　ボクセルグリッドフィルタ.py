#
#ボクセルグリッドフィルタ
#
#点群を一定範囲のボックス内に1つにすることで点の数を減らすやつ
#立方体のグリッドを配置しそのグリッド内の重心だけにする
##

import open3d as o3d
import numpy as np

# Bunnyモデルのメッシュデータのパスを取得し読み込む
bunny_mesh_path = o3d.data.BunnyMesh().path
bunny_mesh = o3d.io.read_triangle_mesh(bunny_mesh_path)

# メッシュを点群に変換
bunny_pcd = bunny_mesh.sample_points_uniformly(number_of_points=2000)

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウドと座標軸を追加
vis.add_geometry(bunny_pcd)

# 描画
vis.run()
vis.destroy_window()


# ボクセルグリッドフィルタ（Open3D）
downpcd_voxel = bunny_pcd.voxel_down_sample(voxel_size=0.02)

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウドと座標軸を追加
vis.add_geometry(downpcd_voxel)

# 描画
vis.run()
vis.destroy_window()

# フィルタリング後の点群の座標を表示
bunny_voxel = np.asarray(downpcd_voxel.points)
print(bunny_voxel)
