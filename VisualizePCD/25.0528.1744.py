#
#読み込んで開くだけ
#全体表示で範囲区切ってもない
#表示窓の大きさ変更#

import pandas as pd
import open3d as o3d
import numpy as np

# CSV読み込み（ヘッダーあり）
df = pd.read_csv('mid360_XYZ_Reflect_data_03-28-13-53-11.csv')

# x,y,zの3列だけ抽出しnumpy配列に
data01 = df.iloc[:, :3].to_numpy(dtype=np.float32)

theta = np.radians(30 + 2.0)
RotationMat = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])

# センサー設置角度分だけ回転（必要なら使う）
data02 = np.dot(RotationMat, data01.T).T

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data02)

cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)


# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウドと座標軸を追加
vis.add_geometry(pcd)
vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()

