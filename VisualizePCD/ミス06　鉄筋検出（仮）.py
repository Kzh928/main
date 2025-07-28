#
#鉄筋検出ミス
#選んだ点を頂点に持つ三角形２つ出てくるだけのやつ
#5面体作りたかったけど5面体作っても
#　鉄筋検出できるほどの精度無い
#5面体で作ったやつきえた#

import csv
import open3d as o3d
import numpy as np
from scipy.spatial import distance

def triangle_area(p1, p2, p3):
    # 三角形の辺の長さを計算
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p3)
    c = np.linalg.norm(p3 - p1)
    
    # ヘロンの公式を用いて面積を計算
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return area

# CSVからデータを読み込む
with open('mid360_XYZdata_06-25-10-03-54.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data01 = np.asarray(data, dtype=np.float32)

# 初期データ数の表示
print(len(data01))

# 回転行列の設定
theta = np.radians(32.0)  # 30度 + 2度
RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0], 
                        [-np.sin(theta), 0, np.cos(theta)]])

# データの回転
data02 = np.dot(RotationMat, data01.T).T

# z軸回りの回転行列の設定
theta_z = np.radians(-3.0)
RotationMat_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                          [np.sin(theta_z), np.cos(theta_z), 0], 
                          [0, 0, 1]])

# データの回転（z軸回り）
data002 = np.dot(RotationMat_z, data02.T).T

# データのフィルタリング
data03 = data002[data002[:, 0] > 0.0]
data04 = data03[data03[:, 1] < 1600.0]
data05 = data04[data04[:, 1] > -2000.0]
data06 = data05[data05[:, 0] < 5000.0]
data07 = data06[data06[:, 2] > -850.0]

# フィルタリング後のデータの数の表示
print(len(data07))

# ユーザーが指定する範囲の入力
y_min, y_max = -2000, 1600
z_min, z_max = -850, 1000
interval = 30.0  # 微小区間のサイズ
max_area = 100.0  # 最大面積の指定

# ポイントクラウドの作成
pcd = o3d.geometry.PointCloud()

# YZ平面方向に区切る
color_map = [
    [1, 0, 0],  # 赤
    [1, 1, 0],  # 黄
    [0, 1, 0],  # 緑
    [0, 0, 1]   # 青
]

selected_points = []
selected_colors = []

for y_start in np.arange(y_min, y_max, interval):
    for z_start in np.arange(z_min, z_max, interval):
        mask = (data07[:, 1] >= y_start) & (data07[:, 1] < y_start + interval) & \
               (data07[:, 2] >= z_start) & (data07[:, 2] < z_start + interval)
        section_points = data07[mask]
        
        if len(section_points) > 0:
            nearest_point = section_points[np.argmin(section_points[:, 0])]
            selected_points.append(nearest_point)
            color = color_map[(int(y_start / interval) + int(z_start / interval)) % len(color_map)]
            selected_colors.append(color)

print(len(selected_points))

pcd.points = o3d.utility.Vector3dVector(np.array(selected_points))
pcd.colors = o3d.utility.Vector3dVector(np.array(selected_colors))

# コーディネートフレームの作成
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# # Visualizerを使用してウィンドウのサイズを設定
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Custom Window", width=1600, height=1200)

# # ポイントクラウドと座標軸を追加
# vis.add_geometry(pcd)
# #vis.add_geometry(cs)
# vis.run()
# vis.destroy_window()

# 使用する点を追跡するためのリスト
remaining_points = selected_points
triangles = []

while len(remaining_points) > 2:
    selected_point2 = remaining_points.pop(np.random.randint(len(remaining_points)))

    dists = distance.cdist([selected_point2], remaining_points, 'euclidean')[0]
    closest_indices = np.argsort(dists)[:4]  # 近傍の4点を選択
    closest_points = [remaining_points[i] for i in closest_indices]

    # 点の選択
    if len(closest_points) >= 4:
        for i in range(len(closest_points) - 1):
            for j in range(i + 1, len(closest_points)):
                # 3点を選び、2つの三角形を作成
                triangle1 = [selected_point2, closest_points[i], closest_points[j]]
                triangle2 = [selected_point2, closest_points[j], closest_points[(j + 1) % len(closest_points)]]

                # 三角形の面積を計算
                area1 = triangle_area(*triangle1)
                area2 = triangle_area(*triangle2)

                if area1 <= max_area:
                    triangles.append(triangle1)
                if area2 <= max_area:
                    triangles.append(triangle2)

    if len(remaining_points) <= 2:
        break

# 三角形のメッシュを作成
if triangles:
    all_vertices = []
    all_triangles = []

    for i, triangle in enumerate(triangles):
        all_vertices.extend(triangle)
        all_triangles.append([3 * i, 3 * i + 1, 3 * i + 2])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(all_vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(all_triangles))

    # 面の色を設定（例としてグレーに設定）
    mesh.paint_uniform_color([0.5, 0.5, 0.5])

    # 法線の計算（両面描画のために必要）
    mesh.compute_vertex_normals()

    # 三角形の数を表示
    print(len(triangles))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Custom Window", width=1600, height=1200)

    # メッシュの両面描画を有効にする
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True

    vis.add_geometry(pcd)
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()
else:
    print("Not enough points to create a surface.")
