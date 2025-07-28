#
#108,110のあわせ
#
#フィルタリング　yz平面で区切る
#それぞれでxに近い方で点抽出
#点選んで近傍点を2つ選んで三角形を表示
#面積が大きいものを排除
#ループで点選び続けて一度選んだ点は選ばないように，2点以下になったら終了
#データの数表示，最初，フィルタリング後，区切った後，三角形の数
#
#抽出されているデータの表示飛ばしてる
#三角形作ってそれらで面を生成が厳しい
#面がばらばらすぎる
#intervalの設定で三角形の面変えれるかも#
#



import csv
import open3d as o3d
import numpy as np
from scipy.spatial import distance

# 三角形の面積を計算する関数
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

#初期データ数の表示
print(len(data01))

# 回転行列の設定（センサーの設置角度は30度）
theta = np.radians(30 + 2.0)
RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0], 
                        [-np.sin(theta), 0, np.cos(theta)]])

# データの回転
data02 = np.dot(RotationMat, data01.T).T

# z軸回りの回転行列の設定（-3度回転）
theta_z = np.radians(-3.0)
RotationMat_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                          [np.sin(theta_z), np.cos(theta_z), 0], 
                          [0, 0, 1]])

# データの回転（z軸回り）
data002 = np.dot(RotationMat_z, data02.T).T

# 条件に基づいてデータをフィルタリング
data03 = data002[data002[:, 0] > 0.0]
data04 = data03[data03[:, 1] < 1600.0]
data05 = data04[data04[:, 1] > -2000.0]
data06 = data05[data05[:, 0] < 5000.0]
data07 = data06[data06[:, 2] > -850.0]

#フィルタリング後のデータの数の表示
print(len(data07))

# ユーザーが指定する範囲の入力
y_min = -2000  # 例: -2000
y_max = 1600   # 例: 1600
z_min = -850   # 例: -850
z_max = 1000   # 例: 1000
max_area = 10000.0  # 最小面積の指定 

# # ユーザーが指定する範囲の入力
# y_min = float(input("Enter y_min: "))  # 例: -2000
# y_max = float(input("Enter y_max: "))  # 例: 1600
# z_min = float(input("Enter z_min: "))  # 例: -850
# z_max = float(input("Enter z_max: "))  # 例: 1000
# max_area  = float(input("Enter max_area: ")) #最大面積の指定

# ポイントクラウドの作成
pcd = o3d.geometry.PointCloud()


# YZ平面方向に区切る
interval = 100.0  # 微小区間のサイズ

# 色を設定するためのマップと区間のリストを作成
color_map = [
    [1, 0, 0],  # 赤
    [1, 1, 0],  # 黄
    [0, 1, 0],  # 緑
    [0, 0, 1]   # 青
]

selected_points = []
selected_colors = []

# YZ平面の区間ごとに処理
for y_start in np.arange(y_min, y_max, interval):
    for z_start in np.arange(z_min, z_max, interval):
        # 現在の区間に属する点を抽出
        mask = (data07[:, 1] >= y_start) & (data07[:, 1] < y_start + interval) & \
               (data07[:, 2] >= z_start) & (data07[:, 2] < z_start + interval)
        section_points = data07[mask]
        
        if len(section_points) > 0:
            # x軸方向で最も手前の点を選択
            nearest_point = section_points[np.argmin(section_points[:, 0])]
            selected_points.append(nearest_point)
            
            # 色を設定
            color = color_map[(int(y_start / interval) + int(z_start / interval)) % len(color_map)]
            selected_colors.append(color)

#選択したデータの数の表示
print(len(selected_points))       

# 選択した点と色をポイントクラウドに設定
pcd.points = o3d.utility.Vector3dVector(np.array(selected_points))
pcd.colors = o3d.utility.Vector3dVector(np.array(selected_colors))

# コーディネートフレームの作成
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# # まず抽出されたデータを表示　枠でかいから大きさ変更
# #o3d.visualization.draw_geometries([pcd, cs])

# # Visualizerを使用してウィンドウのサイズを設定
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定

# # ポイントクラウドと座標軸を追加
# vis.add_geometry(pcd)
# vis.add_geometry(cs)

# # 描画
# vis.run()
# vis.destroy_window()



# 使用する点を追跡するためのリスト
remaining_points = selected_points
triangles = []

# ランダムに点を選び、近傍点と三角形を作成
while len(remaining_points) > 2:
    # ランダムに1点を選択
    selected_point2 = remaining_points.pop(np.random.randint(len(remaining_points)))

    # 選択した点に最も近い2点を見つける
    dists = distance.cdist([selected_point2], remaining_points, 'euclidean')[0]
    closest_indices = np.argsort(dists)[:2]
    closest_points = [remaining_points[i] for i in closest_indices]

    # 三角形の頂点として追加
    triangle = [selected_point2] + closest_points

    # 三角形の面積を計算
    area = triangle_area(np.array(triangle[0]), np.array(triangle[1]), np.array(triangle[2]))

    # 面積が指定された最大面積より小さい場合のみ追加
    if area <= max_area:
        triangles.append(triangle)

    # 選ばれた2点をremaining_pointsから除去
    for i in sorted(closest_indices, reverse=True):
        del remaining_points[i]

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

    #三角形の数を表示
    print(len(triangles))

    # Visualizerを使用してウィンドウのサイズを設定
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定
    
    # メッシュの両面描画を有効にする
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True  # 両面描画を有効にするオプション

    # ポイントクラウドとメッシュを追加
    vis.add_geometry(pcd)
    vis.add_geometry(mesh)
    #vis.add_geometry(cs)

    # 描画
    vis.run()
    vis.destroy_window()
else:
    print("Not enough points to create a surface.")



# # 抽出したデータからランダムに3点を選択
# if len(selected_points) >= 3:
#     random_indices = np.random.choice(len(selected_points), 3, replace=False)
#     triangle_points = np.array(selected_points)[random_indices]

#     # 面（三角形）を作成
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(triangle_points)
#     mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    
#     # 面の色を設定（例としてグレーに設定）
#     mesh.paint_uniform_color([0.5, 0.5, 0.5])

#     # 法線の計算（両面描画のために必要）
#     mesh.compute_vertex_normals()
    
#     # Visualizerを使用してウィンドウのサイズを設定
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定
    
#     # メッシュの両面描画を有効にする
#     opt = vis.get_render_option()
#     opt.mesh_show_back_face = True  # 両面描画を有効にするオプション
    
#     # ポイントクラウドと座標軸と面を追加
#     vis.add_geometry(pcd)
#     vis.add_geometry(cs)
#     vis.add_geometry(mesh)  # 面を追加
    
#     # 描画
#     vis.run()
#     vis.destroy_window()
# else:
#     print("Unable to create surface")
