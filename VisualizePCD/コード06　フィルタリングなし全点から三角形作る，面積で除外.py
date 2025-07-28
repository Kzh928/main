#
#面積の条件を見たさない三角形を除外
#入力した数値より小さいやつだけえらぶ
#一回表示　一旦選んだ点群だけみたい
#座標軸でてない
#全部の点から三角形を作ってる
#一番手前とか区切るとかの処理無しで三角形作るだけ#



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

# ユーザーが指定する範囲の入力
y_min = -2000  # 例: -2000
y_max = 1600   # 例: 1600
z_min = -850   # 例: -850
z_max = 1000   # 例: 1000
min_area = 1000.0  # 最大面積の指定 (例: 50.0)

# ポイントクラウドの作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data07)

# 使用する点を追跡するためのリスト
remaining_points = data07.tolist()
triangles = []

# ランダムに点を選び、近傍点と三角形を作成
while len(remaining_points) > 2:
    # ランダムに1点を選択
    selected_point = remaining_points.pop(np.random.randint(len(remaining_points)))

    # 選択した点に最も近い2点を見つける
    dists = distance.cdist([selected_point], remaining_points, 'euclidean')[0]
    closest_indices = np.argsort(dists)[:2]
    closest_points = [remaining_points[i] for i in closest_indices]

    # 三角形の頂点として追加
    triangle = [selected_point] + closest_points

    # 三角形の面積を計算
    area = triangle_area(np.array(triangle[0]), np.array(triangle[1]), np.array(triangle[2]))

    # 面積が指定された最大面積より小さい場合のみ追加
    if area <= min_area:
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

    # Visualizerを使用してウィンドウのサイズを設定
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定
    
    # メッシュの両面描画を有効にする
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True  # 両面描画を有効にするオプション
    
    # コーディネートフレームの作成
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0000000)
    
    # # ビューコントロールの設定（クリッピング範囲を調整）
    # ctr = vis.get_view_control()
    # ctr.set_zoom(0.00001)  # ズームの設定（調整可能）
    # ctr.set_front([0.0, 0.0, -1.0])  # 視点の設定（調整可能）
    # ctr.set_lookat([0.0, 0.0, 0.0])  # 視点の位置
    # ctr.set_up([0.0, 1.0, 0.0])  # 上方向の設定


    # ポイントクラウドとメッシュを追加
    vis.add_geometry(pcd)
    vis.add_geometry(mesh)
    vis.add_geometry(cs)

    # 描画
    vis.run()
    vis.destroy_window()
else:
    print("Not enough points to create a surface.")
