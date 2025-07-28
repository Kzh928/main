import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import csv

# --- フィルタリング条件（変更可能） ---
filter_params = {
    "x_min": -0.5, "x_max": 0.5,
    "y_min": -0.5, "y_max": 0.5,
    "z_min": -0.5, "z_max": 0.5,
    "min_distance": 0.1  # 原点からの最小距離
}

# --- 任意の基準点（変更可能） ---
reference_point = np.array([-0.295, 0.11, -0.03])  # ここを変更すると別の基準点で距離を測れる

# --- CSVデータ読み込み（BOM対策） ---
filename = '2025.03.12.00.53.17.csv'
with open(filename, encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

data = np.asarray(data, dtype=np.float32)

# --- データが空の場合のチェック ---
if data.size == 0:
    raise ValueError("No Data")

# --- 点群データの抽出（x, y, zのみ） ---
points = data[:, :3]

# --- フィルタリング処理 ---
mask = (
    (points[:, 0] >= filter_params["x_min"]) & (points[:, 0] <= filter_params["x_max"]) &
    (points[:, 1] >= filter_params["y_min"]) & (points[:, 1] <= filter_params["y_max"]) &
    (points[:, 2] >= filter_params["z_min"]) & (points[:, 2] <= filter_params["z_max"]) &
    (np.linalg.norm(points, axis=1) >= filter_params["min_distance"])
)

filtered_points = points[mask]

# --- 指定した基準点に最も近い点を選ぶ ---
tree = KDTree(filtered_points)
_, closest_idx = tree.query(reference_point)  # reference_point に最も近い点のインデックス
closest_point = filtered_points[closest_idx]

# --- KDTreeを使って近傍4点を取得 ---
_, neighbor_indices = tree.query(reference_point, k=5)  # k=5（自身＋近傍4点）

# --- 近傍点の抽出 ---
selected_points = filtered_points[neighbor_indices]

# --- 近傍点との距離を計算 ---
neighbor_distances = np.linalg.norm(selected_points - closest_point, axis=1)

# --- 距離を表示 ---
print(f"Reference Point: {reference_point}")
print(f"Closest Point: {closest_point}")
for i in range(1, len(selected_points)):
    print(f"Near Point {i}: {selected_points[i]}, Distance: {neighbor_distances[i]:.4f}")

# --- Open3Dの点群オブジェクト作成 ---
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)

# --- 可視化用の色設定 ---
colors = np.zeros((len(filtered_points), 3))  # デフォルト（黒）
colors[:, :] = [0.3, 0.3, 0.3]  # 全体をグレーに

# --- 強調表示（基準点に最も近い点を赤，近傍点を黄色）---
colors[closest_idx] = [1, 0, 0]  # 赤
colors[neighbor_indices[1:]] = [1, 1, 0]  # 黄色

pcd.colors = o3d.utility.Vector3dVector(colors)

# --- 可視化 ---
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Filtered Point Cloud", width=1600, height=1200)

# ポイントクラウドと座標軸を追加
vis.add_geometry(pcd)
vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()
