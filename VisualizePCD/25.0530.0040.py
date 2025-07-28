#
#Z軸方向で分割して鉄筋を検出しようとしてる
#
##

import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import threading

# --- ① CSV読み込みと初期回転 ---
df = pd.read_csv('mid360_XYZ_Reflect_data_03-28-13-53-11.csv')
data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
reflect_raw = df.iloc[:, 3].to_numpy(dtype=np.float32)

# 角度調整（30+2度）
theta = np.radians(32.0)
RotationMat = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])
data_rotated = np.dot(RotationMat, data_xyz.T).T

# 距離補正（1.5乗）
distances = np.linalg.norm(data_rotated, axis=1)
reflect_corrected = reflect_raw / (distances**1.5 + 1e-6)

# --- ② PCAで壁面をZ軸に揃える関数 ---
def align_plane_to_axis(points):
    if points.shape[1] != 3:
        raise ValueError(f"点群データは (N, 3) の形状である必要があります．現在の形状: {points.shape}")
    
    centered = points - points.mean(axis=0)
    # フル行列を回避（メモリ節約）
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal_vec = vh[-1]  # 法線ベクトル（Z軸にしたい向き）

    z_axis = np.array([0, 0, 1])
    v = np.cross(normal_vec, z_axis)
    c = np.dot(normal_vec, z_axis)
    s = np.linalg.norm(v)

    if s < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s**2))
    
    aligned_points = np.dot(R, points.T).T
    return aligned_points


# --- ③ 表示関数 ---
def show_histogram(reflect_norm, title):
    plt.figure(figsize=(6, 4))
    plt.hist(reflect_norm, bins=100, range=(0, 1), color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Normalized Reflectance (0–1)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_pointcloud(points, colors, window_name):
    def run_open3d():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1600, height=1200)
        vis.add_geometry(pcd)
        vis.add_geometry(cs)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        vis.run()
        vis.destroy_window()

    thread = threading.Thread(target=run_open3d)
    thread.start()

# --- ④ 処理関数 ---
def process_and_display(x_range, y_range, z_range, window_name, hist_title):
    mask = (
        (data_rotated[:, 0] >= x_range[0]) & (data_rotated[:, 0] <= x_range[1]) &
        (data_rotated[:, 1] >= y_range[0]) & (data_rotated[:, 1] <= y_range[1]) &
        (data_rotated[:, 2] >= z_range[0]) & (data_rotated[:, 2] <= z_range[1])
    )
    filtered_points = data_rotated[mask]
    filtered_reflect = reflect_corrected[mask]

    aligned_points = align_plane_to_axis(filtered_points)

    r_clip_min = np.percentile(filtered_reflect, 0)
    r_clip_max = np.percentile(filtered_reflect, 99.5)
    filtered_reflect = np.clip(filtered_reflect, r_clip_min, r_clip_max)
    reflect_norm = (filtered_reflect - r_clip_min) / (r_clip_max - r_clip_min + 1e-6)

    colors = plt.cm.jet(reflect_norm)[:, :3]

    show_pointcloud(aligned_points, colors, window_name)
    show_histogram(reflect_norm, hist_title)

# --- ⑤ 小領域の例 ---
process_and_display(
    x_range=(3000, 4000),
    y_range=(-500, 700),
    z_range=(-1500, -500),
    window_name="Reflectance Colored (Zoomed & Aligned)",
    hist_title="Reflectance Histogram (Zoomed View)"
)

# --- ⑥ 層分割と各層のヒストグラム ---
def visualize_z_layers_and_hist(points, reflect, n_layers=5, title_prefix="Layer"):
    aligned_points = align_plane_to_axis(points)
    z_vals = aligned_points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_bins = np.linspace(z_min, z_max, n_layers + 1)

    reflect = np.asarray(reflect)
    aligned_reflect = reflect  # 対応する反射強度値（要：同じ順番）

    color_map = plt.cm.get_cmap('tab10', n_layers)
    layer_colors = np.zeros_like(aligned_points)
    
    for i in range(n_layers):
        mask = (z_vals >= z_bins[i]) & (z_vals < z_bins[i + 1])
        layer_colors[mask] = color_map(i)[:3]

        # ヒストグラム（正規化＆クリップ込み）
        r_layer = aligned_reflect[mask]
        if len(r_layer) == 0:
            continue
        r_min = np.percentile(r_layer, 0)
        r_max = np.percentile(r_layer, 99.5)
        r_layer = np.clip(r_layer, r_min, r_max)
        r_norm = (r_layer - r_min) / (r_max - r_min + 1e-6)

        show_histogram(r_norm, f"{title_prefix} {i+1} Histogram")

    show_pointcloud(aligned_points, layer_colors, f"{title_prefix} Colored PointCloud")

# --- ⑦ 実行：全体点群を層分割し，ヒストグラム表示 ---
visualize_z_layers_and_hist(data_rotated, reflect_corrected, n_layers=5, title_prefix="Z-Layer")
