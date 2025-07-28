
#距離補正
#正規化
#カラーマップ
#
#壁面
#
#
#pcaで壁面をZ軸に
#
#正規化するときのパーセンテージで割とかわる#

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
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered)
    normal_vec = vh[-1]  # 法線ベクトル

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

    # PCAで壁面をZ軸に整列
    aligned_points = align_plane_to_axis(filtered_points)

    # 反射強度のクリップ＆正規化（上位1%をカット）
    r_clip_min = np.percentile(filtered_reflect, 0)
    r_clip_max = np.percentile(filtered_reflect, 99.5)
    filtered_reflect = np.clip(filtered_reflect, r_clip_min, r_clip_max)
    reflect_norm = (filtered_reflect - r_clip_min) / (r_clip_max - r_clip_min + 1e-6)

    # 色マップ（Jet）
    colors = plt.cm.jet(reflect_norm)[:, :3]

    # 表示
    show_pointcloud(aligned_points, colors, window_name)
    show_histogram(reflect_norm, hist_title)

# --- ⑤ 実行例（小領域） ---
process_and_display(
    x_range=(3000, 4000),
    y_range=(-500, 700),
    z_range=(-1500, -500),
    window_name="Reflectance Colored (Zoomed & Aligned)",
    hist_title="Reflectance Histogram (Zoomed View)"
)
