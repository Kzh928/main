# ファイル名: visualize_normals.py
# 概要: Open3Dの標準的な法線推定と、頑健な法線推定の結果を
#       それぞれ可視化して比較するためのスクリプト。

import open3d as o3d
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---
FILE_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'
# 頑健な法線推定に使うパラメータ
ROBUST_NORMAL_ITERATIONS = 10 # サンプリング回数
NEIGHBOR_SEARCH_RADIUS = 100.0
NEIGHBOR_MAX_NN = 30

# --- 関数群 ---

def load_and_prepare_pcd(file_path):
    """点群を読み込み、座標変換までを行う"""
    print(f"'{file_path}' を読み込み中...");
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。"); return None
    
    data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
    theta = np.radians(30 + 2.0)
    RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    points = np.dot(RotationMat, data_xyz.T).T
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"読み込み完了。点の数: {len(pcd.points)}");
    return pcd

def estimate_normals_robust(pcd, radius, max_nn, num_iterations, angle_thresh_deg=15.0):
    """RANSACベースの頑健な法線推定"""
    print("\n【計算中】頑健な法線推定... (非常に時間がかかります)")
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    robust_normals = np.zeros_like(points)
    angle_thresh_rad = np.deg2rad(angle_thresh_deg)

    for i in tqdm(range(len(points)), desc="ロバスト法線計算中"):
        k, neighbor_indices, _ = kdtree.search_hybrid_vector_3d(points[i], radius=radius, max_nn=max_nn)
        if k < 3: continue

        neighbor_points = points[neighbor_indices]
        candidate_normals = []
        for _ in range(num_iterations):
            sampled_points = neighbor_points[np.random.choice(k, 3, replace=False)]
            p1, p2, p3 = sampled_points
            v1 = p2 - p1; v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-6: continue
            normal /= norm_len
            candidate_normals.append(normal)
        
        if not candidate_normals: continue

        max_inliers = -1; best_normal = None
        for normal_A in candidate_normals:
            inliers = 0
            for normal_B in candidate_normals:
                dot_product = np.clip(np.dot(normal_A, normal_B), -1.0, 1.0)
                if np.arccos(dot_product) < angle_thresh_rad:
                    inliers += 1
            if inliers > max_inliers:
                max_inliers = inliers
                best_normal = normal_A

        p_to_origin = -points[i]
        if np.dot(best_normal, p_to_origin) < 0:
            best_normal *= -1
        robust_normals[i] = best_normal

    pcd.normals = o3d.utility.Vector3dVector(robust_normals)
    print("計算完了。")

# --- メイン処理 ---
if __name__ == '__main__':
    # 元の点群を読み込み
    pcd_base = load_and_prepare_pcd(FILE_PATH)
    if pcd_base is None: exit()

    # --- 1. 標準的な方法で法線を計算・表示 ---
    print("\n--- 1. Open3D標準の方法による法線 ---")
    pcd_standard = o3d.geometry.PointCloud(pcd_base) # コピーを作成
    print("法線を計算中...")
    pcd_standard.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NEIGHBOR_SEARCH_RADIUS, max_nn=NEIGHBOR_MAX_NN))
    print("ウィンドウを表示します。閉じて次に進んでください。")
    o3d.visualization.draw_geometries([pcd_standard], window_name="Standard Normals")

    # --- 2. 頑健な方法で法線を計算・表示 ---
    print("\n--- 2. 頑健な推定方法による法線 ---")
    pcd_robust = o3d.geometry.PointCloud(pcd_base) # コピーを作成
    estimate_normals_robust(pcd_robust, 
                            radius=NEIGHBOR_SEARCH_RADIUS, 
                            max_nn=NEIGHBOR_MAX_NN, 
                            num_iterations=ROBUST_NORMAL_ITERATIONS)
    print("ウィンドウを表示します。")
    o3d.visualization.draw_geometries([pcd_robust], window_name="Robust Normals")

    print("\n===== 全ての可視化が完了しました =====")
