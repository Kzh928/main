# ファイル名: step7.7_region_growing_robust_fast.py
# 概要: 頑健な法線推定で壁面を検出し、最後に全体表示と抽出部分のみの表示を行う。

import open3d as o3d
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm

# ==============================================================================
# 1. 準備と計算のための関数群
# ==============================================================================

def estimate_normals_robust_fast(pcd, radius, max_nn, num_iterations, consensus_angle_deg):
    print("\n【計算フェーズ】高速版の頑健な法線推定を開始します...")
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    robust_normals = np.zeros_like(points)
    cos_angle_thresh = np.cos(np.deg2rad(consensus_angle_deg))

    for i in tqdm(range(len(points)), desc="ロバスト法線計算中"):
        k, neighbor_indices, _ = kdtree.search_hybrid_vector_3d(points[i], radius=radius, max_nn=max_nn)
        if k < 3: continue

        neighbor_points = points[neighbor_indices]
        
        p1_indices = np.random.choice(k, num_iterations)
        p2_indices = np.random.choice(k, num_iterations)
        p3_indices = np.random.choice(k, num_iterations)

        p1s = neighbor_points[p1_indices]
        p2s = neighbor_points[p2_indices]
        p3s = neighbor_points[p3_indices]

        v1s = p2s - p1s
        v2s = p3s - p1s
        
        candidate_normals = np.cross(v1s, v2s)
        norms = np.linalg.norm(candidate_normals, axis=1)
        
        valid_mask = norms > 1e-6
        if not np.any(valid_mask): continue
        candidate_normals = candidate_normals[valid_mask]
        norms = norms[valid_mask]
        candidate_normals /= norms[:, np.newaxis]

        dot_product_matrix = np.dot(candidate_normals, candidate_normals.T)
        inlier_counts = np.sum(dot_product_matrix > cos_angle_thresh, axis=1)
        
        best_candidate_index = np.argmax(inlier_counts)
        best_normal = candidate_normals[best_candidate_index]

        p_to_origin = -points[i]
        if np.dot(best_normal, p_to_origin) < 0:
            best_normal *= -1
            
        robust_normals[i] = best_normal

    pcd.normals = o3d.utility.Vector3dVector(robust_normals)
    print("頑健な法線推定が完了しました。")

def load_and_prepare_pcd(file_path, robust_normal_params):
    print(f"'{file_path}' を読み込み中...");
    try: df = pd.read_csv(file_path)
    except FileNotFoundError: print(f"エラー: ファイル '{file_path}' が見つかりません。"); return None
    data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32); theta = np.radians(30 + 2.0); RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]); points = np.dot(RotationMat, data_xyz.T).T; pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points)
    estimate_normals_robust_fast(pcd, **robust_normal_params)
    print(f"読み込みと法線計算が完了。点の数: {len(pcd.points)}"); return pcd

def get_initial_seeds(pcd, dist_thresh, normal_x_thresh):
    print(f"\n法線フィルタとRANSACで最初のシードを検出中..."); normals = np.asarray(pcd.normals); wall_candidate_indices = np.where(np.abs(normals[:, 0]) >= normal_x_thresh)[0]
    if len(wall_candidate_indices) < 100: print("警告: 正面向きの点が少なすぎます。"); return None
    pcd_candidates = pcd.select_by_index(wall_candidate_indices); plane_model, seed_indices_local = pcd_candidates.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000)
    if not seed_indices_local: return None
    seed_indices = wall_candidate_indices[seed_indices_local]
    core_pcd = pcd.select_by_index(seed_indices); labels = np.array(core_pcd.cluster_dbscan(eps=150.0, min_points=10, print_progress=False))
    if len(labels) > 0 and labels.max() != -1:
        main_cluster_label = pd.Series(labels[labels!=-1]).mode()[0]; main_cluster_indices = np.where(labels == main_cluster_label)[0]
        seed_indices = np.array(seed_indices)[main_cluster_indices]
    return seed_indices

def calculate_wall_indices(pcd, seed_indices, normal_angle_thresh_deg, search_radius):
    print("\n領域成長で壁面全体を特定中..."); queue = deque(seed_indices); wall_indices = set(seed_indices); all_normals = np.asarray(pcd.normals); kdtree = o3d.geometry.KDTreeFlann(pcd); angle_threshold_rad = np.deg2rad(normal_angle_thresh_deg)
    with tqdm(total=len(np.asarray(pcd.points)), desc="壁面検出中") as pbar:
        pbar.update(len(wall_indices))
        while queue:
            current_idx = queue.popleft(); current_normal = all_normals[current_idx]
            k, neighbor_indices, _ = kdtree.search_radius_vector_3d(pcd.points[current_idx], search_radius)
            for neighbor_idx in neighbor_indices:
                if neighbor_idx in wall_indices: continue
                dot_product = np.clip(np.dot(current_normal, all_normals[neighbor_idx]), -1.0, 1.0)
                if np.arccos(dot_product) < angle_threshold_rad:
                    wall_indices.add(neighbor_idx); queue.append(neighbor_idx); pbar.update(1)
    print("壁面の特定が完了しました。"); return list(wall_indices)

# --- メイン処理 ---
if __name__ == '__main__':
    # --- パラメータ設定 ---
    FILE_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'
    INITIAL_SEED_DIST_THRESH = 15.0
    NORMAL_X_THRESHOLD = 0.9
    NORMAL_ANGLE_THRESHOLD = 5.0
    NEIGHBOR_SEARCH_RADIUS = 100.0
    ROBUST_NORMAL_ITERATIONS = 30
    ROBUST_NORMAL_CONSENSUS_ANGLE = 10.0

    # --- 1. 計算フェーズ ---
    robust_normal_params = {
        "radius": NEIGHBOR_SEARCH_RADIUS, "max_nn": 30,
        "num_iterations": ROBUST_NORMAL_ITERATIONS,
        "consensus_angle_deg": ROBUST_NORMAL_CONSENSUS_ANGLE
    }
    pcd_world = load_and_prepare_pcd(FILE_PATH, robust_normal_params)
    if pcd_world is None: exit()
    
    initial_seeds = get_initial_seeds(pcd_world, INITIAL_SEED_DIST_THRESH, NORMAL_X_THRESHOLD)
    if initial_seeds is None: exit("エラー: 最初のシードが見つかりませんでした。")
    print(f"正面の壁からシードとして {len(initial_seeds)} 点が見つかりました。")

    final_wall_indices = calculate_wall_indices(
        pcd_world, initial_seeds, NORMAL_ANGLE_THRESHOLD, NEIGHBOR_SEARCH_RADIUS)

    if not final_wall_indices:
        exit("エラー: 領域成長の結果、点が検出されませんでした。")

    # --- 2. 最終結果表示フェーズ (全体表示) ---
    print("\n【最終結果表示フェーズ】全体の点群と検出結果を重ねて表示します...")
    final_colors = np.full((len(np.asarray(pcd_world.points)), 3), [0.5, 0.5, 0.5])
    final_colors[final_wall_indices] = [1.0, 0.0, 0.0]
    pcd_world.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.visualization.draw_geometries([pcd_world], window_name="最終結果：全体表示")
    
    # === ▼▼▼【ここから追記】▼▼▼ ===
    # 3. 抽出された壁面点群のみを表示
    print("\n【抽出部分のみ表示フェーズ】検出された壁の点群だけを表示します...")
    
    # final_wall_indices を使って、pcd_world から壁と判断された点のみを選択
    pcd_wall_only = pcd_world.select_by_index(final_wall_indices)
    
    # 選択した点群を一色（赤色）で塗りつぶす
    pcd_wall_only.paint_uniform_color([1.0, 0.0, 0.0])
    
    # 新しいウィンドウで抽出された点群のみを表示
    o3d.visualization.draw_geometries([pcd_wall_only], window_name="最終結果：抽出された壁面のみ")
    # === ▲▲▲【ここまで追記】▲▲▲ ===

    print("\n===== 全ての処理が完了しました =====")
