# ファイル名: step7.6_region_growing_robust_normals.py
# 概要: RANSACベースの頑健な法線推定を導入し、より正確な領域成長を目指す。(AnimationState修正版)
#       法線計算に非常に時間がかかるため注意。

import open3d as o3d
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm

# ==============================================================================
# 1. 準備と計算のための関数群
# ==============================================================================
def estimate_normals_robust(pcd, radius, max_nn, num_iterations, angle_thresh_deg=15.0):
    print("\n【計算フェーズ】頑健な法線推定を開始します... (非常に時間がかかります)")
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

        if best_normal is not None:
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
    estimate_normals_robust(pcd, **robust_normal_params)
    print(f"読み込みと法線計算が完了。点の数: {len(pcd.points)}"); return pcd

def get_initial_seeds(pcd, dist_thresh, normal_x_thresh):
    print(f"\n法線フィルタとRANSACで最初のシードを検出中..."); normals = np.asarray(pcd.normals); wall_candidate_indices = np.where(np.abs(normals[:, 0]) >= normal_x_thresh)[0]
    if len(wall_candidate_indices) < 100: print("警告: 正面向きの点が少なすぎます。"); return None
    pcd_candidates = pcd.select_by_index(wall_candidate_indices); plane_model, seed_indices_local = pcd_candidates.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000); 
    if not seed_indices_local: return None
    seed_indices = wall_candidate_indices[seed_indices_local]
    core_pcd = pcd.select_by_index(seed_indices); labels = np.array(core_pcd.cluster_dbscan(eps=150.0, min_points=10, print_progress=False))
    if len(labels) > 0 and labels.max() != -1:
        main_cluster_label = pd.Series(labels[labels!=-1]).mode()[0]; main_cluster_indices = np.where(labels == main_cluster_label)[0]
        seed_indices = np.array(seed_indices)[main_cluster_indices]
    return seed_indices

def calculate_growth_history(pcd, seed_indices, normal_angle_thresh_deg, search_radius, save_interval):
    print("\n【計算フェーズ】領域成長の全ステップを計算中..."); history = []; queue = deque(seed_indices); wall_indices = set(seed_indices); all_normals = np.asarray(pcd.normals); kdtree = o3d.geometry.KDTreeFlann(pcd); angle_threshold_rad = np.deg2rad(normal_angle_thresh_deg); history.append(list(wall_indices)); points_since_last_save = 0
    with tqdm(total=len(np.asarray(pcd.points)), desc="領域成長中") as pbar:
        pbar.update(len(wall_indices))
        while queue:
            current_idx = queue.popleft(); current_normal = all_normals[current_idx]
            k, neighbor_indices, _ = kdtree.search_radius_vector_3d(pcd.points[current_idx], search_radius)
            for neighbor_idx in neighbor_indices:
                if neighbor_idx in wall_indices: continue
                dot_product = np.clip(np.dot(current_normal, all_normals[neighbor_idx]), -1.0, 1.0)
                if np.arccos(dot_product) < angle_threshold_rad:
                    wall_indices.add(neighbor_idx); queue.append(neighbor_idx); pbar.update(1); points_since_last_save += 1
                    if points_since_last_save >= save_interval: history.append(list(wall_indices)); points_since_last_save = 0
    history.append(list(wall_indices)); print("計算が完了しました。"); return history

# ==============================================================================
# 2. 【再生フェーズ】のクラスと関数
# ==============================================================================
class AnimationState:
    def __init__(self, full_pcd, history_world):
        self.full_pcd_points = np.asarray(full_pcd.points)
        self.history = history_world
        self.current_frame = 0
        self.max_frame = len(history_world) - 1
        self.pcd_to_display = o3d.geometry.PointCloud()
        self.update_geometry()

    def update_geometry(self):
        current_indices = self.history[self.current_frame]
        previous_indices = set(self.history[self.current_frame - 1]) if self.current_frame > 0 else set()
        
        points_to_show = self.full_pcd_points[current_indices]
        colors = np.full_like(points_to_show, [0.8, 0.4, 0.4]) # 既存点の色 (薄い赤)
        
        # ▼▼▼【重要修正】新規追加点の特定方法を高速化 ▼▼▼
        # setの差分計算を使うことで、numpyの全要素比較(isin)よりも遥かに高速になる
        new_indices_set = set(current_indices) - previous_indices
        if new_indices_set: # 新規追加点がある場合のみ
            # current_indicesの中から、新規追加点のインデックスがどこにあるかを探す
            # 辞書を使って高速にマッピング
            current_indices_map = {idx: i for i, idx in enumerate(current_indices)}
            locs_in_current_array = [current_indices_map[idx] for idx in new_indices_set]
            colors[locs_in_current_array] = [1.0, 0.0, 0.0] # 新規追加点の色 (明るい赤)
        
        self.pcd_to_display.points = o3d.utility.Vector3dVector(points_to_show)
        self.pcd_to_display.colors = o3d.utility.Vector3dVector(colors)
        print(f"\rフレーム {self.current_frame}/{self.max_frame} を表示中 (点の数: {len(points_to_show)})", end="")

def go_to_next_frame(vis, state):
    if state.current_frame < state.max_frame:
        state.current_frame += 1; state.update_geometry(); vis.update_geometry(state.pcd_to_display)

def go_to_prev_frame(vis, state):
    if state.current_frame > 0:
        state.current_frame -= 1; state.update_geometry(); vis.update_geometry(state.pcd_to_display)

# ==============================================================================
# 3. メイン処理
# ==============================================================================
if __name__ == '__main__':
    # --- パラメータ設定 ---
    FILE_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'
    INITIAL_SEED_DIST_THRESH = 15.0
    NORMAL_X_THRESHOLD = 0.9
    NEIGHBOR_SEARCH_RADIUS = 100.0
    NORMAL_ANGLE_THRESHOLD = 7.5
    SAVE_STATE_INTERVAL = 500
    ROBUST_NORMAL_ITERATIONS = 10 # 1点あたりのサンプリング回数 (少し下げてテストしやすく)

    # --- 1. 計算フェーズ ---
    robust_normal_params = {"radius": NEIGHBOR_SEARCH_RADIUS, "max_nn": 30, "num_iterations": ROBUST_NORMAL_ITERATIONS}
    pcd_world = load_and_prepare_pcd(FILE_PATH, robust_normal_params)
    if pcd_world is None: exit()
    
    initial_seeds = get_initial_seeds(pcd_world, INITIAL_SEED_DIST_THRESH, NORMAL_X_THRESHOLD)
    if initial_seeds is None: exit("エラー: 最初のシードが見つかりませんでした。")
    print(f"正面の壁からシードとして {len(initial_seeds)} 点が見つかりました。")

    growth_history_world = calculate_growth_history(
        pcd_world, initial_seeds, NORMAL_ANGLE_THRESHOLD, NEIGHBOR_SEARCH_RADIUS, SAVE_STATE_INTERVAL)

    if not growth_history_world or len(growth_history_world[-1]) == 0:
        exit("領域成長の結果、点が検出されませんでした。")

    # --- 2. 再生フェーズ ---
    print("\n\n【再生フェーズ】インタラクティブ再生を開始します...")
    print("  - 'N' キー: 次のフレームへ")
    print("  - 'B' キー: 前のフレームへ")
    print("  - 'Q' キー: 終了")

    state = AnimationState(pcd_world, growth_history_world)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="プログレッシブ表示アニメーション (N:進む, B:戻る)")
    vis.add_geometry(state.pcd_to_display)
    vis.register_key_callback(ord("N"), lambda vis: go_to_next_frame(vis, state))
    vis.register_key_callback(ord("B"), lambda vis: go_to_prev_frame(vis, state))
    vis.run(); vis.destroy_window()

    # --- 3. 最終結果の全体表示フェーズ ---
    print("\n\n【最終結果表示フェーズ】全体の点群と検出結果を重ねて表示します...")
    final_wall_indices = growth_history_world[-1]
    final_colors = np.full((len(np.asarray(pcd_world.points)), 3), [0.5, 0.5, 0.5])
    final_colors[final_wall_indices] = [1.0, 0.0, 0.0]
    pcd_world.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.visualization.draw_geometries([pcd_world], window_name="最終結果：全体表示")
    
    print("\n===== 全ての処理が完了しました =====")
