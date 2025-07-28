# ファイル名: wall_detection_final_custom_coloring.py
# 概要: 領域成長の可視化の際、ユーザーが定義した
#       カスタムのループ範囲ごとに色分けする機能を追加。

import open3d as o3d
import numpy as np
import pandas as pd
import time

# ==============================================================================
# 1. 検出ロジックに関する関数群 (変更なし)
# ==============================================================================
def load_and_prepare_pcd(file_path):
    print(f"'{file_path}' を読み込み中...");
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        return None
    data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
    theta = np.radians(30 + 2.0); RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    points = np.dot(RotationMat, data_xyz.T).T; pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points)
    print(f"読み込み完了。点の数: {len(pcd.points)}")
    return pcd

def get_plane_with_ransac(pcd, dist_thresh, normal_x_thresh=0.9):
    print(f"RANSACで「荒削りな核」を検出中 (dist_thresh={dist_thresh} mm)...");
    if not pcd.has_normals(): pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30))
    normals = np.asarray(pcd.normals); wall_candidate_indices = np.where(np.abs(normals[:, 0]) >= normal_x_thresh)[0]
    pcd_candidates = pcd.select_by_index(wall_candidate_indices)
    if len(pcd_candidates.points) < 100: return None, np.array([])
    plane_model, initial_inlier_indices = pcd_candidates.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000)
    original_indices = wall_candidate_indices[initial_inlier_indices]
    return plane_model, original_indices

def refine_core_with_sor(pcd, raw_core_indices, nb_neighbors, std_ratio):
    print(f"\n統計的ノイズ除去で核を精錬中 (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})...");
    if len(raw_core_indices) == 0: return np.array([])
    raw_core_pcd = pcd.select_by_index(raw_core_indices)
    _, local_refined_indices = raw_core_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    refined_core_indices = raw_core_indices[local_refined_indices]
    print(f"  核の精錬: {len(raw_core_indices)}点 -> {len(refined_core_indices)}点")
    return refined_core_indices

def create_wall_coordinate_system(plane_normal):
    ex_prime = plane_normal / np.linalg.norm(plane_normal); ez_world = np.array([0, 0, 1])
    if np.allclose(np.abs(np.dot(ex_prime, ez_world)), 1.0): ez_world = np.array([0, 1, 0])
    ez_prime = ez_world - np.dot(ez_world, ex_prime) * ex_prime; ez_prime /= np.linalg.norm(ez_prime)
    ey_prime = np.cross(ez_prime, ex_prime)
    T_world_to_local = np.eye(4); T_world_to_local[:3, :3] = np.array([ex_prime, ey_prime, ez_prime])
    T_local_to_world = np.linalg.inv(T_world_to_local)
    return T_world_to_local, T_local_to_world

def filter_pcd_by_prism(pcd_local, core_indices_world, padding_mm, prism_thickness_mm=None):
    print(f"\n壁専用座標系で角柱フィルタを適用中 (padding={padding_mm} mm)...");
    core_points_local = np.asarray(pcd_local.select_by_index(core_indices_world).points)
    if len(core_points_local) == 0: return np.array([]), None
    min_y, max_y = core_points_local[:, 1].min(), core_points_local[:, 1].max(); min_z, max_z = core_points_local[:, 2].min(), core_points_local[:, 2].max()
    min_y_padded, max_y_padded = min_y - padding_mm, max_y + padding_mm; min_z_padded, max_z_padded = min_z - padding_mm, max_z + padding_mm
    all_points_local = np.asarray(pcd_local.points)
    mask_y = (all_points_local[:, 1] >= min_y_padded) & (all_points_local[:, 1] <= max_y_padded)
    mask_z = (all_points_local[:, 2] >= min_z_padded) & (all_points_local[:, 2] <= max_z_padded)
    if prism_thickness_mm is not None:
        print(f"  有限角柱を適用します (厚み: {prism_thickness_mm} mm)")
        center_x = np.mean(core_points_local[:, 0])
        half_thickness = prism_thickness_mm / 2.0
        min_x_padded = center_x - half_thickness
        max_x_padded = center_x + half_thickness
        mask_x = (all_points_local[:, 0] >= min_x_padded) & (all_points_local[:, 0] <= max_x_padded)
    else:
        print("  無限角柱を適用します。")
        mask_x = np.ones(len(all_points_local), dtype=bool)
    prism_indices = np.where(mask_x & mask_y & mask_z)[0]
    print(f"  全{len(pcd_local.points)}点 -> 角柱フィルタ後: {len(prism_indices)}点")
    prism_bounds = (min_y_padded, max_y_padded, min_z_padded, max_z_padded)
    if prism_thickness_mm is not None:
        prism_bounds = (min_x_padded, max_x_padded) + prism_bounds
    return prism_indices, prism_bounds

# ==============================================================================
# 2. 反復的平面拡張アルゴリズム
# ==============================================================================
# ▼▼▼【ここを改修】色分けルールを引数で受け取るように変更 ▼▼▼
def expand_by_iterative_ransac(pcd_local,
                               candidate_indices,
                               initial_core_indices_world,
                               dist_thresh,
                               min_points_for_plane,
                               adjacency_radius,
                               coloring_ranges=None): # カスタムの色分けルール
    print("\n【新アルゴリズム】反復的平面拡張による領域成長を開始します...")
    
    geometries_to_visualize = []
    candidate_pcd = pcd_local.select_by_index(candidate_indices)
    if not candidate_pcd.has_points():
        return [], candidate_pcd
    candidate_tree = o3d.geometry.KDTreeFlann(candidate_pcd)
    map_world_to_candidate = {idx: i for i, idx in enumerate(candidate_indices)}
    core_indices_in_candidate = [map_world_to_candidate[i] for i in initial_core_indices_world if i in map_world_to_candidate]
    core_pcd_in_candidate = candidate_pcd.select_by_index(core_indices_in_candidate)
    core_pcd_in_candidate.paint_uniform_color([1.0, 0.0, 0.0])
    geometries_to_visualize.append(core_pcd_in_candidate)
    final_wall_indices_in_candidate = set(core_indices_in_candidate)
    remaining_indices_in_candidate = set(range(len(candidate_pcd.points))) - final_wall_indices_in_candidate
    
    for i in range(20):
        if not final_wall_indices_in_candidate: break
        boundary_indices_in_candidate = list(final_wall_indices_in_candidate)
        if len(boundary_indices_in_candidate) > 2000:
            boundary_indices_in_candidate = np.random.choice(boundary_indices_in_candidate, 2000, replace=False)
        boundary_points = candidate_pcd.select_by_index(boundary_indices_in_candidate)
        adjacent_indices = set()
        for point in np.asarray(boundary_points.points):
            [k, idx, _] = candidate_tree.search_radius_vector_3d(point, adjacency_radius)
            adjacent_indices.update(idx)
        target_indices = list(adjacent_indices.intersection(remaining_indices_in_candidate))
        if len(target_indices) < min_points_for_plane:
            print("  隣接するターゲットが少ないため終了します。")
            break
        target_pcd = candidate_pcd.select_by_index(target_indices)
        plane_model, new_plane_local_indices = target_pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=200)
        if len(new_plane_local_indices) < min_points_for_plane:
            print(f"  ループ {i+1}: 新しい平面が見つかりませんでした。")
            continue
        new_plane_indices_in_candidate = {target_indices[i] for i in new_plane_local_indices}
        new_segment_pcd = candidate_pcd.select_by_index(list(new_plane_indices_in_candidate))
        
        # ▼▼▼【ここを改修】カスタムの範囲指定に基づいて色を決定 ▼▼▼
        current_loop_num = i + 1
        # デフォルトの色（紫色）
        segment_color = [0.5, 0.0, 0.8] 
        
        if coloring_ranges:
            for start, end, color_val in coloring_ranges:
                if start <= current_loop_num <= end:
                    segment_color = color_val
                    break # 最初に見つかった範囲の色を採用
        
        new_segment_pcd.paint_uniform_color(segment_color)
        geometries_to_visualize.append(new_segment_pcd)
        final_wall_indices_in_candidate.update(new_plane_indices_in_candidate)
        remaining_indices_in_candidate -= new_plane_indices_in_candidate
        print(f"  ループ {i+1}: 新しい平面を発見 (点の数: {len(new_plane_indices_in_candidate)}, 色: {segment_color})")
        time.sleep(0.1)
    print("\n領域成長が完了しました。")
    final_wall_pcd = candidate_pcd.select_by_index(list(final_wall_indices_in_candidate))
    return geometries_to_visualize, final_wall_pcd

# ==============================================================================
# 3. メイン処理
# ==============================================================================
if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    print(f"[INFO] 乱数シードを {seed} に固定しました。")

    # --- パラメータ設定 ---
    FILE_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'
    
    CORE_DIST_THRESH = 15.0; NORMAL_X_THRESH = 0.9
    SOR_NEIGHBORS = 7; SOR_STD_RATIO = 0.3
    CLUSTER_EPS = 150.0; CLUSTER_MIN_POINTS = 10
    FOOTPRINT_PADDING = 200.0
    PRISM_THICKNESS = 2000.0
    ITERATIVE_RANSAC_DIST = 20.0; MIN_POINTS_FOR_PLANE = 50; ADJACENCY_RADIUS = 100.0
    
    # ▼▼▼【色分け制御用パラメータ】▼▼▼
    # 領域成長の際の色分けを、ループ範囲を指定してカスタムします。
    # (開始ループ, 終了ループ, [R, G, B]の色) のタプル(組)のリストで定義します。
    # このリストで指定されなかった範囲は、デフォルトの色（紫）で塗られます。
    COLORING_RANGES = [
        (2, 10, [0.0, 1.0, 0.0]),  # ループ2～4を緑色に
        (11, 18, [0.0, 0.0, 1.0]),  # ループ5～7を青色に
        (19, 30, [1.0, 1.0, 0.0]), # ループ8～15を黄色に
    ]
    
    VIZ_RANGES = [(None, None, True)]
    
    # --- 検出ロジックの実行 ---
    pcd_world = load_and_prepare_pcd(FILE_PATH)
    if pcd_world is None: exit()
    plane_model, raw_core_indices = get_plane_with_ransac(pcd_world, CORE_DIST_THRESH, NORMAL_X_THRESH)
    if plane_model is None or len(raw_core_indices) == 0: exit("エラー: 荒削りな核が見つかりませんでした。")
    core_indices_world = refine_core_with_sor(pcd_world, raw_core_indices, nb_neighbors=SOR_NEIGHBORS, std_ratio=SOR_STD_RATIO)
    if len(core_indices_world) == 0: exit("エラー: SORによる核の精錬後、点が残りませんでした。")
    
    print("\n[改善策] 連結性に基づいて核をさらに精錬します...")
    core_pcd_before_filter = pcd_world.select_by_index(core_indices_world)
    labels = np.array(core_pcd_before_filter.cluster_dbscan(eps=CLUSTER_EPS, min_points=CLUSTER_MIN_POINTS, print_progress=False))
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) > 0:
        main_cluster_label = unique_labels[counts.argmax()]
        main_cluster_local_indices = np.where(labels == main_cluster_label)[0]
        final_core_indices = core_indices_world[main_cluster_local_indices]
        print(f"  核の最終精錬: {len(core_indices_world)}点 -> {len(final_core_indices)}点 (最大の塊のみ抽出)")
        core_indices_world = final_core_indices
    else:
        print("警告: クラスタリングで核の本体が見つかりませんでした。")
        
    core_pcd_world = pcd_world.select_by_index(core_indices_world)
    if not core_pcd_world.has_points(): exit("エラー: 核の最終精錬後、点が残りませんでした。")

    final_plane_model, _ = core_pcd_world.segment_plane(distance_threshold=CORE_DIST_THRESH, ransac_n=3, num_iterations=100)
    T_world_to_local, T_local_to_world = create_wall_coordinate_system(final_plane_model[:3])
    pcd_local = o3d.geometry.PointCloud(pcd_world).transform(T_world_to_local)
    candidate_indices_world, prism_bounds = filter_pcd_by_prism(
        pcd_local, core_indices_world, FOOTPRINT_PADDING, prism_thickness_mm=PRISM_THICKNESS
    )
    
    # ▼▼▼【ここを変更】新しい引数を渡す ▼▼▼
    wall_segments_local, final_wall_local = expand_by_iterative_ransac(
        pcd_local, candidate_indices=candidate_indices_world, initial_core_indices_world=core_indices_world,
        dist_thresh=ITERATIVE_RANSAC_DIST, min_points_for_plane=MIN_POINTS_FOR_PLANE, adjacency_radius=ADJACENCY_RADIUS,
        coloring_ranges=COLORING_RANGES # 新しい色分けルールを渡す
    )

    # ==============================================================================
    # 4. ステップごとの可視化 (変更なし)
    # ==============================================================================
    print("\n===== 分析結果をステップごとに表示します (対象オブジェクトのみ) =====")
    
    # ( ... 可視化の準備と実行部分は変更なし ... )
    core_pcd_world.paint_uniform_color([1, 0, 0])
    candidates_pcd_world = pcd_local.select_by_index(candidate_indices_world).transform(T_local_to_world)
    candidates_pcd_world.paint_uniform_color([0, 1, 0])
    prism_lineset = None
    if prism_bounds is not None:
        if PRISM_THICKNESS is not None and len(prism_bounds) > 4:
            min_x, max_x, min_y, max_y, min_z, max_z = prism_bounds
        else:
            min_y, max_y, min_z, max_z = prism_bounds if len(prism_bounds) == 4 else (0,0,0,0)
            points_in_prism_local = np.asarray(pcd_local.select_by_index(candidate_indices_world).points)
            if points_in_prism_local.size > 0: min_x, max_x = points_in_prism_local[:, 0].min(), points_in_prism_local[:, 0].max()
            else: min_x, max_x = 0, 0
        if min_x != max_x:
            corners = np.array([[min_x, min_y, min_z], [max_x, min_y, min_z], [min_x, max_y, min_z], [max_x, max_y, min_z], [min_x, min_y, max_z], [max_x, min_y, max_z], [min_x, max_y, max_z], [max_x, max_y, max_z]])
            lines = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]])
            prism_lineset = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(corners), lines=o3d.utility.Vector2iVector(lines))
            prism_lineset.transform(T_local_to_world)
            prism_lineset.paint_uniform_color([1, 1, 0])
    final_wall_segments_world = [geom.transform(T_local_to_world) for geom in wall_segments_local]

    print("\n【ステップ1/3】核(赤)のみを表示します。")
    o3d.visualization.draw_geometries([core_pcd_world], "STEP 1: Core Only", width=1600, height=1200)

    print("\n【ステップ2/3】候補点(緑)と角柱(黄)のみを表示します。")
    geometries_step2 = [candidates_pcd_world]
    if prism_lineset: geometries_step2.append(prism_lineset)
    o3d.visualization.draw_geometries(geometries_step2, "STEP 2: Candidates & Prism Only", width=1600, height=1200)

    print("\n【ステップ3】指定された範囲で、最終的な壁面を連続表示します。")
    if not final_wall_segments_world:
        print("表示する壁セグメントがありません。")
    else:
        for i, (start_loop, end_loop, include_core) in enumerate(VIZ_RANGES):
            geometries_to_display = []
            core_segment = final_wall_segments_world[0]
            expansion_segments = final_wall_segments_world[1:]
            start_index = (start_loop - 1) if start_loop is not None else 0
            end_index = end_loop if end_loop is not None else len(expansion_segments)
            if start_index < 0: start_index = 0
            filtered_segments = expansion_segments[start_index:end_index]
            geometries_to_display.extend(filtered_segments)
            if include_core:
                 geometries_to_display.insert(0, core_segment)
            if geometries_to_display:
                title = f"表示 {i+1}/{len(VIZ_RANGES)}: Loop {start_loop or 'start'}-{end_loop or 'end'} (Core: {include_core})"
                print(f"\n--- {title} ---")
                o3d.visualization.draw_geometries(geometries_to_display, title, width=1600, height=1200)
            else:
                print(f"\n--- 表示 {i+1}/{len(VIZ_RANGES)}: 指定範囲に表示するセグメントがありません。 ---")

    print("\n===== 全ての処理が完了しました =====")
