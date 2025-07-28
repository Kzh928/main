# ファイル名: wall_detection_sequential_ransac.py

import open3d as o3d
import numpy as np
import pandas as pd
import time

# (load_and_prepare_pcd, get_plane_with_ransac, expand_by_iterative_ransac の各関数定義は前回と同じ)
def load_and_prepare_pcd(file_path):
    """点群ファイルを読み込み、回転などの前処理を行う"""
    print(f"'{file_path}' を読み込み中...")
    df = pd.read_csv(file_path)
    data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
    theta = np.radians(30 + 2.0)
    RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    points = np.dot(RotationMat, data_xyz.T).T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30))
    return pcd

def get_plane_with_ransac(pcd, normal_x_thresh=0.9, dist_thresh=20.0):
    """法線フィルタとRANSACで平面モデルを検出する"""
    print("RANSACで主要な平面を検出中...")
    normals = np.asarray(pcd.normals)
    wall_candidate_indices = np.where(np.abs(normals[:, 0]) >= normal_x_thresh)[0]
    pcd_candidates = pcd.select_by_index(wall_candidate_indices)
    plane_model, initial_inlier_indices = pcd_candidates.segment_plane(
        distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000)
    return plane_model, wall_candidate_indices[initial_inlier_indices]

def expand_by_iterative_ransac(pcd, initial_indices, dist_thresh=25.0, search_radius=150.0, min_growth_rate=0.01, max_iterations=10):
    """RANSACを繰り返し適用し、領域を成長させる"""
    print("反復RANSACによる領域成長を開始します...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    current_indices = initial_indices
    for i in range(max_iterations):
        print(f"\n--- イテレーション {i+1}/{max_iterations} ---")
        num_before = len(current_indices)
        print(f"現在の点数: {num_before}")
        neighbor_indices = set()
        current_points = np.asarray(pcd.points)[current_indices]
        for point in current_points:
            [k, idx, _] = pcd_tree.search_radius_vector_3d(point, search_radius)
            neighbor_indices.update(idx)
        expansion_candidate_indices = np.array(list(set(current_indices) | neighbor_indices), dtype=np.int32)
        if len(expansion_candidate_indices) <= 3: break
        expansion_candidate_pcd = pcd.select_by_index(expansion_candidate_indices)
        _, new_inlier_indices_local = expansion_candidate_pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=500)
        new_indices = expansion_candidate_indices[new_inlier_indices_local]
        num_after = len(new_indices)
        growth_rate = (num_after - num_before) / num_before if num_before > 0 else 1.0
        print(f"成長後の点数: {num_after} (増加率: {growth_rate:.3f})")
        if growth_rate < min_growth_rate: break
        current_indices = new_indices
        time.sleep(0.5)
    print("\n領域成長が完了しました。")
    return current_indices


# --- ▼▼▼【新機能】逐次RANSACで複数の平面を抽出する関数 ▼▼▼ ---

def extract_multiple_planes(pcd, indices, num_planes=2, min_plane_size=1000, dist_thresh=25.0):
    """
    RANSACを複数回実行し、複数の平面を抽出する
    :param pcd: 全ての点群
    :param indices: 対象となる点のインデックス
    :param num_planes: 抽出したい平面の数
    :param min_plane_size: 平面としてみなす最小の点数
    :param dist_thresh: RANSACの距離閾値
    :return: 抽出されたすべての平面のインデックス
    """
    print("\n逐次RANSACによる平面抽出を開始します...")
    
    # 対象となる点群のコピーを作成
    remaining_indices = list(indices)
    all_found_indices = []
    
    for i in range(num_planes):
        if len(remaining_indices) < min_plane_size:
            print("残りの点群が少ないため、処理を終了します。")
            break

        print(f"\n--- {i+1}番目の平面を探索中... (候補点: {len(remaining_indices)}点) ---")
        
        # 現在の候補点から点群を生成
        candidate_pcd = pcd.select_by_index(remaining_indices)
        
        # RANSACで平面を検出
        _, inlier_indices_local = candidate_pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000)
            
        # 検出された平面のサイズをチェック
        if len(inlier_indices_local) < min_plane_size:
            print(f"検出された平面のサイズが小さすぎるため、この平面は採用しません ({len(inlier_indices_local)} < {min_plane_size})。")
            continue

        print(f"{i+1}番目の平面として {len(inlier_indices_local)} 点を検出しました。")
        
        # 元のpcdにおけるインデックスに変換して保存
        found_indices_global = np.array(remaining_indices)[inlier_indices_local]
        all_found_indices.extend(found_indices_global)
        
        # 次の探索のために、今回見つかった点を候補から除外
        remaining_indices = list(set(remaining_indices) - set(found_indices_global))
        
    return np.array(all_found_indices, dtype=int)


# --- メイン処理 ---
if __name__ == '__main__':
    pcd = load_and_prepare_pcd('mid360_XYZ_Reflect_data_03-28-13-53-11.csv')
    _, initial_wall_indices = get_plane_with_ransac(pcd)

    expanded_indices = expand_by_iterative_ransac(
        pcd,
        initial_wall_indices,
        dist_thresh=25.0,
        search_radius=500.0,
        min_growth_rate=0.01,
        max_iterations=5
    )
    
    # 【新機能の呼び出し】逐次RANSACで平面を2つ抽出
    final_indices = extract_multiple_planes(
        pcd,
        expanded_indices,
        num_planes=2,           # 上下の壁、2つの平面を探す
        min_plane_size=1000,    # 1000点未満の平面は無視する
        dist_thresh=25.0
    )
    
    # 最終結果の表示
    print("\n----------------------------------------------------")
    print("最終的な壁面抽出結果の点群のみを表示します。")
    print("----------------------------------------------------")
    final_wall_cloud = pcd.select_by_index(final_indices)
    if len(final_wall_cloud.points) > 0:
        final_wall_cloud.paint_uniform_color([0, 0, 1]) 
        o3d.visualization.draw_geometries(
            [final_wall_cloud],
            window_name="Final Wall by Sequential RANSAC"
        )
    else:
        print("最終的な壁の点群が見つかりませんでした。")
