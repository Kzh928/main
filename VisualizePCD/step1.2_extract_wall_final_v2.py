# ファイル名: extract_wall_final_v2.py (再表示機能付き)
#２回表示あり
#範囲狭くててっきんとかまだ削れてる#
import open3d as o3d
import numpy as np
import pandas as pd

# --- ★★★ ユーザー設定項目 ★★★ ---
SOURCE_PCD_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'
NORMAL_X_THRESHOLD = 0.9
RANSAC_DISTANCE_THRESHOLD = 20.0
EXPANSION_DISTANCE = 100.0
OUTPUT_WALL_PCD_PATH = 'wall_final_extracted_v2.csv'

# --- メインの処理 ---
if __name__ == '__main__':
    # 1. 点群の読み込みと前処理
    print(f"'{SOURCE_PCD_PATH}' を読み込んでいます...")
    try:
        df = pd.read_csv(SOURCE_PCD_PATH)
        data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
        theta = np.radians(30 + 2.0)
        RotationMat = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        points = np.dot(RotationMat, data_xyz.T).T
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points)
    except FileNotFoundError:
        print(f"エラー: ファイル '{SOURCE_PCD_PATH}' が見つかりません。"); exit()

    # 2. 法線ベクトル計算と、向きによる候補点の絞り込み
    print("法線ベクトルを計算し、正面向きの点（壁候補）を抽出中...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30))
    normals = np.asarray(pcd.normals)
    wall_candidate_indices = np.where(np.abs(normals[:, 0]) >= NORMAL_X_THRESHOLD)[0]
    if len(wall_candidate_indices) == 0:
        print("正面を向いている点が見つかりませんでした。"); exit()
    pcd_candidates = pcd.select_by_index(wall_candidate_indices)
    print(f"壁の候補を {len(pcd_candidates.points)} 点に絞り込みました。")

    # 3. 絞り込んだ候補からRANSACで主要平面を検出
    print("壁候補の中から、RANSACで主要平面を検出中...")
    plane_model, inlier_indices_of_candidates = pcd_candidates.segment_plane(
        distance_threshold=RANSAC_DISTANCE_THRESHOLD, ransac_n=3, num_iterations=1000)
    base_plane_indices = wall_candidate_indices[inlier_indices_of_candidates]
    if len(base_plane_indices) == 0:
        print("壁候補から平面を検出できませんでした。"); exit()
    base_plane_pcd = pcd.select_by_index(base_plane_indices)

    # 4. 検出された壁平面の輪郭（バウンディングボックス）を計算
    wall_bbox = base_plane_pcd.get_axis_aligned_bounding_box()
    y_min, y_max = wall_bbox.min_bound[1], wall_bbox.max_bound[1]
    z_min, z_max = wall_bbox.min_bound[2], wall_bbox.max_bound[2]
    print("検出した壁平面の輪郭を把握しました。")

    # 5. 全ての点と検出平面との距離を計算し、「壁の仲間」を復元
    print(f"検出平面の近傍(距離{EXPANSION_DISTANCE}以内)かつ、壁の輪郭内にある点を復元中...")
    a, b, c, d = plane_model
    distances = np.abs(a*points[:,0] + b*points[:,1] + c*points[:,2] + d) / np.sqrt(a**2+b**2+c**2)
    condition1 = distances <= EXPANSION_DISTANCE
    condition2 = (points[:, 1] >= y_min) & (points[:, 1] <= y_max) & \
                 (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    final_wall_indices = np.where(condition1 & condition2)[0]

    # 6. 結果の分離と可視化
    recovered_indices = np.setdiff1d(final_wall_indices, base_plane_indices)
    background_indices = np.setdiff1d(np.arange(len(points)), final_wall_indices)
    recovered_pcd = pcd.select_by_index(recovered_indices)
    background_pcd = pcd.select_by_index(background_indices)
    base_plane_pcd.paint_uniform_color([1.0, 0, 0])      # 基準平面 = 赤
    recovered_pcd.paint_uniform_color([1.0, 1.0, 0])    # 復元された点 = 黄
    background_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # 背景 = グレー
    print("検出結果を可視化します。（1回目：全体像）")
    o3d.visualization.draw_geometries([base_plane_pcd, recovered_pcd, background_pcd],
        window_name="Wall with Features (Red=Plane, Yellow=Recovered)", width=1600, height=1200)

    # 7. 抽出した「特徴込みの壁面」をファイルに保存
    final_wall_pcd = pcd.select_by_index(final_wall_indices)
    final_wall_points_df = pd.DataFrame(np.asarray(final_wall_pcd.points))
    final_wall_points_df.to_csv(OUTPUT_WALL_PCD_PATH, header=False, index=False)
    print(f"抽出した壁面全体を '{OUTPUT_WALL_PCD_PATH}' に保存しました。")

    # ★★★ ここからが追加箇所 ★★★
    print("\n抽出された壁面だけの最終結果を再表示します。（2回目：抽出結果のみ）")
    o3d.visualization.draw_geometries(
        [base_plane_pcd, recovered_pcd], # 背景(グレー)を除外
        window_name="Final Extracted Wall Only (Red+Yellow)",
        width=1600, height=1200
    )