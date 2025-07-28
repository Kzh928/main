# ファイル名: extract_wall_fully_automated.py (2回表示機能付き)
#RANSACで平面
#厚みを指定して余白と奥行追加した
#
##

import open3d as o3d
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- ★★★ ユーザー設定項目（調整するパラメータ） ★★★ ---
SOURCE_PCD_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'
NORMAL_X_THRESHOLD = 0.9
RANSAC_DISTANCE_THRESHOLD = 20.0
SLAB_THICKNESS = 1000.0 
BOUNDING_BOX_MARGIN = 50.0 
OUTPUT_WALL_PCD_PATH = 'wall_fully_extracted.csv'

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

    # 2. 法線ベクトル計算と候補点の絞り込み
    print("法線ベクトルを計算し、正面向きの点（壁候補）を抽出中...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30))
    normals = np.asarray(pcd.normals)
    wall_candidate_indices = np.where(np.abs(normals[:, 0]) >= NORMAL_X_THRESHOLD)[0]
    pcd_candidates = pcd.select_by_index(wall_candidate_indices)
    if len(pcd_candidates.points) == 0:
        print("正面を向いている点が見つかりませんでした。"); exit()
    print(f"壁の候補を {len(pcd_candidates.points)} 点に絞り込みました。")

    # 3. RANSACで主要平面を検出
    print("壁候補の中から、RANSACで主要平面を検出中...")
    plane_model, base_plane_indices_in_candidates = pcd_candidates.segment_plane(
        distance_threshold=RANSAC_DISTANCE_THRESHOLD, ransac_n=3, num_iterations=1000)
    base_plane_indices = wall_candidate_indices[base_plane_indices_in_candidates]
    base_plane_pcd = pcd.select_by_index(base_plane_indices)
    if len(base_plane_pcd.points) == 0:
        print("基準平面を検出できませんでした。"); exit()
    print(f"基準となる平面を {len(base_plane_pcd.points)} 点、検出しました。")

    # 4. 壁平面の輪郭を自動計算し、マージンを加えて拡張
    wall_bbox = base_plane_pcd.get_axis_aligned_bounding_box()
    y_min = wall_bbox.min_bound[1] - BOUNDING_BOX_MARGIN
    y_max = wall_bbox.max_bound[1] + BOUNDING_BOX_MARGIN
    z_min = wall_bbox.min_bound[2] - BOUNDING_BOX_MARGIN
    z_max = wall_bbox.max_bound[2] + BOUNDING_BOX_MARGIN
    print("壁平面の輪郭を自動計算し、範囲を拡張しました。")

    # 5. 「厚みのある拡張済み四角形（スラブ）」で最終的な点を抽出
    print("最終的な壁面を抽出中...")
    a, b, c, d = plane_model
    distances = np.abs(a*points[:,0] + b*points[:,1] + c*points[:,2] + d) / np.sqrt(a**2+b**2+c**2)
    condition1 = distances <= (SLAB_THICKNESS / 2.0)
    condition2 = (points[:, 1] >= y_min) & (points[:, 1] <= y_max) & \
                 (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    final_wall_indices = np.where(condition1 & condition2)[0]

    # 6. 結果の可視化 (1回目：全体像)
    final_wall_pcd = pcd.select_by_index(final_wall_indices)
    background_pcd = pcd.select_by_index(final_wall_indices, invert=True)
    final_wall_pcd.paint_uniform_color([1.0, 0, 0])      # 抽出された壁 = 赤
    background_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # それ以外 = グレー
    print(f"最終的な壁面として {len(final_wall_pcd.points)} 点を抽出しました。")
    print("結果を可視化します。（1回目：全体像）")
    o3d.visualization.draw_geometries([final_wall_pcd, background_pcd],
        window_name="Fully Automated Wall Extraction (with Context)", width=1600, height=1200)

    # 7. 抽出した壁面をファイルに保存
    pd.DataFrame(np.asarray(final_wall_pcd.points)).to_csv(OUTPUT_WALL_PCD_PATH, header=False, index=False)
    print(f"抽出した壁面全体を '{OUTPUT_WALL_PCD_PATH}' に保存しました。")

    # 8. 最終結果の再表示 (2回目：抽出結果のみ)
    print("\n抽出された壁面だけの最終結果を再表示します。")
    o3d.visualization.draw_geometries(
        [final_wall_pcd], # 背景(グレー)を除外
        window_name="Final Extracted Wall Only",
        width=1600, height=1200
    )