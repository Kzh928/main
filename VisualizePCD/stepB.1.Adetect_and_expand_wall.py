# ファイル名: detect_and_expand_wall.py
#新しいチャット　はじめ
#
##
import open3d as o3d
import numpy as np
import pandas as pd
from scipy import stats # DBSCANでのノイズ除去（オプション）で使用

# --- ★★★ ユーザー設定項目 ★★★ ---

# 壁面を抽出したい、元の点群ファイルのパス
SOURCE_PCD_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'

# RANSACの平面検出パラメータ
# 拡張処理を行うため、ここは元のままでも良い
DISTANCE_THRESHOLD = 20.0

# ★【拡張用パラメータ】平面からの距離の許容範囲
# RANSACのDISTANCE_THRESHOLDと同じか、少し大きめに設定すると拡張範囲が広がる
EXPANSION_DISTANCE_THRESHOLD = 25.0 

# 法線ベクトルによるフィルタリングのパラメータ
NORMAL_X_THRESHOLD = 0.9

# 出力ファイル名
OUTPUT_WALL_PCD_PATH = 'wall_expanded.csv'

# ★【オプション】DBSCANによるノイズ除去を行うか
USE_DBSCAN_FILTER = True
DBSCAN_EPS = 150.0       # クラスタを形成する点間の最大距離
DBSCAN_MIN_POINTS = 50   # クラスタを形成する最小の点数

# --- メインの処理 ---
if __name__ == '__main__':
    # 1. 点群の読み込み
    print(f"'{SOURCE_PCD_PATH}' から点群を読み込んでいます...")
    try:
        df = pd.read_csv(SOURCE_PCD_PATH)
        data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
        theta = np.radians(30 + 2.0)
        RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        points = np.dot(RotationMat, data_xyz.T).T
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    except FileNotFoundError:
        print(f"エラー: ファイル '{SOURCE_PCD_PATH}' が見つかりません。")
        exit()

    # 2. 法線ベクトルの計算
    print("各点の法線ベクトル（表面の向き）を計算中...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30))

    # 3. 法線ベクトルの向きに基づいて「壁の候補」を絞り込む
    normals = np.asarray(pcd.normals)
    wall_candidate_indices = np.where(np.abs(normals[:, 0]) >= NORMAL_X_THRESHOLD)[0]
    
    if len(wall_candidate_indices) == 0:
        print("正面を向いている点が見つかりませんでした。NORMAL_X_THRESHOLDの値を小さくして試してみてください。")
        exit()
        
    pcd_candidates = pcd.select_by_index(wall_candidate_indices)
    print(f"正面向きの点（壁の候補）を {len(pcd_candidates.points)} 点に絞り込みました。")

    # 4. 絞り込んだ候補に対してRANSAC平面検出を実行
    print("壁の候補の中から、最大の平面を検出し、その平面の定義を取得します...")
    plane_model, _ = pcd_candidates.segment_plane(
        distance_threshold=DISTANCE_THRESHOLD,
        ransac_n=3,
        num_iterations=1000
    )
    [a, b, c, d] = plane_model
    print(f"検出された平面の方程式: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # --- ▼▼▼ ここからが拡張処理 ▼▼▼ ---

    # 5. 平面方程式を使い、元の点群全体から壁面を再抽出（拡張）
    print("平面方程式を元に、壁面領域を拡張します...")
    
    # 5-1. 元の点群（回転後）の全点を取得
    all_points = np.asarray(pcd.points)

    # 5-2. 全ての点と平面との距離を計算
    # plane_modelの法線ベクトル(a,b,c)は正規化済みのため、計算が簡略化できる
    distances = np.abs(a * all_points[:, 0] + b * all_points[:, 1] + c * all_points[:, 2] + d)

    # 5-3. 距離が拡張用の閾値以下の点のインデックスを取得
    expanded_wall_indices = np.where(distances < EXPANSION_DISTANCE_THRESHOLD)[0]
    
    final_wall_indices = expanded_wall_indices

    # 6. (オプション) DBSCANクラスタリングでノイズを除去
    if USE_DBSCAN_FILTER and len(expanded_wall_indices) > 0:
        print("DBSCANクラスタリングでノイズを除去します...")
        expanded_pcd = pcd.select_by_index(expanded_wall_indices)
        
        # クラスタリングを実行
        labels = np.array(expanded_pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=True))
        
        # ノイズ（-1）を除いた最大のクラスターのラベルを見つける
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        if len(counts) > 0:
            largest_cluster_label = unique_labels[counts.argmax()]
            
            # 最大のクラスターに属する点のインデックスを最終結果とする
            final_indices_in_expanded = np.where(labels == largest_cluster_label)[0]
            final_wall_indices = expanded_wall_indices[final_indices_in_expanded]
            print(f"最大のクラスター（{len(final_wall_indices)}点）を壁として抽出しました。")
        else:
            print("クラスタが見つかりませんでした。DBSCANのパラメータを調整してください。")
            # クラスタが見つからない場合は、DBSCAN適用前の結果をそのまま使う
            final_wall_indices = expanded_wall_indices


    # --- ▲▲▲ 拡張処理ここまで ▲▲▲ ---

    # 7. 結果の分離と表示
    inlier_cloud = pcd.select_by_index(final_wall_indices)
    outlier_cloud = pcd.select_by_index(final_wall_indices, invert=True)

    if len(inlier_cloud.points) == 0:
        print("壁面を検出できませんでした。")
    else:
        print(f"最終的に壁面として {len(inlier_cloud.points)} 点を抽出しました。")
        inlier_cloud.paint_uniform_color([1.0, 0, 0]) # 壁を赤
        outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5]) # それ以外を灰色
        
        print("検出結果を可視化します。赤色が最終的に壁面として抽出された部分です。")
        o3d.visualization.draw_geometries(
            [inlier_cloud, outlier_cloud],
            window_name="Expanded Wall Detection",
            width=1600, height=1200
        )

        # 8. 抽出した壁面のみをファイルに保存
        wall_points = np.asarray(inlier_cloud.points)
        pd.DataFrame(wall_points).to_csv(OUTPUT_WALL_PCD_PATH, header=['x', 'y', 'z'], index=False)
        print(f"抽出した壁面を '{OUTPUT_WALL_PCD_PATH}' に保存しました。")
