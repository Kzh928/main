# ファイル名: detect_wall_with_orientation.py
#壁面のまっすぐなとこだけ
#鉄筋とか削られたとこ入ってない
#再表示なし



#ひな形
#まっすぐなとこだけ
#ここから拡張していく
#
##


import open3d as o3d
import numpy as np
import pandas as pd

# --- ★★★ ユーザー設定項目 ★★★ ---

# 壁面を抽出したい、元の点群ファイルのパス
SOURCE_PCD_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'

# RANSACの平面検出パラメータ
DISTANCE_THRESHOLD = 20.0 

# 法線ベクトルによるフィルタリングのパラメータ
# 法線ベクトルのX成分の絶対値がこの値以上である点を「正面向き」とみなす
# 1.0に近いほど、完全に正面を向いた点のみを対象とする
NORMAL_X_THRESHOLD = 0.9 

# 出力ファイル名
OUTPUT_WALL_PCD_PATH = 'wall_only_orientation_filtered.csv'

# --- メインの処理 ---
if __name__ == '__main__':
    # 1. 点群の読み込み
    print(f"'{SOURCE_PCD_PATH}' から点群を読み込んでいます...")
    try:
        df = pd.read_csv(SOURCE_PCD_PATH)
        # 前処理（回転）もここで行う
        data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
        theta = np.radians(30 + 2.0)
        RotationMat = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
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
    # 法線のX成分の絶対値が閾値以上の点のインデックスを取得
    wall_candidate_indices = np.where(np.abs(normals[:, 0]) >= NORMAL_X_THRESHOLD)[0]
    
    if len(wall_candidate_indices) == 0:
        print("正面を向いている点が見つかりませんでした。NORMAL_X_THRESHOLDの値を小さくして試してみてください。")
        exit()
        
    pcd_candidates = pcd.select_by_index(wall_candidate_indices)
    print(f"正面向きの点（壁の候補）を {len(pcd_candidates.points)} 点に絞り込みました。")

    # 4. 絞り込んだ候補に対してRANSAC平面検出を実行
    print("壁の候補の中から、最大の平面を検出中...")
    plane_model, inlier_indices_of_candidates = pcd_candidates.segment_plane(
        distance_threshold=DISTANCE_THRESHOLD,
        ransac_n=3,
        num_iterations=1000
    )
    
    # 検出された平面の方程式を表示
    [a, b, c, d] = plane_model
    print(f"検出された平面の方程式: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    # 5. 結果の分離と表示
    # 検出された壁のインデックスを、元の点群のインデックスに変換
    final_wall_indices = wall_candidate_indices[inlier_indices_of_candidates]

    inlier_cloud = pcd.select_by_index(final_wall_indices)
    outlier_cloud = pcd.select_by_index(final_wall_indices, invert=True)

    if len(inlier_cloud.points) == 0:
        print("平面を検出できませんでした。")
    else:
        print(f"壁面として {len(inlier_cloud.points)} 点を抽出しました。")
        inlier_cloud.paint_uniform_color([1.0, 0, 0]) # 壁を赤
        outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5]) # それ以外を灰色
        
        print("検出結果を可視化します。赤色が最終的に壁面として抽出された部分です。")
        o3d.visualization.draw_geometries(
            [inlier_cloud, outlier_cloud],
            window_name="Wall Detection with Orientation Filter",
            width=1600, height=1200
        )

        # 6. 抽出した壁面のみをファイルに保存
        wall_points = np.asarray(inlier_cloud.points)
        # 反射強度などの情報はないため、座標のみ保存
        pd.DataFrame(wall_points).to_csv(OUTPUT_WALL_PCD_PATH, header=False, index=False)
        print(f"抽出した壁面を '{OUTPUT_WALL_PCD_PATH}' に保存しました。")
