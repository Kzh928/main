# 生成日: 2025-06-22
# 目的: 既存の高密度な点群データから、ランダムに点を間引いて、
#       テスト用の低密度な点群データを擬似的に作成する。
# %かえていろいろ作った
# ファイル名: F1_create_sparse_data.py

import open3d as o3d
import numpy as np
import pandas as pd

# --- ★★★ ユーザー設定項目 ★★★ ---

# 元となる、高密度な点群ファイルのパス
SOURCE_PCD_PATH = 'work_area_large.csv'

# 新しく作成する、低密度な点群ファイルのパス
OUTPUT_PCD_PATH = 'sparse_work_area_10percent.csv'

# 残す点の割合 (0.5 = 50%)
DOWNSAMPLE_RATIO = 0.1

# --- メインの処理 ---
if __name__ == '__main__':
    # 1. 元となる点群を読み込む
    print(f"'{SOURCE_PCD_PATH}' を読み込んでいます...")
    try:
        df = pd.read_csv(SOURCE_PCD_PATH, header=None)
        points = df.iloc[:, :3].values
    except FileNotFoundError:
        print(f"エラー: ファイル '{SOURCE_PCD_PATH}' が見つかりません。")
        exit()
    
    print(f"元の点群数: {len(points)}")

    # 2. ランダムにダウンサンプリング
    num_points_to_keep = int(len(points) * DOWNSAMPLE_RATIO)
    print(f"{DOWNSAMPLE_RATIO*100}% にダウンサンプリングします。残す点群数: {num_points_to_keep}")

    # ランダムにインデックスを選択
    indices_to_keep = np.random.choice(len(points), num_points_to_keep, replace=False)
    
    # 新しいデータフレームを作成
    df_sparse = df.iloc[indices_to_keep]

    # 3. 新しいCSVファイルとして保存
    df_sparse.to_csv(OUTPUT_PCD_PATH, header=False, index=False)
    print(f"低密度な点群を '{OUTPUT_PCD_PATH}' に保存しました。")
    
    # 4. 結果を可視化して確認
    print("\n結果を可視化します。")
    print("赤色: 間引かれて残った点 (50%)")
    print("灰色: 削除された点 (50%)")

    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points)

    # 残す点(kept)と、削除する点(removed)を分離する
    pcd_kept = pcd_original.select_by_index(indices_to_keep)
    pcd_removed = pcd_original.select_by_index(indices_to_keep, invert=True)

    # それぞれに色を付ける
    pcd_kept.paint_uniform_color([1.0, 0, 0])      # 残す点は赤色
    pcd_removed.paint_uniform_color([0.5, 0.5, 0.5]) # 削除する点は灰色

    o3d.visualization.draw_geometries(
        [pcd_kept, pcd_removed], # 重ならない2つの点群を表示
        window_name="Downsample Result (Red = Kept, Gray = Removed)",
        width=1600, height=1200
    )
