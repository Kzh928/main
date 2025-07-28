# ファイル名: step2.6C_visualize_all_annotations.py

import open3d as o3d
import numpy as np
import pandas as pd

# --- ★★★ ユーザー設定項目 ★★★ ---

# 色を塗る「土台」となる、一番大きな点群データのパス
BASE_PCD_PATH = 'work_area_large.csv'

# 表示したいアノテーションファイル（.npy）を全てリストアップする
LABEL_FILES_TO_LOAD = [
    'annotated_labels.npy',
    'annotated_labels_negatives.npy',
    'annotated_labels_negatives_B.npy',
    'annotated_labels_negatives_CA.npy',
    'annotated_labels_negatives_CB.npy',
    'annotated_labels_negatives_CC.npy'
]

# --- ラベル定義 ---
LABEL_UNLABELED = 0
LABEL_REBAR = 1
LABEL_NOT_REBAR = 2

# --- 色の定義 ---
COLOR_UNLABELED = [0.5, 0.5, 0.5]  # グレー
COLOR_REBAR = [1.0, 0.0, 0.0]      # 赤
COLOR_NOT_REBAR = [0.0, 0.0, 1.0]  # 青

# --- メインの処理 ---
if __name__ == '__main__':
    # 1. ベースとなる点群を読み込む
    print(f"ベース点群 '{BASE_PCD_PATH}' を読み込んでいます...")
    try:
        points = pd.read_csv(BASE_PCD_PATH, header=None).iloc[:, :3].values
    except FileNotFoundError:
        print(f"エラー: ベース点群ファイル '{BASE_PCD_PATH}' が見つかりません。")
        exit()

    # 2. 最終的に表示する色情報の配列を準備（最初は全て灰色）
    final_colors = np.full((len(points), 3), COLOR_UNLABELED, dtype=float)
    print(f"\n合計 {len(LABEL_FILES_TO_LOAD)} 個のアノテーションファイルを読み込み、重ねていきます...")

    # 3. 各アノテーションファイルを順番に読み込み、色を上書きしていく
    for label_path in LABEL_FILES_TO_LOAD:
        try:
            labels = np.load(label_path)
            print(f"-> '{label_path}' を読み込み中... (ラベル数: {len(labels)})")
            
            # ★★★ 重要 ★★★
            # ベース点群とラベル数が違う場合（＝別の点群でアノテーションした）、エラーを防ぐためにスキップする
            if len(points) != len(labels):
                print(f"   警告: ベース点群の数({len(points)})とラベル数が異なります。このファイルはスキップします。")
                continue

            # 色を更新
            final_colors[labels == LABEL_REBAR] = COLOR_REBAR
            final_colors[labels == LABEL_NOT_REBAR] = COLOR_NOT_REBAR
            print(f"   '{label_path}' の色情報を反映しました。")

        except FileNotFoundError:
            print(f"   警告: アノテーションファイル '{label_path}' が見つかりませんでした。スキップします。")

    # 4. 最終的な点群オブジェクトを作成して可視化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(final_colors)

    print("\n全てのアノテーションを統合した結果を可視化します。")
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="All Combined Annotations",
        width=1600, height=1200
    )