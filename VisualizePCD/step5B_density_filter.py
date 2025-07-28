# ファイル名: step5-A_density_filter.py

import open3d as o3d
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---
# フィルタリングの強さを調整するパラメータ
# この半径内にある他の「赤い点」の数を数える
RADIUS = 30.0 # 単位は点群の座標に依存 (例: 30mm)
# 半径内にこの数以上の仲間がいないと、ノイズとして除去される ここいじる
MIN_NEIGHBORS = 20

# --- ★★★ 入力ファイル（重要） ★★★
# step4で分析した点群のパス
PCD_PATH = 'work_area.csv'
# step4で保存された、v2モデルの予測結果のパス
PREDICTED_LABELS_PATH = 'predicted_labels_f06110053.npy'

# --- ラベル定義 ---
LABEL_REBAR = 0
LABEL_NOT_REBAR = 1

if __name__ == '__main__':
    # 1. データ読み込み
    print("データを読み込んでいます...")
    try:
        points = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values
        pred_labels = np.load(PREDICTED_LABELS_PATH)
    except FileNotFoundError as e:
        print(f"エラー: 必要なファイルが見つかりません。 {e}")
        print("step4を先に実行して、予測結果ファイルを作成してください。")
        exit()

    # 2. まず、フィルター前の状態を可視化する
    print("フィルターをかける前の予測結果を最初に表示します。")
    print("ウィンドウを閉じると、フィルタリング処理が始まります。")
    before_colors = np.full((len(points), 3), 0.5)
    before_colors[pred_labels == LABEL_REBAR] = [1.0, 0.0, 0.0]
    before_colors[pred_labels == LABEL_NOT_REBAR] = [0.0, 0.0, 1.0]
    before_pcd = o3d.geometry.PointCloud(); before_pcd.points = o3d.utility.Vector3dVector(points); before_pcd.colors = o3d.utility.Vector3dVector(before_colors)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
    #o3d.visualization.draw_geometries([before_pcd, axis], window_name="Step5-A: BEFORE Filtering", width=1600, height=1200)

    # 3. 鉄筋と予測された点だけを抽出
    rebar_indices_original = np.where(pred_labels == LABEL_REBAR)[0]
    if len(rebar_indices_original) == 0:
        print("鉄筋と予測された点が見つかりませんでした。フィルタリングを終了します。")
        exit()
    rebar_points = points[rebar_indices_original]
    rebar_pcd = o3d.geometry.PointCloud(); rebar_pcd.points = o3d.utility.Vector3dVector(rebar_points)
    
    # 4. 密度フィルタリングの実行
    print(f"半径{RADIUS}内に{MIN_NEIGHBORS}個以上の近傍点がないかチェック中...")
    rebar_kdtree = o3d.geometry.KDTreeFlann(rebar_pcd)
    denoised_rebar_indices = []
    for i in tqdm(range(len(rebar_points))):
        point = rebar_points[i]
        [k, idx, _] = rebar_kdtree.search_radius_vector_3d(point, RADIUS)
        if k >= MIN_NEIGHBORS:
            denoised_rebar_indices.append(rebar_indices_original[i])
            
    # 5. 新しいラベルを作成
    final_labels = np.full(len(points), LABEL_NOT_REBAR)
    final_labels[denoised_rebar_indices] = LABEL_REBAR

    # フィルタリング前後の赤い点の数を比較して表示
    num_before = len(rebar_indices_original)
    num_after = len(denoised_rebar_indices)
    print("\n--- フィルタリング結果サマリー ---")
    print(f"処理前の赤い点の数: {num_before}")
    print(f"処理後の赤い点の数: {num_after}")
    print(f"除去された点の数: {num_before - num_after}")
    print("--------------------------------\n")

    # 6. 最終結果を可視化
    result_pcd = o3d.geometry.PointCloud(); result_pcd.points = o3d.utility.Vector3dVector(points)
    final_colors = np.full((len(points), 3), 0.5)
    final_colors[final_labels == LABEL_REBAR] = [1.0, 0.0, 0.0]
    final_colors[final_labels == LABEL_NOT_REBAR] = [0.0, 0.0, 1.0]
    result_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    print("フィルタリング後の結果を可視化します。")
    o3d.visualization.draw_geometries([result_pcd], window_name="Step5-A: AFTER Filtering", width=1600, height=1200)