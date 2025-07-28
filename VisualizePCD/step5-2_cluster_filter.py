# ファイル名: step5-B_cluster_filter.py

import open3d as o3d
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---
# フィルタリングの強さを調整するパラメータ
# この距離以内にある点を「同じ塊」の候補とみなす
EPS = 30.0 # 単位は点群の座標に依存 (例: 30mm)
# 塊を構成する最低限の点の数。これ以下のサイズの塊はノイズとして除去される。
MIN_CLUSTER_SIZE = 200

# --- 入力ファイル ---
PCD_PATH = 'work_area_large.csv'
PREDICTED_LABELS_PATH = 'predicted_labels.npy'

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
        print("step4を先に実行して、'predicted_labels.npy'を作成してください。")
        exit()

    # 2. まず、フィルター前の状態を可視化する
    print("フィルターをかける前の予測結果を最初に表示します。")
    print("ウィンドウを閉じると、フィルタリング処理が始まります。")
    before_colors = np.full((len(points), 3), 0.5); before_colors[pred_labels == LABEL_REBAR] = [1.0, 0.0, 0.0]; before_colors[pred_labels == LABEL_NOT_REBAR] = [0.0, 0.0, 1.0]
    before_pcd = o3d.geometry.PointCloud(); before_pcd.points = o3d.utility.Vector3dVector(points); before_pcd.colors = o3d.utility.Vector3dVector(before_colors)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([before_pcd, axis], window_name="Step5-B: BEFORE Filtering", width=1600, height=1200)

    # 3. 鉄筋と予測された点だけを抽出
    rebar_indices_original = np.where(pred_labels == LABEL_REBAR)[0]
    if len(rebar_indices_original) == 0:
        print("鉄筋と予測された点が見つかりませんでした。フィルタリングを終了します。")
        exit()
    rebar_points = points[rebar_indices_original]
    rebar_pcd = o3d.geometry.PointCloud(); rebar_pcd.points = o3d.utility.Vector3dVector(rebar_points)
    
    # 4. クラスタリングの実行 (DBSCAN)
    print(f"EPS={EPS}, MinPoints(クラスタ形成の核)={MIN_CLUSTER_SIZE} でクラスタリングを実行中...")
    # cluster_dbscanのmin_pointsは塊の最小サイズとは少し違うため、ここでは低い値に設定
    cluster_labels = np.array(rebar_pcd.cluster_dbscan(eps=EPS, min_points=10, print_progress=True))
    
    max_label = cluster_labels.max()
    print(f"発見したクラスター（塊）の数: {max_label + 1}")

    # 5. 小さなクラスターを除去
    denoised_rebar_indices = []
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
    # ノイズ（-1）ではない、かつサイズが閾値以上のクラスターだけを維持する
    valid_clusters = unique_labels[(unique_labels != -1) & (counts >= MIN_CLUSTER_SIZE)]
    print(f"維持するクラスターの数: {len(valid_clusters)}")

    for i in range(len(rebar_indices_original)):
        if cluster_labels[i] in valid_clusters:
            denoised_rebar_indices.append(rebar_indices_original[i])

    print(f"フィルタリングの結果: {len(rebar_indices_original)}個の鉄筋点 -> {len(denoised_rebar_indices)}個に")

    # 6. 新しいラベルを作成
    final_labels = np.full(len(points), LABEL_NOT_REBAR)
    final_labels[denoised_rebar_indices] = LABEL_REBAR

    # 7. 最終結果を可視化
    result_pcd = o3d.geometry.PointCloud(); result_pcd.points = o3d.utility.Vector3dVector(points)
    final_colors = np.full((len(points), 3), 0.5)
    final_colors[final_labels == LABEL_REBAR] = [1.0, 0.0, 0.0]
    final_colors[final_labels == LABEL_NOT_REBAR] = [0.0, 0.0, 1.0]
    result_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    print("フィルタリング後の結果を可視化します。")
    o3d.visualization.draw_geometries([result_pcd, axis], window_name="Step5-B: AFTER Filtering (Cluster)", width=1600, height=1200)