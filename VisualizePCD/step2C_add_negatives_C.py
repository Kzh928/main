# ファイル名: step2.1_create_negative_dataset.py
#
#２CAと違って全部を非鉄筋要素として出力するコード
#
##

import open3d as o3d
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---

# サンプルを抽出する元の点群ファイル
SOURCE_PCD_PATH = 'work_area_large.csv'

# 「ここに含まれる点は全て非鉄筋」と定義する座標範囲
# step1.5のヘルパーツールを使って、壁や床などの範囲を調べ、ここに設定してください。
TARGET_BOX = [
    1500.0730163627743, 3002.356954314149,  # Xの範囲
    1562.0, 1804.0,  # Yの範囲
    -907.2713912842048, 51.19574766054114   # Zの範囲
]


# 保存するアノテーションファイル（.npy）の名前
OUTPUT_LABELS_PATH = 'annotated_labels_negatives_CC.npy'
# 出力する学習データセットのファイル名
OUTPUT_DATASET_PATH =  'training_dataset_negatives_CC.pkl'

# --- 他の設定項目 ---
NUM_NEIGHBORS = 64 # 近傍点の数
LABEL_NOT_REBAR = 1 # 非鉄筋のラベル番号（モデル内では0が鉄筋, 1が非鉄筋）

# --- メインの処理 ---
if __name__ == '__main__':
    # 1. 元となる全体の点群データを読み込む
    print(f"'{SOURCE_PCD_PATH}' から元の点群データを読み込んでいます...")
    try:
        points = pd.read_csv(SOURCE_PCD_PATH, header=None).iloc[:, :3].values
    except FileNotFoundError:
        print(f"エラー: 元の点群ファイル '{SOURCE_PCD_PATH}' が見つかりません。")
        exit()
        
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(points)
    kdtree_full = o3d.geometry.KDTreeFlann(pcd_full)
    print("KDTreeを構築しました。")

    # 2. TARGET_BOX内の点を抽出
    box = TARGET_BOX
    target_indices = np.where(
        (points[:, 0] >= box[0]) & (points[:, 0] <= box[1]) &
        (points[:, 1] >= box[2]) & (points[:, 1] <= box[3]) &
        (points[:, 2] >= box[4]) & (points[:, 2] <= box[5])
    )[0]

    if len(target_indices) == 0:
        print("エラー: 指定されたTARGET_BOX内に点が見つかりません。座標範囲を確認してください。")
        exit()
        
    print(f"ターゲット範囲から {len(target_indices)} 個の点を抽出しました。")
    print("これらの点から「非鉄筋」の学習サンプルを作成します...")

    # 3. 学習データセットを作成
    dataset = []
    for i in tqdm(target_indices):
        # ★★★ 重要 ★★★
        # 近傍点は、ターゲット範囲内だけでなく、元の点群全体から探します。
        # これにより、境界部分の「形」を正しく学習できます。
        [k, idx, _] = kdtree_full.search_knn_vector_3d(points[i], NUM_NEIGHBORS)
        
        if k < NUM_NEIGHBORS:
            continue

        neighbors = points[idx]
        normalized_neighbors = neighbors - points[i]
        
        # このスクリプトでは、ラベルは常に「非鉄筋」
        model_label = LABEL_NOT_REBAR
        
        dataset.append((normalized_neighbors.astype(np.float32), model_label))

    # 4. ファイルに保存
    if not dataset:
        print("データセットの作成に失敗しました。")
    else:
        with open(OUTPUT_DATASET_PATH, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"\nデータセットを作成しました。")
        print(f" -> 出力ファイル: '{OUTPUT_DATASET_PATH}'")
        print(f" -> 総サンプル数: {len(dataset)}")
        # このスクリプトで自動生成したラベル情報を.npyファイルとして保存する
        print(f"\nアノテーション結果を '{OUTPUT_LABELS_PATH}' に保存します。")
        output_labels = np.full(len(points), 0, dtype=int) # まず全点を「未定義(0)」に
        output_labels[target_indices] = 2 # ターゲット範囲を「非鉄筋(2)」に設定
        np.save(OUTPUT_LABELS_PATH, output_labels)
        print("保存が完了しました。")