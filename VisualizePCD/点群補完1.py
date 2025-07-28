# ファイル名: A2_create_additive_dataset.py

import pickle
import numpy as np
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---
SOURCE_DATASET_PATH = 'final_training_datasetC.pkl' 
OUTPUT_DATASET_PATH = 'additive_dataset.pkl' # ★ 新しい出力ファイル名
SPARSE_POINTS = 32

if __name__ == '__main__':
    print(f"'{SOURCE_DATASET_PATH}' を読み込んでいます...")
    try:
        with open(SOURCE_DATASET_PATH, 'rb') as f:
            source_dataset = pickle.load(f)
    except FileNotFoundError:
        print(f"エラー: 元となるデータセット '{SOURCE_DATASET_PATH}' が見つかりません。")
        exit()

    new_dataset = []
    skipped_count = 0
    print(f"{len(source_dataset)}個のサンプルから、新しいデータセットを作成します...")

    for i, (points_64, label) in enumerate(tqdm(source_dataset)):
        num_points_in_sample = points_64.shape[0]
        if num_points_in_sample < SPARSE_POINTS:
            skipped_count += 1
            continue

        indices = np.random.choice(num_points_in_sample, SPARSE_POINTS, replace=False)
        points_32 = points_64[indices]
        new_dataset.append((points_32.astype(np.float32), points_64.astype(np.float32)))
    
    if skipped_count > 0:
        print(f"\n[情報] 点の数が足りず、{skipped_count}個のサンプルをスキップしました。")

    with open(OUTPUT_DATASET_PATH, 'wb') as f:
        pickle.dump(new_dataset, f)

    print(f"\n追加生成モデル用の新しいデータセットを作成しました。")
    print(f" -> 出力ファイル: '{OUTPUT_DATASET_PATH}'")
    print(f" -> 総サンプル数: {len(new_dataset)}")