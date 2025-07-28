
# ファイル名: A3_create_strict_dataset.py

import pickle
import numpy as np
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---
SOURCE_DATASET_PATH = 'final_training_datasetC.pkl' 
OUTPUT_DATASET_PATH = 'strict_additive_dataset.pkl' # ★ 新しい出力ファイル名
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
    print(f"{len(source_dataset)}個のサンプルから、新しい厳密なデータセットを作成します...")

    for i, (points_64, label) in enumerate(tqdm(source_dataset)):
        if points_64.shape[0] < SPARSE_POINTS:
            skipped_count += 1
            continue

        # 64点からランダムに32点のインデックスを選ぶ (これが入力になる)
        input_indices = np.random.choice(points_64.shape[0], SPARSE_POINTS, replace=False)
        points_32_input = points_64[input_indices]
        
        # ★★★ 新しいロジック ★★★
        # 入力で使われなかった、残りの点のインデックスを探す (これが正解になる)
        all_indices = np.arange(points_64.shape[0])
        missing_indices = np.setdiff1d(all_indices, input_indices)
        
        # もし、何らかの理由でmissing_indicesが32点でない場合はスキップ
        if len(missing_indices) != 32:
            skipped_count += 1
            continue

        points_32_missing_truth = points_64[missing_indices]
        
        # 新しいデータセットには、(入力32点, 補完すべき正解32点) のペアで保存
        new_dataset.append((points_32_input.astype(np.float32), points_32_missing_truth.astype(np.float32)))
    
    if skipped_count > 0:
        print(f"\n[情報] 点の数が足りず、{skipped_count}個のサンプルをスキップしました。")

    with open(OUTPUT_DATASET_PATH, 'wb') as f:
        pickle.dump(new_dataset, f)

    print(f"\n厳密な答え合わせ用の新しいデータセットを作成しました。")
    print(f" -> 出力ファイル: '{OUTPUT_DATASET_PATH}'")
    print(f" -> 総サンプル数: {len(new_dataset)}")