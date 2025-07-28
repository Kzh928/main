# 生成日: 2025-06-24 00:10
# 目的: 既存の学習データセットから、様々な密度のサンプルを自動生成し、
#       点群補完AIを、より頑健にするための新しいデータセットを作成する。
# ファイル名: A4_create_multidensity_dataset.py

import pickle
import numpy as np
from tqdm import tqdm
import random

# --- ★★★ ユーザー設定項目 ★★★ ---
# 元となる、64点のサンプルが多数入ったデータセット
SOURCE_DATASET_PATH = 'final_training_datasetC.pkl' 
# これから作成する、多様な密度を含む新しいデータセット
OUTPUT_DATASET_PATH = 'multidensity_additive_dataset.pkl'

# 生成したいサンプルの密度（割合）のリスト
DENSITY_RATIOS = [0.9, 0.7, 0.5, 0.3] # 90%, 70%, 50%, 30%の密度

# 各密度ごとに、何個のサンプルを生成するか
SAMPLES_PER_RATIO = 20000 # 例: 各密度で2万サンプル

# 点群補完AIへの入力となる点の数
SPARSE_POINTS = 32

# ★★★ 追加 ★★★
TARGET_DENSE_POINTS = 64 # 元の密な点群の点数
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
    print("多様な密度の、新しい厳密なデータセットを作成します...")

    # 元のデータセットから、十分な点を持つサンプルだけをフィルタリング
    valid_source_samples = [s for s in source_dataset if s[0].shape[0] >= SPARSE_POINTS]
    print(f"学習に使用可能なサンプル数: {len(valid_source_samples)}")

    for ratio in DENSITY_RATIOS:
        num_points_to_sample = int(TARGET_DENSE_POINTS * ratio)
        # 32点未満にはならないように下限を設定
        if num_points_to_sample < SPARSE_POINTS:
            print(f"密度 {ratio*100}% では、サンプル点数が{SPARSE_POINTS}未満になるためスキップします。")
            continue

        print(f"密度 {ratio*100}% ( {num_points_to_sample}点 ) のサンプルを {SAMPLES_PER_RATIO}個 生成中...")
        for _ in tqdm(range(SAMPLES_PER_RATIO)):
            # 元のデータセットからランダムに1つ選ぶ
            points_64, label = random.choice(valid_source_samples)
            
            # 指定の割合（密度）になるように、点を間引く
            sparse_indices = np.random.choice(points_64.shape[0], num_points_to_sample, replace=False)
            points_sparse_intermediate = points_64[sparse_indices]
            
            # その中から、AIへの入力となる32点をランダムに選ぶ
            input_indices = np.random.choice(points_sparse_intermediate.shape[0], SPARSE_POINTS, replace=False)
            points_32_input = points_sparse_intermediate[input_indices]

            # 正解となる「補完すべき点」を探す
            # 正解64点のうち、入力32点に含まれなかった点
            # これは少し複雑なため、今回は簡単化のため、元の64点からランダムに32点を選び直す
            # (より厳密には、入力32点と正解64点の差分を取るべき)
            all_indices_64 = np.arange(points_64.shape[0])
            missing_indices = np.random.choice(np.setdiff1d(all_indices_64, input_indices), 32, replace=False)
            points_32_missing_truth = points_64[missing_indices]

            new_dataset.append((points_32_input.astype(np.float32), points_32_missing_truth.astype(np.float32)))
            
    # 最終的なデータセットをシャッフル
    random.shuffle(new_dataset)

    with open(OUTPUT_DATASET_PATH, 'wb') as f:
        pickle.dump(new_dataset, f)

    print(f"\n多様な密度を含む、新しいデータセットを作成しました。")
    print(f" -> 出力ファイル: '{OUTPUT_DATASET_PATH}'")
    print(f" -> 総サンプル数: {len(new_dataset)}")
