# ファイル名: step2.5_merge_datasets.py (3ファイル統合版)

import pickle

# --- ★★★ ユーザー設定項目 ★★★ ---
# 統合したいデータセットのファイルリスト
DATASET_FILES = [
    'training_dataset.pkl',             # 1つ目のデータセット
    'training_dataset_negatives.pkl',   # 2つ目のデータセット
    'training_dataset_negatives_B.pkl', # 3つ目のデータセット
    'training_dataset_negatives_CC.pkl',
    'training_dataset_negatives_CB.pkl',
    'training_dataset_negatives_CA.pkl'
]

# 統合後のファイル名
FINAL_DATASET_PATH = 'final_training_datasetC.pkl'

if __name__ == '__main__':
    combined_dataset = []
    print("データセットを統合します...")
    
    for file_path in DATASET_FILES:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                combined_dataset.extend(data) # extendでリストを連結
                print(f"'{file_path}' を読み込みました。(サンプル数: {len(data)})")
        except FileNotFoundError:
            print(f"警告: '{file_path}' が見つかりませんでした。スキップします。")
            
    with open(FINAL_DATASET_PATH, 'wb') as f:
        pickle.dump(combined_dataset, f)

    print("\nデータセットを統合しました。")
    print(f" -> 出力ファイル: '{FINAL_DATASET_PATH}'")
    print(f" -> 総サンプル数: {len(combined_dataset)}")