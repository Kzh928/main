# ファイル名: step1.8_merge_all_multiple_labels.py
# 概要: 複数の「鉄筋/非鉄筋」ラベルファイルと、複数の「装置」ラベルファイルを
#      読み込み、1つの最終的な3クラス用アノテーションファイルに結合する。

import numpy as np
import os

# --- ★★★ ユーザー設定項目 ★★★ ---

# 結合元の「鉄筋(1)/非鉄筋(2)」のラベルファイル（複数指定可）
BASE_LABELS_PATHS = [

    'annotated_labels.npy',
    'annotated_labels_negatives.npy',
    'annotated_labels_negatives_B.npy',
    'annotated_labels_negatives_CA.npy',
    'annotated_labels_negatives_CB.npy',
    'annotated_labels_negatives_CC.npy',
]

# ▼▼▼【変更点】装置ラベルも複数指定可能に ▼▼▼
# 新しく作成した「装置(3)」のラベルファイル（複数指定可）
EQUIPMENT_LABELS_PATHS = [
    
    'labels_for_equipment_only.npy',
]

# 結合後の出力ファイル
OUTPUT_LABELS_PATH = 'annotated_labels_3class.npy'

# 元となった点群全体の点の数を確認するために使用
PCD_PATH = 'work_area_large.csv' 

# --- メイン処理 ---
if __name__ == '__main__':
    print("--- ラベル結合処理を開始します ---")

    # 1. 点群の総数を取得
    try:
        with open(PCD_PATH) as f:
            # ヘッダー行を考慮しないシンプルな行数カウント
            total_points = sum(1 for _ in f)
        print(f"元の点群 '{PCD_PATH}' の総点数: {total_points}")
    except FileNotFoundError:
        print(f"エラー: 点群ファイル '{PCD_PATH}' が見つかりません。")
        exit()

    # 2. ベースとなるラベルファイルを読み込み、1つに結合
    merged_base_labels = np.zeros(total_points, dtype=int)
    print("\nベースとなる「鉄筋/非鉄筋」ラベルファイルを結合します...")
    for path in BASE_LABELS_PATHS:
        try:
            labels = np.load(path)
            print(f"  - '{path}' を読み込みました。")
            labeled_indices = np.where(labels > 0)[0]
            merged_base_labels[labeled_indices] = labels[labeled_indices]
        except FileNotFoundError:
            print(f"  - 警告: ファイル '{path}' が見つかりませんでした。スキップします。")
    
    # 3. 装置ラベルファイルを読み込み、1つに結合
    merged_equipment_labels = np.zeros(total_points, dtype=int)
    print("\n「装置」ラベルファイルを結合します...")
    for path in EQUIPMENT_LABELS_PATHS:
        try:
            labels = np.load(path)
            print(f"  - '{path}' を読み込みました。")
            equipment_indices = np.where(labels == 3)[0]
            merged_equipment_labels[equipment_indices] = 3
        except FileNotFoundError:
            print(f"  - 警告: ファイル '{path}' が見つかりませんでした。スキップします。")

    # 4. ラベルの最終結合
    print("\n最終的なラベル結合中...")
    final_merged_labels = np.copy(merged_base_labels)
    equipment_indices_to_overwrite = np.where(merged_equipment_labels == 3)[0]
    final_merged_labels[equipment_indices_to_overwrite] = 3
    
    num_rebar = np.sum(final_merged_labels == 1)
    num_non_rebar = np.sum(final_merged_labels == 2)
    num_equipment = np.sum(final_merged_labels == 3)

    print("結合が完了しました。")
    print("--- 結合後のラベル内訳 ---")
    print(f"  ラベル1 (鉄筋):     {num_rebar} 点")
    print(f"  ラベル2 (非鉄筋):   {num_non_rebar} 点")
    print(f"  ラベル3 (装置):     {num_equipment} 点")
    print("--------------------------")

    # 5. 結果の保存
    np.save(OUTPUT_LABELS_PATH, final_merged_labels)
    print(f"結合したラベルを '{OUTPUT_LABELS_PATH}' に保存しました。")
