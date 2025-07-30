# ファイル名: merge_labels_robust.py
# 概要: 3D座標に基づいて、サイズの異なる複数のラベルファイルを堅牢に結合する。

import numpy as np
import pandas as pd

# --- ★★★ ユーザー設定項目 ★★★ ---

# ▼▼▼ 結合したいファイルペアをここに記述 ▼▼▼
# ( 'ラベルファイル.npy', 'そのラベルの元になった点群ファイル.csv' ) のペアで指定。
# リストの下にあるペアほど優先度が高く、上のラベルを上書きします。
LABELS_TO_MERGE = [
    # --- 優先度 低 ---
    
    # (例) 全体サイズで作成した鉄筋/非鉄筋ラベル
    ('labels_for_equipment_1001.npy', 'equipment_area1001.csv'),
    ('labels_for_equipment_3.npy', 'equipment_area3.csv'),
    ('annotated_labels.npy','work_area.csv'),
    ('annotated_labels_negatives.npy','work_area_large.csv'),
    ('annotated_labels_negatives_B.npy','work_area_large.csv'),
    ('annotated_labels_negatives_CA.npy','work_area_large.csv'),
    ('annotated_labels_negatives_CB.npy','work_area_large.csv'),
    ('annotated_labels_negatives_CC.npy','work_area_large.csv'),

    # (例) 部分的なCSVから作成した装置ラベル
    # ← このようにペアで指定
    
    # --- 優先度 高 ---
]

# 結合後の出力ファイル名
OUTPUT_LABELS_PATH = 'annotated_labels_3class_final_robust_1004.npy'

# 全体点群のCSVファイル
BASE_PCD_PATH = 'work_area_large.csv'

# --- メイン処理 ---
if __name__ == '__main__':
    print("--- 堅牢版 ラベル結合処理を開始します ---")

    # 1. 全体点群を読み込み
    try:
        print(f"ベースとなる点群 '{BASE_PCD_PATH}' を読み込んでいます...")
        base_df = pd.read_csv(BASE_PCD_PATH, header=None, names=['x', 'y', 'z', 'intensity'])
        total_points = len(base_df)
        print(f"総点数: {total_points}")
    except FileNotFoundError:
        print(f"エラー: ベース点群ファイル '{BASE_PCD_PATH}' が見つかりません。")
        exit()

    # 2. 座標を高速検索できる形式に変換
    # 座標を丸めて文字列化し、辞書で管理する（座標 -> インデックス）
    print("座標のインデックスを作成中...（少し時間がかかる場合があります）")
    base_points_map = {f"{row.x:.3f}_{row.y:.3f}_{row.z:.3f}": i for i, row in base_df.iterrows()}

    # 3. 最終的なラベルを格納する空の配列を作成
    final_labels = np.zeros(total_points, dtype=int)
    
    # 4. リストの順番通りにファイルを処理
    print("\n指定されたファイルを順番に結合（上書き）していきます...")
    for label_path, pcd_path in LABELS_TO_MERGE:
        print(f"  - '{label_path}' を処理中...")
        try:
            labels_partial = np.load(label_path)
            points_partial_df = pd.read_csv(pcd_path, header=None, names=['x', 'y', 'z', 'intensity'])
            
            if len(labels_partial) != len(points_partial_df):
                print(f"    - 警告: '{label_path}'と'{pcd_path}'の点数が一致しません。スキップします。")
                continue

            update_count = 0
            # ラベルが付いている点のみを対象にループ
            labeled_indices_partial = np.where(labels_partial > 0)[0]
            for idx_partial in labeled_indices_partial:
                point_info = points_partial_df.iloc[idx_partial]
                coord_key = f"{point_info.x:.3f}_{point_info.y:.3f}_{point_info.z:.3f}"
                
                if coord_key in base_points_map:
                    base_idx = base_points_map[coord_key]
                    final_labels[base_idx] = labels_partial[idx_partial]
                    update_count += 1
            
            print(f"    -> {update_count} 点のラベルを座標ベースで反映しました。")

        except FileNotFoundError:
            print(f"    - 警告: '{label_path}' または '{pcd_path}' が見つかりません。スキップします。")

    # 5. 結合後のラベル内訳を集計・表示・保存
    num_rebar = np.sum(final_labels == 1)
    num_non_rebar = np.sum(final_labels == 2)
    num_equipment = np.sum(final_labels == 3)

    print("\n結合が完了しました。")
    print("--- 結合後のラベル内訳 ---")
    print(f"  ラベル1 (鉄筋):     {num_rebar} 点")
    print(f"  ラベル2 (非鉄筋):   {num_non_rebar} 点")
    print(f"  ラベル3 (装置):     {num_equipment} 点")
    print("--------------------------")

    np.save(OUTPUT_LABELS_PATH, final_labels)
    print(f"結合したラベルを '{OUTPUT_LABELS_PATH}' に保存しました。")
