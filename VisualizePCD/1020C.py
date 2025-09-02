# ファイル名: step1.5b_filter_by_box.py
# 概要: 指定された座標のバウンディングボックス（箱）内部に含まれる点だけを、
#      元の点群から抽出し、新しいCSVファイルとして保存する。

import pandas as pd
import numpy as np

# --- ★★★ ユーザー設定項目 ★★★ ---
# 元となる、全体の点群ファイル
SOURCE_PCD_PATH = 'work_area_large.csv'

# 新しく作成する、切り出し後の点群ファイル名 
OUTPUT_PCD_PATH = 'equipment_area1020C.csv'

# ▼▼▼【ここに貼り付け】▼▼▼
# step1.5で出力された座標を、ここにコピー＆ペーストしてください。

ANNOTATION_TARGET_BOX = [
    2710.8472906039806, 3190.6466772007043,  # Xの範囲
    842.0, 1291.0,  # Yの範囲
    -603.7206701440784, -75.11112301869346   # Zの範囲
]
# --- メイン処理 ---
if __name__ == '__main__':
    print(f"'{SOURCE_PCD_PATH}' を読み込んでいます...")
    try:
        # ヘッダーがないことを想定
        df = pd.read_csv(SOURCE_PCD_PATH, header=None, names=['x', 'y', 'z', 'intensity'])
    except FileNotFoundError:
        print(f"エラー: ファイル '{SOURCE_PCD_PATH}' が見つかりません。")
        exit()

    # 座標の範囲を取得
    min_x, max_x = ANNOTATION_TARGET_BOX[0], ANNOTATION_TARGET_BOX[1]
    min_y, max_y = ANNOTATION_TARGET_BOX[2], ANNOTATION_TARGET_BOX[3]
    min_z, max_z = ANNOTATION_TARGET_BOX[4], ANNOTATION_TARGET_BOX[5]

    print("\n指定された範囲で点群をフィルタリング中...")
    # 指定された範囲内の点だけを抽出
    filtered_df = df[
        (df['x'] >= min_x) & (df['x'] <= max_x) &
        (df['y'] >= min_y) & (df['y'] <= max_y) &
        (df['z'] >= min_z) & (df['z'] <= max_z)
    ]

    if filtered_df.empty:
        print("警告: 指定された範囲に点が見つかりませんでした。")
    else:
        # 結果を新しいCSVファイルに保存
        filtered_df.to_csv(OUTPUT_PCD_PATH, header=False, index=False)
        print(f"フィルタリングが完了しました。")
        print(f"{len(df)} 点 -> {len(filtered_df)} 点に絞り込みました。")
        print(f"結果を '{OUTPUT_PCD_PATH}' に保存しました。")
