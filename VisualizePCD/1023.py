# ファイル名: visualize_overwrite_process.py
# 概要: 座標ベースで紐付けを行いながら、ラベルの重複と上書きが
#       どのように行われるかをステップごとに可視化する検証スクリプト。

import numpy as np
import pandas as pd
import open3d as o3d

# --- ★★★ ユーザー設定項目 ★★★ ---

# ▼▼▼ 検証したいファイルペアをここに記述 ▼▼▼
# ( 'ラベルファイル.npy', 'そのラベルの元になった点群ファイル.csv' ) のペアで指定。
# リストの下にあるペアほど優先度が高く、上のラベルを上書きします。
LABELS_TO_MERGE = [
    # --- 優先度 低 ---
    ('annotated_labels_4class.npy', 'work_area_large.csv'),


]

# 全体点群のCSVファイル
BASE_PCD_PATH = 'work_area_large.csv'

# --- 色の定義 ---
COLOR_MAP = {
    0: [0.5, 0.5, 0.5], # 未分類 (グレー)
    1: [1.0, 0.0, 0.0], # 鉄筋 (赤)
    2: [0.0, 0.0, 1.0], # 非鉄筋 (青)
    3: [0.0, 1.0, 0.0], # 装置 (緑)
}
COLOR_OVERWRITE = [1.0, 0.0, 1.0] # 上書きされる点 (マゼンタ)

def create_pcd_from_labels(points, labels):
    """ラベル配列から色付きの点群を作成するヘルパー関数"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.array([COLOR_MAP.get(label, COLOR_OVERWRITE) for label in labels])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# --- メイン処理 ---
if __name__ == '__main__':
    print("--- 上書き処理の可視化を開始します ---")

    # 1. 全体点群を読み込み
    try:
        print(f"ベースとなる点群 '{BASE_PCD_PATH}' を読み込んでいます...")
        base_df = pd.read_csv(BASE_PCD_PATH, header=None, names=['x', 'y', 'z', 'intensity'])
        base_points = base_df.iloc[:, :3].values
        total_points = len(base_points)
    except FileNotFoundError:
        print(f"エラー: ベース点群ファイル '{BASE_PCD_PATH}' が見つかりません。"); exit()

    # 2. 座標検索用のインデックスを作成
    print("座標のインデックスを作成中...")
    base_points_map = {f"{row.x:.3f}_{row.y:.3f}_{row.z:.3f}": i for i, row in base_df.iterrows()}

    # 3. 最終的なラベルを格納する配列
    final_labels = np.zeros(total_points, dtype=int)
    
    # 4. リストの順番通りにファイルを処理
    for i, (label_path, pcd_path) in enumerate(LABELS_TO_MERGE):
        print("\n" + "="*50)
        print(f"ステップ {i+1}/{len(LABELS_TO_MERGE)}: '{label_path}' を処理します")
        print("="*50)
        
        try:
            labels_partial = np.load(label_path)
            points_partial_df = pd.read_csv(pcd_path, header=None, names=['x', 'y', 'z', 'intensity'])
        except FileNotFoundError:
            print(f"  - 警告: ファイルが見つかりません。スキップします。"); continue

        ## --- 可視化1: Before (現在の状態) ---
        #print("【表示1/3】処理前の状態 (Before) を表示します。")
        #pcd_before = create_pcd_from_labels(base_points, final_labels)
        ## ▼▼▼【変更】ウィンドウサイズを指定 ▼▼▼
        #o3d.visualization.draw_geometries([pcd_before], 
        #                                  window_name=f"Step {i+1} - BEFORE processing {label_path}",
        #                                  width=1600, height=1200)

        # --- これから適用するラベル情報を準備 ---
        indices_to_update_in_base = []
        new_labels_for_update = []
        
        labeled_indices_partial = np.where(labels_partial > 0)[0]
        for idx_partial in labeled_indices_partial:
            point_info = points_partial_df.iloc[idx_partial]
            coord_key = f"{point_info.x:.3f}_{point_info.y:.3f}_{point_info.z:.3f}"
            if coord_key in base_points_map:
                base_idx = base_points_map[coord_key]
                indices_to_update_in_base.append(base_idx)
                new_labels_for_update.append(labels_partial[idx_partial])
        
        if not indices_to_update_in_base:
            print("  - このファイルで更新される点はありませんでした。"); continue

        ## --- 可視化2: Overlap Preview (重複プレビュー) ---
        #print("【表示2/3】上書きプレビューを表示します。（マゼンタ = 上書きされる点）")
        #preview_labels = np.copy(final_labels)
        #overwrite_count = 0
        #for base_idx, new_label in zip(indices_to_update_in_base, new_labels_for_update):
        #    if final_labels[base_idx] > 0 and final_labels[base_idx] != new_label:
        #        preview_labels[base_idx] = -1 # マゼンタで表示するための仮ラベル
        #        overwrite_count += 1
        #    else:
        #        preview_labels[base_idx] = new_label
        #
        #print(f"  - このファイルにより {overwrite_count} 点が上書きされます。")
        
        #pcd_preview = create_pcd_from_labels(base_points, preview_labels)
        ## ▼▼▼【変更】ウィンドウサイズを指定 ▼▼▼
        #o3d.visualization.draw_geometries([pcd_preview],
        #                                  window_name=f"Step {i+1} - OVERLAP PREVIEW (Magenta = Overwritten)",
        #                                  width=1600, height=1200)

        # --- 実際にラベルを更新 ---
        for base_idx, new_label in zip(indices_to_update_in_base, new_labels_for_update):
            final_labels[base_idx] = new_label

        # --- 可視化3: After (処理後の状態) ---
        print("【表示3/3】処理後の状態 (After) を表示します。")
        pcd_after = create_pcd_from_labels(base_points, final_labels)
        # ▼▼▼【変更】ウィンドウサイズを指定 ▼▼▼
        o3d.visualization.draw_geometries([pcd_after], 
                                          window_name=f"Step {i+1} - AFTER processing {label_path}",
                                          width=1600, height=1200)

    print("\n===== 全ての可視化処理が完了しました =====")
