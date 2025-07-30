# ファイル名: step1.5_helper_select_annotation_area.py (修正版)
#範囲を調べるためのもの

import open3d as o3d
import pandas as pd
import numpy as np 

# --- 設定項目 ---
# B: 作業範囲のファイル
WORK_AREA_CSV_PATH = 'work_area_large.csv' 

def select_area_by_picking():
    """
    マウスのピック（クリック）で点を選択し、その領域のバウンディングボックス座標を取得する
    """
    # ファイル読み込み
    try:
        print(f"'{WORK_AREA_CSV_PATH}' を読み込んでいます...")
        df = pd.read_csv(WORK_AREA_CSV_PATH, header=None, names=['x', 'y', 'z', 'intensity'])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(df[['x', 'y', 'z']].values)
    except FileNotFoundError:
        print(f"エラー: '{WORK_AREA_CSV_PATH}' が見つかりません。ステップ1を先に実行してください。")
        return
    
    # ★★★ 変更点 ★★★
    # 点群の基本色を灰色に設定
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    print("\n--- ★★★ 操作方法 ★★★ ---")
    print("1. ウィンドウが開いたら、キーボードの [Shift] キーを押したままにしてください。")
    print("2. [Shift] を押しながら、マウスの左クリックやドラッグで点を選択します。")
    print("3. 選択された点は赤色に変わります。アノテーションしたい領域の点をすべて選択してください。")
    print("4. 選択が終わったら、ウィンドウを閉じてください。")
    print("-----------------------------------")
    
    # 頂点選択機能付きビジュアライザを起動
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name="Pick Annotation Points (Shift + Click/Drag)", width=1600, height=1200)
    vis.add_geometry(pcd)
    vis.run()
    
    picked_points_info = vis.get_picked_points()
    vis.destroy_window()
    
    if len(picked_points_info) == 0:
        print("\n点が選択されませんでした。プログラムを終了します。")
        return

    selected_indices = [info.index for info in picked_points_info]
    selected_points_pcd = pcd.select_by_index(selected_indices)
    
    bbox = selected_points_pcd.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0) # バウンディングボックスは赤色

    min_b = bbox.min_bound
    max_b = bbox.max_bound

    print("\n--- 結果 ---")
    print("選択した領域を囲む座標が計算されました。")
    print("以下のリストをコピーして、step2のANNOTATION_TARGET_BOXに貼り付けてください。")
    
    print("\nANNOTATION_TARGET_BOX = [")
    print(f"    {min_b[0]}, {max_b[0]},  # Xの範囲")
    print(f"    {min_b[1]}, {max_b[1]},  # Yの範囲")
    print(f"    {min_b[2]}, {max_b[2]}   # Zの範囲")
    print("]")
    
    print("\n最終確認のため、作業範囲全体と計算されたBOXを重ねて表示します。")
    
    # ★★★ 変更点 ★★★
    # 座標軸を作成し、サイズを小さく設定
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
    
    # 最終確認の表示
    o3d.visualization.draw_geometries(
        [pcd, bbox, axis], # 灰色点群、赤色BOX、小さい座標軸
        window_name="Final Confirmation",
        width=1600,
        height=1200
    )

if __name__ == '__main__':
    select_area_by_picking()
