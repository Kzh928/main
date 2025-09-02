
# ファイル名: visualize_cache.py
# 概要: 'chipping_area_cache.npz'ファイルを読み込み、
#       保存されている点群データを3Dで可視化する。

import numpy as np
import open3d as o3d

# --- パラメータ設定 ---
# 読み込むキャッシュファイルの名前
CACHE_FILE_PATH = 'chipping_cache.npz'

if __name__ == '__main__':
    print(f"--- キャッシュファイル '{CACHE_FILE_PATH}' を読み込み中 ---")

    # 1. ファイルを読み込む
    try:
        data = np.load(CACHE_FILE_PATH)
        
        # キーワードを指定して各座標データを取得
        chipping_coords = data['chipping_coords']
        wall_coords = data['wall_coords']
        equipment_coords = data['equipment_coords']
        
        print(f"読み込み成功！")
        print(f"  - はつり範囲: {len(chipping_coords)} 点")
        print(f"  - 壁全体: {len(wall_coords)} 点")
        print(f"  - 装置: {len(equipment_coords)} 点")

    except FileNotFoundError:
        print(f"エラー: ファイル '{CACHE_FILE_PATH}' が見つかりません。")
        print("先に 'final_chipping_pipeline_robust.py' を実行してファイルを生成してください。")
        exit()
    except KeyError as e:
        print(f"エラー: ファイル内に必要なデータが見つかりません: {e}")
        exit()

    # 2. 各点群に変換し、色を付ける
    geometries = []

    # 壁全体 (灰色)
    if len(wall_coords) > 0:
        pcd_wall = o3d.geometry.PointCloud()
        pcd_wall.points = o3d.utility.Vector3dVector(wall_coords)
        pcd_wall.paint_uniform_color([0.7, 0.7, 0.7]) # 灰色
        geometries.append(pcd_wall)

    # 装置 (緑色)
    if len(equipment_coords) > 0:
        pcd_equipment = o3d.geometry.PointCloud()
        pcd_equipment.points = o3d.utility.Vector3dVector(equipment_coords)
        pcd_equipment.paint_uniform_color([0.0, 1.0, 0.0]) # 緑色
        geometries.append(pcd_equipment)
        
    # はつり範囲 (黄色)
    if len(chipping_coords) > 0:
        pcd_chipping = o3d.geometry.PointCloud()
        pcd_chipping.points = o3d.utility.Vector3dVector(chipping_coords)
        pcd_chipping.paint_uniform_color([1.0, 1.0, 0.0]) # 黄色
        geometries.append(pcd_chipping)

    # 3. 3Dで可視化
    if not geometries:
        print("警告: 表示できる点群データがありませんでした。")
    else:
        print("\n--- 3Dビューワを起動します ---")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="キャッシュファイル可視化"
        )