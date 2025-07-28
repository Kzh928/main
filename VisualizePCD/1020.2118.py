#
#min,max,intarval
#4色？
#色もっと増やしてかな
#
#色分けのやつ
#
#そもそも一定以上のでいい？
#それとも鉄筋であろうでーたの範囲だけくぎるのが理想か
#データとって範囲区切って傾けて色分け#

import csv
import open3d as o3d
import numpy as np

# CSVファイルの読み込み
with open('2024.10.08.19.57.57.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

# NumPy 配列に変換
data01 = np.asarray(data, dtype=np.float32)

# グラデーションの設定
min_value = float(input("min: "))  # 最小値
max_value = float(input("max: "))  # 最大値
num_intervals = int(input("intarval: "))  # 色の間隔
#0.255

# 各間隔の幅を計算
interval_width = (max_value - min_value) / num_intervals

# 色を設定するための配列を初期化
colors = np.zeros((data01.shape[0], 3))  # RGBのための配列

# 各点の反射強度に基づいて色を決定
for i in range(data01.shape[0]):
    intensity = data01[i][3]  # 4列目をリフレクションの強度とする

    # 指定した範囲内の強度に色を設定
    if intensity < min_value:
        colors[i] = [0, 0, 1]  # 最小値未満は青
    elif intensity > max_value:
        colors[i] = [1, 0, 0]  # 最大値を超える場合は赤
    else:
        # 色のインデックスを計算
        index = int((intensity - min_value) / interval_width)
        # グラデーションに基づいて色を設定
        fraction = (intensity - min_value) / interval_width - index  # 割合を計算

        # 色の範囲を決定（青→緑→黄→赤）
        if index < num_intervals // 3:  # 青から緑
            r = 0
            g = fraction
            b = 1 - fraction
        elif index < 2 * (num_intervals // 3):  # 緑から黄
            r = fraction
            g = 1
            b = 0
        else:  # 黄から赤
            r = 1
            g = 1 - fraction
            b = 0

        colors[i] = [r, g, b]

# 点群データを作成
data002 = data01[:, :3]  # 最初の3列を使用
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data002)
pcd.colors = o3d.utility.Vector3dVector(colors)  # 点群に色を追加

# 座標軸の作成
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウドと座標軸を追加
vis.add_geometry(pcd)
vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()
