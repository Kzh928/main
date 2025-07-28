#
#出力窓大きさ変更
#importされているか置いた文字列を定義できているか
#pcd,cs
#
#
#
##


#import open3d as o3d
#import numpy as np


# ポイントクラウドと座標軸を作成するコード
# pcd = ...
# cs = ...

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウドと座標軸を追加
vis.add_geometry(pcd)
vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()
