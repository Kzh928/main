#
#
#
#回転フィルタリング
#
#-5から5くらいで壁全部入る
#
#数値プログラム中に入力しないといけないタイプ，先行入力じゃないやつ
#
##

import csv
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # カラーマップを利用

# CSVファイルの読み込み
filename = '2024.12.18.02.10.21.csv'  # CSVファイル名
with open(filename) as f:
    reader = csv.reader(f)
    data = [row for row in reader]

# NumPy 配列に変換
data = np.asarray(data, dtype=np.float32)

# --- データの範囲を確認 ---
print("Original Data Range:")
print(f"x: {np.min(data[:, 0]):.2f} to {np.max(data[:, 0]):.2f}")
print(f"y: {np.min(data[:, 1]):.2f} to {np.max(data[:, 1]):.2f}")
print(f"z: {np.min(data[:, 2]):.2f} to {np.max(data[:, 2]):.2f}")

# --- フィルタリング条件の入力 ---
print("\nEnter filtering range (leave blank for no filtering):")
x_min = input("x min: ")
x_max = input("x max: ")
y_min = input("y min: ")
y_max = input("y max: ")
z_min = input("z min: ")
z_max = input("z max: ")

# 入力値が空でない場合のみ数値化
x_min = float(x_min) if x_min else None
x_max = float(x_max) if x_max else None
y_min = float(y_min) if y_min else None
y_max = float(y_max) if y_max else None
z_min = float(z_min) if z_min else None
z_max = float(z_max) if z_max else None

# フィルタリング処理
filtered_data = data
if x_min is not None:
    filtered_data = filtered_data[filtered_data[:, 0] >= x_min]
if x_max is not None:
    filtered_data = filtered_data[filtered_data[:, 0] <= x_max]
if y_min is not None:
    filtered_data = filtered_data[filtered_data[:, 1] >= y_min]
if y_max is not None:
    filtered_data = filtered_data[filtered_data[:, 1] <= y_max]
if z_min is not None:
    filtered_data = filtered_data[filtered_data[:, 2] >= z_min]
if z_max is not None:
    filtered_data = filtered_data[filtered_data[:, 2] <= z_max]

# --- フィルタリング後のデータの範囲を確認 ---
print("\nFiltered Data Range:")
print(f"x: {np.min(filtered_data[:, 0]):.2f} to {np.max(filtered_data[:, 0]):.2f}")
print(f"y: {np.min(filtered_data[:, 1]):.2f} to {np.max(filtered_data[:, 1]):.2f}")
print(f"z: {np.min(filtered_data[:, 2]):.2f} to {np.max(filtered_data[:, 2]):.2f}")

# --- 回転行列の設定 ---
theta = np.radians(30 + 2.0)  # センサー設置角度（30度に微調整）
RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0], 
                        [-np.sin(theta), 0, np.cos(theta)]])

# データの回転
rotated_data = np.dot(RotationMat, filtered_data[:, :3].T).T  # x, y, z のみ回転

# z軸回りの回転行列の設定（-3度回転）
theta_z = np.radians(-3.0)
RotationMat_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                          [np.sin(theta_z), np.cos(theta_z), 0], 
                          [0, 0, 1]])

# データの回転（z軸回り）
rotated_data = np.dot(RotationMat_z, rotated_data.T).T

# グラデーションの設定（反射強度による色分け）
min_value = float(input("min (intensity range): "))  # 最小値
max_value = float(input("max (intensity range): "))  # 最大値

# 反射強度を正規化（0～1にスケール変換）
normalized_intensity = (filtered_data[:, 3] - min_value) / (max_value - min_value)
normalized_intensity = np.clip(normalized_intensity, 0, 1)  # 範囲外をクリップ

# カラーマップを選択
cmap = cm.get_cmap("jet")  # カラーマップを変更可能

# カラーマップからRGB値を取得
colors = cmap(normalized_intensity)[:, :3]  # RGBの最初の3成分を抽出

# 点群データを作成
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(rotated_data)  # 回転後の座標を設定
pcd.colors = o3d.utility.Vector3dVector(colors)  # 点群に色を設定

# 座標軸の作成（確認用）
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Filtered Point Cloud", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウドと座標軸を追加
vis.add_geometry(pcd)
vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()
