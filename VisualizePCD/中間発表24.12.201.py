#
#
#フィルタリング，いろみOKのエレベーター扉　棒付き
#2024.12.18.02.39.06.csv
#
#test202  2024.12.18.02.09.45        との比較   201～205 中間資料
##


import csv
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # カラーマップを利用

# --- フィルタリング条件の設定 ---
filter_conditions = {
    "x_min": -5.0,     # x座標の最小値
    "x_max": 5.0,  # x座標の最大値
    "y_min": -2.0, # y座標の最小値
    "y_max": 3.0,  # y座標の最大値
    "z_min": -5.0,  # z座標の最小値
    "z_max": None     # z座標の最大値（Noneの場合制限なし）
}


# CSVファイルの読み込み
filename = '2024.12.18.02.39.06.csv'  # CSVファイル名
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
print(f"intensity: {np.min(data[:, 3]):.2f} to {np.max(data[:, 3]):.2f}")  # 反射強度の範囲を確認

# --- フィルタリング処理 ---
filtered_data = data
if filter_conditions["x_min"] is not None:
    filtered_data = filtered_data[filtered_data[:, 0] >= filter_conditions["x_min"]]
if filter_conditions["x_max"] is not None:
    filtered_data = filtered_data[filtered_data[:, 0] <= filter_conditions["x_max"]]
if filter_conditions["y_min"] is not None:
    filtered_data = filtered_data[filtered_data[:, 1] >= filter_conditions["y_min"]]
if filter_conditions["y_max"] is not None:
    filtered_data = filtered_data[filtered_data[:, 1] <= filter_conditions["y_max"]]
if filter_conditions["z_min"] is not None:
    filtered_data = filtered_data[filtered_data[:, 2] >= filter_conditions["z_min"]]
if filter_conditions["z_max"] is not None:
    filtered_data = filtered_data[filtered_data[:, 2] <= filter_conditions["z_max"]]

# --- フィルタリング後のデータの範囲を確認 ---
print("\nFiltered Data Range:")
print(f"x: {np.min(filtered_data[:, 0]):.2f} to {np.max(filtered_data[:, 0]):.2f}")
print(f"y: {np.min(filtered_data[:, 1]):.2f} to {np.max(filtered_data[:, 1]):.2f}")
print(f"z: {np.min(filtered_data[:, 2]):.2f} to {np.max(filtered_data[:, 2]):.2f}")
print(f"intensity: {np.min(filtered_data[:, 3]):.2f} to {np.max(filtered_data[:, 3]):.2f}")  # 反射強度の範囲を確認

# --- 回転行列の設定 ---
theta = np.radians(30 + 2.0)  # センサー設置角度（30度に微調整）
RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0], 
                        [-np.sin(theta), 0, np.cos(theta)]])


#データのクリップ
print(len(filtered_data))
a = []
for i in range(0,len(filtered_data),1):
    if(filtered_data[i,0] > 0.3 and filtered_data[i,1] < 1.0 and filtered_data[i,2] > 0.2 and  filtered_data[i,2] < 1.5):
        a.append(filtered_data[i,:])
filtered_data = np.array(a[:])

print(filtered_data[10,:])

print(len(filtered_data))

avg = np.average(filtered_data.T[3])
std = np.std(filtered_data.T[3])

print(f"average: {avg}, standatd: {std}, min: {np.min(filtered_data.T[3])}, max: {np.min(filtered_data.T[3])}")

# データの回転
rotated_data = np.dot(RotationMat, filtered_data[:, :3].T).T  # x, y, z のみ回転

# z軸回りの回転行列の設定（-3度回転）
theta_z = np.radians(-3.0)
RotationMat_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                          [np.sin(theta_z), np.cos(theta_z), 0], 
                          [0, 0, 1]])

# データの回転（z軸回り）
rotated_data = np.dot(RotationMat_z, rotated_data.T).T




# --- グラデーションの設定（反射強度による色分け） ---
min_value = np.min(filtered_data[:, 3])  # 反射強度の最小値
max_value = np.max(filtered_data[:, 3])  # 反射強度の最大値

times = 3
min_value = avg-times*std # 反射強度の最小値
max_value = avg+times*std  # 反射強度の最大値

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
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Filtered Point Cloud", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウドと座標軸を追加
vis.add_geometry(pcd)
vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()
