# ファイル名: step1_process_and_create_work_area.py

import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# --- 設定項目 ---
# A: 元となる巨大なCSVファイルのパス
ORIGINAL_CSV_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'

# B: 保存する作業範囲ファイル名
WORK_AREA_CSV_PATH = 'work_area_large.csv'

# --- ここからご提示いただいたコード ---

# CSV読み込み（ヘッダーあり）
print(f"'{ORIGINAL_CSV_PATH}' を読み込み中...")
df = pd.read_csv(ORIGINAL_CSV_PATH)

# x, y, z, reflect列の抽出
data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
reflect_raw = df.iloc[:, 3].to_numpy(dtype=np.float32)

# 回転（センサー角度補正）
print("回転処理を実行中...")
theta = np.radians(30 + 2.0)
RotationMat = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])
data_rotated = np.dot(RotationMat, data_xyz.T).T

# 点ごとの距離計算
distances = np.linalg.norm(data_rotated, axis=1)

# 距離補正（強度 ÷ 距離²）
reflect_corrected = reflect_raw / (distances**2 + 1e-6)  # 0除算防止

# 任意の範囲指定　　#作業現場　周りのものあり　　下の方のareaの名前も変更
#work_area_large.csv
print("指定範囲でフィルタリング中...")
x_min, x_max = 0, 10000
y_min, y_max = -1800, 2000
z_min, z_max = -1700, 4000



# 任意の範囲指定　　　壁面だけ　work_area.csv
# x_min, x_max = 3000, 4000
# y_min, y_max = -500, 1000
# z_min, z_max = -1500, -500


# マスク
mask = (
    (data_rotated[:, 0] >= x_min) & (data_rotated[:, 0] <= x_max) &
    (data_rotated[:, 1] >= y_min) & (data_rotated[:, 1] <= y_max) &
    (data_rotated[:, 2] >= z_min) & (data_rotated[:, 2] <= z_max)
)
filtered_points = data_rotated[mask]
filtered_reflect = reflect_corrected[mask]
print(f"フィルタリング結果: {len(df)}点 -> {len(filtered_points)}点")


# ★★★ ここからが追加部分 ★★★

# 保存するデータを作成
# 座標(x,y,z)と反射強度を結合して、4列のデータにする
# filtered_reflectを(N,)から(N,1)に変形してから結合
data_to_save = np.hstack([filtered_points, filtered_reflect.reshape(-1, 1)])

# pandasのDataFrameに変換
df_to_save = pd.DataFrame(data_to_save)

# CSVファイルとして保存（ヘッダーとインデックスは不要）
print(f"処理結果を '{WORK_AREA_CSV_PATH}' に保存中...")
df_to_save.to_csv(WORK_AREA_CSV_PATH, header=False, index=False)
print("保存が完了しました。")

# ★★★ 追加部分ここまで ★★★


# --- ここから表示処理（内容は変更なし） ---

# Min-Max正規化
r_min = filtered_reflect.min()
r_max = filtered_reflect.max()
reflect_norm = (filtered_reflect - r_min) / (r_max - r_min + 1e-6)

# Jetカラーマップ変換
colors = plt.cm.jet(reflect_norm)[:, :3]

# 点群と色
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(filtered_points)
pcd1.colors = o3d.utility.Vector3dVector(colors)

# 座標軸
cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0)

# 表示
print("処理結果を可視化します...")
vis1 = o3d.visualization.Visualizer()
vis1.create_window(window_name="Work Area (B) Preview", width=1600, height=1200)
vis1.add_geometry(pcd1)
vis1.add_geometry(cs)

opt = vis1.get_render_option()
opt.point_size = 3.0

vis1.run()
vis1.destroy_window()