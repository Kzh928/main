#
#
#SVMを使うとして，異常検知にone-class-svmを使用
#サンプルデータを作成してそれに適用してるコード
##
##


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import open3d as o3d



# === サンプルデータ生成 ===
# 鉄筋（ラベル: 1）と非鉄筋（ラベル: 0）のデータを作成
np.random.seed(42)
num_points = 1000

# 鉄筋: 高い反射強度、線状の形状を模倣
x_steel = np.random.uniform(0, 1, num_points)
y_steel = np.random.uniform(0, 0.2, num_points)
z_steel = np.random.uniform(0, 1, num_points)
intensity_steel = np.random.uniform(0.8, 1.0, num_points)
data_steel = np.stack([x_steel, y_steel, z_steel, intensity_steel], axis=1)

# 非鉄筋: 低い反射強度、不規則な形状を模倣
x_other = np.random.uniform(0, 1, num_points)
y_other = np.random.uniform(0, 1, num_points)
z_other = np.random.uniform(0, 1, num_points)
intensity_other = np.random.uniform(0.1, 0.5, num_points)
data_other = np.stack([x_other, y_other, z_other, intensity_other], axis=1)

# データとラベルを結合
data = np.vstack([data_steel, data_other])
labels = np.array([1] * num_points + [0] * num_points)  # 鉄筋: 1, 非鉄筋: 0

# === データ前処理 ===
# 特徴量: x, y, z, intensity
X = data
y = labels

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# スケール変換（標準化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === SVMを使用した分類 ===
# SVMモデルを定義
svc_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svc_model.fit(X_train_scaled, y_train)

# 予測
y_pred = svc_model.predict(X_test_scaled)

# 結果の評価
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# === One-Class SVMを使用した異常検知 ===
# 鉄筋（ラベル: 1）のデータのみを使用してモデルを学習
steel_data = X_train_scaled[y_train == 1]

# One-Class SVMモデルを定義
one_class_model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
one_class_model.fit(steel_data)

# 異常検知（1: 正常、-1: 異常）
ocsvm_predictions = one_class_model.predict(X_test_scaled)

# 異常値をカウント
normal_count = np.sum(ocsvm_predictions == 1)
anomaly_count = np.sum(ocsvm_predictions == -1)

print("\nOne-Class SVM Results:")
print(f"Number of normal data : {normal_count}")
print(f"Number of abnormal data : {anomaly_count}")

# === 結果の可視化 ===
# 正常データと異常データを別々にプロット
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 2], c=ocsvm_predictions, cmap='coolwarm', s=10, alpha=0.7)
plt.title("One-Class SVM Results: Normal (1) vs Anomaly (-1)")
plt.xlabel("X Coordinate")
plt.ylabel("Z Coordinate")
plt.colorbar(label="Prediction (1: Normal, -1: Anomaly)")
plt.show()

# === 点群データの可視化 ===
# 正常データ（青色）、異常データ（赤色）に色分け
colors = ['blue' if label == 1 else 'red' for label in ocsvm_predictions]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(X_test[:, :3])  # x, y, z のみ使用
pcd.colors = o3d.utility.Vector3dVector([[0, 0, 1] if label == 1 else [1, 0, 0] for label in ocsvm_predictions])



# Visualizerを使用してウィンドウのサイズを設定
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Custom Window", width=1600, height=1200)  # ウィンドウサイズを設定

# ポイントクラウドと座標軸を追加
vis.add_geometry(pcd)
# vis.add_geometry(cs)

# 描画
vis.run()
vis.destroy_window()

