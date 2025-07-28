# ファイル名: step4.10_analyze_pca_and_angle.py
# 生成日時: 2025-07-12 21:16
# 概要: 全点群データから得られた評価スコアに対し、主成分分析(PCA)を実行して
#      分布の主軸を特定します。さらに、各点が主軸からなす回転角を算出し、
#      その結果を色分けした散布図として可視化します。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA # 主成分分析のために追加

# --- ★★★ ユーザー設定項目 ★★★ ---
MODEL_PATH = 'pointnet_model_vC.pth'
PCD_PATH = 'work_area_large.csv'
NUM_NEIGHBORS = 64
MAX_POINTS_TO_PROCESS = 0 # 0ですべての点を処理

# --- モデル定義 (変更なし) ---
class TNet(nn.Module):
    # ... (省略)
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x

class PointNet(nn.Module):
    # ... (省略)
    def __init__(self, num_classes=2): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return x


# --- メインの処理 ---
if __name__ == '__main__':
    # 1. スコアの取得 (step4.9と同じ)
    # ... (長いので省略)
    device = torch.device("cpu"); model = PointNet(num_classes=2).to(device); model.load_state_dict(torch.load(MODEL_PATH, map_location=device)); model.eval(); print(f"'{MODEL_PATH}' から学習済みモデルを読み込みました。"); points_to_process = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values; print(f"全{len(points_to_process)}点を対象に処理を開始します。"); pcd_to_process = o3d.geometry.PointCloud(); pcd_to_process.points = o3d.utility.Vector3dVector(points_to_process); kdtree = o3d.geometry.KDTreeFlann(pcd_to_process); all_scores = []; colors = np.full((len(points_to_process), 3), [0.5, 0.5, 0.5]); print("推論を実行し、各点のスコアを取得・分類中...");
    with torch.no_grad():
        for i in tqdm(range(len(points_to_process)), desc="スコア取得中"):
            current_point = points_to_process[i]; k, idx, _ = kdtree.search_knn_vector_3d(current_point, NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: continue
            neighbors = points_to_process[idx, :]; normalized_neighbors = neighbors - current_point; input_tensor = torch.tensor(normalized_neighbors.T, dtype=torch.float32).unsqueeze(0).to(device); scores = model(input_tensor).cpu().numpy().flatten(); all_scores.append(scores)
    scores_np = np.array(all_scores)
    rebar_scores = scores_np[:, 0]
    not_rebar_scores = scores_np[:, 1]
    
    # ----------------------------------------------------------------
    # 2. 主成分分析(PCA)による主軸の計算
    # ----------------------------------------------------------------
    print("\n--- 主成分分析(PCA)を実行 ---")
    # PCAの準備。n_components=2は2次元で分析するという意味。
    pca = PCA(n_components=2)
    # スコアデータをPCAに学習させ、主軸を計算させる
    pca.fit(scores_np)
    
    # === 数式解説: 主軸 ===
    # 主軸とは、データのばらつきが最も大きい方向を示すベクトル。
    # PCAによって、データの共分散行列の固有ベクトルが求められる。
    # 第一主成分(pca.components_[0])が、最も大きい固有値に対応する固有ベクトルであり、
    # これがデータの主軸の方向を示す。
    # vec_v1 = (v1_x, v1_y)
    principal_axis_vector = pca.components_[0]
    
    # データの中心（平均値）もPCAから取得できる
    # vec_s_mean = (s_rebar_mean, s_not_rebar_mean)
    mean_score = pca.mean_
    
    print(f"データの中心 (平均スコア): ({mean_score[0]:.2f}, {mean_score[1]:.2f})")
    print(f"主軸の方向ベクトル: ({principal_axis_vector[0]:.2f}, {principal_axis_vector[1]:.2f})")

    # ----------------------------------------------------------------
    # 3. 各点の主軸に対する回転角の計算
    # ----------------------------------------------------------------
    print("--- 各点の回転角を計算 ---")
    
    # === 数式解説: 回転角 ===
    # 1. 各スコアの平均値を0にする（中心化）
    #    vec_s_prime = vec_s - vec_s_mean
    centered_scores = scores_np - mean_score
    
    # 2. 主軸(v1)と、それに直交する軸(v2)を基準とした新しい座標を計算
    #    v2は第二主成分(pca.components_[1])として得られる
    #    new_x = vec_s_prime ・ vec_v1  (主軸方向の射影長)
    #    new_y = vec_s_prime ・ vec_v2  (主軸に直交する方向の射影長)
    transformed_scores = pca.transform(scores_np)
    
    # 3. 新しい座標(new_x, new_y)から角度を計算
    #    theta = arctan2(new_y, new_x)
    #    arctan2は、-piから+pi (-180°から+180°)の範囲で正しい象限の角度を返す
    angles_rad = np.arctan2(transformed_scores[:, 1], transformed_scores[:, 0])
    angles_deg = np.degrees(angles_rad) # ラジアンを度に変換

    # ----------------------------------------------------------------
    # 4. 結果の可視化
    # ----------------------------------------------------------------
    print("--- 結果を可視化 ---")
    plt.figure(figsize=(12, 10))
    
    # 回転角(angles_deg)に応じて色分けした散布図をプロット
    # cmap='hsv'は色の変化が分かりやすいカラーマップ
    scatter = plt.scatter(rebar_scores, not_rebar_scores, s=5, alpha=0.5, c=angles_deg, cmap='hsv')
    
    # 主軸の線を描画
    # 線の長さを決めるため、スコアの広がりを取得
    axis_range = max(rebar_scores.max() - rebar_scores.min(), not_rebar_scores.max() - not_rebar_scores.min())
    # 中心の点から、主軸ベクトルの両方向に線を伸ばす
    p1 = mean_score - principal_axis_vector * axis_range / 2
    p2 = mean_score + principal_axis_vector * axis_range / 2
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", lw=2, label="主軸 (第一主成分)")
    
    # グラフの装飾
    plt.title("評価スコアの主成分分析と回転角の可視化", fontsize=16)
    plt.xlabel("鉄筋らしさのスコア (Logits)", fontsize=12)
    plt.ylabel("非鉄筋らしさのスコア (Logits)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    
    # カラーバーを追加
    cbar = plt.colorbar(scatter)
    cbar.set_label("主軸からの回転角 (°)", fontsize=12)
    
    plt.show()
