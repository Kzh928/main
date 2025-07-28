# ファイル名: step4.11_analyze_pca_and_distance.py
# 生成日時: 2025-07-12 21:54
# 概要: PCAで算出した主軸に対し、各点がどれだけ垂直に離れているか（垂直距離）を計算し、
#      その距離に応じて点を色分けすることで、例外的な点を可視化する。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA

# --- ★★★ ユーザー設定項目 ★★★ ---
MODEL_PATH = 'pointnet_model_vC.pth'
PCD_PATH = 'work_area_large.csv'
NUM_NEIGHBORS = 64
MAX_POINTS_TO_PROCESS = 0 # 0ですべての点を処理

# --- モデル定義 (省略) ---
class TNet(nn.Module):
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x
class PointNet(nn.Module):
    def __init__(self, num_classes=2): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return x

# --- メインの処理 ---
if __name__ == '__main__':
    # 1. スコアの取得 (省略)
    device = torch.device("cpu"); model = PointNet(num_classes=2).to(device); model.load_state_dict(torch.load(MODEL_PATH, map_location=device)); model.eval(); print(f"'{MODEL_PATH}' から学習済みモデルを読み込みました。"); points_to_process = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values; print(f"全{len(points_to_process)}点を対象に処理を開始します。"); pcd_to_process = o3d.geometry.PointCloud(); pcd_to_process.points = o3d.utility.Vector3dVector(points_to_process); kdtree = o3d.geometry.KDTreeFlann(pcd_to_process); all_scores = []; colors = np.full((len(points_to_process), 3), [0.5, 0.5, 0.5]); print("推論を実行し、各点のスコアを取得・分類中...");
    with torch.no_grad():
        for i in tqdm(range(len(points_to_process)), desc="スコア取得中"):
            current_point = points_to_process[i]; k, idx, _ = kdtree.search_knn_vector_3d(current_point, NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: continue
            neighbors = points_to_process[idx, :]; normalized_neighbors = neighbors - current_point; input_tensor = torch.tensor(normalized_neighbors.T, dtype=torch.float32).unsqueeze(0).to(device); scores = model(input_tensor).cpu().numpy().flatten(); all_scores.append(scores)
    scores_np = np.array(all_scores)
    rebar_scores = scores_np[:, 0]
    not_rebar_scores = scores_np[:, 1]
    
    # 2. 主成分分析(PCA)の実行 (変更なし)
    print("\n--- 主成分分析(PCA)を実行 ---")
    pca = PCA(n_components=2)
    pca.fit(scores_np)
    principal_axis_vector = pca.components_[0]
    mean_score = pca.mean_

    # ----------------------------------------------------------------
    # 3. 各点の主軸からの「垂直距離」を計算
    # ----------------------------------------------------------------
    print("--- 各点の主軸からの垂直距離を計算 ---")
    
    # === 数式解説: 垂直距離 ===
    # PCAのtransformメソッドは、データを主成分を基底とする新しい座標系に変換する。
    # 変換後の1列目: 第一主成分方向の座標値
    # 変換後の2列目: 第二主成分（主軸に直交）方向の座標値
    # この2列目の値の絶対値が、主軸からの垂直距離に相当する。
    # distance = | vec_s_prime ・ vec_v2 |
    
    transformed_scores = pca.transform(scores_np)
    distances_from_axis = np.abs(transformed_scores[:, 1])

    # ----------------------------------------------------------------
    # 4. 結果の可視化
    # ----------------------------------------------------------------
    print("--- 結果を可視化 ---")
    plt.figure(figsize=(12, 10))
    
    # ▼▼▼【変更点】▼▼▼
    # 垂直距離(distances_from_axis)に応じて色分けした散布図をプロット
    # cmap='viridis'は、値が低いと紫、高いと黄色になるカラーマップ
    scatter = plt.scatter(rebar_scores, not_rebar_scores, s=5, alpha=0.5, c=distances_from_axis, cmap='viridis')
    
    # 主軸の線を描画 (変更なし)
    axis_range = max(rebar_scores.max() - rebar_scores.min(), not_rebar_scores.max() - not_rebar_scores.min())
    p1 = mean_score - principal_axis_vector * axis_range / 2
    p2 = mean_score + principal_axis_vector * axis_range / 2
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", lw=2, label="主軸 (第一主成分)")
    
    # グラフの装飾
    plt.title("評価スコアの主軸からの垂直距離の可視化", fontsize=16)
    plt.xlabel("鉄筋らしさのスコア (Logits)", fontsize=12)
    plt.ylabel("非鉄筋らしさのスコア (Logits)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    
    # ▼▼▼【変更点】▼▼▼
    # カラーバーのラベルを変更
    cbar = plt.colorbar(scatter)
    cbar.set_label("主軸からの垂直距離", fontsize=12)
    
    plt.show()
