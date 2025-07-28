# ファイル名: step4.12_highlight_outliers.py
# 生成日時: 2025-07-12 22:15
# 概要: 主軸からの垂直距離を計算し、距離が大きい上位N点を数値でリストアップすると同時に、
#      散布図上でも赤色のマーカーで強調表示する。

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
# 強調表示・リストアップする上位点の数
NUM_TOP_OUTLIERS = 20

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
    # ...
    device = torch.device("cpu"); model = PointNet(num_classes=2).to(device); model.load_state_dict(torch.load(MODEL_PATH, map_location=device)); model.eval(); print(f"'{MODEL_PATH}' から学習済みモデルを読み込みました。"); points_to_process = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values; print(f"全{len(points_to_process)}点を対象に処理を開始します。"); pcd_to_process_o3d = o3d.geometry.PointCloud(); pcd_to_process_o3d.points = o3d.utility.Vector3dVector(points_to_process); kdtree = o3d.geometry.KDTreeFlann(pcd_to_process_o3d); all_scores = [];
    with torch.no_grad():
        for i in tqdm(range(len(points_to_process)), desc="スコア取得中"):
            current_point = points_to_process[i]; k, idx, _ = kdtree.search_knn_vector_3d(current_point, NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: all_scores.append([np.nan, np.nan]); continue
            neighbors = points_to_process[idx, :]; normalized_neighbors = neighbors - current_point; input_tensor = torch.tensor(normalized_neighbors.T, dtype=torch.float32).unsqueeze(0).to(device); scores = model(input_tensor).cpu().numpy().flatten(); all_scores.append(scores)
    scores_np = np.array(all_scores)
    # NaN（計算できなかった点）を除外
    valid_indices = ~np.isnan(scores_np).any(axis=1)
    scores_np = scores_np[valid_indices]
    points_to_process = points_to_process[valid_indices]

    rebar_scores = scores_np[:, 0]
    not_rebar_scores = scores_np[:, 1]
    
    # 2. PCAと垂直距離の計算 (省略)
    pca = PCA(n_components=2); pca.fit(scores_np); principal_axis_vector = pca.components_[0]; mean_score = pca.mean_
    transformed_scores = pca.transform(scores_np)
    distances_from_axis = np.abs(transformed_scores[:, 1])

    # ----------------------------------------------------------------
    # 3. ▼▼▼【新機能】垂直距離が大きい上位N点の情報を数値で出力 ▼▼▼
    # ----------------------------------------------------------------
    print(f"\n--- 主軸からの垂直距離が大きい上位{NUM_TOP_OUTLIERS}点 ---")
    
    # 距離でソートし、上位N点のインデックスを取得
    sorted_indices = np.argsort(distances_from_axis)
    top_outlier_indices = sorted_indices[-NUM_TOP_OUTLIERS:] # 末尾からN個
    
    # 見やすいようにヘッダーを定義
    print(f"{'Rank':>4} | {'PointID':>7} | {'X':>8} | {'Y':>8} | {'Z':>8} | {'Rebarスコア':>12} | {'NonRebarスコア':>14} | {'垂直距離':>10}")
    print("-" * 100)
    
    # 距離が大きい順（降順）に表示
    rank = 1
    for i in reversed(top_outlier_indices):
        point_coords = points_to_process[i]
        rebar_sc = rebar_scores[i]
        not_rebar_sc = not_rebar_scores[i]
        dist = distances_from_axis[i]
        # f-stringを使って綺麗にフォーマット
        print(f"{rank:4d} | {i:7d} | {point_coords[0]:8.2f} | {point_coords[1]:8.2f} | {point_coords[2]:8.2f} | {rebar_sc:12.2f} | {not_rebar_sc:14.2f} | {dist:10.4f}")
        rank += 1

    # ----------------------------------------------------------------
    # 4. 結果の可視化
    # ----------------------------------------------------------------
    print("\n--- 結果を可視化 ---")
    plt.figure(figsize=(12, 10))
    
    # 全ての点を距離に応じてプロット
    scatter = plt.scatter(rebar_scores, not_rebar_scores, s=10, alpha=0.4, c=distances_from_axis, cmap='viridis')
    
    # ▼▼▼【新機能】上位N点を赤の×で強調表示 ▼▼▼
    plt.scatter(
        rebar_scores[top_outlier_indices], 
        not_rebar_scores[top_outlier_indices],
        s=100, marker='x', color='red', label=f"上位{NUM_TOP_OUTLIERS}点の例外点"
    )

    # 主軸の線を描画
    axis_range = max(rebar_scores.max() - rebar_scores.min(), not_rebar_scores.max() - not_rebar_scores.min())
    p1 = mean_score - principal_axis_vector * axis_range / 2
    p2 = mean_score + principal_axis_vector * axis_range / 2
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", lw=2, label="主軸 (第一主成分)")
    
    # グラフの装飾
    plt.title("評価スコアの主軸からの垂直距離の可視化（例外点ハイライト）", fontsize=16)
    plt.xlabel("鉄筋らしさのスコア (Logits)", fontsize=12)
    plt.ylabel("非鉄筋らしさのスコア (Logits)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    
    cbar = plt.colorbar(scatter)
    cbar.set_label("主軸からの垂直距離", fontsize=12)
    
    print("グラフを表示します。")
    print("グラフのツールバーにある虫眼鏡マークを使って、インタラクティブに拡大・縮小が可能です。")
    plt.show()
