# ファイル名: step4.9_visualize_full_dataset.py
# 生成日時: 2025-07-12 20:22
# 概要: ランダムサンプリングを行わず、全ての点群データを対象にスコア計算と可視化を行う。
#      処理に時間がかかる可能性がある。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import japanize_matplotlib

# --- ★★★ ユーザー設定項目 ★★★ ---
MODEL_PATH = 'pointnet_model_vC.pth'
PCD_PATH = 'work_area_large.csv'
NUM_NEIGHBORS = 64

# ▼▼▼【重要】サンプリングを無効化するため、0に設定 ▼▼▼
MAX_POINTS_TO_PROCESS = 0

# --- モデル定義 (変更なし) ---
class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k)
        self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x)
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1)
        x = x + iden; x = x.view(-1, self.k, self.k)
        return x

class PointNet(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNet, self).__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x):
        trans_input = self.input_transform(x)
        x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1)
        x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x

# --- メインの処理 ---
if __name__ == '__main__':
    # 1. モデルとデータの準備
    device = torch.device("cpu")
    model = PointNet(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。")
        exit()
    model.eval()
    print(f"'{MODEL_PATH}' から学習済みモデルを読み込みました。")

    try:
        points_to_process = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values
    except FileNotFoundError:
        print(f"エラー: 点群ファイル '{PCD_PATH}' が見つかりません。")
        exit()

    print(f"全{len(points_to_process)}点を対象に処理を開始します。")

    print("近傍探索用のKD-Treeを構築中...")
    pcd_to_process = o3d.geometry.PointCloud()
    pcd_to_process.points = o3d.utility.Vector3dVector(points_to_process)
    kdtree = o3d.geometry.KDTreeFlann(pcd_to_process)
    
    # 2. スコアの取得と分類結果の保存
    all_scores = []
    colors = np.full((len(points_to_process), 3), [0.5, 0.5, 0.5]) 
    
    print("推論を実行し、各点のスコアを取得・分類中...")
    with torch.no_grad():
        for i in tqdm(range(len(points_to_process)), desc="分類中"):
            current_point = points_to_process[i]
            [k, idx, _] = kdtree.search_knn_vector_3d(current_point, NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS:
                continue
            
            neighbors = points_to_process[idx, :]
            normalized_neighbors = neighbors - current_point
            input_tensor = torch.tensor(normalized_neighbors.T, dtype=torch.float32).unsqueeze(0).to(device)
            
            scores = model(input_tensor).cpu().numpy().flatten()
            all_scores.append(scores)
            
            rebar_score = scores[0]
            not_rebar_score = scores[1]
            
            if rebar_score > not_rebar_score:
                colors[i] = [1, 0, 0]  # 赤 (鉄筋)
            else:
                colors[i] = [0, 0, 1]  # 青 (非鉄筋)

    # 3. 3D点群の表示
    print("\n--- 3D点群の可視化 ---")
    pcd_result = o3d.geometry.PointCloud()
    pcd_result.points = o3d.utility.Vector3dVector(points_to_process)
    pcd_result.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_result], window_name="Full Dataset Classification Result")

    # 4. 2D散布図の表示
    if not all_scores:
        print("スコアを取得できなかったため、散布図は表示できません。")
        exit()
        
    print("\n--- 2D散布図の可視化 ---")
    scores_np = np.array(all_scores)
    rebar_scores = scores_np[:, 0]
    not_rebar_scores = scores_np[:, 1]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(rebar_scores, not_rebar_scores, s=5, alpha=0.3)
    plt.title("鉄筋/非鉄筋 評価スコア分布 (全データ)", fontsize=16)
    plt.xlabel("鉄筋らしさのスコア (Logits)", fontsize=12)
    plt.ylabel("非鉄筋らしさのスコア (Logits)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='grey', linewidth=0.8)
    plt.axvline(0, color='grey', linewidth=0.8)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
