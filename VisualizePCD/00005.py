# ファイル名: step5.3_analyze_score_angle_symmetric.py
# 概要: 1. AIの評価スコアから主軸を計算する。
#      2. 各点の主軸からの角度を計算し、絶対値（0°～180°）に基づいて
#         色を付けた散布図を生成する。
#      3. 角度の分布が分かるヒストグラム（度数分布表）をコンソールに出力する。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- ★★★ ユーザー設定項目 ★★★ ---
# 2クラス分類用に学習させたモデルのパス
MODEL_PATH = 'pointnet_model_vC.pth' # 2クラス分類モデルを指定してください
# 分類したい点群ファイルのパス
PCD_PATH = 'work_area_large.csv'

# その他設定
NUM_NEIGHBORS = 64
NUM_CLASSES = 2

# --- モデル定義 (2クラス分類用) ---
class TNet(nn.Module):
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x
class PointNet(nn.Module):
    def __init__(self, num_classes=2): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return x # log_softmaxを適用する前のlogitsを返す

# --- メインの処理 ---
if __name__ == '__main__':
    # 1. モデルとデータの準備
    device = torch.device("cpu"); model = PointNet(num_classes=NUM_CLASSES).to(device)
    try: model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError: print(f"エラー: モデル '{MODEL_PATH}' が見つかりません。"); exit()
    model.eval(); print(f"'{MODEL_PATH}' から学習済みモデルを読み込みました。")
    
    try: points = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values
    except FileNotFoundError: print(f"エラー: 点群 '{PCD_PATH}' が見つかりません。"); exit()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)); kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # 2. 全点の評価スコア(logits)を取得
    all_logits = []
    print("全点の評価スコアを計算中...");
    with torch.no_grad():
        for i in tqdm(range(len(points)), desc="スコア計算中"):
            [k, idx, _] = kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: all_logits.append([0, 0]); continue
            neighbors = points[idx]; normalized_neighbors = neighbors - points[i]
            input_tensor = torch.from_numpy(normalized_neighbors.T).float().unsqueeze(0).to(device)
            logits = model(input_tensor)
            all_logits.append(logits.squeeze().cpu().numpy())
    all_logits = np.array(all_logits)

    # 3. 主成分分析(PCA)で主軸を計算
    pca = PCA(n_components=2)
    pca.fit(all_logits)
    main_axis_vector = pca.components_[0]
    
    # 4. 各点の主軸からの角度を計算
    angles = []
    for logit in all_logits:
        angle_rad = np.arctan2(logit[1], logit[0]) - np.arctan2(main_axis_vector[1], main_axis_vector[0])
        angles.append(np.rad2deg(angle_rad))
    angles = np.array(angles)
    
    # ▼▼▼【変更点1】角度を0～180度の範囲に変換 ▼▼▼
    # -180～180度を、絶対値を取ることで0～180度に変換
    symmetric_angles = np.abs(angles)

    # 5. 散布図のプロット
    plt.rcParams['font.family'] = 'Meiryo' # Windowsの場合
    plt.figure(figsize=(12, 10))
    
    # カラーマップの設定 (0度:紫 -> 180度:黄)
    cmap = mpl.colormaps['viridis_r']
    norm = mpl.colors.Normalize(vmin=0, vmax=180)
    
    sc = plt.scatter(all_logits[:, 0], all_logits[:, 1], c=symmetric_angles, cmap=cmap, s=5, alpha=0.5)
    
    # 主軸の描画
    line_range = np.array([-400, 400])
    main_axis_line = line_range[:, np.newaxis] * main_axis_vector
    plt.plot(main_axis_line[:, 0], main_axis_line[:, 1], color='red', linewidth=2, label='主軸 (第一主成分)')
    
    plt.title('評価スコアの主成分分析と「対称な」回転角の可視化', fontsize=16)
    plt.xlabel('鉄筋らしさのスコア (Logits)', fontsize=12)
    plt.ylabel('非鉄筋らしさのスコア (Logits)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.axis('equal')
    
    # カラーバーの設定
    cbar = plt.colorbar(sc, orientation='vertical', pad=0.02)
    cbar.set_label('主軸からの絶対角度 (°)', fontsize=12)
    plt.show()

    # ▼▼▼【変更点2】角度のヒストグラム（度数分布表）を作成・表示 ▼▼▼
    print("\n--- 角度の度数分布表 ---")
    # 10度ごとに区切る
    bins = np.arange(0, 181, 10)
    hist, bin_edges = np.histogram(symmetric_angles, bins=bins)
    
    print(f"{'角度範囲(度)':<15} | {'点の数':<10}")
    print("-" * 30)
    for i in range(len(hist)):
        print(f"{bin_edges[i]:>3.0f} - {bin_edges[i+1]:<7.0f} | {hist[i]:<10,}")
    print("-" * 30)
