# ファイル名: step5.6_analyze_score_angle_origin_base.py
# 概要: PCAで算出した主軸の「方向」だけを利用し、原点(0,0)を通る直線を基準として、
#      各点の相対角度を計算・可視化する。

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
MODEL_PATH = 'pointnet_model_vC.pth'
PCD_PATH = 'work_area_large.csv'
NUM_NEIGHBORS = 64
NUM_CLASSES = 2
NUM_TOP_OUTLIERS = 10

# --- モデル定義 (省略) ---
class TNet(nn.Module):
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x
class PointNet(nn.Module):
    def __init__(self, num_classes=2): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return x

if __name__ == '__main__':
    # 1. スコア計算 (省略)
    device = torch.device("cpu"); model = PointNet(num_classes=NUM_CLASSES).to(device); model.load_state_dict(torch.load(MODEL_PATH, map_location=device)); model.eval(); print(f"'{MODEL_PATH}' から学習済みモデルを読み込みました。");
    try: points = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values
    except FileNotFoundError: print(f"エラー: 点群 '{PCD_PATH}' が見つかりません。"); exit()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)); kdtree = o3d.geometry.KDTreeFlann(pcd)
    all_logits = []; print("全点の評価スコアを計算中...");
    with torch.no_grad():
        for i in tqdm(range(len(points)), desc="スコア計算中"):
            [k, idx, _] = kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: all_logits.append([np.nan, np.nan]); continue
            neighbors = points[idx]; normalized_neighbors = neighbors - points[i]; input_tensor = torch.from_numpy(normalized_neighbors.T).float().unsqueeze(0).to(device); logits = model(input_tensor); all_logits.append(logits.squeeze().cpu().numpy())
    all_logits = np.array(all_logits)
    valid_indices = ~np.isnan(all_logits).any(axis=1); all_logits = all_logits[valid_indices]
    
    # 2. PCAで主軸の「方向」を計算
    pca = PCA(n_components=2)
    pca.fit(all_logits)
    main_axis_vector = pca.components_[0] # これが主軸の方向ベクトル
    
    # ▼▼▼【変更点】角度計算のロジックを変更 ▼▼▼
    # 主軸ベクトルの角度を計算
    main_axis_angle_rad = np.arctan2(main_axis_vector[1], main_axis_vector[0])
    
    # 各点の、原点からの角度を計算
    points_angle_rad = np.arctan2(all_logits[:, 1], all_logits[:, 0])
    
    # 主軸の角度からの相対角度を計算
    relative_angles_rad = points_angle_rad - main_axis_angle_rad
    
    # 角度を -180 ~ +180 の範囲に正規化
    angles_deg = np.rad2deg(relative_angles_rad)
    angles_deg = (angles_deg + 180) % 360 - 180
    
    # 色分けのために絶対値（0～180）に変換
    symmetric_angles = np.abs(angles_deg)

    # 3. 散布図のプロット
    plt.rcParams['font.family'] = 'Meiryo'
    plt.figure(figsize=(12, 10))
    cmap = mpl.colormaps['viridis_r']; norm = mpl.colors.Normalize(vmin=0, vmax=180)
    sc = plt.scatter(all_logits[:, 0], all_logits[:, 1], c=symmetric_angles, cmap=cmap, s=5, alpha=0.5)
    
    # ▼▼▼【変更点】原点を通る基準線を描画 ▼▼▼
    line_range = np.array([-1200, 400])
    origin_axis_line = line_range[:, np.newaxis] * main_axis_vector
    plt.plot(origin_axis_line[:, 0], origin_axis_line[:, 1], color='red', linewidth=2, label='基準軸 (原点通過)')
    
    plt.title('評価スコアと「原点基準の相対角度」の可視化', fontsize=16)
    plt.xlabel('鉄筋らしさのスコア (Logits)', fontsize=12); plt.ylabel('非鉄筋らしさのスコア (Logits)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend(); plt.axis('equal')
    cbar = plt.colorbar(sc, orientation='vertical', pad=0.02); cbar.set_label('基準軸からの絶対角度 (°)', fontsize=12)
    plt.show()

    # 4. 角度のヒストグラム表示 (変更なし)
    print("\n--- 角度の度数分布表 ---")
    bins = np.arange(0, 181, 10); hist, bin_edges = np.histogram(symmetric_angles, bins=bins)
    print(f"{'角度範囲(度)':<15} | {'点の数':<10}"); print("-" * 30)
    for i in range(len(hist)): print(f"{bin_edges[i]:>3.0f} - {bin_edges[i+1]:<7.0f} | {hist[i]:<10,}")
    print("-" * 30)
