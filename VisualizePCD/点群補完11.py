# 生成日: 2025-06-23
# 目的: 高密度な点群データを入力とし、内部で指定の割合に間引いた後、
#       点群補完AIを「使わなかった」場合の性能を評価する、自己完結型の比較実験用パイプライン。
# ファイル名: G3_test_without_upsampling_integrated.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---
# 間引きたい元の、高密度な点群ファイル
SOURCE_PCD_PATH = 'work_area_large.csv'
# 間引いて残す点の割合 (0.9 = 90%)
DOWNSAMPLE_RATIO = 0.6

# 使用する鉄筋検出モデル
DETECTOR_MODEL_PATH = 'pointnet_model_vC.pth'

# パラメータ
DETECTOR_NEIGHBORS = 64
FILTER_RADIUS = 30.0
FILTER_MIN_NEIGHBORS = 10


# --- モデル定義1: 点群補完AI ---
class StrictAdditiveGenerator(nn.Module):
    def __init__(self, input_points=32, additional_points=32):
        super(StrictAdditiveGenerator, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_points*3,256), nn.ReLU(), nn.Linear(256,128))
        self.decoder = nn.Sequential(nn.Linear(128,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU(), nn.Linear(512,additional_points*3))
        self.additional_points = additional_points
    def forward(self, x):
        x_flat = x.view(x.size(0), -1); features = self.encoder(x_flat); generated_32_flat = self.decoder(features)
        generated_32 = generated_32_flat.view(x.size(0), self.additional_points, 3); return generated_32

# --- モデル定義2: 鉄筋検出AI ---
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
        return F.log_softmax(x, dim=1)

# --- メインのパイプライン処理 ---
if __name__ == '__main__':
    # --- 0. 準備 ---
    device = torch.device("cpu")
    detector_model = PointNet(num_classes=2).to(device)
    try:
        detector_model.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location=device))
        print(f"'{DETECTOR_MODEL_PATH}' から学習済みモデルを読み込みました。")
    except FileNotFoundError as e:
        print(f"エラー: 鉄筋検出モデル '{DETECTOR_MODEL_PATH}' が見つかりません。"); exit()
    detector_model.eval()

    # --- 1. 高密度データを読み込み、指定の割合に間引く ---
    try:
        source_df = pd.read_csv(SOURCE_PCD_PATH, header=None)
        print(f"\n'{SOURCE_PCD_PATH}' を読み込み、{DOWNSAMPLE_RATIO*100:.0f}%に間引きます...")
    except FileNotFoundError:
        print(f"エラー: 入力点群ファイル '{SOURCE_PCD_PATH}' が見つかりません。"); exit()
    
    num_to_keep = int(len(source_df) * DOWNSAMPLE_RATIO)
    sparse_df = source_df.sample(n=num_to_keep)
    sparse_points = sparse_df.iloc[:, :3].values
    print(f" -> 低密度な点群 ({len(sparse_points)}点) をメモリ上に作成しました。")

    # --- 2. 点群補完なしで、鉄筋検出を直接実行 ---
    print("\n[実験] 点群補完なしで、鉄筋検出を直接実行します...")
    sparse_pcd = o3d.geometry.PointCloud(); sparse_pcd.points = o3d.utility.Vector3dVector(sparse_points)
    sparse_kdtree = o3d.geometry.KDTreeFlann(sparse_pcd)
    final_labels = np.full(len(sparse_points), 1, dtype=int)

    with torch.no_grad():
        for i in tqdm(range(len(sparse_points)), desc="  推論中"):
            [k, idx, _] = sparse_kdtree.search_knn_vector_3d(sparse_points[i], DETECTOR_NEIGHBORS)
            local_points = sparse_points[idx]
            if k < DETECTOR_NEIGHBORS:
                padding = np.repeat(local_points[-1:], DETECTOR_NEIGHBORS - k, axis=0)
                local_points = np.vstack([local_points, padding])
            
            normalized_64 = local_points - sparse_points[i]
            input_tensor_64 = torch.tensor(normalized_64.T, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = detector_model(input_tensor_64)
            _, predicted = torch.max(outputs.data, 1)
            final_labels[i] = predicted.cpu().numpy()[0]

    print(f" -> 鉄筋検出完了。鉄筋と予測された点数: {np.sum(final_labels == 0)}")

    # --- 3. 後処理と可視化 ---
    filtered_labels = final_labels # 後処理は一旦スキップ
    print("\n後処理なしの、AIの生の予測結果を可視化します。")
    
    final_colors = np.full((len(sparse_points), 3), 0.5)
    final_colors[filtered_labels == 0] = [1.0, 0.0, 0.0]
    final_colors[filtered_labels == 1] = [0.0, 0.0, 1.0]
    final_pcd = o3d.geometry.PointCloud(); final_pcd.points = o3d.utility.Vector3dVector(sparse_points); final_pcd.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.visualization.draw_geometries([final_pcd], window_name="Result WITHOUT Upsampling", width=1600, height=1200)
    
    print("\nパイプライン処理が完了しました。")
