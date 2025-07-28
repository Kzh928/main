# 生成日: 2025-06-23
# 目的: 低密度な点群データに対し、点群補完と鉄筋検出を「効率的に」
#       連続実行する、最適化されたエンドツーエンドの推論パイプライン。
# ファイル名: G2_final_pipeline_optimized.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---
SPARSE_PCD_PATH = 'sparse_work_area_50percent.csv'
UPSAMPLER_MODEL_PATH = 'repulsion_model.pth'
DETECTOR_MODEL_PATH = 'pointnet_model_vC.pth'

SPARSE_NEIGHBORS = 32
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
        super(TNet, self).__init__(); self.k=k
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
        x = x + iden; x = x.view(-1, self.k, self.k); return x

class PointNet(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNet, self).__init__()
        self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x):
        trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1)
        x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024)
        x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# --- メインのパイプライン処理 ---
if __name__ == '__main__':
    device = torch.device("cpu")
    upsampler_model = StrictAdditiveGenerator().to(device)
    detector_model = PointNet(num_classes=2).to(device)
    try:
        upsampler_model.load_state_dict(torch.load(UPSAMPLER_MODEL_PATH, map_location=device))
        detector_model.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location=device))
    except FileNotFoundError as e:
        print(f"エラー: モデルファイルが見つかりません。 {e}"); exit()
    upsampler_model.eval(); detector_model.eval()

    try:
        sparse_points = pd.read_csv(SPARSE_PCD_PATH, header=None).iloc[:, :3].values
        print(f"'{SPARSE_PCD_PATH}' を読み込み、処理を開始します。")
    except FileNotFoundError:
        print(f"エラー: 入力点群ファイル '{SPARSE_PCD_PATH}' が見つかりません。"); exit()

    # --- 新しい効率的な統合ループ ---
    print("\n[統合処理] 高密度化と鉄筋検出を連続で実行します...")
    sparse_pcd = o3d.geometry.PointCloud(); sparse_pcd.points = o3d.utility.Vector3dVector(sparse_points)
    sparse_kdtree = o3d.geometry.KDTreeFlann(sparse_pcd)
    
    pred_labels = np.full(len(sparse_points), 1, dtype=int)

    with torch.no_grad():
        for i in tqdm(range(len(sparse_points)), desc="  統合処理中"):
            [k, idx, _] = sparse_kdtree.search_knn_vector_3d(sparse_points[i], SPARSE_NEIGHBORS)
            local_sparse_points = sparse_points[idx]
            if k < SPARSE_NEIGHBORS:
                padding = np.repeat(local_sparse_points[-1:], SPARSE_NEIGHBORS - k, axis=0)
                local_sparse_points = np.vstack([local_sparse_points, padding])

            input_tensor_32 = torch.from_numpy(local_sparse_points).unsqueeze(0).to(device).float()
            generated_32_points = upsampler_model(input_tensor_32).squeeze(0).cpu().numpy()
            
            combined_64_points = np.vstack([local_sparse_points, generated_32_points])
            
            normalized_64 = combined_64_points - np.mean(combined_64_points, axis=0)
            input_tensor_64 = torch.tensor(normalized_64.T, dtype=torch.float32).unsqueeze(0).to(device)
            
            outputs = detector_model(input_tensor_64)
            _, predicted = torch.max(outputs.data, 1)
            pred_labels[i] = predicted.cpu().numpy()[0]

    print(f" -> 鉄筋検出完了。鉄筋と予測された点数: {np.sum(pred_labels == 0)}")

    # --- 後処理 ---
    print("\n[後処理] 密度フィルタで、予測結果のノイズを除去します...")
    LABEL_REBAR = 0; LABEL_NOT_REBAR = 1
    rebar_indices = np.where(pred_labels == LABEL_REBAR)[0]
    if len(rebar_indices) > 0:
        rebar_pcd = sparse_pcd.select_by_index(rebar_indices)
        rebar_kdtree = o3d.geometry.KDTreeFlann(rebar_pcd)
        denoised_rebar_indices = []
        for i in tqdm(range(len(rebar_pcd.points)), desc="  フィルタリング中"):
            [k, _, _] = rebar_kdtree.search_radius_vector_3d(rebar_pcd.points[i], FILTER_RADIUS)
            if k >= FILTER_MIN_NEIGHBORS:
                denoised_rebar_indices.append(rebar_indices[i])
        filtered_labels = np.full(len(sparse_points), LABEL_NOT_REBAR); filtered_labels[denoised_rebar_indices] = LABEL_REBAR
        print(f" -> フィルタリング完了。最終的な鉄筋点数: {len(denoised_rebar_indices)}")
    else:
        filtered_labels = pred_labels
        print(" -> 鉄筋と予測された点がなかったため、フィルタリングをスキップしました。")

    # --- 最終結果の可視化 ---
    print("\n最終結果を可視化します。")
    final_colors = np.full((len(sparse_points), 3), 0.5)
    final_colors[filtered_labels == LABEL_REBAR] = [1.0, 0.0, 0.0]
    final_colors[filtered_labels == LABEL_NOT_REBAR] = [0.0, 0.0, 1.0]
    final_pcd = o3d.geometry.PointCloud(); final_pcd.points = o3d.utility.Vector3dVector(sparse_points); final_pcd.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.visualization.draw_geometries([final_pcd], window_name="Optimized End-to-End Test Result", width=1600, height=1200)
    
    print("\nパイプライン処理が完了しました。")
