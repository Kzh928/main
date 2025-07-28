# ファイル名: step8.0_wall_and_rebar_detection.py
# 概要: 壁面検出と鉄筋検出を統合したパイプライン。
#       まず領域成長で壁面を特定し、その壁面領域内でのみ
#       PointNetによる3クラス分類（鉄筋/非鉄筋/装置）を実行する。

import open3d as o3d
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. PointNetモデルの定義 (学習時と全く同じ)
# ==============================================================================
class TNet(nn.Module):
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x

class PointNet(nn.Module):
    def __init__(self, num_classes=3): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return F.log_softmax(x, dim=1)

# ==============================================================================
# 2. 壁面検出のための関数群
# ==============================================================================
def load_and_prepare_pcd(file_path):
    print(f"'{file_path}' を読み込み中...");
    try: df = pd.read_csv(file_path)
    except FileNotFoundError: print(f"エラー: ファイル '{file_path}' が見つかりません。"); return None
    data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32); theta = np.radians(30 + 2.0); RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]); points = np.dot(RotationMat, data_xyz.T).T; pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points); print("法線を計算中..."); pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30)); print(f"読み込み完了。点の数: {len(pcd.points)}"); return pcd

def get_initial_seeds(pcd, dist_thresh, normal_x_thresh):
    print(f"\n法線フィルタとRANSACで最初のシードを検出中..."); normals = np.asarray(pcd.normals); wall_candidate_indices = np.where(np.abs(normals[:, 0]) >= normal_x_thresh)[0]
    if len(wall_candidate_indices) < 100: print("警告: 正面向きの点が少なすぎます。"); return None
    pcd_candidates = pcd.select_by_index(wall_candidate_indices); plane_model, seed_indices_local = pcd_candidates.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000)
    if not seed_indices_local: return None
    seed_indices = wall_candidate_indices[seed_indices_local]
    core_pcd = pcd.select_by_index(seed_indices); labels = np.array(core_pcd.cluster_dbscan(eps=150.0, min_points=10, print_progress=False))
    if len(labels) > 0 and labels.max() != -1:
        main_cluster_label = pd.Series(labels[labels!=-1]).mode()[0]; main_cluster_indices = np.where(labels == main_cluster_label)[0]
        seed_indices = np.array(seed_indices)[main_cluster_indices]
    return seed_indices

def calculate_wall_indices(pcd, seed_indices, normal_angle_thresh_deg, search_radius):
    print("\n領域成長で壁面全体を特定中..."); queue = deque(seed_indices); wall_indices = set(seed_indices); all_normals = np.asarray(pcd.normals); kdtree = o3d.geometry.KDTreeFlann(pcd); angle_threshold_rad = np.deg2rad(normal_angle_thresh_deg)
    with tqdm(total=len(np.asarray(pcd.points)), desc="壁面検出中") as pbar:
        pbar.update(len(wall_indices))
        while queue:
            current_idx = queue.popleft(); current_normal = all_normals[current_idx]
            k, neighbor_indices, _ = kdtree.search_radius_vector_3d(pcd.points[current_idx], search_radius)
            for neighbor_idx in neighbor_indices:
                if neighbor_idx in wall_indices: continue
                dot_product = np.clip(np.dot(current_normal, all_normals[neighbor_idx]), -1.0, 1.0)
                if np.arccos(dot_product) < angle_threshold_rad:
                    wall_indices.add(neighbor_idx); queue.append(neighbor_idx); pbar.update(1)
    print("壁面の特定が完了しました。"); return list(wall_indices)

# ==============================================================================
# 3. メイン処理
# ==============================================================================
if __name__ == '__main__':
    # --- パラメータ設定 ---
    WALL_PCD_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'
    REBAR_MODEL_PATH = 'pointnet_model_3class_final_robust.pth'
    
    # 壁面検出用パラメータ
    INITIAL_SEED_DIST_THRESH = 15.0
    NORMAL_X_THRESHOLD = 0.9
    NEIGHBOR_SEARCH_RADIUS = 100.0
    NORMAL_ANGLE_THRESHOLD = 7.5
    
    # 鉄筋検出用パラメータ
    NUM_NEIGHBORS = 64
    NUM_CLASSES = 3

    # --- ステップ1: 壁面検出 ---
    pcd_world = load_and_prepare_pcd(WALL_PCD_PATH)
    if pcd_world is None: exit()
    
    initial_seeds = get_initial_seeds(pcd_world, INITIAL_SEED_DIST_THRESH, NORMAL_X_THRESHOLD)
    if initial_seeds is None: exit("エラー: 壁面の最初のシードが見つかりませんでした。")
    print(f"正面の壁からシードとして {len(initial_seeds)} 点が見つかりました。")

    final_wall_indices = calculate_wall_indices(
        pcd_world, initial_seeds, NORMAL_ANGLE_THRESHOLD, NEIGHBOR_SEARCH_RADIUS)

    if not final_wall_indices:
        exit("エラー: 領域成長の結果、壁面が検出されませんでした。")
    print(f"合計 {len(final_wall_indices)} 点が壁として特定されました。")

    # --- ステップ2: 鉄筋検出 ---
    print("\nPointNetモデルを読み込んでいます...")
    device = torch.device("cpu")
    model = PointNet(num_classes=NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(REBAR_MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: モデルファイル '{REBAR_MODEL_PATH}' が見つかりません。"); exit()
    model.eval()
    print("モデルの読み込み完了。")

    points = np.asarray(pcd_world.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_world)
    
    # 最終的な色を格納する配列 (デフォルトは灰色)
    final_colors = np.full((len(points), 3), [0.5, 0.5, 0.5])
    color_map = {
        0: [1.0, 0.0, 0.0], # 鉄筋 -> 赤
        1: [0.0, 0.0, 1.0], # 非鉄筋 -> 青
        2: [0.0, 1.0, 0.0]  # 装置 -> 緑
    }

    print("\n壁面領域内で鉄筋検出を実行中...")
    with torch.no_grad():
        # 壁と特定された点に対してのみループ
        for i in tqdm(final_wall_indices, desc="鉄筋検出中"):
            [k, idx, _] = kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: continue
            
            neighbors = points[idx]
            normalized_neighbors = neighbors - points[i]
            input_tensor = torch.from_numpy(normalized_neighbors.T).float().unsqueeze(0).to(device)
            
            outputs = model(input_tensor)
            _, predicted_label = torch.max(outputs.data, 1)
            
            # 壁の点の色を、分類結果に応じて更新
            final_colors[i] = color_map[predicted_label.item()]

    # --- ステップ3: 最終結果の可視化 ---
    print("\n処理完了。最終結果を表示します。")
    pcd_world.colors = o3d.utility.Vector3dVector(final_colors)
    o3d.visualization.draw_geometries([pcd_world], window_name="統合結果: 壁面内の鉄筋検出")

    print("\n===== 全ての処理が完了しました =====")
