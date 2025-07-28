# ファイル名: step10.0_final_pipeline.py
# 概要: 3クラス分類と壁面検出を並列実行し、結果を統合して
#       最終的なはつり範囲を特定する、完全自動化パイプライン。

import open3d as o3d
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

# ==============================================================================
# 1. PointNetモデルの定義 (変更なし)
# ==============================================================================
class TNet(nn.Module):
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x

class PointNet(nn.Module):
    def __init__(self, num_classes=3): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return F.log_softmax(x, dim=1)

# ==============================================================================
# 2. 壁面検出のための関数群 (変更なし)
# ==============================================================================
def load_and_prepare_pcd(file_path):
    print(f"'{file_path}' を読み込み中...");
    try: df = pd.read_csv(file_path, header=None)
    except FileNotFoundError: print(f"エラー: ファイル '{file_path}' が見つかりません。"); return None, None
    data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32); theta = np.radians(30 + 2.0); RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]); points = np.dot(RotationMat, data_xyz.T).T; pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points); print("法線を計算中..."); pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30)); print(f"読み込み完了。点の数: {len(pcd.points)}"); return pcd, points

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
    # --- ★★★ ユーザー設定項目 ★★★ ---
    FULL_PCD_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'
    REBAR_MODEL_PATH = 'pointnet_model_3class_final_robust.pth'
    
    # --- 壁面検出用パラメータ ---
    INITIAL_SEED_DIST_THRESH = 15.0; NORMAL_X_THRESHOLD = 0.9
    NEIGHBOR_SEARCH_RADIUS = 100.0; NORMAL_ANGLE_THRESHOLD = 7.5
    
    # --- 鉄筋検出用パラメータ ---
    NUM_NEIGHBORS = 64

    # --- 共通のデータ読み込み ---
    pcd_world, points_world = load_and_prepare_pcd(FULL_PCD_PATH)
    if pcd_world is None: exit()

    # =================================================================
    # フェーズ1: 全体への3クラス分類
    # =================================================================
    print("\n--- フェーズ1: 全体への3クラス分類 ---")
    device = torch.device("cpu"); model = PointNet(num_classes=3).to(device)
    try: model.load_state_dict(torch.load(REBAR_MODEL_PATH, map_location=device))
    except FileNotFoundError: print(f"エラー: モデルファイル '{REBAR_MODEL_PATH}' が見つかりません。"); exit()
    model.eval()
    
    kdtree_world = o3d.geometry.KDTreeFlann(pcd_world)
    all_points_labels = np.zeros(len(points_world), dtype=int)

    with torch.no_grad():
        for i in tqdm(range(len(points_world)), desc="3クラス分類中 (全体)"):
            [k, idx, _] = kdtree_world.search_knn_vector_3d(points_world[i], NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: continue
            neighbors = points_world[idx]; normalized_neighbors = neighbors - points_world[i]
            input_tensor = torch.from_numpy(normalized_neighbors.T).float().unsqueeze(0).to(device)
            outputs = model(input_tensor); _, predicted_label = torch.max(outputs.data, 1)
            all_points_labels[i] = predicted_label.item()

    # =================================================================
    # フェーズ2: 全体への壁面検出
    # =================================================================
    print("\n--- フェーズ2: 壁面検出 ---")
    initial_seeds = get_initial_seeds(pcd_world, INITIAL_SEED_DIST_THRESH, NORMAL_X_THRESHOLD)
    if initial_seeds is None: exit("エラー: 壁面の最初のシードが見つかりませんでした。")
    final_wall_indices = calculate_wall_indices(pcd_world, initial_seeds, NORMAL_ANGLE_THRESHOLD, NEIGHBOR_SEARCH_RADIUS)
    if not final_wall_indices: exit("エラー: 領域成長の結果、壁面が検出されませんでした。")
    print(f"合計 {len(final_wall_indices)} 点が壁として特定されました。")

    # =================================================================
    # フェーズ3: 結果の統合と、はつり範囲のマッピング
    # =================================================================
    print("\n--- フェーズ3: 2つの結果を統合し、はつり範囲をマッピング ---")
    
    # -1:壁でない, 0:鉄筋, 1:非鉄筋(残す), 2:装置, 4:はつり範囲
    display_labels = np.full(len(points_world), -1, dtype=int)
    
    # まず、壁と判断された点に、3クラス分類の結果を割り当てる
    display_labels[final_wall_indices] = all_points_labels[final_wall_indices]
    
    # はつり判定の対象となる「壁であり、かつ非鉄筋」の点のインデックスを取得
    non_rebar_on_wall_indices = np.where(display_labels == 1)[0] # 1は非鉄筋
    # 比較対象となる「全ての鉄筋」のインデックスを取得
    rebar_indices = np.where(all_points_labels == 0)[0] # 0は鉄筋
    
    if len(rebar_indices) > 0 and len(non_rebar_on_wall_indices) > 0:
        rebar_points_yz = points_world[rebar_indices][:, 1:]; rebar_kdtree_2d = cKDTree(rebar_points_yz)
        non_rebar_on_wall_yz = points_world[non_rebar_on_wall_indices][:, 1:]
        
        distances, nearest_rebar_indices_in_list = rebar_kdtree_2d.query(non_rebar_on_wall_yz, k=1)
        
        non_rebar_on_wall_x = points_world[non_rebar_on_wall_indices][:, 0]
        nearest_rebar_x = points_world[rebar_indices[nearest_rebar_indices_in_list]][:, 0]
        
        chipping_mask = non_rebar_on_wall_x < nearest_rebar_x
        chipping_indices = non_rebar_on_wall_indices[chipping_mask]
        
        LABEL_CHIPPING = 4
        display_labels[chipping_indices] = LABEL_CHIPPING
        print(f"{len(chipping_indices)} 点が「削るべき範囲」として特定されました。")
    else:
        print("鉄筋または壁中の非鉄筋が見つからなかったため、はつり範囲の計算をスキップ。")

    # =================================================================
    # フェーズ4: 最終結果の可視化
    # =================================================================
    print("\n処理完了。最終結果を表示します。")
    
    # --- 表示1: 詳細な4(+1)色表示 ---
    color_map_detail = {
        -1: [0.3, 0.3, 0.3], # 壁でない (ダークグレー)
        0: [1.0, 0.0, 0.0],  # 鉄筋 (赤)
        1: [0.0, 0.0, 1.0],  # 非鉄筋 (青)
        2: [0.0, 1.0, 0.0],  # 装置 (緑)
        4: [1.0, 1.0, 0.0]   # はつり範囲 (黄)
    }
    detail_colors = np.array([color_map_detail.get(label, [0.1,0.1,0.1]) for label in display_labels])
    pcd_world.colors = o3d.utility.Vector3dVector(detail_colors)
    print("\nまず、詳細な分類結果（4色+背景）を表示します。")
    o3d.visualization.draw_geometries([pcd_world], window_name="最終結果(1/2): 詳細分類")

    # --- 表示2: シンプルな2(+1)色表示 ---
    color_map_binary = {
        -1: [0.3, 0.3, 0.3], # 壁でない (ダークグレー)
        4: [1.0, 1.0, 0.0]   # はつり範囲 (黄)
    }
    binary_colors = np.array([color_map_binary.get(label, [0.6, 0.6, 0.6]) for label in display_labels]) # 削らない点はライトグレー
    pcd_world.colors = o3d.utility.Vector3dVector(binary_colors)
    print("\n次に「削る/削らない」のシンプルな判断結果（2色+背景）を表示します。")
    o3d.visualization.draw_geometries([pcd_world], window_name="最終結果(2/2): はつり範囲")
    
    print("\n===== 全ての処理が完了しました =====")
