# ファイル名: step10.1_final_pipeline_robust_chipping.py
# 概要: はつり範囲の判定ロジックを、近傍の複数鉄筋との平均深度比較に改良し、
#       ノイズに強い、より安定した結果を目指す最終版パイプライン。

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
    
    # 壁面検出用パラメータ
    INITIAL_SEED_DIST_THRESH = 15.0; NORMAL_X_THRESHOLD = 0.9
    NEIGHBOR_SEARCH_RADIUS = 100.0; NORMAL_ANGLE_THRESHOLD = 7.5
    
    # 鉄筋検出用パラメータ
    NUM_NEIGHBORS = 64

    # ▼▼▼【新パラメータ】はつり範囲判定で、平均を取るための近傍鉄筋の数 ▼▼▼
    CHIPPING_REBAR_NEIGHBORS = 10

    # (フェーズ1, 2は変更なし)
    pcd_world, points_world = load_and_prepare_pcd(FULL_PCD_PATH)
    if pcd_world is None: exit()
    initial_seeds = get_initial_seeds(pcd_world, INITIAL_SEED_DIST_THRESH, NORMAL_X_THRESHOLD)
    if initial_seeds is None: exit("エラー: 壁面の最初のシードが見つかりませんでした。")
    final_wall_indices = calculate_wall_indices(pcd_world, initial_seeds, NORMAL_ANGLE_THRESHOLD, NEIGHBOR_SEARCH_RADIUS)
    if not final_wall_indices: exit("エラー: 領域成長の結果、壁面が検出されませんでした。")
    pcd_wall_only = pcd_world.select_by_index(final_wall_indices)
    if pcd_wall_only.is_empty(): exit("エラー: 抽出された壁の点群が空です。")
    print("\n--- フェーズ2: 壁面内の3クラス分類 ---")
    device = torch.device("cpu"); model = PointNet(num_classes=3).to(device)
    try: model.load_state_dict(torch.load(REBAR_MODEL_PATH, map_location=device))
    except FileNotFoundError: print(f"エラー: モデルファイル '{REBAR_MODEL_PATH}' が見つかりません。"); exit()
    model.eval()
    points_wall = np.asarray(pcd_wall_only.points); kdtree_wall = o3d.geometry.KDTreeFlann(pcd_wall_only)
    wall_labels = np.zeros(len(points_wall), dtype=int)
    with torch.no_grad():
        for i in tqdm(range(len(points_wall)), desc="3クラス分類中"):
            [k, idx, _] = kdtree_wall.search_knn_vector_3d(points_wall[i], NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: continue
            neighbors = points_wall[idx]; normalized_neighbors = neighbors - points_wall[i]
            input_tensor = torch.from_numpy(normalized_neighbors.T).float().unsqueeze(0).to(device)
            outputs = model(input_tensor); _, predicted_label = torch.max(outputs.data, 1)
            wall_labels[i] = predicted_label.item()

    # =================================================================
    # フェーズ3: はつり範囲のマッピング (ロジック改良版)
    # =================================================================
    print("\n--- フェーズ3: はつり範囲のマッピング (改良版ロジック) ---")
    
    rebar_indices = np.where(wall_labels == 0)[0]
    non_rebar_indices = np.where(wall_labels == 1)[0]
    final_wall_labels = np.copy(wall_labels)
    LABEL_CHIPPING = 4
    
    if len(rebar_indices) > 0 and len(non_rebar_indices) > 0:
        rebar_points_yz = points_wall[rebar_indices][:, 1:]
        rebar_kdtree_2d = cKDTree(rebar_points_yz)
        non_rebar_points_yz = points_wall[non_rebar_indices][:, 1:]
        
        # ▼▼▼【ロジック変更】最も近い1点ではなく、指定した数の近傍点を検索 ▼▼▼
        distances, nearest_rebar_indices_in_list = rebar_kdtree_2d.query(
            non_rebar_points_yz, k=CHIPPING_REBAR_NEIGHBORS
        )
        
        # k=1の場合と形状を合わせる
        if CHIPPING_REBAR_NEIGHBORS == 1:
            nearest_rebar_indices_in_list = np.expand_dims(nearest_rebar_indices_in_list, axis=1)

        # ▼▼▼【ロジック変更】見つけた複数の鉄筋のX座標の「平均値」を計算 ▼▼▼
        # nearest_rebar_indices_in_list の各行が、ある非鉄筋点に対する近傍鉄筋リストのインデックス
        # points_wall[rebar_indices[...]] で、それらの鉄筋の3D座標を取得
        nearest_rebar_coords = points_wall[rebar_indices[nearest_rebar_indices_in_list]]
        # X座標の平均を計算 (axis=1 は各非鉄筋点ごとに平均を取る)
        avg_nearest_rebar_x = np.mean(nearest_rebar_coords[:, :, 0], axis=1)
        
        non_rebar_x = points_wall[non_rebar_indices][:, 0]
        
        # ▼▼▼【ロジック変更】非鉄筋のX座標と「鉄筋の平均X座標」を比較 ▼▼▼
        chipping_mask = non_rebar_x < avg_nearest_rebar_x
        chipping_indices_in_wall = non_rebar_indices[chipping_mask]
        
        final_wall_labels[chipping_indices_in_wall] = LABEL_CHIPPING
        print(f"{len(chipping_indices_in_wall)} 点が「削るべき範囲」として特定されました。")
    else:
        print("鉄筋または非鉄筋が見つからなかったため、はつり範囲の計算をスキップしました。")

    # (フェーズ4, 5は変更なし)
    print("\n処理完了。最終結果を表示します。")
    color_map_final = {0: [1.0, 0.0, 0.0], 1: [0.0, 0.0, 1.0], 2: [0.0, 1.0, 0.0], 4: [1.0, 1.0, 0.0]}
    final_colors = np.array([color_map_final.get(label, [0.5,0.5,0.5]) for label in final_wall_labels])
    pcd_wall_only.colors = o3d.utility.Vector3dVector(final_colors)
    print("\nまず、詳細な分類結果（4色）を表示します。")
    o3d.visualization.draw_geometries([pcd_wall_only], window_name="最終結果(1/2): 詳細分類 (改良版)")
    print("\n次に「削る/削らない」のシンプルな判断結果（2色）を表示します。")
    colors_binary = np.full((len(points_wall), 3), [0.5, 0.5, 0.5])
    colors_binary[final_wall_labels == LABEL_CHIPPING] = [1.0, 1.0, 0.0]
    pcd_wall_only.colors = o3d.utility.Vector3dVector(colors_binary)
    o3d.visualization.draw_geometries([pcd_wall_only], window_name="最終結果(2/2): はつり範囲 (改良版)")
    
    print("\n===== 全ての処理が完了しました =====")
