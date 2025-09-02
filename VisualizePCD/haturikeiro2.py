# ファイル名: path_planning_dilation_fix.py
# 概要: グリッドマップを膨張させることで点群の連結性を確保し、安定した経路を生成する。

import open3d as o3d
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
# ▼▼▼【修正】膨張処理のためにscipyからbinary_dilationをインポート ▼▼▼
from scipy.ndimage import binary_dilation

# --- PointNetモデル定義 & 補助関数 (変更なし) ---
# (クラス定義、関数定義は変更がないため省略します)
class TNet(nn.Module):
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x
class PointNet(nn.Module):
    def __init__(self, num_classes=4): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return F.log_softmax(x, dim=1)
def load_and_prepare_pcd(file_path):
    print(f"'{file_path}' を読み込み中...");
    try: df = pd.read_csv(file_path, header=None)
    except FileNotFoundError: print(f"エラー: ファイル '{file_path}' が見つかりません。"); return None, None
    data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
    theta = np.radians(30 + 2.0)
    RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    points = np.dot(RotationMat, data_xyz.T).T
    pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(points)
    print("法線を計算中..."); pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30))
    print(f"読み込み完了。点の数: {len(pcd.points)}"); return pcd, points
def get_initial_seeds(pcd, dist_thresh, normal_x_thresh):
    print(f"\n法線フィルタとRANSACで最初のシードを検出中...");
    normals = np.asarray(pcd.normals); wall_candidate_indices = np.where(np.abs(normals[:, 0]) >= normal_x_thresh)[0]
    if len(wall_candidate_indices) < 100: print("警告: 正面向きの点が少なすぎます。"); return None
    pcd_candidates = pcd.select_by_index(wall_candidate_indices)
    plane_model, seed_indices_local = pcd_candidates.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000)
    if not seed_indices_local: return None
    seed_indices = wall_candidate_indices[seed_indices_local]
    core_pcd = pcd.select_by_index(seed_indices)
    labels = np.array(core_pcd.cluster_dbscan(eps=150.0, min_points=10, print_progress=False))
    if len(labels) > 0 and labels.max() != -1:
        main_cluster_label = pd.Series(labels[labels!=-1]).mode()[0]
        main_cluster_indices = np.where(labels == main_cluster_label)[0]
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
                if np.arccos(dot_product) < angle_threshold_rad: wall_indices.add(neighbor_idx); queue.append(neighbor_idx); pbar.update(1)
    print("壁面の特定が完了しました。"); return list(wall_indices)


# --- メイン処理 ---
if __name__ == '__main__':
    # =================================================================
    # フェーズ1, 2, 3: はつり範囲の特定
    # =================================================================
    FULL_PCD_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'
    REBAR_MODEL_PATH = 'pointnet_model_4class_1235.pth'
    INITIAL_SEED_DIST_THRESH = 15.0; NORMAL_X_THRESHOLD = 0.9
    NEIGHBOR_SEARCH_RADIUS = 100.0; NORMAL_ANGLE_THRESHOLD = 7.5
    NUM_NEIGHBORS = 64
    REBAR_TO_EQUIPMENT_THRESH_MM = 20.0
    REBAR_LAYER_DBSCAN_EPS_MM = 30.0
    REBAR_LAYER_MIN_POINTS = 20
    
    NOZZLE_DIAMETER_MM = 10.0
    CHIPPING_AREA_CLUSTER_EPS_MM = 50.0
    PATH_GRID_RESOLUTION_MM = 2.0 

    pcd_world, points_world = load_and_prepare_pcd(FULL_PCD_PATH)
    if pcd_world is None:
        exit()
    initial_seeds = get_initial_seeds(pcd_world, INITIAL_SEED_DIST_THRESH, NORMAL_X_THRESHOLD)
    if initial_seeds is None: exit("エラー: 壁面の最初のシードが見つかりませんでした。")
    final_wall_indices = calculate_wall_indices(pcd_world, initial_seeds, NORMAL_ANGLE_THRESHOLD, NEIGHBOR_SEARCH_RADIUS)
    if not final_wall_indices: exit("エラー: 領域成長の結果、壁面が検出されませんでした。")
    
    # ... (フェーズ2, 3 の処理は変更がないため省略)
    device = torch.device("cpu"); model = PointNet(num_classes=4).to(device)
    model.load_state_dict(torch.load(REBAR_MODEL_PATH, map_location=device))
    model.eval(); kdtree_world = o3d.geometry.KDTreeFlann(pcd_world); all_points_labels = np.zeros(len(points_world), dtype=int)
    with torch.no_grad():
        for i in tqdm(range(len(points_world)), desc="4クラス分類中 (全体)"):
            [k, idx, _] = kdtree_world.search_knn_vector_3d(points_world[i], NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: continue
            neighbors = points_world[idx]; normalized_neighbors = neighbors - points_world[i]
            input_tensor = torch.from_numpy(normalized_neighbors.T).float().unsqueeze(0).to(device)
            outputs = model(input_tensor); _, predicted_label = torch.max(outputs.data, 1)
            all_points_labels[i] = predicted_label.item()

    print("\n--- フェーズ3: 堅牢なはつり範囲の特定 ---")
    chipping_indices = []
    wall_plane_model = [1, 0, 0, 0] 
    wall_mask = np.zeros(len(points_world), dtype=bool); wall_mask[final_wall_indices] = True
    true_wall_points_indices = np.where((wall_mask) & (all_points_labels == 1))[0]
    if len(true_wall_points_indices) > 0:
        pcd_true_wall = pcd_world.select_by_index(true_wall_points_indices)
        wall_plane_model, _ = pcd_true_wall.segment_plane(distance_threshold=INITIAL_SEED_DIST_THRESH, ransac_n=3, num_iterations=1000)
    A, B, C, D = wall_plane_model
    refined_labels = np.copy(all_points_labels)
    rebar_indices_initial = np.where(all_points_labels == 0)[0]
    if A > 0: A, B, C, D = -A, -B, -C, -D
    points_to_check = points_world[rebar_indices_initial]
    distances = A * points_to_check[:, 0] + B * points_to_check[:, 1] + C * points_to_check[:, 2] + D
    misclassified_indices = rebar_indices_initial[distances > REBAR_TO_EQUIPMENT_THRESH_MM]
    refined_labels[misclassified_indices] = 2
    rebar_indices_refined = np.where(refined_labels == 0)[0]
    rebar_points_refined = points_world[rebar_indices_refined]
    if len(rebar_points_refined) > REBAR_LAYER_MIN_POINTS:
        rebar_x_coords = rebar_points_refined[:, 0].reshape(-1, 1)
        db = DBSCAN(eps=REBAR_LAYER_DBSCAN_EPS_MM, min_samples=5).fit(rebar_x_coords)
        cluster_labels = db.labels_
        unique_labels = set(cluster_labels)
        reliable_layers = []
        if -1 in unique_labels: unique_labels.remove(-1)
        for k in unique_labels:
            class_member_mask = (cluster_labels == k)
            if np.sum(class_member_mask) >= REBAR_LAYER_MIN_POINTS:
                layer_points = rebar_points_refined[class_member_mask]
                reliable_layers.append({'avg_depth': np.mean(layer_points[:, 0])})
        if reliable_layers:
            main_rebar_layer = max(reliable_layers, key=lambda x: x['avg_depth'])
            X_ref = main_rebar_layer['avg_depth']
            chipping_candidate_indices = np.where((wall_mask) & (refined_labels == 1))[0]
            chipping_candidate_points = points_world[chipping_candidate_indices]
            chipping_mask = chipping_candidate_points[:, 0] < X_ref
            chipping_indices = chipping_candidate_indices[chipping_mask].tolist()
    
    # =================================================================
    # フェーズ4: 経路計画
    # =================================================================
    print("\n--- フェーズ4: はつり経路の計画 ---")
    
    if len(chipping_indices) == 0:
        print("はつり範囲が特定されませんでした。")
        exit()

    # --- 2D写像とグルーピング ---
    plane_normal = np.array([A, B, C])
    up_vector = np.array([0, 1, 0])
    if np.linalg.norm(np.cross(plane_normal, up_vector)) < 1e-6: up_vector = np.array([0, 0, 1])
    x_basis = np.cross(up_vector, plane_normal); x_basis /= np.linalg.norm(x_basis)
    y_basis = np.cross(plane_normal, x_basis); y_basis /= np.linalg.norm(y_basis)
    chipping_points_3d = points_world[chipping_indices]
    origin_3d = np.mean(chipping_points_3d, axis=0)
    chipping_points_2d = np.vstack([np.dot(chipping_points_3d - origin_3d, x_basis), np.dot(chipping_points_3d - origin_3d, y_basis)]).T
    db_2d = DBSCAN(eps=CHIPPING_AREA_CLUSTER_EPS_MM, min_samples=10).fit(chipping_points_2d)
    area_labels = db_2d.labels_
    
    path_linesets = []
    unique_area_labels = set(area_labels)
    if -1 in unique_area_labels: unique_area_labels.remove(-1)

    print(f"{len(unique_area_labels)}個のはつり領域に対して、空白を避ける経路を生成します。")
    for label in unique_area_labels:
        # --- ステップA: 高解像度グリッドマップの作成 ---
        area_mask = (area_labels == label)
        area_points_2d = chipping_points_2d[area_mask]
        
        resolution = PATH_GRID_RESOLUTION_MM
        min_2d = np.min(area_points_2d, axis=0)
        
        grid_dims = np.ceil((np.max(area_points_2d, axis=0) - min_2d) / resolution).astype(int) + 1
        grid = np.zeros((grid_dims[1], grid_dims[0]), dtype=bool)

        grid_coords = np.floor((area_points_2d - min_2d) / resolution).astype(int)
        grid[grid_coords[:, 1], grid_coords[:, 0]] = True

        # ▼▼▼【修正】グリッドを「膨張」させて、点同士のわずかな隙間を埋める ▼▼▼
        grid = binary_dilation(grid, iterations=1)

        # --- ステップB: 深さ優先探索(DFS)で経路を生成 ---
        visited = np.zeros_like(grid, dtype=bool)
        path_grid_coords = []
        
        start_ys, start_xs = np.where(grid)
        if len(start_xs) == 0: continue
        
        start_idx = np.lexsort((start_xs, start_ys))[0]
        start_node = (start_ys[start_idx], start_xs[start_idx])
        
        stack = [start_node]
        
        while stack:
            y, x = stack[-1]
            if not visited[y, x]:
                visited[y, x] = True
                path_grid_coords.append((x, y))

            if y % 2 == 0: 
                neighbors = [(y, x+1), (y+1, x+1), (y+1, x), (y+1, x-1), (y, x-1), (y-1, x-1), (y-1, x), (y-1, x+1)]
            else: 
                neighbors = [(y, x-1), (y-1, x-1), (y-1, x), (y-1, x+1), (y, x+1), (y+1, x+1), (y+1, x), (y+1, x-1)]

            found_next = False
            for ny, nx in neighbors:
                if 0 <= ny < grid_dims[1] and 0 <= nx < grid_dims[0] and grid[ny, nx] and not visited[ny, nx]:
                    stack.append((ny, nx))
                    found_next = True
                    break
            
            if not found_next:
                stack.pop()

        # --- ステップC: 3D経路に変換 ---
        if len(path_grid_coords) > 1:
            # 経路の単純化を削除
            path_2d = np.array(path_grid_coords) * resolution + min_2d
            path_3d = np.array([origin_3d + p[0] * x_basis + p[1] * y_basis for p in path_2d])
            path_3d += plane_normal * 10.0
            
            lines = [[i, i + 1] for i in range(len(path_3d) - 1)]
            lineset = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(path_3d),
                lines=o3d.utility.Vector2iVector(lines)
            )
            lineset.colors = o3d.utility.Vector3dVector([[1, 0.5, 0] for _ in range(len(lines))])
            path_linesets.append(lineset)

    # --- ステップD: 最終可視化 ---
    print("最終結果: はつり範囲と経路を3Dで表示します。")
    chipping_pcd = pcd_world.select_by_index(chipping_indices)
    chipping_pcd.paint_uniform_color([1, 1, 0])
    
    geometries_to_draw = [chipping_pcd] + path_linesets
    o3d.visualization.draw_geometries(geometries_to_draw, window_name="はつり経路計画（膨張版）")

    print("\n===== 全ての処理が完了しました =====")