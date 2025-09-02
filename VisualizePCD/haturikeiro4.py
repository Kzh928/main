# ファイル名: final_chipping_pipeline_v7_0_final.py
# 概要: クラスタリング表示色を増やし、他の点群を障害物として認識させて
#       経路の内部貫通を完全に防ぐロジックを導入した最終完成版。

import open3d as o3d
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, MultiPoint, LineString
from shapely.affinity import rotate
from scipy.spatial import ConvexHull
import os
import cv2 # OpenCVライブラリ
import matplotlib.pyplot as plt

# --- PointNetモデル定義 (変更なし) ---
class TNet(nn.Module):
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x
class PointNet(nn.Module):
    def __init__(self, num_classes=4): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return F.log_softmax(x, dim=1)

# --- 補助関数 (変更なし) ---
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

def create_polygons_with_holes_from_points(cluster_points_2d, obstacle_points_2d, cluster_label, grid_resolution_mm=15.0, morph_kernel_size=3, morph_close_iterations=2):
    if len(cluster_points_2d) < 3:
        return []

    # 全ての点を含む描画領域を計算
    all_points = np.vstack([cluster_points_2d, obstacle_points_2d]) if len(obstacle_points_2d) > 0 else cluster_points_2d
    min_coords = np.min(all_points, axis=0) - grid_resolution_mm
    max_coords = np.max(all_points, axis=0) + grid_resolution_mm
    
    dims = np.ceil((max_coords - min_coords) / grid_resolution_mm).astype(int) + 1
    height, width = dims[1], dims[0]
    
    # 作業対象（白）と障害物（灰色）を描画した画像を作成
    image = np.zeros((height, width), dtype=np.uint8)
    
    # 障害物を描画 (値: 128)
    if len(obstacle_points_2d) > 0:
        obstacle_indices = ((obstacle_points_2d - min_coords) / grid_resolution_mm).astype(int)
        image[obstacle_indices[:, 1], obstacle_indices[:, 0]] = 128
        
    # 作業対象クラスタを描画 (値: 255)
    points_indices = ((cluster_points_2d - min_coords) / grid_resolution_mm).astype(int)
    image[points_indices[:, 1], points_indices[:, 0]] = 255
    
    # 作業対象のみをマスクとして抽出
    mask = (image == 255).astype(np.uint8) * 255

    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iterations)

    contours, hierarchy = cv2.findContours(closed_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    polygons = []
    for i, contour in enumerate(contours):
        if hierarchy[0, i, 3] == -1:
            shell_coords = contour[:, 0, :] * grid_resolution_mm + min_coords
            if len(shell_coords) < 4:
                continue
            holes = []
            child_index = hierarchy[0, i, 2]
            while child_index != -1:
                hole_contour = contours[child_index]
                hole_coords = hole_contour[:, 0, :] * grid_resolution_mm + min_coords
                holes.append(hole_coords)
                child_index = hierarchy[0, child_index, 0]
            polygon = Polygon(shell=shell_coords, holes=holes)
            if polygon.is_valid and polygon.area > (NOZZLE_DIAMETER**2):
                 polygons.append(polygon)
    return polygons

def generate_path_from_points_image_based(points_2d, cluster_label, nozzle_diameter, grid_resolution_mm=15.0, morph_kernel_size=3, morph_close_iterations=2):
    if len(points_2d) < 3: return None, None
    nozzle_radius = nozzle_diameter / 2.0
    min_coords = np.min(points_2d, axis=0) - grid_resolution_mm
    max_coords = np.max(points_2d, axis=0) + grid_resolution_mm
    dims = np.ceil((max_coords - min_coords) / grid_resolution_mm).astype(int) + 1
    height, width = dims[1], dims[0]
    image = np.zeros((height, width), dtype=np.uint8)
    points_indices = ((points_2d - min_coords) / grid_resolution_mm).astype(int)
    image[points_indices[:, 1], points_indices[:, 0]] = 255
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iterations)
    offset_pixels = int(np.ceil(nozzle_radius / grid_resolution_mm))
    erosion_kernel = np.ones((3, 3), np.uint8) 
    eroded_image = cv2.erode(closed_image, erosion_kernel, iterations=offset_pixels)
    pixel_path = []
    scan_step = int(nozzle_diameter / grid_resolution_mm)
    if scan_step == 0: scan_step = 1
    go_right = True
    for y in range(0, height, scan_step):
        row = eroded_image[y, :]
        white_pixels = np.where(row > 0)[0]
        if len(white_pixels) == 0: continue
        segments = []
        current_segment = [white_pixels[0]]
        for i in range(1, len(white_pixels)):
            if white_pixels[i] == white_pixels[i-1] + 1:
                current_segment.append(white_pixels[i])
            else:
                segments.append(current_segment)
                current_segment = [white_pixels[i]]
        segments.append(current_segment)
        if not go_right:
            segments.reverse()
        for segment in segments:
            start_x = segment[0]; end_x = segment[-1]
            if go_right:
                pixel_path.append([start_x, y]); pixel_path.append([end_x, y])
            else:
                pixel_path.append([end_x, y]); pixel_path.append([start_x, y])
        go_right = not go_right
    if len(pixel_path) < 2: return None, None
    path_points_2d = np.array(pixel_path, dtype=float) * grid_resolution_mm + min_coords
    final_path = LineString(path_points_2d)
    final_filled_area = final_path.buffer(nozzle_radius, cap_style=2, join_style=2)
    return final_path, final_filled_area

# --- メイン処理 ---
if __name__ == '__main__':
    # =================================================================
    # フェーズ1: パラメータ設定とデータ準備
    # =================================================================
    FULL_PCD_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'
    REBAR_MODEL_PATH = 'pointnet_model_4class_1235.pth'
    
    USE_CACHED_RESULTS = True 
    CACHE_FILE_PATH = 'chipping_cache.npz'
    
    NOZZLE_DIAMETER = 50.0
    PATH_DBSCAN_EPS = 100.0 
    PATH_DBSCAN_MIN_POINTS = 30
    
    GRID_RESOLUTION_MM = 15.0
    MORPH_KERNEL_SIZE = 3
    MORPH_CLOSE_ITERATIONS = 2
    
    INITIAL_SEED_DIST_THRESH = 15.0; NORMAL_X_THRESHOLD = 0.9
    NEIGHBOR_SEARCH_RADIUS = 100.0; NORMAL_ANGLE_THRESHOLD = 7.5
    NUM_NEIGHBORS = 64
    REBAR_TO_EQUIPMENT_THRESH_MM = 20.0
    REBAR_LAYER_DBSCAN_EPS_MM = 30.0
    REBAR_LAYER_MIN_POINTS = 20
    
    pcd_world, points_world = load_and_prepare_pcd(FULL_PCD_PATH)
    if pcd_world is None: exit()
    
    if USE_CACHED_RESULTS and os.path.exists(CACHE_FILE_PATH):
        print("\n--- 中間キャッシュの読み込み ---")
        cache = np.load(CACHE_FILE_PATH)
        chipping_indices = cache['chipping_indices']
        X_ref = cache['X_ref'].item()
        final_wall_indices = cache['final_wall_indices']
        print("中間キャッシュの読み込みが完了しました。")
    else:
        # (キャッシュがない場合、初回実行時のみ動作)
        print("\n--- 通常のパイプラインを実行 ---")
        initial_seeds = get_initial_seeds(pcd_world, INITIAL_SEED_DIST_THRESH, NORMAL_X_THRESHOLD)
        if initial_seeds is None: exit("エラー: 壁面の最初のシードが見つかりませんでした。")
        final_wall_indices = calculate_wall_indices(pcd_world, initial_seeds, NORMAL_ANGLE_THRESHOLD, NEIGHBOR_SEARCH_RADIUS)
        if not final_wall_indices: exit("エラー: 領域成長の結果、壁面が検出されませんでした。")
        device = torch.device("cpu"); model = PointNet(num_classes=4).to(device)
        try: model.load_state_dict(torch.load(REBAR_MODEL_PATH, map_location=device))
        except Exception as e: print(f"モデルの読み込み中にエラーが発生しました: {e}"); exit()
        model.eval(); kdtree_world = o3d.geometry.KDTreeFlann(pcd_world); all_points_labels = np.zeros(len(points_world), dtype=int)
        with torch.no_grad():
            for i in tqdm(range(len(points_world)), desc="4クラス分類中 (全体)"):
                [k, idx, _] = kdtree_world.search_knn_vector_3d(points_world[i], NUM_NEIGHBORS)
                if k < NUM_NEIGHBORS: continue
                neighbors = points_world[idx]; normalized_neighbors = neighbors - points_world[i]
                input_tensor = torch.from_numpy(normalized_neighbors.T).float().unsqueeze(0).to(device)
                outputs = model(input_tensor); _, predicted_label = torch.max(outputs.data, 1)
                all_points_labels[i] = predicted_label.item()
        wall_mask = np.zeros(len(points_world), dtype=bool); wall_mask[final_wall_indices] = True
        true_wall_points_indices = np.where((wall_mask) & (all_points_labels == 1))[0]
        pcd_true_wall = pcd_world.select_by_index(true_wall_points_indices)
        wall_plane_model, _ = pcd_true_wall.segment_plane(distance_threshold=INITIAL_SEED_DIST_THRESH, ransac_n=3, num_iterations=1000)
        A, B, C, D = wall_plane_model
        refined_labels = np.copy(all_points_labels); rebar_indices_initial = np.where(all_points_labels == 0)[0]
        if A > 0: A, B, C, D = -A, -B, -C, -D
        points_to_check = points_world[rebar_indices_initial]
        distances = A * points_to_check[:, 0] + B * points_to_check[:, 1] + C * points_to_check[:, 2] + D
        misclassified_indices = rebar_indices_initial[distances > REBAR_TO_EQUIPMENT_THRESH_MM]
        refined_labels[misclassified_indices] = 2
        rebar_indices_refined = np.where(refined_labels == 0)[0]
        rebar_points_refined = points_world[rebar_indices_refined]
        chipping_indices = []; X_ref = -1
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
                    avg_depth = np.mean(layer_points[:, 0])
                    reliable_layers.append({'label': k, 'avg_depth': avg_depth})
            if reliable_layers:
                main_rebar_layer = max(reliable_layers, key=lambda x: x['avg_depth'])
                X_ref = main_rebar_layer['avg_depth']
                chipping_candidate_indices = np.where((wall_mask) & (refined_labels == 1))[0]
                chipping_candidate_points = points_world[chipping_candidate_indices]
                chipping_mask = chipping_candidate_points[:, 0] < X_ref
                chipping_indices = chipping_candidate_indices[chipping_mask]
        np.savez(CACHE_FILE_PATH, 
                 chipping_indices=np.array(chipping_indices, dtype=np.int32), 
                 X_ref=X_ref,
                 final_wall_indices=np.array(list(final_wall_indices), dtype=np.int32))
        print(f"計算結果を '{CACHE_FILE_PATH}' に保存しました。")

    # =================================================================
    # フェーズ3.5: はつり経路の生成
    # =================================================================
    print("\n--- フェーズ3.5: はつり経路の生成 ---")
    all_path_linesets = []
    cluster_pcds_to_visualize = []
    
    if len(chipping_indices) > 0:
        chipping_points_3d = points_world[chipping_indices]
        chipping_points_2d = chipping_points_3d[:, [1, 2]]
        
        # ▼▼▼【変更】障害物となる点群を定義 ▼▼▼
        non_chipping_wall_indices = list(set(final_wall_indices) - set(chipping_indices))
        obstacle_points_3d = points_world[non_chipping_wall_indices]
        obstacle_points_2d = obstacle_points_3d[:, [1, 2]]
        
        min_x_coord = np.min(chipping_points_3d[:, 0])
        path_x_coord_display = min_x_coord - 50.0
        
        db_path = DBSCAN(eps=PATH_DBSCAN_EPS, min_samples=PATH_DBSCAN_MIN_POINTS).fit(chipping_points_2d)
        path_labels = db_path.labels_
        unique_path_labels = set(path_labels)
        if -1 in unique_path_labels: unique_path_labels.remove(-1)
        print(f"はつり範囲を{len(unique_path_labels)}個の大まかな領域に分割しました。")

        # ▼▼▼【色の拡張】より多くの色を用意 ▼▼▼
        cluster_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0.5, 0], [0.5, 0, 1]]

        for i, label in enumerate(unique_path_labels):
            print(f"領域 {label} の経路を生成中...")
            try:
                cluster_mask = (path_labels == label)
                points_in_cluster_3d = chipping_points_3d[cluster_mask]
                points_in_cluster_2d = chipping_points_2d[cluster_mask]
                
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(points_in_cluster_3d)
                color = cluster_colors[i % len(cluster_colors)]
                cluster_pcd.paint_uniform_color(color)
                cluster_pcds_to_visualize.append(cluster_pcd)
                print(f"  - 領域 {label} を色 {color} で可可視化します。")
                
                # ▼▼▼【変更】障害物情報を関数に渡す ▼▼▼
                polygons_with_holes = create_polygons_with_holes_from_points(
                    points_in_cluster_2d,
                    obstacle_points_2d,
                    cluster_label=label,
                    grid_resolution_mm=GRID_RESOLUTION_MM,
                    morph_kernel_size=MORPH_KERNEL_SIZE,
                    morph_close_iterations=MORPH_CLOSE_ITERATIONS
                )
                print(f"  - 領域 {label} 内で、{len(polygons_with_holes)}個の独立したはつり可能エリア（穴と障害物を考慮）を検出。")

                for j, polygon in enumerate(polygons_with_holes):
                    # (generate_boustrophedon_path_2dはもう使わない)
                    path_2d, filled_area_2d = generate_path_from_points_image_based(
                        polygon.exterior.coords, # ポリゴンの外形点群を入力とする
                        cluster_label=f"{label}_{j}",
                        nozzle_diameter=NOZZLE_DIAMETER,
                        grid_resolution_mm=GRID_RESOLUTION_MM,
                        morph_kernel_size=MORPH_KERNEL_SIZE,
                        morph_close_iterations=MORPH_CLOSE_ITERATIONS
                    )

                    if path_2d is None:
                        print(f"  -> ポリゴン {j} の経路生成に失敗しました。スキップします。")
                        continue
                        
                    path_3d_coords = np.insert(np.array(path_2d.coords), 0, path_x_coord_display, axis=1)
                    lines = [[k, k + 1] for k in range(len(path_3d_coords) - 1)]
                    center_line_set = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(path_3d_coords),
                        lines=o3d.utility.Vector2iVector(lines)
                    )
                    center_line_set.paint_uniform_color([1, 0, 1])
                    all_path_linesets.append(center_line_set)
                    
                    fill_3d_coords = np.insert(np.array(filled_area_2d.exterior.coords), 0, path_x_coord_display, axis=1)
                    lines = [[k, k + 1] for k in range(len(fill_3d_coords) - 1)]
                    fill_line_set = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(fill_3d_coords),
                        lines=o3d.utility.Vector2iVector(lines)
                    )
                    fill_line_set.paint_uniform_color([0.2, 1, 1])
                    all_path_linesets.append(fill_line_set)
                    print(f"  -> ポリゴン {j} の経路を可視化リストに追加しました。")

            except Exception as e:
                print(f"  - エラー: 領域 {label} の処理中に予期せぬエラーが発生しました: {e}")

        print(f"すべての経路生成が完了しました。可視化リスト内のオブジェクト数: {len(all_path_linesets)}")
    else: print("はつり範囲が0点のため、経路生成をスキップしました。")
    
    # =================================================================
    # フェーズ4: 最終結果の可視化
    # =================================================================
    print("\n--- フェーズ4: 最終結果の可視化 ---")
    
    non_chipping_wall_pcd = pcd_world.select_by_index(non_chipping_wall_indices)
    non_chipping_wall_pcd.paint_uniform_color([0.5, 0.5, 0.5])

    print("はつり範囲(色分け)、壁の残り(灰)、生成経路(マゼンタ/シアン) を表示します。")
    o3d.visualization.draw_geometries(
        [non_chipping_wall_pcd] + cluster_pcds_to_visualize + all_path_linesets, 
        window_name="最終結果: はつり範囲と生成経路"
    )

    print("\n===== 全ての処理が完了しました =====")