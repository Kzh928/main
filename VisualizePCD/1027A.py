# ファイル名: final_chipping_pipeline_robust.py
# 概要: 点群データから、壁面を検出し、その中の鉄筋・コンクリート壁面・装置・床を分類。
#       「掛け算」アプローチとロバストな基準面計算により、はつり範囲を正確に特定する。

import open3d as o3d
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
# ▼▼▼【追加】クラスタリングのためにDBSCANをインポート ▼▼▼
from sklearn.cluster import DBSCAN

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

# --- メイン処理 ---
if __name__ == '__main__':
    # =================================================================
    # フェーズ1: パラメータ設定とデータ準備
    # =================================================================
    FULL_PCD_PATH = 'mid360_XYZ_Reflect_data_03-28-13-53-11.csv'
    REBAR_MODEL_PATH = 'pointnet_model_4class_1235.pth' # 4クラスモデルを指定
    
    # --- 壁面検出パラメータ ---
    INITIAL_SEED_DIST_THRESH = 15.0; NORMAL_X_THRESHOLD = 0.9
    NEIGHBOR_SEARCH_RADIUS = 100.0; NORMAL_ANGLE_THRESHOLD = 7.5
    NUM_NEIGHBORS = 64
    
    # --- ▼▼▼【追加】新しいロジック用のパラメータ ▼▼▼ ---
    # 「掛け算」用：壁サーフェスからこの距離以上手前にある「鉄筋」は「装置」と見なす
    REBAR_TO_EQUIPMENT_THRESH_MM = 20.0
    # 「ロバストな基準面」用：鉄筋層クラスタリングのパラメータ
    REBAR_LAYER_DBSCAN_EPS_MM = 30.0 # 同じ層と見なす鉄筋間の最大距離
    REBAR_LAYER_MIN_POINTS = 20      # 「信頼できる層」と見なすための最小点数

    pcd_world, points_world = load_and_prepare_pcd(FULL_PCD_PATH)
    if pcd_world is None: exit()
    initial_seeds = get_initial_seeds(pcd_world, INITIAL_SEED_DIST_THRESH, NORMAL_X_THRESHOLD)
    if initial_seeds is None: exit("エラー: 壁面の最初のシードが見つかりませんでした。")
    final_wall_indices = calculate_wall_indices(pcd_world, initial_seeds, NORMAL_ANGLE_THRESHOLD, NEIGHBOR_SEARCH_RADIUS)
    if not final_wall_indices: exit("エラー: 領域成長の結果、壁面が検出されませんでした。")
    
    # =================================================================
    # フェーズ2: 全体への4クラス分類 (変更なし)
    # =================================================================
    print("\n--- フェーズ2: 全体への4クラス分類 ---")
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

    # =================================================================
    # ▼▼▼【ここから新しいロジック】▼▼▼
    # フェーズ3: 堅牢なはつり範囲の特定
    # =================================================================
    print("\n--- フェーズ3: 堅牢なはつり範囲の特定 ---")
    
    # --- ステップ2-1: 壁の「基準サーフェス」を計算 ---
    print("ステップ2-1: 壁の基準サーフェスを計算中...")
    wall_mask = np.zeros(len(points_world), dtype=bool); wall_mask[final_wall_indices] = True
    true_wall_points_indices = np.where((wall_mask) & (all_points_labels == 1))[0]
    pcd_true_wall = pcd_world.select_by_index(true_wall_points_indices)
    # RANSACで壁の平面方程式 Ax+By+Cz+D=0 を取得
    wall_plane_model, _ = pcd_true_wall.segment_plane(distance_threshold=INITIAL_SEED_DIST_THRESH, ransac_n=3, num_iterations=1000)
    A, B, C, D = wall_plane_model
    print(f"基準サーフェス: {A:.2f}x + {B:.2f}y + {C:.2f}z + {D:.2f} = 0")
    
    # --- ステップ2-2: 「掛け算」による分類ラベルの補正 (改善版) ---
    print("ステップ2-2: 「掛け算」アプローチで分類結果を補正中...")
    refined_labels = np.copy(all_points_labels)
    rebar_indices_initial = np.where(all_points_labels == 0)[0]

    # --- 壁の平面方程式を使った厳密な距離計算 ---
    # 平面の法線ベクトル(A,B,C)がスキャナ側(X座標が小さい方)を向いていると計算が楽なので、向きを調整
    if A > 0:
        A, B, C, D = -A, -B, -C, -D

    # 鉄筋と判定された点について、壁サーフェスからの「符号付き距離」を計算
    points_to_check = points_world[rebar_indices_initial]
    # 距離 = Ax + By + Cz + D  (この値が正なら手前、負なら奥)
    distances = A * points_to_check[:, 0] + B * points_to_check[:, 1] + C * points_to_check[:, 2] + D

    # 平面よりも「手前」に、閾値以上離れている点のみを「装置」と見なす
    misclassified_indices = rebar_indices_initial[distances > REBAR_TO_EQUIPMENT_THRESH_MM]
    refined_labels[misclassified_indices] = 2 # 2は装置のラベル
    print(f"{len(misclassified_indices)}個の「鉄筋」を「装置」として補正しました。")

    
    # --- ステップ3: ロバストな鉄筋層の特定 ---
    print("ステップ3: 深度クラスタリングで本命の鉄筋層を特定中...")
    # 補正後のラベルで鉄筋点を抽出
    rebar_indices_refined = np.where(refined_labels == 0)[0]
    rebar_points_refined = points_world[rebar_indices_refined]

    if len(rebar_points_refined) > REBAR_LAYER_MIN_POINTS:
        # X座標（深度）のみで1Dクラスタリング
        rebar_x_coords = rebar_points_refined[:, 0].reshape(-1, 1)
        db = DBSCAN(eps=REBAR_LAYER_DBSCAN_EPS_MM, min_samples=5).fit(rebar_x_coords)
        cluster_labels = db.labels_
        
        # 各クラスターの情報を集計
        unique_labels = set(cluster_labels)
        reliable_layers = []
        if -1 in unique_labels: unique_labels.remove(-1) # ノイズ(-1)は除外
        
        for k in unique_labels:
            class_member_mask = (cluster_labels == k)
            num_points_in_cluster = np.sum(class_member_mask)
            
            # 「信頼できる層」か判断
            if num_points_in_cluster >= REBAR_LAYER_MIN_POINTS:
                layer_points = rebar_points_refined[class_member_mask]
                avg_depth = np.mean(layer_points[:, 0])
                reliable_layers.append({'label': k, 'avg_depth': avg_depth, 'points': layer_points, 'indices': rebar_indices_refined[class_member_mask]})
                print(f"  - 信頼できる鉄筋層(ラベル{k})を発見: {num_points_in_cluster}点, 平均深度 {avg_depth:.2f} mm")

        # --- ステップ4: はつり範囲の最終決定 ---
        if reliable_layers:
            # 最も深度が深い「本命の層」を選ぶ
            main_rebar_layer = max(reliable_layers, key=lambda x: x['avg_depth'])
            print(f"本命の鉄筋層(ラベル{main_rebar_layer['label']})を選定。")
            
            # 本命の層から基準面を計算
            X_ref = main_rebar_layer['avg_depth']
            print(f"ロバストな基準面を X = {X_ref:.2f} mm に決定。")
            
            # はつり対象となる壁面上のコンクリート点を特定
            chipping_candidate_indices = np.where((wall_mask) & (refined_labels == 1))[0]
            chipping_candidate_points = points_world[chipping_candidate_indices]
            
            # 基準面より手前にある点を「はつり範囲」とする
            chipping_mask = chipping_candidate_points[:, 0] < X_ref
            chipping_indices = chipping_candidate_indices[chipping_mask]
            
            print(f"{len(chipping_indices)} 点が「はつり範囲」として特定されました。")
        else:
            chipping_indices = []
            print("警告: 信頼できる鉄筋層が見つかりませんでした。はつり範囲は0点です。")
    else:
        chipping_indices = []
        print("警告: 鉄筋候補の点数が少なすぎるため、はつり範囲の計算をスキップします。")

    # --- 最終的な表示用ラベルの作成 ---
    display_labels = np.full(len(points_world), -1, dtype=int)
    # 壁面上の点は、補正後の分類結果を反映
    display_labels[wall_mask] = refined_labels[wall_mask]
    # 壁面でない点は、鉄筋と装置のみ表示（床などはグレーアウト）
    non_wall_mask = ~wall_mask
    rebar_or_equip_mask = (refined_labels == 0) | (refined_labels == 2)
    display_labels[non_wall_mask & rebar_or_equip_mask] = refined_labels[non_wall_mask & rebar_or_equip_mask]

    # はつり範囲のラベルを最終的に上書き
    if len(chipping_indices) > 0:
        display_labels[chipping_indices] = 4 # 4ははつり範囲のラベル
        
    # === ▲▲▲【ここまで新しいロジック】▲▲▲ ===

    # =================================================================
    # フェーズ4: 最終結果の可視化 (変更なし)
    # =================================================================
    print("\n--- フェーズ4: 最終結果の4段階可視化 ---")
    color_map_detail = {
        -1: [0.3, 0.3, 0.3], 0: [1.0, 0.0, 0.0], 1: [0.0, 0.0, 1.0], 
        2: [0.0, 1.0, 0.0], 3: [1.0, 0.0, 1.0], 4: [1.0, 1.0, 0.0]
    }
    detail_colors = np.array([color_map_detail.get(label, [0.3,0.3,0.3]) for label in display_labels])
    pcd_world.colors = o3d.utility.Vector3dVector(detail_colors)
    print("\n[1/4] 詳細な分類結果（全体像）を表示します。")
    o3d.visualization.draw_geometries([pcd_world], window_name="最終結果(1/4): 詳細分類 (全体像)")

    color_map_binary = {-1: [0.3, 0.3, 0.3], 4: [1.0, 1.0, 0.0]}
    binary_colors = np.array([color_map_binary.get(label, [0.6, 0.6, 0.6]) for label in display_labels])
    pcd_world.colors = o3d.utility.Vector3dVector(binary_colors)
    print("\n[2/4] 「削る/削らない」のシンプルな判断結果（全体像）を表示します。")
    o3d.visualization.draw_geometries([pcd_world], window_name="最終結果(2/4): はつり範囲 (全体像)")
    
    print("\n--- 壁面のみの表示 ---")
    pcd_wall_only = pcd_world.select_by_index(final_wall_indices)
    labels_wall_only = display_labels[final_wall_indices]
    
    detail_colors_wall_only = np.array([color_map_detail.get(label, [0.3,0.3,0.3]) for label in labels_wall_only])
    pcd_wall_only.colors = o3d.utility.Vector3dVector(detail_colors_wall_only)
    print("\n[3/4] 詳細な分類結果（壁のみ）を表示します。")
    #o3d.visualization.draw_geometries([pcd_wall_only], window_name="最終結果(3/4): 詳細分類 (壁のみ)")

    binary_colors_wall_only = np.array([color_map_binary.get(label, [0.6, 0.6, 0.6]) for label in labels_wall_only])
    pcd_wall_only.colors = o3d.utility.Vector3dVector(binary_colors_wall_only)
    print("\n[4/4] 「削る/削らない」のシンプルな判断結果（壁のみ）を表示します。")
    #o3d.visualization.draw_geometries([pcd_wall_only], window_name="最終結果(4/4): はつり範囲 (壁のみ)")

    print("\n===== 全ての処理が完了しました =====")