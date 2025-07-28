# ファイル名: pipeline.py (対話的範囲選択機能付き・完全版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import tkinter
from tkinter import filedialog

# --- モデル定義 ---
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

# --- 各ステップの処理を関数化 ---

def select_area_interactively(points):
    """対話的に範囲を選択し、そのバウンディングボックスを返す"""
    print("\n[ステップ0] 対話的な範囲選択を開始します。")
    print("Shiftキーを押しながら左クリックで、分析したいエリアの点をいくつか選択してください。")
    print("選択が終わったら、ウィンドウを閉じるか 'Q'キー を押してください。")

    pcd_to_select = o3d.geometry.PointCloud()
    pcd_to_select.points = o3d.utility.Vector3dVector(points)
    
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name="Interactive Area Selection (Shift+Click to select)", width=1600, height=1200)
    vis.add_geometry(pcd_to_select)
    vis.run()
    picked_indices_info = vis.get_picked_points()
    vis.destroy_window()

    if not picked_indices_info:
        print("点が選択されませんでした。プログラムを終了します。")
        return None

    picked_indices = [p.index for p in picked_indices_info]
    selected_points = points[picked_indices]
    selected_pcd = o3d.geometry.PointCloud(); selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
    
    aabb = selected_pcd.get_axis_aligned_bounding_box()
    min_b = aabb.min_bound
    max_b = aabb.max_bound
    
    margin = 50.0
    crop_box = [
        min_b[0] - margin, max_b[0] + margin,
        min_b[1] - margin, max_b[1] + margin,
        min_b[2] - margin, max_b[2] + margin,
    ]
    
    print("範囲が選択されました。")
    return crop_box

def preprocess_and_crop(raw_csv_path, crop_box):
    """ステップ1の処理: 前処理と切り出し"""
    print(f"\n[ステップ1] 前処理と切り出しを開始: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    data_xyz = df.iloc[:, :3].to_numpy(dtype=np.float32)
    theta = np.radians(30 + 2.0)
    RotationMat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    data_rotated = np.dot(RotationMat, data_xyz.T).T
    mask = ( (data_rotated[:, 0] >= crop_box[0]) & (data_rotated[:, 0] <= crop_box[1]) &
             (data_rotated[:, 1] >= crop_box[2]) & (data_rotated[:, 1] <= crop_box[3]) &
             (data_rotated[:, 2] >= crop_box[4]) & (data_rotated[:, 2] <= crop_box[5]) )
    filtered_points = data_rotated[mask]
    print(f" -> 切り出し後の点群数: {len(filtered_points)}")
    if len(filtered_points) == 0:
        print("警告: 切り出し範囲に点が存在しません。")
    return filtered_points

def run_inference(points, model_path, num_neighbors=64):
    """ステップ4の処理: AIによる推論"""
    print(f"\n[ステップ4] 推論を開始 (モデル: {model_path})")
    device = torch.device("cpu")
    model = PointNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    pcd_for_kdtree = o3d.geometry.PointCloud(); pcd_for_kdtree.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_for_kdtree)
    num_points = len(points)
    pred_labels = np.full(num_points, 1, dtype=int)
    with torch.no_grad():
        for i in tqdm(range(num_points), desc="  推論中"):
            [k, idx, _] = kdtree.search_knn_vector_3d(points[i], num_neighbors)
            if k < num_neighbors: continue
            neighbors = points[idx]
            normalized_neighbors = neighbors - points[i]
            input_tensor = torch.tensor(normalized_neighbors.T, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            pred_labels[i] = predicted.cpu().numpy()[0]
    print(f" -> 推論完了。鉄筋と予測された点数: {np.sum(pred_labels == 0)}")
    return pred_labels

def filter_by_density(points, pred_labels, radius, min_neighbors):
    """ステップ5の処理: 密度ベースのフィルタリング"""
    print(f"\n[ステップ5] 密度フィルタリングを開始 (Radius={radius}, MinNeighbors={min_neighbors})")
    LABEL_REBAR = 0; LABEL_NOT_REBAR = 1
    rebar_indices_original = np.where(pred_labels == LABEL_REBAR)[0]
    if len(rebar_indices_original) == 0: return pred_labels
    rebar_points = points[rebar_indices_original]
    rebar_pcd = o3d.geometry.PointCloud(); rebar_pcd.points = o3d.utility.Vector3dVector(rebar_points)
    rebar_kdtree = o3d.geometry.KDTreeFlann(rebar_pcd)
    denoised_rebar_indices = []
    for i in tqdm(range(len(rebar_points)), desc="  フィルタリング中"):
        [k, _, _] = rebar_kdtree.search_radius_vector_3d(rebar_points[i], radius)
        if k >= min_neighbors:
            denoised_rebar_indices.append(rebar_indices_original[i])
    final_labels = np.full(len(points), LABEL_NOT_REBAR)
    final_labels[denoised_rebar_indices] = LABEL_REBAR
    print(f" -> フィルタリング完了。処理後の鉄筋点数: {len(denoised_rebar_indices)}")
    return final_labels

def visualize_result(points, labels):
    """最終結果の可視化"""
    print("\n最終結果を可視化します...")
    LABEL_REBAR = 0; LABEL_NOT_REBAR = 1
    colors = np.full((len(points), 3), 0.5)
    colors[labels == LABEL_REBAR] = [1.0, 0.0, 0.0]
    colors[labels == LABEL_NOT_REBAR] = [0.0, 0.0, 1.0]
    result_pcd = o3d.geometry.PointCloud(); result_pcd.points = o3d.utility.Vector3dVector(points); result_pcd.colors = o3d.utility.Vector3dVector(colors)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([result_pcd, axis], window_name="Automated Pipeline Result", width=1600, height=1200)



# --- パイプラインの本体 ---
if __name__ == '__main__':
    # 1. ファイル選択ダイアログを開いて入力ファイルを取得
    root = tkinter.Tk()
    root.withdraw() # 小さなルートウィンドウは非表示にする
    input_csv_path = filedialog.askopenfilename(
        title="分析したい生の点群データCSVを選択してください",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not input_csv_path:
        print("ファイルが選択されなかったので、処理を終了します。")
    else:
        print(f"入力ファイル: {input_csv_path}")
        # 使用するモデルは、ここで固定で指定
        model_path = 'pointnet_model_v2.pth'
        
        # 2. 生データを読み込み
        print("生の点群データを読み込んでいます...")
        try:
            raw_points = pd.read_csv(input_csv_path).iloc[:, :3].to_numpy(dtype=np.float32)
        except FileNotFoundError:
            print(f"エラー: 入力ファイル '{input_csv_path}' が見つかりません。")
            exit()

        # 3. 対話的に範囲を選択
        crop_box = select_area_interactively(raw_points)
        
        if crop_box:
            # 4. 前処理と切り出し
            processed_points = preprocess_and_crop(input_csv_path, crop_box)
            
            if len(processed_points) > 0:
                # 5. 推論
                predicted_labels = run_inference(processed_points, model_path)
                
                # 6. 後処理フィルタリング (パラメータはここで調整)
                filter_radius = 30.0
                filter_neighbors = 10
                final_labels = filter_by_density(processed_points, predicted_labels, filter_radius, filter_neighbors)

                # 7. 可視化
                visualize_result(processed_points, final_labels)
        
        print("\nパイプライン処理が完了しました。")