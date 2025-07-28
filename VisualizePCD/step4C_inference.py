# ファイル名: step4C_inference.py (vCモデル使用版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---
MODEL_PATH = 'pointnet_model_vC.pth'
PCD_PATH = 'work_area_large.csv' # または 'work_area.csv' など、試したいファイル
NUM_NEIGHBORS = 64

# --- ラベル定義 ---
LABEL_REBAR = 0
LABEL_NOT_REBAR = 1

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

# --- メインの処理 ---
if __name__ == '__main__':
    device = torch.device("cpu")
    model = PointNet(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"'{MODEL_PATH}' から学習済みモデルを読み込みました。")
    except FileNotFoundError:
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。step3を先に実行してください。")
        exit()
    model.eval()

    try:
        points = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values
        print(f"'{PCD_PATH}' から点群データを読み込みました。")
    except FileNotFoundError:
        print(f"エラー: 点群ファイル '{PCD_PATH}' が見つかりません。")
        exit()
        
    pcd_for_kdtree = o3d.geometry.PointCloud()
    pcd_for_kdtree.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_for_kdtree)
    print("KDTreeを構築しました。")

    num_points = len(points)
    pred_labels = np.full(num_points, LABEL_NOT_REBAR, dtype=int) 

    print("推論を実行中... (全点をチェックするため時間がかかります)")
    with torch.no_grad():
        for i in tqdm(range(num_points)):
            [k, idx, _] = kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: continue
            
            neighbors = points[idx]
            normalized_neighbors = neighbors - points[i]
            input_tensor = torch.tensor(normalized_neighbors.T, dtype=torch.float32).unsqueeze(0).to(device)
            
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            pred_labels[i] = predicted.cpu().numpy()[0]

    print("推論が完了しました。結果を可視化します。")
    colors = np.full((len(points), 3), 0.5) 
    colors[pred_labels == LABEL_REBAR] = [1.0, 0.0, 0.0]
    colors[pred_labels == LABEL_NOT_REBAR] = [0.0, 0.0, 1.0]

    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(points)
    result_pcd.colors = o3d.utility.Vector3dVector(colors)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Inference Result (Red: Rebar, Blue: Not-Rebar)", width=1600, height=1200)
    vis.add_geometry(result_pcd)
    #vis.add_geometry(axis)
    vis.run()
    vis.destroy_window()

    # 新しい予測結果を保存する
    np.save('predicted_labels_vC.npy', pred_labels)
    print("新しい予測結果を 'predicted_labels_vC.npy' に保存しました。")