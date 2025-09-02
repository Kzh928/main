# ファイル名: step4.1_verify_4class_model.py
# 概要: 学習済みの「4クラス分類」モデルを使い、点群を色分けして分類結果を可視化する。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---
# 4クラス分類用に学習させたモデルのパス
MODEL_PATH = 'pointnet_model_4class_1235.pth'
# 分類したい点群ファイルのパス
PCD_PATH = 'work_area_large.csv'

# その他設定
NUM_NEIGHBORS = 64 # 学習時と同じ32を推奨
NUM_CLASSES = 4    # 3から4に変更

# --- モデル定義 (学習時と全く同じ) ---
class TNet(nn.Module):
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x

class PointNet(nn.Module):
    def __init__(self, num_classes=4): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return F.log_softmax(x, dim=1)

# --- メインの処理 ---
if __name__ == '__main__':
    # 1. モデルとデータの準備
    device = torch.device("cpu")
    model = PointNet(num_classes=NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。")
        exit()
    model.eval()
    print(f"'{MODEL_PATH}' から学習済みモデルを読み込みました。")

    try:
        points = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values
    except FileNotFoundError:
        print(f"エラー: 点群ファイル '{PCD_PATH}' が見つかりません。")
        exit()
        
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # 2. 推論と色分け
    colors = np.full((len(points), 3), [0.5, 0.5, 0.5]) # デフォルトは灰色
    
    # カラーマップを4クラス用に拡張
    # モデルの出力 (0,1,2,3) に対応する色を定義
    color_map = {
        0: [1.0, 0.0, 0.0], # 鉄筋 -> 赤
        1: [0.0, 0.0, 1.0], # 壁面 -> 青
        2: [0.0, 1.0, 0.0], # 装置 -> 緑
        3: [0.0, 1.0, 1.0]  # 床   -> シアン
    }

    print("推論を実行し、点群を色分けしています...")
    with torch.no_grad():
        for i in tqdm(range(len(points)), desc="分類中"):
            [k, idx, _] = kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS:
                continue
            
            neighbors = points[idx]
            normalized_neighbors = neighbors - points[i]
            input_tensor = torch.from_numpy(normalized_neighbors.T).float().unsqueeze(0).to(device)
            
            outputs = model(input_tensor)
            _, predicted_label = torch.max(outputs.data, 1)
            
            # 予測されたラベルに対応する色を割り当て
            colors[i] = color_map.get(predicted_label.item(), [1.0, 1.0, 1.0]) # 不明なラベルは白

    # 3. 結果の可視化
    print("推論完了。結果を可視化します。")
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="4クラス分類 結果")

    print("\n===== 全ての処理が完了しました =====")