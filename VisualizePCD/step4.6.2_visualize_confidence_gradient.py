import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- ★★★ ユーザー設定項目 ★★★ ---
# ご指定の条件を反映
MODEL_PATH = 'pointnet_model_vC.pth'
PCD_PATH = 'work_area_large.csv'
NUM_NEIGHBORS = 64

# --- モデル定義 (変更なし) ---
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
        
    pcd_for_kdtree = o3d.geometry.PointCloud(); pcd_for_kdtree.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_for_kdtree)
    
    # 各点の「不確実性」を保存するリスト
    uncertainty_scores = np.full(len(points), -1.0, dtype=float)

    print("推論を実行し、各点の「不確実性」を計算中...")
    with torch.no_grad():
        for i in tqdm(range(len(points))):
            [k, idx, _] = kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: continue
            
            neighbors = points[idx]
            normalized_neighbors = neighbors - points[i]
            input_tensor = torch.tensor(normalized_neighbors.T, dtype=torch.float32).unsqueeze(0).to(device)
            
            outputs = model(input_tensor)
            probabilities = torch.exp(outputs)
            
            # 「鉄筋」クラス(インデックス0)の確率を取得
            rebar_prob = probabilities[0, 0].cpu().numpy()
            # 不確実性を計算 (0.5に近いほど1.0に、0.0や1.0に近いほど0.0になる)
            uncertainty = 1.0 - abs(rebar_prob - 0.5) * 2
            uncertainty_scores[i] = uncertainty

    print("推論完了。不確実性に応じて色を付け、可視化します。")
    
    # 確率に基づいて色を決定
    # 'jet' カラーマップを使用: 0.0(青) -> 0.5(緑) -> 1.0(赤)
    cmap = plt.get_cmap('jet')
    # 確率が計算できなかった点(-1)は灰色にする
    colors = np.full((len(points), 3), [0.5, 0.5, 0.5])

    valid_indices = np.where(uncertainty_scores != -1.0)[0]
    # RGBAからRGBだけを取り出す
    colors[valid_indices] = cmap(uncertainty_scores[valid_indices])[:, :3]

    # 可視化
    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(points)
    result_pcd.colors = o3d.utility.Vector3dVector(colors)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])

    o3d.visualization.draw_geometries(
        [result_pcd, axis],
        window_name="Prediction Uncertainty",
        width=1600, height=1200
    )
