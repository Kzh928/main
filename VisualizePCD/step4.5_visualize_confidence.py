# ファイル名: step4.5_visualize_confidence.py (グラフ範囲指定版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import japanize_matplotlib

# --- ★★★ ユーザー設定項目 ★★★ ---
MODEL_PATH = 'pointnet_model.pth'
PCD_PATH = 'work_area.csv'
LABELS_PATH = 'annotated_labels.npy'
NUM_NEIGHBORS = 64

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
        return x

# --- メインの処理 ---
if __name__ == '__main__':
    # ... (モデル準備、データ準備、推論処理は変更なしです) ...
    device = torch.device("cpu")
    model = PointNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"'{MODEL_PATH}' から学習済みモデルを読み込みました。")
    points = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values
    try:
        true_labels = np.load(LABELS_PATH)
        print(f"'{LABELS_PATH}' から正解ラベルを読み込みました。")
    except FileNotFoundError:
        true_labels = None
    pcd_for_kdtree = o3d.geometry.PointCloud(); pcd_for_kdtree.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_for_kdtree)
    all_logits = []; processed_indices = []
    print("推論を実行し、各点のLogitスコアを取得中...")
    with torch.no_grad():
        for i in tqdm(range(len(points))):
            [k, idx, _] = kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: continue
            neighbors = points[idx]
            normalized_neighbors = neighbors - points[i]
            input_tensor = torch.tensor(normalized_neighbors.T, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(input_tensor)
            all_logits.append(logits.cpu().numpy().flatten())
            processed_indices.append(i)
    logits_np = np.array(all_logits)
    processed_indices = np.array(processed_indices)
    
    if true_labels is not None and len(processed_indices) > 0:
        target_true_labels = true_labels[processed_indices]
        rebar_indices = np.where(target_true_labels == 1)[0]
        not_rebar_indices = np.where(target_true_labels == 2)[0]
        rebar_logits = logits_np[rebar_indices]
        not_rebar_logits = logits_np[not_rebar_indices]
        print("\n--- スコアの数値分析結果 ---")
        if len(rebar_logits) > 0:
            print(f"\n[正解が「鉄筋」の点] ({len(rebar_logits)}点)")
            print(f"  鉄筋らしさスコア(Y):   Min={rebar_logits[:, 0].min():.2f}, Max={rebar_logits[:, 0].max():.2f}, Avg={rebar_logits[:, 0].mean():.2f}")
            print(f"  非鉄筋らしさスコア(X): Min={rebar_logits[:, 1].min():.2f}, Max={rebar_logits[:, 1].max():.2f}, Avg={rebar_logits[:, 1].mean():.2f}")
        if len(not_rebar_logits) > 0:
            print(f"\n[正解が「非鉄筋」の点] ({len(not_rebar_logits)}点)")
            print(f"  鉄筋らしさスコア(Y):   Min={not_rebar_logits[:, 0].min():.2f}, Max={not_rebar_logits[:, 0].max():.2f}, Avg={not_rebar_logits[:, 0].mean():.2f}")
            print(f"  非鉄筋らしさスコア(X): Min={not_rebar_logits[:, 1].min():.2f}, Max={not_rebar_logits[:, 1].max():.2f}, Avg={not_rebar_logits[:, 1].mean():.2f}")
        print("--------------------------\n")
        
        print("グラフを作成・表示します... (合計3枚)")
        
        # グラフ1: 全体図（指定範囲に拡大）
        plt.figure(figsize=(10, 10))
        plt.scatter(rebar_logits[:, 1], rebar_logits[:, 0], c='red', alpha=0.5, label="正解が「鉄筋」の点")
        plt.scatter(not_rebar_logits[:, 1], not_rebar_logits[:, 0], c='blue', alpha=0.5, label="正解が「非鉄筋」の点")
        plt.title("モデルの予測スコア プロット (全体図・拡大)", fontsize=16)
        plt.xlabel("「非鉄筋」らしさのスコア (Logit)", fontsize=12); plt.ylabel("「鉄筋」らしさのスコア (Logit)", fontsize=12)
        plt.grid(True)
        # ★★★ ここからが修正箇所 ★★★
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.plot([-15, 15], [-15, 15], 'g--', label="判断の境界線 (y = x)")
        # ★★★ ここまでが修正箇所 ★★★
        plt.axhline(0, color='black', linewidth=0.5); plt.axvline(0, color='black', linewidth=0.5)
        plt.legend(); plt.gca().set_aspect('equal', adjustable='box')
        
        # グラフ2: 「鉄筋」の点のみ拡大
        if len(rebar_logits) > 0:
            plt.figure(figsize=(8, 8))
            plt.scatter(rebar_logits[:, 1], rebar_logits[:, 0], c='red', alpha=0.5)
            plt.title("正解が「鉄筋」の点のスコア分布 (拡大図)", fontsize=16)
            plt.xlabel("「非鉄筋」らしさのスコア (Logit)", fontsize=12); plt.ylabel("「鉄筋」らしさのスコア (Logit)", fontsize=12)
            plt.grid(True); plt.axhline(0, color='black', linewidth=0.5); plt.axvline(0, color='black', linewidth=0.5)
            plt.gca().set_aspect('equal', adjustable='box')

        # グラフ3: 「非鉄筋」の点のみ拡大
        if len(not_rebar_logits) > 0:
            plt.figure(figsize=(8, 8))
            plt.scatter(not_rebar_logits[:, 1], not_rebar_logits[:, 0], c='blue', alpha=0.5)
            plt.title("正解が「非鉄筋」の点のスコア分布 (拡大図)", fontsize=16)
            plt.xlabel("「非鉄筋」らしさのスコア (Logit)", fontsize=12); plt.ylabel("「鉄筋」らしさのスコア (Logit)", fontsize=12)
            plt.grid(True); plt.axhline(0, color='black', linewidth=0.5); plt.axvline(0, color='black', linewidth=0.5)
            plt.gca().set_aspect('equal', adjustable='box')

        plt.show()
    else:
        print("プロットするためのアノテーション済みデータが見つかりませんでした。")