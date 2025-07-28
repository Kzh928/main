# ファイル名: step4.1_evaluate_3class_model.py
# 概要: 学習済みの3クラスモデルの性能を詳細に評価する。全体の正解率に加え、
#      どのクラスを何と間違えたかが一目でわかる「混同行列」を出力する。

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# --- ★★★ ユーザー設定項目 ★★★ ---
MODEL_PATH = 'pointnet_model_3class_weighted.pth'
PCD_PATH = 'work_area_large.csv'
LABELS_PATH = 'annotated_labels_3class.npy' # 答え合わせに使うラベルファイル
NUM_NEIGHBORS = 64
NUM_CLASSES = 3

# --- モデル定義 (学習時と全く同じ) ---
class TNet(nn.Module):
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x
class PointNet(nn.Module):
    def __init__(self, num_classes=3): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return F.log_softmax(x, dim=1)

# --- メイン処理 ---
if __name__ == '__main__':
    # 1. モデルとデータの準備
    device = torch.device("cpu"); model = PointNet(num_classes=NUM_CLASSES).to(device);
    try: model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError: print(f"エラー: モデル '{MODEL_PATH}' が見つかりません。"); exit()
    model.eval(); print(f"'{MODEL_PATH}' から学習済みモデルを読み込みました。")
    
    points = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values
    try: labels_true_all = np.load(LABELS_PATH)
    except FileNotFoundError: print(f"エラー: ラベル '{LABELS_PATH}' が見つかりません。"); exit()

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # 評価対象となる、ラベルが振られた点のインデックスを取得
    eval_indices = np.where(labels_true_all > 0)[0]
    if len(eval_indices) == 0: print("評価対象のラベル付き点が見つかりません。"); exit()
    
    # 2. 推論の実行
    labels_pred = []
    labels_true = []
    
    print(f"評価を開始します (対象: {len(eval_indices)}点)...")
    with torch.no_grad():
        for i in tqdm(eval_indices, desc="評価中"):
            [k, idx, _] = kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS)
            if k < NUM_NEIGHBORS: continue
            
            neighbors = points[idx]; normalized_neighbors = neighbors - points[i]
            input_tensor = torch.from_numpy(normalized_neighbors.T).float().unsqueeze(0).to(device)
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            
            labels_pred.append(predicted.item())
            # 正解ラベルも0-indexedに変換 (1,2,3 -> 0,1,2)
            labels_true.append(labels_true_all[i] - 1)

    # 3. 結果の集計と表示
    target_names = ['rebar (鉄筋)', 'not_rebar (非鉄筋)', 'equipment (装置)']
    
    print("\n\n" + "="*50)
    print("           モデル性能評価レポート")
    print("="*50)
    
    # 詳細レポート (各クラスの適合率、再現率、F1スコア)
    report = classification_report(labels_true, labels_pred, target_names=target_names)
    print(report)
    
    # 混同行列 (Confusion Matrix)
    print("\n--- 混同行列 (Confusion Matrix) ---")
    cm = confusion_matrix(labels_true, labels_pred)
    print("行(↓): 正解ラベル / 列(→): 予測ラベル")
    print("          ", " ".join([f"{name.split(' ')[0]:>10}" for name in target_names]))
    print("          " + "-"*35)
    for i, row in enumerate(cm):
        row_str = f"{target_names[i].split(' ')[0]:<10}|"
        for val in row:
            row_str += f" {val:10d}"
        print(row_str)
    print("--------------------------------------")
    print("\n===== 評価完了 =====")
