# ファイル名: step3.2_train_3class_with_logging.py
# 概要: 3クラスモデルの学習を行い、エポックごとの損失と精度を
#      CSVファイルに自動で記録する。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from tqdm import tqdm
import os

# --- ★★★ ユーザー設定項目 ★★★ ---
PCD_PATH = 'work_area_large.csv'
LABELS_PATH = 'annotated_labels_3class.npy'
NUM_NEIGHBORS = 64

NUM_CLASSES = 3
NUM_EPOCHS = 50 # 50エポックに設定
BATCH_SIZE = 32
LEARNING_RATE = 0.001

OUTPUT_MODEL_PATH = 'pointnet_model_3class.pth'
# ▼▼▼【新機能】ログの保存先ファイル名 ▼▼▼
LOG_FILE_PATH = 'training_log.csv'

# --- モデル定義 (変更なし) ---
class TNet(nn.Module):
    # ... (内容はstep3.1と同じ)
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x

class PointNet(nn.Module):
    # ... (内容はstep3.1と同じ)
    def __init__(self, num_classes=2): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return F.log_softmax(x, dim=1)

# --- データセットの定義 (変更なし) ---
class PointCloudDataset(Dataset):
    # ... (内容はstep3.1と同じ)
    def __init__(self, points, labels, kdtree): self.points = points; self.labels = labels; self.kdtree = kdtree; self.labeled_indices = np.where(self.labels > 0)[0]
    def __len__(self): return len(self.labeled_indices)
    def __getitem__(self, idx): point_idx = self.labeled_indices[idx]; point = self.points[point_idx]; k, indices, _ = self.kdtree.search_knn_vector_3d(point, NUM_NEIGHBORS); neighbors = self.points[indices]; normalized_neighbors = neighbors - point; data = torch.from_numpy(normalized_neighbors.T).float(); label = torch.tensor(self.labels[point_idx] - 1, dtype=torch.long); return data, label

# --- メインの学習処理 ---
if __name__ == '__main__':
    # 1. データの準備
    # ... (内容はstep3.1と同じ)
    print("データを読み込み中..."); points = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values
    try: labels = np.load(LABELS_PATH)
    except FileNotFoundError: print(f"エラー: ラベルファイル '{LABELS_PATH}' が見つかりません。"); exit()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)); kdtree = o3d.geometry.KDTreeFlann(pcd); dataset = PointCloudDataset(points, labels, kdtree); dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    if len(dataset) == 0: print("エラー: 学習対象となるラベル付きの点がデータ内に見つかりませんでした。"); exit()
    print(f"学習データ準備完了。学習対象の点数: {len(dataset)}")
    
    # 2. モデル、損失関数、最適化手法の定義
    # ... (内容はstep3.1と同じ)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"使用デバイス: {device}")
    model = PointNet(num_classes=NUM_CLASSES).to(device); criterion = nn.NLLLoss(); optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ▼▼▼【新機能】ログを保存するためのリストを準備 ▼▼▼
    history = []

    # 3. 学習ループ
    print("学習を開始します...")
    for epoch in range(NUM_EPOCHS):
        model.train(); running_loss = 0.0; correct_preds = 0; total_preds = 0
        pbar = tqdm(dataloader, desc=f"エポック {epoch+1}/{NUM_EPOCHS}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(); outputs = model(data); loss = criterion(outputs, target); loss.backward(); optimizer.step()
            running_loss += loss.item(); _, predicted = torch.max(outputs.data, 1); total_preds += target.size(0); correct_preds += (predicted == target).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = (correct_preds / total_preds) * 100
        print(f"エポック {epoch+1} 完了 - 損失: {epoch_loss:.4f}, 精度: {epoch_acc:.2f}%")
        
        # ▼▼▼【新機能】このエポックの結果をリストに追加 ▼▼▼
        history.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'accuracy': epoch_acc
        })

    # 4. 学習済みモデルの保存
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"\n学習が完了しました。モデルを '{OUTPUT_MODEL_PATH}' に保存しました。")

    # ▼▼▼【新機能】学習履歴をCSVファイルに保存 ▼▼▼
    history_df = pd.DataFrame(history)
    history_df.to_csv(LOG_FILE_PATH, index=False)
    print(f"学習ログを '{LOG_FILE_PATH}' に保存しました。")
