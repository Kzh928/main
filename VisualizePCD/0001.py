# ファイル名: step3.8_train_with_weighted_sampler.py
# 概要: WeightedRandomSampler（復元抽出）で3クラスモデルの学習と評価を行う。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
# ▼▼▼【変更】WeightedRandomSampler をインポート ▼▼▼
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import open3d as o3d
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# --- ★★★ ユーザー設定項目 ★★★ ---
PCD_PATH = 'work_area_large.csv'
LABELS_PATH = 'annotated_labels_3class.npy'
NUM_NEIGHBORS = 64
NUM_CLASSES = 3
NUM_EPOCHS = 15 # ひとまず200エポックで様子見
BATCH_SIZE = 32 # 通常のバッチサイズを指定
LEARNING_RATE = 0.001
OUTPUT_MODEL_PATH = 'pointnet_model_3class_weighted_sampler.pth'
LOG_FILE_PATH = 'training_log_weighted_sampler.csv'

# --- モデル定義 (変更なし) ---
class TNet(nn.Module):
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x
class PointNet(nn.Module):
    def __init__(self, num_classes=3): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return F.log_softmax(x, dim=1)

# --- データセットの定義 (変更なし) ---
class PointCloudDataset(Dataset):
    def __init__(self, points, labels, kdtree): self.points = points; self.labels = labels; self.kdtree = kdtree; self.labeled_indices = np.where(self.labels > 0)[0]; self.dataset_labels = self.labels[self.labeled_indices] - 1
    def __len__(self): return len(self.labeled_indices)
    def __getitem__(self, idx):
        point_idx = self.labeled_indices[idx]; point = self.points[point_idx]
        k, indices, _ = self.kdtree.search_knn_vector_3d(point, NUM_NEIGHBORS)
        if k < NUM_NEIGHBORS:
            indices_to_add = np.random.choice(indices, NUM_NEIGHBORS - k, replace=True)
            indices = np.append(indices, indices_to_add)
        neighbors = self.points[indices]
        normalized_neighbors = neighbors - point; data = torch.from_numpy(normalized_neighbors.T).float()
        label = torch.tensor(self.labels[point_idx] - 1, dtype=torch.long); return data, label

# --- メイン処理 ---
if __name__ == '__main__':
    # 1. データ準備フェーズ
    print("データを読み込み中..."); points = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values
    try: labels = np.load(LABELS_PATH)
    except FileNotFoundError: print(f"エラー: ラベルファイル '{LABELS_PATH}' が見つかりません。"); exit()
    
    print("\n--- データセットの構成 ---")
    num_rebar = np.sum(labels == 1); num_not_rebar = np.sum(labels == 2); num_equipment = np.sum(labels == 3)
    print(f"  - 鉄筋 (クラス0): {num_rebar} 点\n  - 非鉄筋 (クラス1): {num_not_rebar} 点\n  - 装置 (クラス2): {num_equipment} 点")
    print("--------------------------\n")
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)); kdtree = o3d.geometry.KDTreeFlann(pcd)
    dataset = PointCloudDataset(points, labels, kdtree)
    if len(dataset) == 0: print("エラー: 学習対象となるラベル付きの点がデータ内に見つかりませんでした。"); exit()

    # ▼▼▼【変更】WeightedRandomSampler の準備 ▼▼▼
    # クラスごとのサンプル数を計算
    class_counts = np.array([num_rebar, num_not_rebar, num_equipment])
    # 各クラスの重みを計算 (サンプル数の逆数)
    class_weights = 1. / class_counts
    # データセット内の各サンプルに対応する重みを割り当て
    sample_weights = np.array([class_weights[label] for label in dataset.dataset_labels])
    
    # サンプラーを作成
    # num_samples: 1エポックでサンプリングする総数。データセット長と同じにするのが一般的。
    # replacement=True: 復元抽出を許可
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

    # ▼▼▼【変更】DataLoaderでsamplerとbatch_sizeを指定 (shuffleはFalse) ▼▼▼
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)
    print(f"WeightedRandomSamplerを使用します。1エポックあたりのバッチ数: {len(dataloader)}")


    # 2. 学習フェーズ (変更なし)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"使用デバイス: {device}")
    model = PointNet(num_classes=NUM_CLASSES).to(device); criterion = nn.NLLLoss(); optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    print("\n学習を開始します...")
    for epoch in range(NUM_EPOCHS):
        model.train(); running_loss = 0.0; correct_preds = 0; total_preds = 0
        pbar = tqdm(dataloader, desc=f"エポック {epoch+1}/{NUM_EPOCHS}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device); optimizer.zero_grad(); outputs = model(data); loss = criterion(outputs, target); loss.backward(); optimizer.step()
            running_loss += loss.item(); _, predicted = torch.max(outputs.data, 1); total_preds += target.size(0); correct_preds += (predicted == target).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        epoch_loss = running_loss / len(dataloader); epoch_acc = (correct_preds / total_preds) * 100
        print(f"エポック {epoch+1} 完了 - 損失: {epoch_loss:.4f}, 精度 (バッチ内): {epoch_acc:.2f}%")
        history.append({'epoch': epoch + 1, 'loss': epoch_loss, 'accuracy': epoch_acc})
    
    # 3. 評価フェーズ (変更なし)
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    pd.DataFrame(history).to_csv(LOG_FILE_PATH, index=False)
    print(f"\n学習完了。モデルを'{OUTPUT_MODEL_PATH}'、ログを'{LOG_FILE_PATH}'に保存しました。")
    print("\n--- モデル性能評価 (全教師データに対する評価) ---")
    model.eval(); labels_pred = []; labels_true = []
    eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for data, target in tqdm(eval_dataloader, desc="評価中"):
            data, target = data.to(device), target.to(device); outputs = model(data); _, predicted = torch.max(outputs.data, 1)
            labels_pred.extend(predicted.cpu().numpy()); labels_true.extend(target.cpu().numpy())
    target_names = ['rebar (鉄筋/0)', 'not_rebar (非鉄筋/1)', 'equipment (装置/2)']
    print("\n" + "="*50); print("           最終モデル 性能評価レポート"); print("="*50)
    report = classification_report(labels_true, labels_pred, target_names=target_names, zero_division=0)
    print(report)
    print("\n--- 混同行列 (Confusion Matrix) ---")
    cm = confusion_matrix(labels_true, labels_pred)
    print("行(↓): 正解ラベル / 列(→): 予測ラベル")
    print("           ", " ".join([f"{name.split(' ')[0]:>10}" for name in target_names]))
    print("          " + "-"*35)
    for i, row in enumerate(cm):
        if i < len(target_names):
            row_str = f"{target_names[i].split(' ')[0]:<10}|";
            for val in cm[i]: row_str += f" {val:10d}"
            print(row_str)
    print("--------------------------------------\n")
    print("===== 全ての処理が完了しました =====")
