# ファイル名: step3.7_train_with_balanced_batch_4class.py
# 概要: クラス比率を固定したバッチで4クラスモデルの学習と評価を行う。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import open3d as o3d
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import random

# --- ★★★ ユーザー設定項目 ★★★ ---
PCD_PATH = 'work_area_large.csv'
# ▼▼▼【注意】4クラス（1:鉄筋, 2:壁面, 3:装置, 4:床）に対応したラベルファイルが必要です ▼▼▼
LABELS_PATH = 'annotated_labels_4class.npy'

NUM_NEIGHBORS = 64
# ▼▼▼【変更】クラス数を4に変更 ▼▼▼
NUM_CLASSES = 4
NUM_EPOCHS = 500

# ▼▼▼【変更】バッチ内の各クラスのサンプル数を4クラス用に設定 ▼▼▼
# バッチ内の各クラスのサンプル数 (合計32になるように調整)
# この比率はデータセットの構成や重視したいクラスに応じて調整してください
BATCH_RATIO = {
    0: 6,  # 鉄筋 (ラベル1に対応)
    1: 10, # コンクリート壁面 (ラベル2に対応)
    2: 12,  # 装置 (ラベル3に対応)
    3: 5, # 床 (ラベル4に対応)
}
BATCH_SIZE = sum(BATCH_RATIO.values())

LEARNING_RATE = 0.001
OUTPUT_MODEL_PATH = 'pointnet_model_4class.pth' # ◀◀◀【変更推奨】モデル名を変更
LOG_FILE_PATH = 'training_log_4class.csv' # ◀◀◀【変更推奨】ログファイル名を変更


# --- モデル定義 (変更なし) ---
# PointNetの出力層のユニット数は、インスタンス化時に NUM_CLASSES を渡すことで
# 自動的に4クラス対応になるため、クラス定義自体の変更は不要です。
class TNet(nn.Module):
    def __init__(self, k=3): super(TNet, self).__init__(); self.k=k; self.conv1 = nn.Conv1d(k,64,1); self.conv2 = nn.Conv1d(64,128,1); self.conv3 = nn.Conv1d(128,1024,1); self.fc1 = nn.Linear(1024,512); self.fc2 = nn.Linear(512,256); self.fc3 = nn.Linear(256,k*k); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.bn4 = nn.BatchNorm1d(512); self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x): batchsize = x.size(0); x = F.relu(self.bn1(self.conv1(x))); x = F.relu(self.bn2(self.conv2(x))); x = F.relu(self.bn3(self.conv3(x))); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn4(self.fc1(x))); x = F.relu(self.bn5(self.fc2(x))); x = self.fc3(x); iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1,self.k*self.k).repeat(batchsize,1); x = x + iden; x = x.view(-1, self.k, self.k); return x
class PointNet(nn.Module):
    def __init__(self, num_classes=4): super(PointNet, self).__init__(); self.input_transform = TNet(k=3); self.feature_transform = TNet(k=64); self.conv1 = nn.Conv1d(3, 64, 1); self.conv2 = nn.Conv1d(64, 128, 1); self.conv3 = nn.Conv1d(128, 1024, 1); self.bn1 = nn.BatchNorm1d(64); self.bn2 = nn.BatchNorm1d(128); self.bn3 = nn.BatchNorm1d(1024); self.fc1 = nn.Linear(1024, 512); self.fc2 = nn.Linear(512, 256); self.fc3 = nn.Linear(256, num_classes); self.dropout = nn.Dropout(p=0.3); self.bn_fc1 = nn.BatchNorm1d(512); self.bn_fc2 = nn.BatchNorm1d(256)
    def forward(self, x): trans_input = self.input_transform(x); x = torch.bmm(x.transpose(2,1), trans_input).transpose(2,1); x = F.relu(self.bn1(self.conv1(x))); trans_feat = self.feature_transform(x); x = torch.bmm(x.transpose(2,1), trans_feat).transpose(2,1); x = F.relu(self.bn2(self.conv2(x))); x = self.bn3(self.conv3(x)); x = torch.max(x, 2, keepdim=True)[0]; x = x.view(-1, 1024); x = F.relu(self.bn_fc1(self.fc1(x))); x = F.relu(self.bn_fc2(self.dropout(self.fc2(x)))); x = self.fc3(x); return F.log_softmax(x, dim=1)


# --- データセットの定義 (変更なし) ---
# ラベル値が1,2,3,4...と増えても、-1することで0,1,2,3...に変換されるため、
# このクラス定義自体の変更は不要です。
class PointCloudDataset(Dataset):
    def __init__(self, points, labels, kdtree):
        self.points = points
        self.labels = labels
        self.kdtree = kdtree
        # ラベルが1以上（教師データ）のインデックスのみを対象とする
        self.labeled_indices = np.where(self.labels > 0)[0]
        # DataLoaderで使うために、教師データのラベル（0, 1, 2, 3に変換済み）も保持
        self.dataset_labels = self.labels[self.labeled_indices] - 1

    def __len__(self):
        return len(self.labeled_indices)

    def __getitem__(self, idx):
        # idxはself.labeled_indicesのインデックス
        point_idx = self.labeled_indices[idx]
        point = self.points[point_idx]

        # 近傍点検索
        k, indices, _ = self.kdtree.search_knn_vector_3d(point, NUM_NEIGHBORS)

        # 近傍点が足りない場合の処理
        if k < NUM_NEIGHBORS:
            neighbors = self.points[indices]
            indices_to_add = np.random.choice(indices, NUM_NEIGHBORS - k, replace=True)
            neighbors_to_add = self.points[indices_to_add]
            neighbors = np.vstack((neighbors, neighbors_to_add))
        else:
            neighbors = self.points[indices]

        # 正規化
        normalized_neighbors = neighbors - point
        data = torch.from_numpy(normalized_neighbors.T).float()

        # ラベル (1, 2, 3, 4) を (0, 1, 2, 3) に変換
        label = torch.tensor(self.labels[point_idx] - 1, dtype=torch.long)
        return data, label

# --- バランスドバッチサンプラーの定義 (変更なし) ---
# BATCH_RATIOのキーが増えても動的に対応するため、変更不要です。
class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset_labels, batch_ratio):
        self.dataset_labels = dataset_labels
        self.batch_ratio = batch_ratio
        self.batch_size = sum(batch_ratio.values())

        # クラスごとにインデックスを分類
        self.indices_by_class = {}
        for class_id in batch_ratio.keys():
            self.indices_by_class[class_id] = np.where(self.dataset_labels == class_id)[0]

        # 全てのクラスに十分なデータがあるか確認
        for class_id, num_required in batch_ratio.items():
            if len(self.indices_by_class[class_id]) < num_required:
                raise ValueError(f"クラス {class_id} のサンプル数が不足しています。必要数: {num_required}, 実際: {len(self.indices_by_class[class_id])}")

        # バッチの総数を決定
        self.num_batches = min(
            len(self.indices_by_class[class_id]) // num_required
            for class_id, num_required in batch_ratio.items() if num_required > 0
        )

    def __iter__(self):
        # エポックごとに各クラスのインデックスをシャッフル
        shuffled_indices = {}
        for class_id, indices in self.indices_by_class.items():
            shuffled_indices[class_id] = np.random.permutation(indices)

        # バッチを生成
        for i in range(self.num_batches):
            batch = []
            for class_id, num_samples in self.batch_ratio.items():
                start_idx = i * num_samples
                end_idx = (i + 1) * num_samples
                batch.extend(shuffled_indices[class_id][start_idx:end_idx])

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# --- メイン処理 ---
if __name__ == '__main__':
    # =================================================================
    # 1. データ準備フェーズ
    # =================================================================
    print("データを読み込み中..."); points = pd.read_csv(PCD_PATH, header=None).iloc[:, :3].values
    try: labels = np.load(LABELS_PATH)
    except FileNotFoundError: print(f"エラー: ラベルファイル '{LABELS_PATH}' が見つかりません。"); exit()

    # ▼▼▼【変更】データセットの構成表示を4クラス対応に ▼▼▼
    print("\n--- データセットの構成 ---")
    labeled_indices = np.where(labels > 0)[0]
    num_total_labeled = len(labeled_indices)
    num_rebar = np.sum(labels == 1)
    num_wall = np.sum(labels == 2)     # 「非鉄筋」から「壁面」に変更
    num_equipment = np.sum(labels == 3)
    num_floor = np.sum(labels == 4)      # 「床」クラスを追加
    print(f"ラベル付き総サンプル数: {num_total_labeled}")
    # クラスIDを0, 1, 2, 3として表示
    print(f" - 鉄筋 (クラス0):         {num_rebar} 点 ({num_rebar/num_total_labeled:.2%})")
    print(f" - コンクリート壁面 (クラス1): {num_wall} 点 ({num_wall/num_total_labeled:.2%})")
    print(f" - 装置 (クラス2):         {num_equipment} 点 ({num_equipment/num_total_labeled:.2%})")
    print(f" - 床 (クラス3):           {num_floor} 点 ({num_floor/num_total_labeled:.2%})")
    print("--------------------------\n")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)); kdtree = o3d.geometry.KDTreeFlann(pcd)
    dataset = PointCloudDataset(points, labels, kdtree)
    if len(dataset) == 0: print("エラー: 学習対象となるラベル付きの点がデータ内に見つかりませんでした。"); exit()

    try:
        balanced_sampler = BalancedBatchSampler(dataset.dataset_labels, BATCH_RATIO)
    except ValueError as e:
        print(f"エラー: バランスドバッチサンプラーの初期化に失敗しました。\n{e}")
        print("各クラスのサンプル数がバッチ比率の要求を満たしているか確認してください。")
        exit()

    dataloader = DataLoader(dataset, batch_sampler=balanced_sampler)

    # ▼▼▼【変更】バッチ比率の表示を4クラス対応に ▼▼▼
    print(f"バッチ比率 -> 鉄筋:{BATCH_RATIO[0]}, 壁面:{BATCH_RATIO[1]}, 装置:{BATCH_RATIO[2]}, 床:{BATCH_RATIO[3]}")
    print(f"1エポックあたりのバッチ数: {len(dataloader)}")


    # =================================================================
    # 2. 学習フェーズ
    # =================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"使用デバイス: {device}")
    model = PointNet(num_classes=NUM_CLASSES).to(device); criterion = nn.NLLLoss(); optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    print("\n学習を開始します...")
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
        print(f"エポック {epoch+1} 完了 - 損失: {epoch_loss:.4f}, 精度 (バッチ内): {epoch_acc:.2f}%")
        history.append({'epoch': epoch + 1, 'loss': epoch_loss, 'accuracy': epoch_acc})

    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    pd.DataFrame(history).to_csv(LOG_FILE_PATH, index=False)
    print(f"\n学習完了。モデルを'{OUTPUT_MODEL_PATH}'、ログを'{LOG_FILE_PATH}'に保存しました。")

    # =================================================================
    # 3. 評価フェーズ
    # =================================================================
    print("\n--- モデル性能評価 (全教師データに対する評価) ---")
    model.eval()

    labels_pred = []; labels_true = []
    eval_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for data, target in tqdm(eval_dataloader, desc="評価中"):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            labels_pred.extend(predicted.cpu().numpy())
            labels_true.extend(target.cpu().numpy())

    # ▼▼▼【変更】クラス名を4クラス対応に ▼▼▼
    target_names = ['rebar (鉄筋/0)', 'wall (壁面/1)', 'equipment (装置/2)', 'floor (床/3)']

    print("\n" + "="*60); print("                  最終モデル 性能評価レポート"); print("="*60)
    report = classification_report(labels_true, labels_pred, target_names=target_names, zero_division=0)
    print(report)

    print("\n--- 混同行列 (Confusion Matrix) ---")
    cm = confusion_matrix(labels_true, labels_pred)
    print("行(↓): 正解ラベル / 列(→): 予測ラベル")
    # 表示が長くなるため、ヘッダーの形式を少し調整
    header = "             " + "".join([f"{name.split(' ')[0]:>12}" for name in target_names])
    print(header)
    print(" " + "-" * (len(header)-1))
    for i, row in enumerate(cm):
        if i < len(target_names):
            row_str = f"{target_names[i].split(' ')[0]:<12}|"
            for val in cm[i]:
                row_str += f" {val:11d}"
            print(row_str)
    print(" " + "-" * (len(header)-1) + "\n")
    print("===== 全ての処理が完了しました =====")