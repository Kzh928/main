# ファイル名: step3_train_pointnet.py

# 'final_training_datasetC.pkl' を読み込んでいます...
# 訓練データ数: 109698, 検証データ数: 27425
# 使用デバイス: cpu
# 学習を開始します...
# エポック 1/25 | 訓練損失: 0.0385 | 訓練精度: 98.97% | 検証精度: 98.87%
# エポック 2/25 | 訓練損失: 0.0271 | 訓練精度: 98.99% | 検証精度: 98.98%
# エポック 3/25 | 訓練損失: 0.0224 | 訓練精度: 99.04% | 検証精度: 99.15%
# エポック 4/25 | 訓練損失: 0.0196 | 訓練精度: 99.10% | 検証精度: 99.29%
# エポック 5/25 | 訓練損失: 0.0182 | 訓練精度: 99.15% | 検証精度: 99.27%
# エポック 6/25 | 訓練損失: 0.0170 | 訓練精度: 99.21% | 検証精度: 99.34%
# エポック 7/25 | 訓練損失: 0.0162 | 訓練精度: 99.27% | 検証精度: 99.40%
# エポック 8/25 | 訓練損失: 0.0152 | 訓練精度: 99.31% | 検証精度: 99.41%
# エポック 9/25 | 訓練損失: 0.0139 | 訓練精度: 99.39% | 検証精度: 99.34%
# エポック 10/25 | 訓練損失: 0.0131 | 訓練精度: 99.45% | 検証精度: 99.49%
# エポック 11/25 | 訓練損失: 0.0130 | 訓練精度: 99.44% | 検証精度: 99.32%
# エポック 12/25 | 訓練損失: 0.0122 | 訓練精度: 99.48% | 検証精度: 99.50%
# エポック 13/25 | 訓練損失: 0.0116 | 訓練精度: 99.52% | 検証精度: 99.57%
# エポック 14/25 | 訓練損失: 0.0110 | 訓練精度: 99.54% | 検証精度: 99.60%
# エポック 15/25 | 訓練損失: 0.0104 | 訓練精度: 99.60% | 検証精度: 99.56%
# エポック 16/25 | 訓練損失: 0.0098 | 訓練精度: 99.61% | 検証精度: 99.62%
# エポック 17/25 | 訓練損失: 0.0098 | 訓練精度: 99.62% | 検証精度: 99.59%
# エポック 18/25 | 訓練損失: 0.0092 | 訓練精度: 99.62% | 検証精度: 99.56%
# エポック 19/25 | 訓練損失: 0.0089 | 訓練精度: 99.65% | 検証精度: 99.68%
# エポック 20/25 | 訓練損失: 0.0085 | 訓練精度: 99.67% | 検証精度: 99.65%
# エポック 21/25 | 訓練損失: 0.0078 | 訓練精度: 99.69% | 検証精度: 99.54%
# エポック 22/25 | 訓練損失: 0.0081 | 訓練精度: 99.66% | 検証精度: 99.61%
# エポック 23/25 | 訓練損失: 0.0081 | 訓練精度: 99.68% | 検証精度: 99.56%
# エポック 24/25 | 訓練損失: 0.0074 | 訓練精度: 99.69% | 検証精度: 99.71%
# エポック 25/25 | 訓練損失: 0.0074 | 訓練精度: 99.72% | 検証精度: 99.52%
# 学習が完了しました。
# 学習済みモデルを 'pointnet_model_vC.pth' に保存しました。
# Press any key to continue . . .

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import pickle

# --- ★★★ ユーザー設定項目 ★★★ ---
# 学習のエポック数（教科書を何周読むか）
EPOCHS = 25
# バッチサイズ（一度に何個のサンプルを見るか）
BATCH_SIZE = 32
# 学習率（どれくらいの勢いで学習を進めるか）
LEARNING_RATE = 0.001
# データセットファイル
DATASET_FILE = 'final_training_datasetC.pkl'
# 学習済みモデルの保存先
MODEL_SAVE_PATH = 'pointnet_model_vC.pth' # v2など、新しい名前を付ける

# --- PointNetのモデル定義 ---

class TNet(nn.Module):
    """PointNetの入力と特徴を整列させるためのT-Net"""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k*self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNet(nn.Module):
    """PointNetの本体"""
    def __init__(self, num_classes=2):
        super(PointNet, self).__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        trans_input = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), trans_input).transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# --- PyTorchのデータセットクラス ---
class PointCloudDataset(torch.utils.data.Dataset):
    """学習用データセットをPyTorchで扱えるようにするクラス"""
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        points, label = self.dataset[index]
        # PointNetは (バッチサイズ, チャンネル数, 点数) の入力を期待するので、(3, 64) に変形
        points = torch.from_numpy(points.T) 
        label = torch.tensor(label, dtype=torch.long)
        return points, label

# --- メインの学習処理 ---
if __name__ == '__main__':
    # 1. データセットの準備
    print(f"'{DATASET_FILE}' を読み込んでいます...")
    full_dataset = PointCloudDataset(DATASET_FILE)
    
    # 訓練用と検証用にデータを分割 (80%を訓練用、20%を検証用)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    print(f"訓練データ数: {len(train_dataset)}, 検証データ数: {len(test_dataset)}")
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. モデル、損失関数、最適化手法の定義
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    model = PointNet(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 3. 学習ループ
    print("学習を開始します...")
    for epoch in range(EPOCHS):
        # --- 訓練モード ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            points, labels = data
            points, labels = points.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total
        
        # --- 検証モード ---
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in test_loader:
                points, labels = data
                points, labels = points.to(device), labels.to(device)
                outputs = model(points)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total

        print(f'エポック {epoch + 1}/{EPOCHS} | 訓練損失: {running_loss / len(train_loader):.4f} | ' \
              f'訓練精度: {train_accuracy:.2f}% | 検証精度: {test_accuracy:.2f}%')

    print('学習が完了しました。')

    # 4. 学習済みモデルの保存
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"学習済みモデルを '{MODEL_SAVE_PATH}' に保存しました。")