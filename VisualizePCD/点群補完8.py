# ファイル名: C4_train_with_repulsion.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
INPUT_DATASET_PATH = 'strict_additive_dataset.pkl'
MODEL_SAVE_PATH = 'repulsion_model.pth' # ★ 新しいモデル名

# ★★★ 新しいパラメータ ★★★
# Repulsion Lossの重み。この値を大きくすると、点を散らす力が強くなる
REPULSION_WEIGHT = 0.1

# --- モデル定義 ---
class StrictAdditiveGenerator(nn.Module):
    def __init__(self, input_points=32, additional_points=32):
        super(StrictAdditiveGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_points * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, additional_points * 3)
        )
        self.additional_points = additional_points

    def forward(self, x):
        # このモデルは、追加する32点だけを生成する
        x_flat = x.view(x.size(0), -1)
        features = self.encoder(x_flat)
        generated_32_flat = self.decoder(features)
        generated_32 = generated_32_flat.view(x.size(0), self.additional_points, 3)
        return generated_32

# --- データセットクラス ---
class StrictAdditiveDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        input_32, missing_32 = self.data[idx]
        return torch.from_numpy(input_32), torch.from_numpy(missing_32)

# --- 損失関数 ---
def chamfer_distance(p1, p2):
    """ 点群P1とP2の間のChamfer Distanceを計算 """
    dist1 = torch.cdist(p1, p2)
    dist2 = torch.cdist(p2, p1)
    min_dist1, _ = torch.min(dist1, dim=2)
    min_dist2, _ = torch.min(dist2, dim=2)
    loss = min_dist1.mean() + min_dist2.mean()
    return loss

def repulsion_loss(points, k=5, h=0.03):
    """ 生成された点群内の点同士が近すぎる場合のペナルティを計算 """
    dist_matrix = torch.cdist(points, points)
    # 自分自身(距離0)を除いた、最も近いk個の点を探す
    knn_dist = torch.topk(dist_matrix, k + 1, dim=2, largest=False).values[:, :, 1:]
    # hより近い点に強いペナルティを与える (hは半径のようなもの)
    penalty = torch.exp(- (knn_dist ** 2) / (h ** 2))
    loss = torch.mean(penalty)
    return loss

# --- メインの学習処理 ---
if __name__ == '__main__':
    device = torch.device("cpu")
    try:
        dataset = StrictAdditiveDataset(INPUT_DATASET_PATH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"'{INPUT_DATASET_PATH}' を読み込みました。総サンプル数: {len(dataset)}")
    except FileNotFoundError:
        print(f"エラー: データセットファイル '{INPUT_DATASET_PATH}' が見つかりません。ステップA3を先に実行してください。")
        exit()
    
    model = StrictAdditiveGenerator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nRepulsion Lossを導入した、新しいモデルの学習を開始します...")
    for epoch in range(EPOCHS):
        model.train()
        total_chamfer = 0
        total_repulsion = 0
        for input_batch, missing_truth_batch in tqdm(dataloader, desc=f"エポック {epoch+1}/{EPOCHS}"):
            input_batch, missing_truth_batch = input_batch.to(device), missing_truth_batch.to(device)
            
            optimizer.zero_grad()
            generated_32_batch = model(input_batch)
            
            # 2種類の損失を計算
            chamfer_loss_val = chamfer_distance(generated_32_batch, missing_truth_batch)
            repulsion_loss_val = repulsion_loss(generated_32_batch)
            
            # 2つの損失を重み付けして合計
            total_loss = chamfer_loss_val + REPULSION_WEIGHT * repulsion_loss_val
            
            total_loss.backward()
            optimizer.step()
            
            total_chamfer += chamfer_loss_val.item()
            total_repulsion += repulsion_loss_val.item()
        
        avg_chamfer = total_chamfer / len(dataloader)
        avg_repulsion = total_repulsion / len(dataloader)
        print(f"エポック {epoch+1} 完了 | 平均Chamfer損失: {avg_chamfer:.6f} | 平均Repulsion損失: {avg_repulsion:.6f}")

    print("学習が完了しました。")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"学習済みモデルを '{MODEL_SAVE_PATH}' に保存しました。")