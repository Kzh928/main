# ファイル名: C2_train_additive_model.py

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
INPUT_DATASET_PATH = 'additive_dataset.pkl' # ★ ステップAで作成したファイル
MODEL_SAVE_PATH = 'additive_model.pth'      # ★ 新しいモデル名

# --- モデル定義 ---
class AdditiveGenerator(nn.Module):
    def __init__(self, input_points=32, additional_points=32):
        super(AdditiveGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_points * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        # ★★★ 出力層を32点分(32*3=96)に変更 ★★★
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, additional_points * 3) 
        )
        self.additional_points = additional_points

    def forward(self, x):
        # xは (バッチサイズ, 32, 3)
        x_flat = x.view(x.size(0), -1) # (バッチサイズ, 32*3) に
        features = self.encoder(x_flat)
        generated_32_flat = self.decoder(features)
        # (バッチサイズ, 32*3) -> (バッチサイズ, 32, 3) に
        generated_32 = generated_32_flat.view(x.size(0), self.additional_points, 3)
        
        # ★★★ 入力32点と生成32点を結合 ★★★
        combined_64 = torch.cat([x, generated_32], dim=1)
        return combined_64

# --- データセットクラス (変更なし) ---
class AdditiveDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f: self.data = pickle.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        sparse, dense = self.data[idx]; return torch.from_numpy(sparse), torch.from_numpy(dense)

# --- 損失関数 (Chamfer Distance) (変更なし) ---
def chamfer_distance(p1, p2):
    dist1 = torch.cdist(p1, p2); dist2 = torch.cdist(p2, p1)
    min_dist1, _ = torch.min(dist1, dim=2); min_dist2, _ = torch.min(dist2, dim=2)
    loss = min_dist1.mean() + min_dist2.mean(); return loss

# --- メインの学習処理 ---
if __name__ == '__main__':
    device = torch.device("cpu")
    dataset = AdditiveDataset(INPUT_DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = AdditiveGenerator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("追加生成モデルの学習を開始します... (時間がかかります)")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for sparse_batch, dense_batch in tqdm(dataloader, desc=f"エポック {epoch+1}/{EPOCHS}"):
            sparse_batch, dense_batch = sparse_batch.to(device), dense_batch.to(device)
            
            optimizer.zero_grad()
            # ★★★ モデルは結合後の64点を返す ★★★
            generated_combined_batch = model(sparse_batch)
            # ★★★ 答え合わせは、結合後64点 vs 正解64点 ★★★
            loss = chamfer_distance(generated_combined_batch, dense_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"エポック {epoch+1} 完了 | 平均損失 (Chamfer Distance): {avg_loss:.6f}")

    print("学習が完了しました。")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"学習済みモデルを '{MODEL_SAVE_PATH}' に保存しました。")