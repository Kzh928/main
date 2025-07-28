# ファイル名: D3_test_strict_model.py

import torch
import torch.nn as nn
import numpy as np
import pickle
import random
import open3d as o3d
from torch.utils.data import Dataset, DataLoader

# --- ★★★ ユーザー設定項目 ★★★ ---
MODEL_PATH = 'strict_additive_model.pth'
DATASET_PATH = 'strict_additive_dataset.pkl'
NUM_SAMPLES_TO_SHOW = 5
X_SPACING = 300.0
Y_SPACING = 300.0

# --- モデル定義 (C3の学習スクリプトと全く同じ) ---
class StrictAdditiveGenerator(nn.Module):
    def __init__(self, input_points=32, additional_points=32):
        super(StrictAdditiveGenerator, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_points*3,256), nn.ReLU(), nn.Linear(256,128))
        self.decoder = nn.Sequential(nn.Linear(128,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU(), nn.Linear(512,additional_points*3))
        self.additional_points = additional_points
    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        features = self.encoder(x_flat)
        generated_32_flat = self.decoder(features)
        generated_32 = generated_32_flat.view(x.size(0), self.additional_points, 3)
        return generated_32

# --- データセットクラス (C3の学習スクリプトと全く同じ) ---
class StrictAdditiveDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f: self.data = pickle.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        input_32, missing_32 = self.data[idx]; return torch.from_numpy(input_32), torch.from_numpy(missing_32)

# --- メインのテスト処理 ---
if __name__ == '__main__':
    # 1. モデルの準備
    device = torch.device("cpu")
    model = StrictAdditiveGenerator().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。"); exit()
    model.eval()
    print(f"'{MODEL_PATH}' から学習済みモデルを読み込みました。")
    
    # 2. テスト用データセットの準備
    try:
        dataset = StrictAdditiveDataset(DATASET_PATH)
        if len(dataset) < NUM_SAMPLES_TO_SHOW:
            NUM_SAMPLES_TO_SHOW = len(dataset)
        test_samples = random.sample(list(dataset), NUM_SAMPLES_TO_SHOW)
        print(f"'{DATASET_PATH}' からランダムに{NUM_SAMPLES_TO_SHOW}個のテストサンプルを選びました。")
    except FileNotFoundError:
        print(f"エラー: データセットファイル '{DATASET_PATH}' が見つかりません。"); exit()

    # 3. 可視化するジオメトリを保存するリスト
    geometries_to_draw = []
    print("AIに32点から不足分の32点を生成させています...")
    with torch.no_grad():
        for i, (input_32_tensor, missing_32_truth_tensor) in enumerate(test_samples):
            input_32_np = input_32_tensor.cpu().numpy()
            
            # AIによる「追加の32点」の生成（推論）
            generated_32_tensor = model(input_32_tensor.unsqueeze(0).to(device))
            generated_32_np = generated_32_tensor.squeeze(0).cpu().numpy()
            
            # ★★★ ここが新しいロジック ★★★
            # AIの生成結果（64点） = 元の32点 + AIが生成した32点
            generated_combined_points = np.vstack([input_32_np, generated_32_np])
            # 正解（64点） = 元の32点 + 正解の残り32点
            dense_points_truth_np = np.vstack([input_32_np, missing_32_truth_tensor.cpu().numpy()])
            # ★★★ ここまで ★★★

            y_offset = -i * Y_SPACING

            # 入力 (青)
            pcd_input = o3d.geometry.PointCloud()
            pcd_input.points = o3d.utility.Vector3dVector(input_32_np)
            pcd_input.paint_uniform_color([0, 0, 1])
            pcd_input.translate((0, y_offset, 0))
            geometries_to_draw.append(pcd_input)

            # AIの生成結果 (赤)
            pcd_generated = o3d.geometry.PointCloud()
            pcd_generated.points = o3d.utility.Vector3dVector(generated_combined_points)
            pcd_generated.paint_uniform_color([1, 0, 0])
            pcd_generated.translate((X_SPACING, y_offset, 0))
            geometries_to_draw.append(pcd_generated)

            # 正解 (緑)
            pcd_truth = o3d.geometry.PointCloud()
            pcd_truth.points = o3d.utility.Vector3dVector(dense_points_truth_np)
            pcd_truth.paint_uniform_color([0, 1, 0])
            pcd_truth.translate((X_SPACING * 2, y_offset, 0))
            geometries_to_draw.append(pcd_truth)

    print("\n結果を可視化します。")
    print("各行が1つのテスト結果で、左から [入力(青,32点)], [AIの生成結果(赤,64点)], [正解(緑,64点)] です。")
    o3d.visualization.draw_geometries(
        geometries_to_draw,
        window_name=f"{NUM_SAMPLES_TO_SHOW} Strict Additive Model Test Results",
        width=1200, height=400 + (NUM_SAMPLES_TO_SHOW -1) * 150
    )