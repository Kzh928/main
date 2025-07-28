# ファイル名: E1_analyze_upsampling_density.py

import torch
import torch.nn as nn
import numpy as np
import pickle
import random
import open3d as o3d
from torch.utils.data import Dataset

# --- ★★★ ユーザー設定項目 ★★★ ---
MODEL_PATH = 'strict_additive_model.pth'
DATASET_PATH = 'strict_additive_dataset.pkl'
NUM_SAMPLES_TO_ANALYZE = 5 # 分析するサンプル数

# --- モデル定義 (C3の学習スクリプトと全く同じ) ---
class StrictAdditiveGenerator(nn.Module):
    def __init__(self, input_points=32, additional_points=32):
        super(StrictAdditiveGenerator, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_points*3,256), nn.ReLU(), nn.Linear(256,128))
        self.decoder = nn.Sequential(nn.Linear(128,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU(), nn.Linear(512,additional_points*3))
        self.additional_points = additional_points
    def forward(self, x):
        x_flat = x.view(x.size(0), -1); features = self.encoder(x_flat); generated_32_flat = self.decoder(features)
        generated_32 = generated_32_flat.view(x.size(0), self.additional_points, 3); return generated_32

# --- データセットクラス (C3の学習スクリプトと全く同じ) ---
class StrictAdditiveDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f: self.data = pickle.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        input_32, missing_32 = self.data[idx]; return torch.from_numpy(input_32), torch.from_numpy(missing_32)

def analyze_density(pcd):
    """点群の2つの密度指標を計算する関数"""
    if len(pcd.points) == 0:
        return 0, 0
    # 1. 平均最近傍距離
    distances = pcd.compute_nearest_neighbor_distance()
    avg_nn_dist = np.mean(distances)
    
    # 2. 重心からの平均距離
    centroid = pcd.get_center()
    avg_centroid_dist = np.mean(np.linalg.norm(np.asarray(pcd.points) - centroid, axis=1))
    
    return avg_nn_dist, avg_centroid_dist

# --- メインの分析処理 ---
if __name__ == '__main__':
    # 1. モデルとデータの準備
    device = torch.device("cpu")
    model = StrictAdditiveGenerator().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        dataset = StrictAdditiveDataset(DATASET_PATH)
        test_samples = random.sample(list(dataset), NUM_SAMPLES_TO_ANALYZE)
    except FileNotFoundError as e:
        print(f"エラー: 必要なファイルが見つかりません。 {e}"); exit()
    model.eval()
    
    print(f"--- {NUM_SAMPLES_TO_ANALYZE}個のサンプルについて、密度を数値化します ---")
    
    with torch.no_grad():
        for i, (input_32_tensor, missing_32_truth_tensor) in enumerate(test_samples):
            input_32_np = input_32_tensor.cpu().numpy()
            
            # AIによる「追加の32点」の生成
            generated_32_tensor = model(input_32_tensor.unsqueeze(0).to(device))
            generated_32_np = generated_32_tensor.squeeze(0).cpu().numpy()
            
            # 各点群を準備
            pcd_input = o3d.geometry.PointCloud(); pcd_input.points = o3d.utility.Vector3dVector(input_32_np)
            
            generated_combined_points = np.vstack([input_32_np, generated_32_np])
            pcd_generated = o3d.geometry.PointCloud(); pcd_generated.points = o3d.utility.Vector3dVector(generated_combined_points)

            dense_points_truth_np = np.vstack([input_32_np, missing_32_truth_tensor.cpu().numpy()])
            pcd_truth = o3d.geometry.PointCloud(); pcd_truth.points = o3d.utility.Vector3dVector(dense_points_truth_np)

            # 各点群の密度を計算
            nn_in, cd_in = analyze_density(pcd_input)
            nn_gen, cd_gen = analyze_density(pcd_generated)
            nn_truth, cd_truth = analyze_density(pcd_truth)

            # 結果を出力
            print(f"\n--- サンプル {i+1} ---")
            print(f"  [入力 (32点)]    平均最近傍距離: {nn_in:.2f} | 重心からの平均距離: {cd_in:.2f}")
            print(f"  [AI生成 (64点)]  平均最近傍距離: {nn_gen:.2f} | 重心からの平均距離: {cd_gen:.2f}")
            print(f"  [正解 (64点)]    平均最近傍距離: {nn_truth:.2f} | 重心からの平均距離: {cd_truth:.2f}")