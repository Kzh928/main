# 生成日: 2025-06-25
# 目的: 点群補完AIが、「枠線」という構造的な形状を
#       正しく復元できるか、その単体性能をテストする。
# ファイル名: H3_unittest_square_frame.py

import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import random

# --- ★★★ ユーザー設定項目 ★★★ ---
MODEL_PATH = 'repulsion_model.pth' 
NUM_PATTERNS_TO_SHOW = 5

# --- モデル定義 (変更なし) ---
class StrictAdditiveGenerator(nn.Module):
    def __init__(self, input_points=32, additional_points=32):
        super(StrictAdditiveGenerator, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_points*3,256), nn.ReLU(), nn.Linear(256,128))
        self.decoder = nn.Sequential(nn.Linear(128,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU(), nn.Linear(512,additional_points*3))
        self.additional_points = additional_points
    def forward(self, x):
        x_flat = x.view(x.size(0), -1); features = self.encoder(x_flat); generated_32_flat = self.decoder(features)
        generated_32 = generated_32_flat.view(x.size(0), self.additional_points, 3); return generated_32

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

    # 2. テスト用の「正方形の枠線」データを作成
    print(f"テスト用の合成データ（正方形の枠線）を {NUM_PATTERNS_TO_SHOW} パターン作成・評価します...")
    num_points_per_edge = 16
    edge_coords = np.linspace(-50, 50, num_points_per_edge, endpoint=False)
    z = np.zeros(num_points_per_edge)
    top_edge = np.vstack([edge_coords, np.full_like(edge_coords, 50), z]).T
    bottom_edge = np.vstack([edge_coords, np.full_like(edge_coords, -50), z]).T
    left_edge = np.vstack([np.full_like(edge_coords, -50), edge_coords, z]).T
    right_edge = np.vstack([np.full_like(edge_coords, 50), edge_coords, z]).T
    points_64_truth = np.vstack([top_edge, bottom_edge, left_edge, right_edge])
    
    geometries_to_draw = []
    with torch.no_grad():
        for i in range(NUM_PATTERNS_TO_SHOW):
            # 32点のまばらな入力を作成
            input_indices = np.random.choice(64, 32, replace=False)
            points_32_input = points_64_truth[input_indices]
            
            # AIによる点群生成（推論）
            input_tensor = torch.from_numpy(points_32_input).unsqueeze(0).to(device).float()
            generated_32_tensor = model(input_tensor)
            generated_32_points = generated_32_tensor.squeeze(0).cpu().numpy()
            generated_combined_points = np.vstack([points_32_input, generated_32_points])

            # 結果の可視化準備
            y_offset = -i * 150
            pcd_input = o3d.geometry.PointCloud(); pcd_input.points = o3d.utility.Vector3dVector(points_32_input)
            pcd_input.paint_uniform_color([0, 0, 1]); pcd_input.translate((0, y_offset, 0))
            geometries_to_draw.append(pcd_input)
            pcd_generated = o3d.geometry.PointCloud(); pcd_generated.points = o3d.utility.Vector3dVector(generated_combined_points)
            pcd_generated.paint_uniform_color([1, 0, 0]); pcd_generated.translate((200, y_offset, 0))
            geometries_to_draw.append(pcd_generated)
            pcd_truth_vis = o3d.geometry.PointCloud(); pcd_truth_vis.points = o3d.utility.Vector3dVector(points_64_truth)
            pcd_truth_vis.paint_uniform_color([0, 1, 0]); pcd_truth_vis.translate((400, y_offset, 0))
            geometries_to_draw.append(pcd_truth_vis)

    # 全パターンの可視化
    print("\n結果を可視化します。")
    print("各行が1つのランダムパターンでのテスト結果です。")
    print("左から [穴あき枠線(青)], [AIの復元結果(赤)], [完璧な枠線(緑)] です。")
    o3d.visualization.draw_geometries(
        geometries_to_draw,
        window_name=f"{NUM_PATTERNS_TO_SHOW} Patterns - Frame Test",
        width=1200, height=800
    )
