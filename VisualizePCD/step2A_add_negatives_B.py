# ファイル名: step2_add_negatives_B.py 追加分　側面のみ
import open3d as o3d
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm

# --- ★★★ ユーザー設定項目 ★★★ ---

# B: ステップ1で作成した「作業範囲」のCSVファイル
WORK_AREA_CSV_PATH = 'work_area_large.csv' 

# C: この中でアノテーションを行う、という領域を指定します。
# work_area.csv を一度表示してみて、鉄筋とその周辺が含まれるように座標を調整してください。
# ANNOTATION_TARGET_BOX = [
#     -1.0, 1.0,   # ラベル付けしたいXの範囲 (min, max)
#     -0.5, 0.5,   # ラベル付けしたいYの範囲 (min, max)
#     -0.2, 0.2    # ラベル付けしたいZの範囲 (min, max)
# ]

ANNOTATION_TARGET_BOX = [
    2001.311961172848, 2519.6492291477407,  # Xの範囲
    -1312.0, -1100.0,  # Yの範囲
    -598.7830383125777, -94.6667521445312   # Zの範囲
]

# --- 他の設定項目 ---
# ★★★ ファイル名を新しいものに変更 ★★★
LABELS_FILE_PATH = 'annotated_labels_negatives_B.npy'
OUTPUT_DATASET_FILE = 'training_dataset_negatives_B.pkl'
# ★★★ ここまで ★★★
NUM_NEIGHBORS = 64 # 近傍点の数

# --- ラベルと色の定義 ---
LABEL_UNLABELED = 0
LABEL_REBAR = 1
LABEL_NOT_REBAR = 2
COLOR_UNLABELED = [0.5, 0.5, 0.5]  # グレー
COLOR_REBAR = [1.0, 0.0, 0.0]      # 赤
COLOR_NOT_REBAR = [0.0, 0.0, 1.0]  # 青
COLOR_CURRENT = [0.0, 1.0, 0.0]    # 緑 (注目点)

class InteractiveAnnotator:
    """対話的アノテーションを行うためのクラス"""
    def __init__(self):
        self.points = None
        self.labels = None
        self.pcd = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        
        self.target_indices = None  # アノテーション対象となる点のインデックスリスト
        self.current_target_pos = 0 # target_indices内の現在位置
        self.current_point_index = 0 # points全体の中での現在位置

    def run_annotation(self):
        """アノテーションツールを起動する"""
        if not self._load_data():
            return
        self._setup_visualizer()
        self.vis.run()

    def _load_data(self):
        """点群データと、既存ならラベルデータを読み込む"""
        try:
            print(f"作業範囲ファイル '{WORK_AREA_CSV_PATH}' を読み込み中...")
            df = pd.read_csv(WORK_AREA_CSV_PATH, header=None, names=['x', 'y', 'z', 'intensity'])
            self.points = df[['x', 'y', 'z']].values
            self.pcd.points = o3d.utility.Vector3dVector(self.points)
        except FileNotFoundError:
            print(f"エラー: '{WORK_AREA_CSV_PATH}' が見つかりません。ステップ1を先に実行してください。")
            return False

        box = ANNOTATION_TARGET_BOX
        self.target_indices = np.where(
            (self.points[:, 0] >= box[0]) & (self.points[:, 0] <= box[1]) &
            (self.points[:, 1] >= box[2]) & (self.points[:, 1] <= box[3]) &
            (self.points[:, 2] >= box[4]) & (self.points[:, 2] <= box[5])
        )[0]
        
        if len(self.target_indices) == 0:
            print("エラー: ANNOTATION_TARGET_BOX内に点が見つかりません。座標範囲を確認してください。")
            return False
        print(f"アノテーション対象は {len(self.target_indices)} 点に絞られました。")

        if os.path.exists(LABELS_FILE_PATH):
            print(f"'{LABELS_FILE_PATH}' から既存のラベルを読み込みました。")
            self.labels = np.load(LABELS_FILE_PATH)
        else:
            print("既存のラベルファイルが見つかりません。新しいラベルを作成します。")
            self.labels = np.full(len(self.points), LABEL_UNLABELED, dtype=int)
        
        self.current_point_index = self.target_indices[self.current_target_pos]
        return True

    def _update_colors(self, update_geometry=False):
        """点群の色を更新する"""
        colors = np.array([COLOR_UNLABELED] * len(self.points))
        colors[self.labels == LABEL_REBAR] = COLOR_REBAR
        colors[self.labels == LABEL_NOT_REBAR] = COLOR_NOT_REBAR
        colors[self.current_point_index] = COLOR_CURRENT
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        if update_geometry:
            self.vis.update_geometry(self.pcd)
        self.vis.update_renderer()
        self._print_status()

    def _print_status(self):
        """現在の進捗状況を表示する"""
        total_targets = len(self.target_indices)
        labeled_targets = np.sum(self.labels[self.target_indices] != LABEL_UNLABELED)
        rebar = np.sum(self.labels == LABEL_REBAR)
        not_rebar = np.sum(self.labels == LABEL_NOT_REBAR)
        print(f"\r進捗: {labeled_targets}/{total_targets} | 注目点: {self.current_point_index} | 鉄筋(赤): {rebar} | 非鉄筋(青): {not_rebar}", end="")

    def _set_label(self, label):
        """現在の注目点にラベルを付けて次に進む"""
        self.labels[self.current_point_index] = label
        self._next_point(self.vis)

    def _next_point(self, vis):
        """アノテーション対象リストの次の点へ移動する"""
        if self.current_target_pos + 1 < len(self.target_indices):
            self.current_target_pos += 1
            self.current_point_index = self.target_indices[self.current_target_pos]
            self._update_colors(True)

    def _prev_point(self, vis):
        """アノテーション対象リストの前の点へ戻る"""
        if self.current_target_pos > 0:
            self.current_target_pos -= 1
            self.current_point_index = self.target_indices[self.current_target_pos]
            self._update_colors(True)

    def _save_labels(self, vis):
        """現在のラベル情報をファイルに保存する"""
        np.save(LABELS_FILE_PATH, self.labels)
        print(f"\nラベルを '{LABELS_FILE_PATH}' に保存しました。")
        self._print_status()

    def _load_labels(self, vis):
        """ファイルからラベル情報を読み込む"""
        if os.path.exists(LABELS_FILE_PATH):
            self.labels = np.load(LABELS_FILE_PATH)
            print(f"\n'{LABELS_FILE_PATH}' からラベルを再読み込みしました。")
            self._update_colors(True)
        else:
            print(f"\n保存ファイルが見つかりません。")

    def _quit(self, vis):
        """ビジュアライザを終了する"""
        vis.destroy_window()

    def _setup_visualizer(self):
        """キー操作やウィンドウの設定を行う"""
        self.vis.create_window(window_name="Interactive Annotator", width=1600, height=1200)
        self.vis.add_geometry(self.pcd)
        
        key_to_callback = {
            ord("1"): lambda v: self._set_label(LABEL_REBAR),
            ord("2"): lambda v: self._set_label(LABEL_NOT_REBAR),
            ord("N"): self._next_point,
            ord("B"): self._prev_point,
            ord("S"): self._save_labels,
            ord("L"): self._load_labels,
            ord("Q"): self._quit
        }
        for key, callback in key_to_callback.items():
            self.vis.register_key_callback(key, callback)

        self._update_colors()
        print("\n--- 操作方法 ---")
        print("1: 鉄筋 (赤) としてラベル付け")
        print("2: 非鉄筋 (青) としてラベル付け")
        print("N: 次の点へ   B: 前の点へ戻る")
        print("S: 進捗を保存  L: 保存した進捗を読み込む")
        print("Q: 終了")
        print("----------------")

def create_training_dataset_from_labels():
    """保存されたラベルから学習データセットを作成する"""
    print("学習用データセットの作成を開始します。")
    try:
        points = pd.read_csv(WORK_AREA_CSV_PATH, header=None).iloc[:, :3].values
        labels = np.load(LABELS_FILE_PATH)
    except FileNotFoundError as e:
        print(f"エラー: 必要なファイルが見つかりません。 {e}")
        return

    dataset = []
    target_indices = np.where(labels > 0)[0]
    
    if len(target_indices) == 0:
        print("エラー: アノテーションされた点が見つかりません。")
        return

    pcd_for_kdtree = o3d.geometry.PointCloud()
    pcd_for_kdtree.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_for_kdtree)
    
    print("近傍点を探索し、データセットを作成中...")
    for i in tqdm(target_indices):
        [k, idx, _] = kdtree.search_knn_vector_3d(points[i], NUM_NEIGHBORS)
        if k < NUM_NEIGHBORS: continue

        neighbors = points[idx]
        normalized_neighbors = neighbors - points[i]
        model_label = labels[i] - 1  # 鉄筋(1)->0, 非鉄筋(2)->1
        dataset.append((normalized_neighbors.astype(np.float32), model_label))
    
    if not dataset:
        print("データセットの作成に失敗しました。")
        return

    print(f"データセットを作成しました。総サンプル数: {len(dataset)}")
    with open(OUTPUT_DATASET_FILE, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"データセットを '{OUTPUT_DATASET_FILE}' に保存しました。")

if __name__ == '__main__':
    choice = input("何をしますか？ [1] アノテーションを開始/再開 [2] 保存済みラベルから学習データを作成: ")
    if choice == '1':
        annotator = InteractiveAnnotator()
        annotator.run_annotation()
    elif choice == '2':
        create_training_dataset_from_labels()
    else:
        print("無効な選択です。'1'または'2'を入力してください。")