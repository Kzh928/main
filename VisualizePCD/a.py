# ファイル名: step2.3_interactive_annotator_highlight.py
# 概要: 注目点をハイライト表示し、視点移動を簡単に行えるように改善したアノテーター。

import open3d as o3d
import numpy as np
import pandas as pd
import os

# --- ★★★ ユーザー設定項目 ★★★ ---

# 作業対象の点群CSVファイル
WORK_AREA_CSV_PATH = 'equipment_area2.csv'

# ラベル情報の保存/読み込み先
LABELS_FILE_PATH = 'labels_for_equipment_2.npy'

# アノテーション対象範囲
ANNOTATION_TARGET_BOX = [
    -50000, 50000,  # Xの範囲 (min, max)
    -50000, 50000,  # Yの範囲 (min, max)
    -50000, 50000   # Zの範囲 (min, max)
]

# ▼▼▼【追加】ハイライト表示用の設定 ▼▼▼
# ハイライト用の球体の半径 (点群のスケールに合わせて調整)
HIGHLIGHT_RADIUS = 30.0
# ハイライト用の球体の色 (マゼンタ)
HIGHLIGHT_COLOR = [1.0, 0.0, 1.0]
# ▼▼▼【変更】注目点の色をハイライト色と統一 ▼▼▼
COLOR_CURRENT = HIGHLIGHT_COLOR


# --- ラベルと色の定義 ---
LABEL_UNLABELED = 0
LABEL_REBAR = 1
LABEL_NOT_REBAR = 2
LABEL_EQUIPMENT = 3
COLOR_UNLABELED = [0.5, 0.5, 0.5] # グレー
COLOR_REBAR = [1.0, 0.0, 0.0]     # 赤
COLOR_NOT_REBAR = [0.0, 0.0, 1.0] # 青
COLOR_EQUIPMENT = [0.0, 1.0, 0.0] # 緑

class InteractiveAnnotator:
    def __init__(self):
        self.points = None
        self.labels = None
        self.pcd = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.target_indices = None
        self.current_target_pos = 0
        self.current_point_index = 0
        # ▼▼▼【追加】ハイライト用の球体を管理する変数を追加 ▼▼▼
        self.highlight_sphere = None

    def run_annotation(self):
        if not self._load_data(): return
        self._setup_visualizer()
        self.vis.run()

    def _load_data(self):
        try:
            df = pd.read_csv(WORK_AREA_CSV_PATH, header=None, names=['x', 'y', 'z', 'intensity'])
            self.points = df[['x', 'y', 'z']].values
            self.pcd.points = o3d.utility.Vector3dVector(self.points)
        except FileNotFoundError:
            print(f"エラー: '{WORK_AREA_CSV_PATH}' が見つかりません。")
            return False

        box = ANNOTATION_TARGET_BOX
        self.target_indices = np.where(
            (self.points[:, 0] >= box[0]) & (self.points[:, 0] <= box[1]) &
            (self.points[:, 1] >= box[2]) & (self.points[:, 1] <= box[3]) &
            (self.points[:, 2] >= box[4]) & (self.points[:, 2] <= box[5])
        )[0]

        if len(self.target_indices) == 0: print("エラー: ANNOTATION_TARGET_BOX内に点が見つかりません。"); return False
        print(f"アノテーション対象は {len(self.target_indices)} 点です。")

        if os.path.exists(LABELS_FILE_PATH):
            print(f"'{LABELS_FILE_PATH}' から既存のラベルを読み込みました。")
            self.labels = np.load(LABELS_FILE_PATH)
        else:
            print("新しいラベルファイルを作成します。")
            self.labels = np.full(len(self.points), LABEL_UNLABELED, dtype=int)

        self.current_point_index = self.target_indices[self.current_target_pos]
        return True

    def _update_colors(self, update_geometry=True):
        colors = np.array([COLOR_UNLABELED] * len(self.points))
        colors[self.labels == LABEL_REBAR] = COLOR_REBAR
        colors[self.labels == LABEL_NOT_REBAR] = COLOR_NOT_REBAR
        colors[self.labels == LABEL_EQUIPMENT] = COLOR_EQUIPMENT
        # 注目点自体の色もハイライト色に
        colors[self.current_point_index] = COLOR_CURRENT
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        if update_geometry: self.vis.update_geometry(self.pcd)
        self._update_highlight_sphere() # ハイライト球を更新
        self._print_status()

    # ▼▼▼【新機能】ハイライト用の球体を更新する関数 ▼▼▼
    def _update_highlight_sphere(self):
        # 以前の球体があれば削除
        if self.highlight_sphere is not None:
            self.vis.remove_geometry(self.highlight_sphere, reset_bounding_box=False)

        # 新しい球体を作成
        center_point = self.points[self.current_point_index]
        self.highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=HIGHLIGHT_RADIUS)
        self.highlight_sphere.paint_uniform_color(HIGHLIGHT_COLOR)
        self.highlight_sphere.translate(center_point)

        # 新しい球体をビジュアライザに追加
        self.vis.add_geometry(self.highlight_sphere, reset_bounding_box=False)

    def _print_status(self):
        total_targets = len(self.target_indices)
        labeled_targets = np.sum(self.labels[self.target_indices] != LABEL_UNLABELED)
        rebar = np.sum(self.labels == LABEL_REBAR)
        not_rebar = np.sum(self.labels == LABEL_NOT_REBAR)
        equipment = np.sum(self.labels == LABEL_EQUIPMENT)
        status_str = f"\r進捗: {labeled_targets}/{total_targets} | 注目点ID: {self.current_point_index} | 鉄筋(1): {rebar} | 非鉄筋(2): {not_rebar} | 装置(3): {equipment}"
        print(status_str, end="")

    def _set_label(self, label):
        self.labels[self.current_point_index] = label
        self._next_point(self.vis)

    def _next_point(self, vis):
        if self.current_target_pos + 1 < len(self.target_indices):
            self.current_target_pos += 1
            self.current_point_index = self.target_indices[self.current_target_pos]
            self._update_colors()

    def _prev_point(self, vis):
        if self.current_target_pos > 0:
            self.current_target_pos -= 1
            self.current_point_index = self.target_indices[self.current_target_pos]
            self._update_colors()

    # ▼▼▼【新機能】注目点に視点を移動する関数 ▼▼▼
    def _center_view(self, vis):
        ctr = self.vis.get_view_control()
        ctr.set_lookat(self.points[self.current_point_index])
        print("\n視点を注目点に移動しました。")
        self._print_status()

    def _save_labels(self, vis):
        np.save(LABELS_FILE_PATH, self.labels)
        print(f"\nラベルを '{LABELS_FILE_PATH}' に保存しました。")
        self._print_status()

    def _quit(self, vis):
        vis.destroy_window()

    def _setup_visualizer(self):
        self.vis.create_window(window_name="Interactive Annotator (Highlight Ver.)", width=1600, height=1200)
        self.vis.add_geometry(self.pcd)

        # ▼▼▼【変更】キー操作に「C」を追加 ▼▼▼
        key_to_callback = {
            ord("1"): lambda v: self._set_label(LABEL_REBAR),
            ord("2"): lambda v: self._set_label(LABEL_NOT_REBAR),
            ord("3"): lambda v: self._set_label(LABEL_EQUIPMENT),
            ord("N"): self._next_point,
            ord("B"): self._prev_point,
            ord("C"): self._center_view, # ★注目点へ視点移動
            ord("S"): self._save_labels,
            ord("Q"): self._quit
        }
        for key, callback in key_to_callback.items():
            self.vis.register_key_callback(key, callback)

        # 最初の色とハイライトを更新
        self._update_colors()

        print("\n--- 操作方法 ---")
        print("1: 鉄筋 (赤) としてラベル付け")
        print("2: 非鉄筋 (青) としてラベル付け")
        print("3: 装置 (緑) としてラベル付け")
        print("------------------------------")
        print("N: 次の点へ (スキップ)")
        print("B: 前の点へ戻る")
        print("C: ★注目点へ視点を移動する★")
        print("------------------------------")
        print("S: 進捗を保存")
        print("Q: 終了")
        print("------------------------------")

if __name__ == '__main__':
    annotator = InteractiveAnnotator()
    annotator.run_annotation()
