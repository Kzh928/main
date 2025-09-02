# ファイル名: step2.7_two_stage_annotator_4class.py
# 概要: 「鉄筋」「壁面」「装置」「床」の4クラスをアノテーションするための
#       範囲選択と個別ラベリングを組み合わせた高効率アノテーター。

import open3d as o3d
import numpy as np
import pandas as pd
import os

# --- ★★★ ユーザー設定項目 ★★★ ---
WORK_AREA_CSV_PATH = 'equipment_area1020C.csv'
# ラベルファイルのパス (4クラス用の新しいファイル名)
LABELS_FILE_PATH = 'annotated_labels_4class_1020C.npy'

# --- ▼▼▼【変更点】新しいラベル体系の定義 ▼▼▼ ---
LABEL_UNLABELED = 0;   COLOR_UNLABELED = [0.5, 0.5, 0.5]
LABEL_REBAR = 1;       COLOR_REBAR = [1.0, 0.0, 0.0]     # 鉄筋 (赤)
LABEL_WALL = 2;        COLOR_WALL = [0.0, 0.0, 1.0]      # 壁面 (青)
LABEL_EQUIPMENT = 3;   COLOR_EQUIPMENT = [0.0, 1.0, 0.0] # 装置 (緑)
LABEL_FLOOR = 5;       COLOR_FLOOR = [0.0, 1.0, 1.0]     # 床 (シアン)
HIGHLIGHT_RADIUS = 15.0; HIGHLIGHT_COLOR = [1.0, 0.0, 1.0]

class TwoStageAnnotator:
    def __init__(self):
        self.points = None; self.labels = None; self.pcd = o3d.geometry.PointCloud()
        self.vis = None; self.target_indices = None; self.current_target_pos = 0
        self.current_point_index = 0; self.highlight_sphere = None

    def run(self):
        if not self._load_data(): return
        self._run_bulk_annotation_phase()
        self._run_detailed_annotation_phase()
        print("\n===== 全てのアノテーションプロセスが完了しました =====")

    def _load_data(self):
        try:
            df = pd.read_csv(WORK_AREA_CSV_PATH, header=None, names=['x', 'y', 'z'])
            self.points = df[['x', 'y', 'z']].values
            self.pcd.points = o3d.utility.Vector3dVector(self.points)
        except FileNotFoundError:
            print(f"エラー: '{WORK_AREA_CSV_PATH}' が見つかりません。"); return False
        if os.path.exists(LABELS_FILE_PATH):
            print(f"'{LABELS_FILE_PATH}' から既存のラベルを読み込みました。")
            self.labels = np.load(LABELS_FILE_PATH)
        else:
            print("新しいラベルファイルを作成します。")
            self.labels = np.full(len(self.points), LABEL_UNLABELED, dtype=int)
        return True

    def _update_pcd_colors(self):
        colors = np.array([COLOR_UNLABELED] * len(self.points))
        colors[self.labels == LABEL_REBAR] = COLOR_REBAR
        colors[self.labels == LABEL_WALL] = COLOR_WALL
        colors[self.labels == LABEL_EQUIPMENT] = COLOR_EQUIPMENT
        colors[self.labels == LABEL_FLOOR] = COLOR_FLOOR
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def _run_bulk_annotation_phase(self):
        while True:
            print("\n" + "="*20); print("  フェーズ1: 一括ラベリング"); print("="*20)
            self._update_pcd_colors()
            vis_select = o3d.visualization.VisualizerWithVertexSelection()
            vis_select.create_window(window_name="フェーズ1: 範囲選択 (Shift+Drag)", width=1600, height=1200)
            vis_select.add_geometry(self.pcd)
            print("\n--- 操作方法 (フェーズ1) ---")
            print("1. [Shift]+ドラッグで、ラベル付けしたい範囲を選択します（選択点は赤くなります）。")
            print("2. 選択が終わったら、このウィンドウを閉じてください。")
            print("--------------------------")
            vis_select.run()
            picked_points_info = vis_select.get_picked_points()
            vis_select.destroy_window()
            if len(picked_points_info) > 0:
                selected_indices = [info.index for info in picked_points_info]
                print(f"\n{len(selected_indices)} 点が選択されました。")
                while True:
                    # ▼▼▼【変更点】選択肢を4クラスに変更 ▼▼▼
                    label_choice = input("この範囲にどのラベルを付けますか？ (1:鉄筋, 2:壁面, 3:装置, 5:床, C:キャンセル) > ")
                    if label_choice in ["1", "2", "3", "5"]:
                        self.labels[selected_indices] = int(label_choice)
                        print(f"選択範囲にラベル {label_choice} を付与しました。")
                        break
                    elif label_choice.upper() == 'C':
                        print("ラベリングをキャンセルしました。"); break
                    else:
                        print("無効な入力です。")
            else:
                print("\n点が選択されませんでした。")
            self._print_label_summary()
            user_input = input("\n続けて範囲選択を行いますか？ (Y:はい / N:いいえ、個別作業へ) > ")
            if user_input.upper() == 'N':
                print("\nフェーズ1を終了し、フェーズ2へ移行します。")
                self._save_labels(); return

    def _run_detailed_annotation_phase(self):
        print("\n" + "="*20); print("  フェーズ2: 個別ラベリング"); print("="*20)
        self.target_indices = np.where(self.labels == LABEL_UNLABELED)[0]
        if len(self.target_indices) == 0:
            print("未分類の点はありません。アノテーションは完了しています。"); return
        print(f"個別アノテーションの対象は、残りの {len(self.target_indices)} 点です。")
        self.current_target_pos = 0; self.current_point_index = self.target_indices[self.current_target_pos]
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="フェーズ2: 個別ラベリング", width=1600, height=1200)
        self.vis.add_geometry(self.pcd)
        
        # ▼▼▼【変更点】キー操作を4クラスに変更 ▼▼▼
        key_to_callback = {
            ord("1"): lambda v: self._set_label_detailed(LABEL_REBAR),
            ord("2"): lambda v: self._set_label_detailed(LABEL_WALL),
            ord("3"): lambda v: self._set_label_detailed(LABEL_EQUIPMENT),
            ord("5"): lambda v: self._set_label_detailed(LABEL_FLOOR),
            ord("N"): self._next_point, ord("B"): self._prev_point,
            ord("C"): self._center_view, ord("S"): lambda v: self._save_labels(),
            ord("Q"): self._quit
        }
        for key, callback in key_to_callback.items():
            self.vis.register_key_callback(key, callback)
        self._update_colors_detailed()
        print("\n--- 操作方法 (フェーズ2) ---")
        print("1:鉄筋, 2:壁面, 3:装置, 5:床 | N:次へ | B:前へ | C:視点移動 | S:保存 | Q:終了")
        print("--------------------------")
        self.vis.run()

    def _update_colors_detailed(self, update_geometry=True):
        self._update_pcd_colors()
        self.pcd.colors[self.current_point_index] = HIGHLIGHT_COLOR
        if update_geometry: self.vis.update_geometry(self.pcd)
        self._update_highlight_sphere()
        self._print_label_summary(detailed=True)

    def _update_highlight_sphere(self):
        if self.highlight_sphere is not None: self.vis.remove_geometry(self.highlight_sphere, False)
        center_point = self.points[self.current_point_index]
        self.highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=HIGHLIGHT_RADIUS)
        self.highlight_sphere.paint_uniform_color(HIGHLIGHT_COLOR)
        self.highlight_sphere.translate(center_point)
        self.vis.add_geometry(self.highlight_sphere, False)

    def _print_label_summary(self, detailed=False):
        rebar = np.sum(self.labels == LABEL_REBAR)
        wall = np.sum(self.labels == LABEL_WALL)
        equipment = np.sum(self.labels == LABEL_EQUIPMENT)
        floor = np.sum(self.labels == LABEL_FLOOR)
        unlabeled = np.sum(self.labels == LABEL_UNLABELED)
        if detailed:
            status_str = f"\r進捗: {self.current_target_pos + 1}/{len(self.target_indices)} | 1:鉄筋({rebar}) 2:壁面({wall}) 3:装置({equipment}) 5:床({floor}) 未分類({unlabeled})"
        else:
            status_str = f"現在のラベル数 -> 1:鉄筋({rebar}) 2:壁面({wall}) 3:装置({equipment}) 5:床({floor}) 未分類({unlabeled})"
        print(status_str, end="")

    def _set_label_detailed(self, label):
        self.labels[self.current_point_index] = label; self._next_point(self.vis)
    def _next_point(self, vis):
        if self.current_target_pos + 1 < len(self.target_indices):
            self.current_target_pos += 1; self.current_point_index = self.target_indices[self.current_target_pos]; self._update_colors_detailed()
    def _prev_point(self, vis):
        if self.current_target_pos > 0:
            self.current_target_pos -= 1; self.current_point_index = self.target_indices[self.current_target_pos]; self._update_colors_detailed()
    def _center_view(self, vis):
        ctr = self.vis.get_view_control(); ctr.set_lookat(self.points[self.current_point_index]); print("\n"); self._print_label_summary(detailed=True)
    def _save_labels(self):
        np.save(LABELS_FILE_PATH, self.labels); print(f"\nラベルを '{LABELS_FILE_PATH}' に保存しました。"); self._print_label_summary()
    def _quit(self, vis):
        self._save_labels(); vis.destroy_window()

if __name__ == '__main__':
    annotator = TwoStageAnnotator()
    annotator.run()