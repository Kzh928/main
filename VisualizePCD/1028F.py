# ファイル名: step2.6_reclassify_non_rebar_v3.py
# 概要: 「非鉄筋(2)」を「壁面(2)」「装置(3)」「床(5)」に再分類する。
#       【v3】ラベル番号の重複による集計エラーを修正した最終版。
#       【バグ修正】個別作業で処理済みの点に戻って再ラベル付けした際のエラーを修正。

import open3d as o3d
import numpy as np
import pandas as pd
import os

# --- ★★★ ユーザー設定項目 ★★★ ---
WORK_AREA_CSV_PATH = 'work_area_large.csv'
INPUT_LABELS_PATH = 'annotated_labels_negatives.npy'
OUTPUT_LABELS_PATH = 'annotated_labels_4class_F.npy'
# --- 新しいラベル体系の定義 ---
LABEL_UNLABELED = 0;   COLOR_UNLABELED = [0.6, 0.6, 0.6]
LABEL_REBAR = 1;       COLOR_REBAR = [1.0, 0.0, 0.0]
LABEL_WALL = 2;        COLOR_WALL = [0.0, 0.0, 1.0]
LABEL_EQUIPMENT = 3;   COLOR_EQUIPMENT = [0.0, 1.0, 0.0]
LABEL_FLOOR = 5;       COLOR_FLOOR = [0.0, 1.0, 1.0] # シアン
HIGHLIGHT_RADIUS = 15.0; HIGHLIGHT_COLOR = [1.0, 0.0, 1.0]

class RelabelAnnotator:
    def __init__(self):
        self.points = None; self.labels = None; self.pcd = o3d.geometry.PointCloud()
        self.vis = None; self.target_indices = None; self.current_target_pos = 0
        self.current_point_index = 0; self.highlight_sphere = None
        self.OLD_NOT_REBAR_LABEL = 2
        self.unprocessed_indices = set()

    def run(self):
        if not self._load_data(): return
        self._run_bulk_relabel_phase()
        self._run_detailed_relabel_phase()
        print("\n===== 全ての再分類プロセスが完了しました =====")

    def _load_data(self):
        try:
            df = pd.read_csv(WORK_AREA_CSV_PATH, header=None, names=['x', 'y', 'z'])
            self.points = df[['x', 'y', 'z']].values
            self.pcd.points = o3d.utility.Vector3dVector(self.points)
        except FileNotFoundError: print(f"エラー: '{WORK_AREA_CSV_PATH}' が見つかりません。"); return False
        try:
            print(f"'{INPUT_LABELS_PATH}' から既存のラベルを読み込みました。")
            self.labels = np.load(INPUT_LABELS_PATH)
            self.unprocessed_indices = set(np.where(self.labels == self.OLD_NOT_REBAR_LABEL)[0])
        except FileNotFoundError: print(f"エラー: '{INPUT_LABELS_PATH}' が見つかりません。"); return False
        return True

    def _update_pcd_colors(self):
        colors = np.full((len(self.points), 3), COLOR_UNLABELED)
        colors[self.labels == LABEL_REBAR] = COLOR_REBAR
        colors[self.labels == LABEL_WALL] = COLOR_WALL
        colors[self.labels == LABEL_EQUIPMENT] = COLOR_EQUIPMENT
        colors[self.labels == LABEL_FLOOR] = COLOR_FLOOR
        colors[list(self.unprocessed_indices)] = COLOR_WALL # 未処理の点は青でハイライト
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def _run_bulk_relabel_phase(self):
        while True:
            print("\n" + "="*25); print("  フェーズ1: 一括再分類"); print("="*25)
            self._update_pcd_colors()
            vis_select = o3d.visualization.VisualizerWithVertexSelection()
            vis_select.create_window(window_name="フェーズ1: 非鉄筋(青)の範囲を選択 (Shift+Drag)", width=1600, height=1200)
            vis_select.add_geometry(self.pcd)
            print("\n--- 操作方法 (フェーズ1) ---")
            print("1. 青色で表示されている「未処理の非鉄筋」の中から、分類し直したい範囲を選択します。")
            print("2. 選択が終わったら、このウィンドウを閉じてください。")
            print("--------------------------")
            vis_select.run()
            picked_points_info = vis_select.get_picked_points()
            vis_select.destroy_window()

            if len(picked_points_info) > 0:
                selected_indices = {info.index for info in picked_points_info}
                final_indices_to_relabel = list(selected_indices.intersection(self.unprocessed_indices))
                
                if not final_indices_to_relabel:
                    print("\n選択範囲に、再分類対象の「未処理の非鉄筋(青)」が含まれていませんでした。")
                else:
                    print(f"\n選択範囲内の {len(final_indices_to_relabel)} 点の「非鉄筋」を再分類します。")
                    while True:
                        label_choice = input("この範囲をどれに分類しますか？ (2:壁面, 3:装置, 5:床, C:キャンセル) > ")
                        if label_choice in ["2", "3", "5"]:
                            self.labels[final_indices_to_relabel] = int(label_choice)
                            self.unprocessed_indices -= set(final_indices_to_relabel)
                            print(f"選択範囲に新しいラベル {label_choice} を付与しました。")
                            break
                        elif label_choice.upper() == 'C': print("キャンセルしました。"); break
                        else: print("無効な入力です。")
            else:
                print("\n点が選択されませんでした。")
            self._print_label_summary()
            user_input = input("\n続けて範囲選択を行いますか？ (Y:はい / N:いいえ、個別作業へ) > ")
            if user_input.upper() == 'N':
                print("\nフェーズ1を終了し、フェーズ2へ移行します。")
                self._save_labels(); return

    def _run_detailed_relabel_phase(self):
        print("\n" + "="*25); print("  フェーズ2: 個別再分類"); print("="*25)
        self.target_indices = list(self.unprocessed_indices)
        if not self.target_indices:
            print("再分類対象の点はありません。アノテーションは完了しています。")
            self._save_labels(); return
        print(f"個別アノテーションの対象は、残りの {len(self.target_indices)} 点です。")
        self.current_target_pos = 0; self.current_point_index = self.target_indices[self.current_target_pos]
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="フェーズ2: 個別再分類", width=1600, height=1200)
        self.vis.add_geometry(self.pcd)
        key_to_callback = {ord("2"): lambda v: self._set_label_detailed(LABEL_WALL), ord("3"): lambda v: self._set_label_detailed(LABEL_EQUIPMENT), ord("5"): lambda v: self._set_label_detailed(LABEL_FLOOR), ord("N"): self._next_point, ord("B"): self._prev_point, ord("C"): self._center_view, ord("S"): lambda v: self._save_labels(), ord("Q"): self._quit}
        for key, callback in key_to_callback.items():
            self.vis.register_key_callback(key, callback)
        self._update_colors_detailed()
        print("\n--- 操作方法 (フェーズ2) ---"); print("2:壁面, 3:装置, 5:床 | N:次へ | B:前へ | C:視点移動 | S:保存 | Q:終了"); print("--------------------------")
        self.vis.run()

    def _update_colors_detailed(self, update_geometry=True):
        self._update_pcd_colors()
        if self.target_indices:
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
        unprocessed_count = len(self.unprocessed_indices)
        if detailed:
            status_str = f"\r進捗: {self.current_target_pos + 1}/{len(self.target_indices)} | 壁面:{wall}, 装置:{equipment}, 床:{floor} | (未処理:{unprocessed_count})"
        else:
            print(f"\n現在のラベル数 -> 鉄筋:{rebar}, 壁面:{wall}, 装置:{equipment}, 床:{floor}, (未処理:{unprocessed_count})")

    def _set_label_detailed(self, label):
        self.labels[self.current_point_index] = label
        # ▼▼▼【バグ修正】 .remove() から .discard() に変更 ▼▼▼
        self.unprocessed_indices.discard(self.current_point_index)
        self._next_point(self.vis)

    def _next_point(self, vis):
        if self.current_target_pos + 1 < len(self.target_indices):
            self.current_target_pos += 1
            self.current_point_index = self.target_indices[self.current_target_pos]
            self._update_colors_detailed()
    def _prev_point(self, vis):
        if self.current_target_pos > 0:
            self.current_target_pos -= 1
            self.current_point_index = self.target_indices[self.current_target_pos]
            self._update_colors_detailed()
    def _center_view(self, vis):
        ctr = self.vis.get_view_control(); ctr.set_lookat(self.points[self.current_point_index])
    def _save_labels(self):
        np.save(OUTPUT_LABELS_PATH, self.labels)
        print(f"\nラベルを '{OUTPUT_LABELS_PATH}' に保存しました。"); self._print_label_summary()
    def _quit(self, vis):
        self._save_labels(); vis.destroy_window()

if __name__ == '__main__':
    annotator = RelabelAnnotator()
    annotator.run()