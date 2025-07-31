# ファイル名: step2.4_two_stage_annotator.py
# 概要: 範囲選択による一括ラベリングと、残りの点の個別ラベリングを組み合わせた高効率アノテーター。

import open3d as o3d
import numpy as np
import pandas as pd
import os

# --- ★★★ ユーザー設定項目 ★★★ ---
# 今回は全体（work_area_large.csv）を対象にするのがオススメ
WORK_AREA_CSV_PATH = 'equipment_area1010_B.csv'
# ラベルファイルのパス (既存のものを読み込むか、新規作成されます)
LABELS_FILE_PATH = 'labels_for_equipment_1010_B.npy'

# --- ラベルと色の定義 (変更なし) ---
LABEL_UNLABELED = 0; COLOR_UNLABELED = [0.5, 0.5, 0.5]
LABEL_REBAR = 1;     COLOR_REBAR = [1.0, 0.0, 0.0]
LABEL_NOT_REBAR = 2; COLOR_NOT_REBAR = [0.0, 0.0, 1.0]
LABEL_EQUIPMENT = 3; COLOR_EQUIPMENT = [0.0, 1.0, 0.0]
HIGHLIGHT_RADIUS = 30.0; HIGHLIGHT_COLOR = [1.0, 0.0, 1.0]

class TwoStageAnnotator:
    def __init__(self):
        self.points = None
        self.labels = None
        self.pcd = o3d.geometry.PointCloud()
        self.vis = None
        self.target_indices = None
        self.current_target_pos = 0
        self.current_point_index = 0
        self.highlight_sphere = None

    def run(self):
        if not self._load_data(): return
        self._run_bulk_annotation_phase()
        self._run_detailed_annotation_phase()
        print("\n===== 全てのアノテーションプロセスが完了しました =====")

    def _load_data(self):
        try:
            df = pd.read_csv(WORK_AREA_CSV_PATH, header=None, names=['x', 'y', 'z', 'intensity'])
            self.points = df[['x', 'y', 'z']].values
            self.pcd.points = o3d.utility.Vector3dVector(self.points)
        except FileNotFoundError:
            print(f"エラー: '{WORK_AREA_CSV_PATH}' が見つかりません。")
            return False

        if os.path.exists(LABELS_FILE_PATH):
            print(f"'{LABELS_FILE_PATH}' から既存のラベルを読み込みました。")
            self.labels = np.load(LABELS_FILE_PATH)
        else:
            print("新しいラベルファイルを作成します。")
            self.labels = np.full(len(self.points), LABEL_UNLABELED, dtype=int)
        return True

    def _update_pcd_colors(self):
        """点群全体の色を現在のラベル状態に合わせて更新"""
        colors = np.array([COLOR_UNLABELED] * len(self.points))
        colors[self.labels == LABEL_REBAR] = COLOR_REBAR
        colors[self.labels == LABEL_NOT_REBAR] = COLOR_NOT_REBAR
        colors[self.labels == LABEL_EQUIPMENT] = COLOR_EQUIPMENT
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    # --- フェーズ1: 一括ラベリング ---
    def _run_bulk_annotation_phase(self):
        while True:
            print("\n" + "="*20)
            print("  フェーズ1: 一括ラベリング")
            print("="*20)
            self._update_pcd_colors()

            vis_select = o3d.visualization.VisualizerWithVertexSelection()
            vis_select.create_window(window_name="フェーズ1: 範囲選択 (Shift+Drag)", width=1600, height=1200)
            vis_select.add_geometry(self.pcd)
            print("\n--- 操作方法 (フェーズ1) ---")
            print("1. [Shift]キーを押しながらマウスでドラッグし、一括でラベル付けしたい範囲を選択します（選択された点は赤くなります）。")
            print("2. 選択が終わったら、このウィンドウを閉じてください。")
            print("--------------------------")
            vis_select.run()

            picked_points_info = vis_select.get_picked_points()
            vis_select.destroy_window()

            if len(picked_points_info) == 0:
                print("\n点が選択されませんでした。")
            else:
                selected_indices = [info.index for info in picked_points_info]
                print(f"\n{len(selected_indices)} 点が選択されました。")
                while True:
                    label_choice = input("この範囲にどのラベルを付けますか？ (1:鉄筋, 2:非鉄筋, 3:装置, C:キャンセル) > ")
                    if label_choice in ["1", "2", "3"]:
                        self.labels[selected_indices] = int(label_choice)
                        print(f"選択範囲にラベル {label_choice} を付与しました。")
                        break
                    elif label_choice.upper() == 'C':
                        print("ラベリングをキャンセルしました。")
                        break
                    else:
                        print("無効な入力です。1, 2, 3, C のいずれかを入力してください。")

            self._print_label_summary()
            
            while True:
                user_input = input("続けて範囲選択を行いますか？ (Y:はい / N:いいえ、個別作業へ進む) > ")
                if user_input.upper() == 'Y':
                    break
                elif user_input.upper() == 'N':
                    print("\nフェーズ1を終了し、フェーズ2（個別ラベリング）へ移行します。")
                    self._save_labels()
                    return
                else:
                    print("無効な入力です。YかNを入力してください。")

    # --- フェーズ2: 個別ラベリング ---
    def _run_detailed_annotation_phase(self):
        print("\n" + "="*20)
        print("  フェーズ2: 個別ラベリング")
        print("="*20)
        
        # ▼▼▼【重要】まだラベルが付いていない点だけを対象にする ▼▼▼
        self.target_indices = np.where(self.labels == LABEL_UNLABELED)[0]
        
        if len(self.target_indices) == 0:
            print("未分類の点はありません。アノテーションは完了しています。")
            return

        print(f"個別アノテーションの対象は、残りの {len(self.target_indices)} 点です。")
        self.current_target_pos = 0
        self.current_point_index = self.target_indices[self.current_target_pos]

        # ビジュアライザのセットアップと実行
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="フェーズ2: 個別ラベリング", width=1600, height=1200)
        self.vis.add_geometry(self.pcd)
        
        key_to_callback = {
            ord("1"): lambda v: self._set_label_detailed(LABEL_REBAR),
            ord("2"): lambda v: self._set_label_detailed(LABEL_NOT_REBAR),
            ord("3"): lambda v: self._set_label_detailed(LABEL_EQUIPMENT),
            ord("N"): self._next_point, ord("B"): self._prev_point,
            ord("C"): self._center_view, ord("S"): lambda v: self._save_labels(),
            ord("Q"): self._quit
        }
        for key, callback in key_to_callback.items():
            self.vis.register_key_callback(key, callback)

        self._update_colors_detailed()

        print("\n--- 操作方法 (フェーズ2) ---")
        print("1,2,3: ラベル付け | N:次へ | B:前へ | C:視点移動 | S:保存 | Q:終了")
        print("--------------------------")
        self.vis.run()

    # --- フェーズ2で使うヘルパー関数群 (step2.3のものを流用・一部改変) ---
    def _update_colors_detailed(self, update_geometry=True):
        self._update_pcd_colors()
        # 注目点の色だけ上書き
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
        not_rebar = np.sum(self.labels == LABEL_NOT_REBAR)
        equipment = np.sum(self.labels == LABEL_EQUIPMENT)
        unlabeled = np.sum(self.labels == LABEL_UNLABELED)
        if detailed:
            status_str = f"\r進捗: {self.current_target_pos + 1}/{len(self.target_indices)} | 注目点ID: {self.current_point_index} | 鉄筋(1):{rebar} 非鉄筋(2):{not_rebar} 装置(3):{equipment} 未分類:{unlabeled}"
        else:
            status_str = f"現在のラベル数 -> 鉄筋(1):{rebar}, 非鉄筋(2):{not_rebar}, 装置(3):{equipment}, 未分類:{unlabeled}"
        print(status_str, end="")

    def _set_label_detailed(self, label):
        self.labels[self.current_point_index] = label
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
        ctr = self.vis.get_view_control()
        ctr.set_lookat(self.points[self.current_point_index])
        print("\n視点を注目点に移動しました。"); self._print_label_summary(detailed=True)

    def _save_labels(self):
        np.save(LABELS_FILE_PATH, self.labels)
        print(f"\nラベルを '{LABELS_FILE_PATH}' に保存しました。")
        self._print_label_summary()

    def _quit(self, vis):
        self._save_labels()
        vis.destroy_window()

if __name__ == '__main__':
    annotator = TwoStageAnnotator()
    annotator.run()
