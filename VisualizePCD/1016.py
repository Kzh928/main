# ファイル名: plot_training_log.py
# 概要: 学習スクリプトが出力したログCSVファイルを読み込み、
#       損失と精度の推移を折れ線グラフとして表示・保存する。

import pandas as pd
import matplotlib.pyplot as plt

# --- ★★★ ユーザー設定項目 ★★★ ---

# グラフ化したいログCSVファイルのパス
LOG_FILE_PATH = 'training_log_final_robust_1010.csv'
# グラフを画像として保存する場合のファイル名
OUTPUT_IMAGE_PATH = 'training_progress_graph_1010.png'

# --- メイン処理 ---
if __name__ == '__main__':
    # 1. ログファイルを読み込み
    try:
        df = pd.read_csv(LOG_FILE_PATH)
        print(f"'{LOG_FILE_PATH}' を読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ログファイル '{LOG_FILE_PATH}' が見つかりません。")
        exit()

    # 2. グラフの作成
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 3. 損失のグラフ（左側のY軸）
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(df['epoch'], df['loss'], color='tab:blue', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # 4. 精度のグラフ（右側のY軸）
    ax2 = ax1.twinx()  # 2つ目のY軸を共有
    ax2.set_ylabel('Accuracy (%)', color='tab:red')
    ax2.plot(df['epoch'], df['accuracy'], color='tab:red', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 5. グラフのタイトルと凡例
    plt.title('Training Progress (Loss and Accuracy)')
    fig.tight_layout() # レイアウトを調整
    
    # 6. グラフを画像として保存
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"グラフを '{OUTPUT_IMAGE_PATH}' に保存しました。")

    # 7. グラフを画面に表示
    print("グラフをウィンドウに表示します。")
    plt.show()
