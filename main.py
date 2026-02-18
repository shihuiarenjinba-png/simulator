import sys
import os
import pandas as pd

# ---------------------------------------------------------
# 【設定エリア】分析したい指標をここで選択します
# ---------------------------------------------------------
# 複数の「窓」をここで定義します。
# 選択肢の例: 'DI_Coincident', 'CI_Coincident', 'VIX', 'USD_JPY'
TARGET_INDICATORS = [
    'DI_Coincident',  # 景気動向指数（一致指数）
    'VIX',            # 恐怖指数
    'USD_JPY'         # ドル円為替
]

# ---------------------------------------------------------
# 1. パス設定 (モジュールの読み込みを確実にする)
# ---------------------------------------------------------
# 現在のフォルダの 'src' をPythonの検索パスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from src.data_loader import DataLoader
    from src.analyzer import EconomicAnalyzer
    from src.modeler import EconomicModeler
except ImportError as e:
    print(f"\n[CRITICAL ERROR] モジュールの読み込みに失敗しました。")
    print(f"詳細: {e}")
    print("ヒント: 仮想環境が有効か、srcフォルダ内にファイルがあるか確認してください。")
    sys.exit(1)

def main():
    print("\n" + "="*60)
    print("  Economic Factor Attribution System - MAIN SEQUENCE")
    print("="*60)

    # =========================================================
    # STEP 1: データの収集と統合 (Data Loading)
    # =========================================================
    print("\n[STEP 1] データの収集と統合を開始します...")
    try:
        loader = DataLoader()
        # 内閣府、FF5、市場データを全て取得・結合
        df_master = loader.get_merged_data()
        
        if df_master is None or df_master.empty:
            print("[ERROR] データが取得できませんでした。プログラムを終了します。")
            return
            
        print(f"  -> 成功: {len(df_master)} ヶ月分のデータを確保しました。")
        print(f"  -> 期間: {df_master.index.min().date()} ～ {df_master.index.max().date()}")

    except Exception as e:
        print(f"[ERROR] データ収集プロセスで予期せぬエラーが発生しました: {e}")
        return

    # =========================================================
    # STEP 2: リード・ラグ分析 (Lag Analysis)
    # =========================================================
    print("\n[STEP 2] 最適ラグ（時間差）の特定を開始します...")
    try:
        analyzer = EconomicAnalyzer(df_master)
        
        # データセットに存在するターゲットだけをフィルタリング（安全策）
        valid_targets = [t for t in TARGET_INDICATORS if t in df_master.columns]
        
        if not valid_targets:
            print("[ERROR] 指定されたターゲット指標がデータ内に見つかりません。")
            print(f"  利用可能な列: {df_master.columns.tolist()}")
            return

        # マルチターゲット分析の実行
        lag_results = analyzer.run_multi_target_analysis(valid_targets)
        
        # 簡易結果表示
        print(f"  -> {len(lag_results)} つの指標について最適ラグを特定しました。")

    except Exception as e:
        print(f"[ERROR] 分析プロセスでエラーが発生しました: {e}")
        return

    # =========================================================
    # STEP 3: モデリングと寄与度算出 (Modeling & Attribution)
    # =========================================================
    print("\n[STEP 3] 重回帰分析による寄与度分解を開始します...")
    
    # 成功したターゲットごとに、寄与度を計算してレポートを出力
    for target_col in lag_results.keys():
        try:
            print(f"\n[{target_col} 分析開始] ----------------------------------")
            
            # 1. データの整形（最適ラグを適用して時間をずらす）
            df_aligned = analyzer.get_aligned_dataset(target_col)
            
            # 2. モデラーの初期化（慎重にデータセットを渡す）
            modeler = EconomicModeler(df_aligned)
            
            # 3. 学習（寄与度計算）
            modeler.train_model()
            
            # 4. レポート表示
            modeler.print_summary(target_col)
            
        except Exception as e:
            print(f"  [WARNING] {target_col} の分析中に問題が発生しました: {e}")
            continue # 一つのエラーで全部止めず、次へ進む

    print("\n" + "="*60)
    print("  すべてのプロセスが正常に完了しました。")
    print("="*60)

if __name__ == "__main__":
    main()
