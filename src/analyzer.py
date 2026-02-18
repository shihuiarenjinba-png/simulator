import pandas as pd
import numpy as np

class EconomicAnalyzer:
    def __init__(self, df):
        """
        :param df: data_loaderによって結合された、全指標を含むDataFrame
        """
        self.df = df
        
        # 分析に使用する説明変数（Fama-French 5 Factors）
        # data_loaderのカラム名に合わせて調整してください
        self.factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        
        # 分析結果を格納する辞書
        self.results = {}

    def find_optimal_lags(self, target_col, max_lag=12):
        """
        指定されたターゲット（例: 'DI_Coincident'）に対して、
        各ファクターが「何ヶ月前に」最も強く相関するか（最適ラグ）を探索する。
        
        :param target_col: 分析対象の指標名（例: 'DI_Coincident', 'VIX'）
        :param max_lag: 探索する最大ラグ（月数）。デフォルトは1年（12ヶ月）。
        :return: {ファクター名: {'lag': 最適ラグ, 'corr': 相関係数}} の辞書
        """
        if target_col not in self.df.columns:
            print(f"Error: Target column '{target_col}' not found in dataset.")
            return None

        print(f"Analyzing optimal lags for target: {target_col} ...")
        
        best_lags = {}

        for factor in self.factors:
            if factor not in self.df.columns:
                continue

            best_corr = 0.0
            best_lag = 0
            
            # ラグ0ヶ月〜max_lagヶ月までを総当たりでチェック
            for lag in range(max_lag + 1):
                # ファクターをlagヶ月分だけ「過去」にずらして、現在のターゲットと比較する
                # shift(lag): データを行方向にずらす。
                # 例: lag=3 の場合、「3ヶ月前のファクター値」と「今の景気」の相関を見る
                
                # 欠損値を除外して計算（重要）
                series_factor_lagged = self.df[factor].shift(lag)
                valid_idx = series_factor_lagged.notna() & self.df[target_col].notna()
                
                if valid_idx.sum() < 24: # データ点数が少なすぎる場合はスキップ（最低2年分）
                    continue

                corr = self.df.loc[valid_idx, target_col].corr(series_factor_lagged[valid_idx])
                
                # 「相関の絶対値」が最大になるラグを採用する
                # （強い逆相関も重要なシグナルのため）
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            
            best_lags[factor] = {
                'optimal_lag': best_lag,
                'correlation': best_corr
            }
            
        return best_lags

    def analyze_multi_targets(self, target_list):
        """
        複数のターゲット（リスト形式）を受け取り、それぞれの分析結果を一括で行う。
        
        :param target_list: 例 ['DI_Coincident', 'VIX', 'USD_JPY']
        :return: 全体の分析結果辞書
        """
        print(f"Starting Multi-Target Analysis for: {target_list}")
        
        for target in target_list:
            lag_result = self.find_optimal_lags(target)
            if lag_result:
                self.results[target] = lag_result
        
        return self.results

    def get_lagged_dataset(self, target_col):
        """
        【モデリング用】
        特定した最適ラグを適用して、回帰分析にすぐに使える形のDataFrameを作成して返す。
        Modeler（次の工程）で使用する。
        """
        if target_col not in self.results:
            print(f"Analysis for {target_col} not found. Run analyze_multi_targets first.")
            return None
            
        lag_info = self.results[target_col]
        df_shifted = pd.DataFrame(index=self.df.index)
        
        # 目的変数（ターゲット）はそのまま
        df_shifted[target_col] = self.df[target_col]
        
        # 説明変数（ファクター）を最適ラグ分だけずらして配置
        for factor, info in lag_info.items():
            lag = info['optimal_lag']
            col_name = f"{factor}_Lag{lag}"
            df_shifted[col_name] = self.df[factor].shift(lag)
            
        # ずらしたことで発生したNaNを削除
        return df_shifted.dropna()

# --- 動作確認用ブロック ---
if __name__ == "__main__":
    # ここではデータローダーを呼び出して実際に動くかテストします
    # ※ディレクトリ構造に合わせて import data_loader の記述が必要
    try:
        from data_loader import DataLoader
        
        # 1. データ取得
        loader = DataLoader()
        # テスト用に最新データ取得（ネット接続が必要）
        df = loader.get_merged_data()
        
        # 2. 分析エンジンの起動
        analyzer = EconomicAnalyzer(df)
        
        # 3. 複数ターゲットを指定して実行！
        # ユーザー様のご要望通り、DI, VIX, 為替をセット
        targets = ['DI_Coincident', 'VIX', 'USD_JPY']
        
        # カラム名が正しいかチェック（VIXやUSD_JPYがない場合の保険）
        available_targets = [t for t in targets if t in df.columns]
        
        results = analyzer.analyze_multi_targets(available_targets)
        
        # 4. 結果の表示
        import pprint
        print("\n--- Analysis Results (Optimal Lags) ---")
        pprint.pprint(results)
        
    except ImportError:
        print("data_loader.py not found or dependencies missing.")
    except Exception as e:
        print(f"An error occurred: {e}")
