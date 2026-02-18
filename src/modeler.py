import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class EconomicModeler:
    def __init__(self, df):
        """
        :param df: analyzerによってラグ調整されたデータフレーム
                   (最後の列がターゲット、それ以外がファクターと想定)
        """
        self.df = df
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.results = {}

    def train_model(self):
        """
        【慎重実装】重回帰モデルの学習と寄与度算出
        """
        # データが空でないかチェック
        if self.df is None or len(self.df) < 12:
            print("Error: Not enough data to train model (minimum 12 months required).")
            return None

        # 1. データ分割 (X: 説明変数, y: 目的変数)
        # 最後の列をターゲット(y)、それ以外をファクター(X)とする
        X = self.df.iloc[:, 1:]  # 1列目以降（ラグ付きファクター）
        y = self.df.iloc[:, 0]   # 0列目（ターゲット: DIなど）
        
        feature_names = X.columns.tolist()

        # 2. データの標準化 (Standardization)
        # これを行わないと、単位が大きい変数(HMLなど)の寄与度が歪んでしまう
        X_scaled = self.scaler.fit_transform(X)
        
        # yも標準化することで、係数の解釈を「標準化偏回帰係数」にする
        # yは1次元配列なので reshape が必要
        y_scaled = StandardScaler().fit_transform(y.values.reshape(-1, 1)).flatten()

        # 3. モデル学習 (OLS: 最小二乗法)
        self.model.fit(X_scaled, y_scaled)

        # 4. 統計情報の抽出
        # 決定係数 (R^2): モデルがどれくらい「景気」を説明できているか (max 1.0)
        r_squared = self.model.score(X_scaled, y_scaled)
        
        # 偏回帰係数 (Coefficients): これが各ファクターの「寄与度（重み）」
        coefficients = self.model.coef_
        
        # 5. 結果のまとめ
        # 寄与度を絶対値の大きさ順にソートして見やすくする
        contributions = []
        for name, coef in zip(feature_names, coefficients):
            contributions.append({
                'factor': name,
                'weight': coef,             # 係数（プラスなら景気押し上げ、マイナスなら押し下げ）
                'impact': abs(coef)         # 影響力の大きさ（絶対値）
            })
        
        # 影響力の大きい順に並び替え
        contributions.sort(key=lambda x: x['impact'], reverse=True)

        self.results = {
            'r_squared': r_squared,
            'intercept': self.model.intercept_,
            'contributions': contributions
        }

        return self.results

    def print_summary(self, target_name):
        """
        分析結果をレポート形式で表示
        """
        if not self.results:
            print("No model trained yet.")
            return

        print(f"\n{'='*50}")
        print(f"  MODEL REPORT: {target_name}")
        print(f"{'='*50}")
        
        r2 = self.results['r_squared']
        print(f"Model Accuracy (R^2): {r2:.4f}")
        if r2 > 0.5:
            print("  -> Excellent! The factors explain the movement well.")
        elif r2 > 0.3:
            print("  -> Good. There is a meaningful relationship.")
        else:
            print("  -> Low. Other external factors might be stronger.")
            
        print(f"\n[ Factor Contributions (Impact) ]")
        print(f"{'Factor Name':<15} | {'Weight':<8} | {'Influence'}")
        print(f"{'-'*15}-+-{'-'*8}-+-{'-'*20}")
        
        for item in self.results['contributions']:
            name = item['factor']
            weight = item['weight']
            
            # ビジュアルバーの作成
            bar_len = int(abs(weight) * 10)
            bar = ('█' * bar_len)
            direction = "POS" if weight > 0 else "NEG" # Positive / Negative
            
            print(f"{name:<15} | {weight:>8.3f} | {direction} {bar}")

# --- 動作確認用ブロック ---
if __name__ == "__main__":
    try:
        from data_loader import DataLoader
        from analyzer import EconomicAnalyzer
        
        print("1. Loading Data...")
        loader = DataLoader()
        df_master = loader.get_merged_data()
        
        print("2. Analyzing Lags...")
        analyzer = EconomicAnalyzer(df_master)
        
        # テスト対象: DI一致指数
        target = 'DI_Coincident'
        analyzer.find_optimal_lags(target)
        
        # モデリング用のデータセットを取得（ここが重要）
        df_model_input = analyzer.get_aligned_dataset(target)
        
        print(f"3. Modeling '{target}' ...")
        modeler = EconomicModeler(df_model_input)
        modeler.train_model()
        modeler.print_summary(target)

    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure 'scikit-learn' is installed via pip.")
    except Exception as e:
        print(f"An error occurred: {e}")
