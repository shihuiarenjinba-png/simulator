import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import io
import datetime

from src.factor_library import load_factor_dataset, load_portfolio_dataset

class DataLoader:
    def __init__(self, factor_source="Japan 5 Factors", portfolio_source="なし"):
        # データ取得開始日
        self.start_date = '1990-01-01'
        self.end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        self.factor_source = factor_source
        self.portfolio_source = portfolio_source
        
        # 内閣府 景気動向指数のページ
        self.cabinet_url = "https://www.esri.cao.go.jp/jp/stat/di/di.html"

    def fetch_cabinet_office_data(self):
        """
        【全自動】内閣府のウェブサイトから「長期系列」のエクセルを探索・ダウンロードし、
        DI・CI指数を抽出して整形する。
        """
        print(f"[{datetime.datetime.now().time()}] Accessing Cabinet Office website...")
        
        try:
            # 1. HTMLを取得して解析
            res = requests.get(self.cabinet_url)
            res.encoding = res.apparent_encoding
            soup = BeautifulSoup(res.text, 'html.parser')

            # 2. 「長期系列」かつ「.xlsx」を含むリンクを探す
            target_link = None
            for a in soup.find_all('a', href=True):
                # リンクテキストかファイル名に「長期系列」が含まれるものを探す
                if '長期系列' in a.text and 'xlsx' in a['href']:
                    target_link = a['href']
                    break
            
            if not target_link:
                print("Warning: 'Long-term series' Excel link not found. Trying fallback logic...")
                # 見つからない場合の予備ロジック（ファイル名に 'reig' が含まれることが多い）
                for a in soup.find_all('a', href=True):
                    if 'reig' in a['href'] and 'xlsx' in a['href']:
                        target_link = a['href']
                        break

            if not target_link:
                raise Exception("Target Excel link could not be found.")

            # URLの構築（相対パス対応）
            if not target_link.startswith('http'):
                base_url = "https://www.esri.cao.go.jp/jp/stat/di/"
                # hrefが "./" で始まる場合の処理
                file_name = target_link.split('/')[-1]
                target_url = base_url + file_name
            else:
                target_url = target_link

            print(f"Found Excel URL: {target_url}")

            # 3. エクセルをメモリ上にダウンロード
            file_res = requests.get(target_url)
            file_res.raise_for_status()
            
            # 4. Pandasで読み込み（ヘッダー等は無視して全読み込み）
            with io.BytesIO(file_res.content) as f:
                df = pd.read_excel(f, header=None)

            # --- データ整形ロジック ---
            # 戦略: 2列目(B列)に「数値の月(1~12)」が入っている行をデータ行とみなす
            # かつ、1列目(A列)に「西暦(19xx, 20xx)」が入っていること。
            
            # 数値変換を試みる
            df[1] = pd.to_numeric(df[1], errors='coerce') # 年
            df[2] = pd.to_numeric(df[2], errors='coerce') # 月

            # 年と月がNaNでない行を抽出
            data_rows = df.dropna(subset=[1, 2]).copy()
            
            # さらに「年」が1980以上であることを条件にする（誤検知防止）
            data_rows = data_rows[data_rows[1] > 1900]

            # 必要な列をピンポイントで抽出
            # 通常の長期系列Excelの列構成（0始まりのインデックス）:
            # Col 1: Year
            # Col 2: Month
            # Col 4: CI Coincident (CI一致指数)
            # Col 9: DI Coincident (DI一致指数)
            # ※エクセルの列がズレている場合に備えて確認が必要ですが、標準的にはこの位置です。
            
            target_cols = {
                1: 'Year',
                2: 'Month',
                4: 'CI_Coincident',
                9: 'DI_Coincident'
            }
            
            # 必要な列だけ抜き出し
            df_clean = data_rows[list(target_cols.keys())].rename(columns=target_cols)
            
            # 日付インデックス作成 (毎月1日とする)
            df_clean['Date'] = pd.to_datetime(
                df_clean['Year'].astype(int).astype(str) + '-' + 
                df_clean['Month'].astype(int).astype(str) + '-01'
            )
            df_clean.set_index('Date', inplace=True)
            
            # 不要列削除と数値化
            df_clean.drop(['Year', 'Month'], axis=1, inplace=True)
            df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
            
            print("Cabinet Office data fetched successfully.")
            return df_clean

        except Exception as e:
            print(f"Error fetching Cabinet Office data: {e}")
            return pd.DataFrame()

    def fetch_fama_french(self):
        """
        Kenneth French Data Library またはローカルファイルから 5 ファクターを取得
        """
        print(f"[{datetime.datetime.now().time()}] Fetching Fama-French 5 Factors ({self.factor_source})...")
        try:
            df_ff = load_factor_dataset(self.factor_source, start_date=self.start_date, end_date=self.end_date)

            print("Fama-French data fetched.")
            return df_ff
            
        except Exception as e:
            print(f"Error fetching Fama-French data: {e}")
            return pd.DataFrame()

    def fetch_market_data(self):
        """
        Yahoo FinanceからVIXと為替を取得
        """
        print(f"[{datetime.datetime.now().time()}] Fetching Market Indicators (VIX, USD/JPY)...")
        try:
            series_frames = []
            ticker_map = {
                "^VIX": "VIX",
                "JPY=X": "USD_JPY",
            }
            for ticker, label in ticker_map.items():
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                if df is None or df.empty:
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    if "Close" in df.columns.get_level_values(0):
                        df = df["Close"]
                    elif "Adj Close" in df.columns.get_level_values(0):
                        df = df["Adj Close"]

                if isinstance(df, pd.Series):
                    series = df
                elif "Close" in df.columns:
                    series = df["Close"]
                elif "Adj Close" in df.columns:
                    series = df["Adj Close"]
                else:
                    continue

                monthly = series.resample("MS").mean().to_frame(name=label)
                series_frames.append(monthly)

            if not series_frames:
                return pd.DataFrame()

            df_monthly = pd.concat(series_frames, axis=1)
            print("Market indicators fetched.")
            return df_monthly
            
        except Exception as e:
            print(f"Error fetching Market data: {e}")
            return pd.DataFrame()

    def fetch_portfolio_data(self):
        """
        ローカルの portfolio ファイルが指定されていれば取得
        """
        if not self.portfolio_source or self.portfolio_source == "なし":
            return pd.DataFrame()

        print(f"[{datetime.datetime.now().time()}] Fetching Portfolio Dataset ({self.portfolio_source})...")
        try:
            df_portfolio = load_portfolio_dataset(self.portfolio_source, start_date=self.start_date, end_date=self.end_date)
            print("Portfolio dataset fetched.")
            return df_portfolio
        except Exception as e:
            print(f"Error fetching Portfolio data: {e}")
            return pd.DataFrame()

    def get_merged_data(self):
        """
        【メイン実行関数】全てのデータを結合して返す
        """
        # 1. 各ソースからデータ取得
        df_cabinet = self.fetch_cabinet_office_data()
        df_ff = self.fetch_fama_french()
        df_market = self.fetch_market_data()
        df_portfolio = self.fetch_portfolio_data()

        if df_ff is None or df_ff.empty:
            return pd.DataFrame()

        final_df = df_ff.copy()
        for df in [df_market, df_cabinet, df_portfolio]:
            if df is not None and not df.empty:
                final_df = final_df.join(df, how='left')

        final_df = final_df[~final_df.index.duplicated(keep="first")].sort_index()

        # 水準値のままだとファクター回帰と相性が悪いので、変化率・前月差も作る
        if 'VIX' in final_df.columns:
            final_df['VIX_change_pct'] = final_df['VIX'].pct_change() * 100.0
        if 'USD_JPY' in final_df.columns:
            final_df['USD_JPY_change_pct'] = final_df['USD_JPY'].pct_change() * 100.0
        if 'DI_Coincident' in final_df.columns:
            final_df['DI_Coincident_change'] = final_df['DI_Coincident'].diff()
        if 'CI_Coincident' in final_df.columns:
            final_df['CI_Coincident_change'] = final_df['CI_Coincident'].diff()
        
        print(f"Data merge complete. Final shape: {final_df.shape}")
        print(f"Data range: {final_df.index.min()} to {final_df.index.max()}")
        
        return final_df

# --- 動作テスト用ブロック ---
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.get_merged_data()
    
    print("\n--- Head of Data ---")
    print(df.head())
    print("\n--- Tail of Data ---")
    print(df.tail())
    
    # データをCSVに保存して確認したければ以下をコメントアウト解除
    # df.to_csv("merged_data_check.csv")
