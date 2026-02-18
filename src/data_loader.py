import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import io
import datetime

class DataLoader:
    def __init__(self):
        # データ取得開始日
        self.start_date = '1990-01-01'
        self.end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
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
        Kenneth French Data Libraryから日本の5ファクターを取得
        """
        print(f"[{datetime.datetime.now().time()}] Fetching Fama-French 5 Factors (Japan)...")
        try:
            # 日本の5ファクター
            ds = web.DataReader('Japan_5_Factors', 'famafrench', start=self.start_date, end=self.end_date)
            df_ff = ds[0] # 月次リターン
            
            # IndexをTimestampに変換
            df_ff.index = df_ff.index.to_timestamp()
            
            # 月末日付を「翌月の1日」または「当月の1日」に揃える
            # ここでは内閣府データに合わせて「当月の1日」に補正します
            df_ff.index = df_ff.index + pd.offsets.MonthBegin(-1) + pd.offsets.MonthBegin(1)
            
            # カラム名を扱いやすく変更
            df_ff.columns = [c.strip() for c in df_ff.columns]
            
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
            tickers = ['^VIX', 'JPY=X']
            # auto_adjust=Trueで配当調整後終値などを自動取得
            df = yf.download(tickers, start=self.start_date, end=self.end_date, progress=False)
            
            # 'Close' または 'Adj Close' を取得（yfinanceのバージョンによる違いを吸収）
            if 'Adj Close' in df.columns:
                df = df['Adj Close']
            elif 'Close' in df.columns:
                df = df['Close']
            
            # 名前変更
            df.rename(columns={'^VIX': 'VIX', 'JPY=X': 'USD_JPY'}, inplace=True)
            
            # 日次データを月次平均にリサンプリング (Month Start)
            df_monthly = df.resample('MS').mean()
            
            print("Market indicators fetched.")
            return df_monthly
            
        except Exception as e:
            print(f"Error fetching Market data: {e}")
            return pd.DataFrame()

    def get_merged_data(self):
        """
        【メイン実行関数】全てのデータを結合して返す
        """
        # 1. 各ソースからデータ取得
        df_cabinet = self.fetch_cabinet_office_data()
        df_ff = self.fetch_fama_french()
        df_market = self.fetch_market_data()
        
        # 2. 結合 (Inner Join)
        # まずはFFとMarket
        df_merged = df_ff.join(df_market, how='inner')
        
        # 次に内閣府
        final_df = df_merged.join(df_cabinet, how='inner')
        
        # 欠損値削除
        final_df.dropna(inplace=True)
        
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
