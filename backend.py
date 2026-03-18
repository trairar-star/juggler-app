import os
import pandas as pd
import numpy as np
import lightgbm as lgb # type: ignore
import streamlit as st # type: ignore
import gspread
import unicodedata
from google.oauth2.service_account import Credentials

# 定数定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_FILE = os.path.join(BASE_DIR, 'service_account.json')
SPREADSHEET_KEY = '1ylt9mdIkKKk6YRcZh4O05O7fPF4d2BU6VXzboP_vs5s'
SHEET_NAME = 'juggler_raw'
HISTORY_CACHE_FILE = os.path.join(BASE_DIR, 'history_cache.parquet')

# ---------------------------------------------------------
# 機種スペック情報
# ---------------------------------------------------------
MACHINE_SPECS = {
    "ウルトラミラクルジャグラー": {
        "設定1": {"BIG": 267.5, "REG": 425.6, "合算": 164.3},
        "設定4": {"BIG": 242.7, "REG": 322.8, "合算": 138.6},
        "設定5": {"BIG": 233.2, "REG": 297.9, "合算": 130.8},
        "設定6": {"BIG": 216.3, "REG": 277.7, "合算": 121.6},
    },
    "ゴーゴージャグラー3": {
        "設定1": {"BIG": 259.0, "REG": 354.2, "合算": 149.6},
        "設定4": {"BIG": 254.0, "REG": 268.6, "合算": 130.5},
        "設定5": {"BIG": 247.3, "REG": 247.3, "合算": 123.7},
        "設定6": {"BIG": 234.9, "REG": 234.9, "合算": 117.4},
    },
    "ジャグラーガールズSS": {
        "設定1": {"BIG": 273.1, "REG": 381.0, "合算": 159.1},
        "設定4": {"BIG": 250.1, "REG": 281.3, "合算": 132.4},
        "設定5": {"BIG": 243.6, "REG": 270.8, "合算": 128.3},
        "設定6": {"BIG": 226.0, "REG": 252.1, "合算": 119.2},
    },
    "ネオアイムジャグラーEX": {
        "設定1": {"BIG": 273.1, "REG": 439.8, "合算": 168.5},
        "設定4": {"BIG": 259.0, "REG": 315.1, "合算": 142.2},
        "設定5": {"BIG": 259.0, "REG": 255.0, "合算": 128.5},
        "設定6": {"BIG": 255.0, "REG": 255.0, "合算": 127.5},
    },
    "ハッピージャグラーVIII": {
        "設定1": {"BIG": 273.1, "REG": 397.2, "合算": 161.8},
        "設定4": {"BIG": 254.0, "REG": 300.6, "合算": 137.7},
        "設定5": {"BIG": 239.2, "REG": 273.1, "合算": 127.5},
        "設定6": {"BIG": 226.0, "REG": 256.0, "合算": 120.0},
    },
    "ファンキージャグラー2KT": {
        "設定1": {"BIG": 266.4, "REG": 439.8, "合算": 165.9},
        "設定4": {"BIG": 249.2, "REG": 322.8, "合算": 140.6},
        "設定5": {"BIG": 240.1, "REG": 299.3, "合算": 133.2},
        "設定6": {"BIG": 219.9, "REG": 262.1, "合算": 119.6},
    },
    "マイジャグラーV": {
        "設定1": {"BIG": 273.1, "REG": 409.6, "合算": 163.8},
        "設定4": {"BIG": 254.0, "REG": 290.0, "合算": 135.4},
        "設定5": {"BIG": 240.1, "REG": 268.6, "合算": 126.8},
        "設定6": {"BIG": 229.1, "REG": 229.1, "合算": 114.6},
    },
    "ジャグラー（デフォルト）": {
        "設定1": {"BIG": 273.1, "REG": 439.8, "合算": 168.5},
        "設定4": {"BIG": 259.0, "REG": 315.1, "合算": 142.2},
        "設定5": {"BIG": 259.0, "REG": 255.0, "合算": 128.5},
        "設定6": {"BIG": 255.0, "REG": 255.0, "合算": 127.5},
    }
}

def get_machine_specs():
    return MACHINE_SPECS

def get_matched_spec_key(machine_name, specs):
    """機種名から最も一致するスペックキーを探す。見つからなければデフォルトを返す"""
    if not isinstance(machine_name, str) or not machine_name:
        return "ジャグラー（デフォルト）"
    if machine_name in specs:
        return machine_name
    for spec_key in specs.keys():
        if spec_key == "ジャグラー（デフォルト）": continue
        chk_word = spec_key.split('ジャグラー')[0] if 'ジャグラー' in spec_key else spec_key
        if not chk_word: chk_word = "ガールズ" if "ガールズ" in spec_key else spec_key
        if chk_word and chk_word in machine_name:
            return spec_key
    return "ジャグラー（デフォルト）"

# ---------------------------------------------------------
# データ読み込み・保存関数 (Model / Logic)
# ---------------------------------------------------------
@st.cache_resource(ttl=3300)
def _get_gspread_client():
    """認証クライアントを取得する共通関数"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    
    # 1. Streamlit CloudのSecrets機能を確認
    try:
        if "gcp_service_account" in st.secrets:
            creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
            return gspread.authorize(creds)
    except Exception:
        # secrets.tomlが存在しない場合は例外を無視してローカルJSONでの認証へ進む
        pass
    
    # 2. ローカルのJSONファイルを確認
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
        return gspread.authorize(creds)
    
    else:
        raise FileNotFoundError("認証情報が見つかりません。st.secrets または service_account.json を設定してください。")

@st.cache_data(ttl=3600)
def load_data():
    """Googleスプレッドシートから生の稼働データを読み込む"""
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet(SHEET_NAME)
        
        # 速度改善: get_all_records() はパースが重いため get_all_values() で取得する
        data = worksheet.get_all_values()
        if not data or len(data) < 2: return pd.DataFrame()
        raw_df = pd.DataFrame(data[1:], columns=data[0])
        
        if raw_df.empty: return pd.DataFrame()

        # 前処理
        raw_df.columns = [str(c).strip() for c in raw_df.columns]
        date_col = '対象日付'
        if date_col not in raw_df.columns: return pd.DataFrame()

        # --- 1週間以上前のデータは更新されない仕様を利用した高速化 (ローカルキャッシュ) ---
        raw_df['tmp_date'] = pd.to_datetime(raw_df[date_col], errors='coerce')
        latest_date = raw_df['tmp_date'].max()
        if pd.isna(latest_date):
            latest_date = pd.Timestamp.now()
        
        # 7日前の日付を「確定済みデータ（フリーズ）」の境界とする
        freeze_threshold = latest_date - pd.Timedelta(days=7)

        history_df = pd.DataFrame()
        if os.path.exists(HISTORY_CACHE_FILE):
            try:
                history_df = pd.read_parquet(HISTORY_CACHE_FILE, engine='pyarrow')
                # キャッシュ内のデータが古すぎる/新しすぎる場合を考慮し、確定済み範囲のみ残す
                if date_col in history_df.columns:
                    history_df = history_df[history_df[date_col] < freeze_threshold]
                else:
                    history_df = pd.DataFrame()
            except:
                pass

        if not history_df.empty:
            max_history_date = history_df[date_col].max()
            # 履歴キャッシュより新しいデータのみをパース対象にする（劇的な高速化）
            target_raw_df = raw_df[raw_df['tmp_date'] > max_history_date].copy()
        else:
            target_raw_df = raw_df.copy()

        target_raw_df = target_raw_df.drop(columns=['tmp_date'])
        
        if target_raw_df.empty and not history_df.empty:
            return history_df

        # --- 重い前処理（新規データのみ実行されるため一瞬で終わる） ---
        rename_map = {'REG回数': 'REG', 'BIG回数': 'BIG', '店舗名': '店名'}
        target_raw_df = target_raw_df.rename(columns=rename_map)

        if '機種名' in target_raw_df.columns:
            target_raw_df['機種名'] = target_raw_df['機種名'].apply(lambda x: unicodedata.normalize('NFKC', str(x)) if pd.notna(x) else x)

        def convert_prob(val):
            if pd.isna(val) or str(val).strip() == '': return 0.0
            val_str = str(val).strip()
            if '/' in val_str:
                try:
                    n, d = val_str.split('/')
                    return float(n) / float(d) if float(d) != 0 else 0.0
                except: return 0.0
            try:
                v = float(val)
                return 1.0 / v if v > 1.0 else v
            except: return 0.0

        for col in ['合成確率', 'BIG確率', 'REG確率']:
            if col in target_raw_df.columns:
                target_raw_df[col] = target_raw_df[col].apply(convert_prob)

        num_cols = ['台番号', '累計ゲーム', 'BIG', 'REG', '差枚', '末尾番号', '最終ゲーム']
        for col in num_cols:
            if col in target_raw_df.columns:
                target_raw_df[col] = target_raw_df[col].replace('', np.nan) # 空文字をNaNに変換
                target_raw_df[col] = pd.to_numeric(target_raw_df[col], errors='coerce').fillna(0)
        
        target_raw_df[date_col] = pd.to_datetime(target_raw_df[date_col], errors='coerce')
        target_raw_df = target_raw_df.dropna(subset=[date_col])
        
        # --- キャッシュと新規データを結合 ---
        if not history_df.empty:
            df = pd.concat([history_df, target_raw_df], ignore_index=True)
        else:
            df = target_raw_df

        # --- データ型のダウンキャストによるメモリ最適化 ---
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        # 次回のために、確定済みデータをキャッシュファイルに保存
        frozen_df = df[df[date_col] < freeze_threshold]
        if not frozen_df.empty:
            try:
                frozen_df.to_parquet(HISTORY_CACHE_FILE, engine='pyarrow')
            except Exception:
                pass # 保存エラーは無視（Streamlit Cloud環境等への配慮）
            
        return df
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_prediction_log():
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('prediction_log')
        return pd.DataFrame(worksheet.get_all_records())
    except: return pd.DataFrame()

def save_prediction_log(df):
    if df.empty:
        st.warning("保存するデータがありません。")
        return False
    if 'prediction_score' in df.columns:
        shop_col = '店名' if '店名' in df.columns else ('店舗名' if '店舗名' in df.columns else None)
        if shop_col:
            df = df.sort_values('prediction_score', ascending=False).groupby(shop_col).head(10)
        else:
            df = df.sort_values('prediction_score', ascending=False).head(10)
        if df.empty:
            st.warning("保存する推奨台がありません。")
            return False
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        log_sheet_name = 'prediction_log'
        
        try: 
            worksheet = sh.worksheet(log_sheet_name)
            existing_data = worksheet.get_all_values()
            if existing_data:
                header = existing_data[0]
            else:
                header = ['実行日時', '予測対象日', '対象日付', '店名', '台番号', '機種名', 'prediction_score', 'おすすめ度', '予測差枚数', '根拠', 'ai_version']
        except: 
            worksheet = sh.add_worksheet(title=log_sheet_name, rows="1000", cols="20")
            header = ['実行日時', '予測対象日', '対象日付', '店名', '台番号', '機種名', 'prediction_score', 'おすすめ度', '予測差枚数', '根拠', 'ai_version']
            existing_data = []

        if 'ai_version' not in header: header.append('ai_version')
        if '予測対象日' not in header: header.append('予測対象日')
            
        save_df = df.copy()
        save_df['実行日時'] = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        if 'ai_version' not in save_df.columns:
            save_df['ai_version'] = "不明"
            
        if 'next_date' in save_df.columns:
            save_df['予測対象日'] = save_df['next_date']
        else:
            save_df['予測対象日'] = save_df['対象日付'] + pd.Timedelta(days=1)
            
        for col in save_df.columns:
            if pd.api.types.is_datetime64_any_dtype(save_df[col]):
                save_df[col] = save_df[col].dt.strftime('%Y-%m-%d')

        valid_cols = [c for c in header if c in save_df.columns]
        for c in header:
            if c not in valid_cols and c in save_df.columns:
                valid_cols.append(c)
                header.append(c)
                
        save_df = save_df[valid_cols]
        save_df = save_df.fillna('')
        
        # 今回保存する予測対象日のリストを取得（上書き判定用）
        target_dates = save_df['予測対象日'].unique().tolist()
        
        final_data = [header]
        if existing_data and len(existing_data) > 1:
            try:
                date_col_idx = existing_data[0].index('予測対象日')
                for row in existing_data[1:]:
                    row_padded = row + [''] * (len(header) - len(row))
                    # 予測対象日が今回保存するものと同じ場合はスキップ（古いデータを削除）
                    if len(row) > date_col_idx and row[date_col_idx] in target_dates:
                        continue
                    final_data.append(row_padded)
            except ValueError:
                # 予測対象日列がない場合はそのまま残す
                for row in existing_data[1:]:
                    final_data.append(row + [''] * (len(header) - len(row)))
                    
        # 新しいデータを追加
        final_data.extend(save_df.values.tolist())
        
        # シートをクリアして一括更新
        worksheet.clear()
        try:
            worksheet.update(values=final_data, range_name='A1')
        except TypeError:
            try:
                worksheet.update('A1', final_data)
            except Exception:
                worksheet.update(final_data)
                
        st.success(f"予測結果（各店舗Top10）を '{log_sheet_name}' シートに保存（上書き）しました！")
        return True
    except Exception as e: 
        st.error(f"保存エラー: {e}")
        return False

@st.cache_data(ttl=3600)
def load_shop_events():
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('shop_events')
        df = pd.DataFrame(worksheet.get_all_records())
        if not df.empty and 'イベント日付' in df.columns:
            df['イベント日付'] = pd.to_datetime(df['イベント日付'])
        return df
    except: return pd.DataFrame()

def save_shop_event(shop_name, event_date, event_name, event_rank):
    if not shop_name or not event_name:
        st.warning("店舗名とイベント名を入力してください。")
        return False
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        try: 
            worksheet = sh.worksheet('shop_events')
            if 'イベントランク' not in worksheet.row_values(1):
                worksheet.update_cell(1, len(worksheet.row_values(1)) + 1, 'イベントランク')
        except: 
            worksheet = sh.add_worksheet(title='shop_events', rows="1000", cols="6")
            worksheet.append_row(['登録日時', '店名', 'イベント日付', 'イベント名', '備考', 'イベントランク'])
        
        timestamp = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        date_str = event_date.strftime('%Y-%m-%d')
        worksheet.append_row([timestamp, shop_name, date_str, event_name, '', event_rank])
        return True
    except Exception as e:
        st.error(f"イベント保存エラー: {e}")
        return False

def update_shop_event(old_shop_name, old_event_date, old_event_name, new_shop_name, new_event_date, new_event_name, new_event_rank):
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('shop_events')
        all_values = worksheet.get_all_values()
        if not all_values: return False
        header = all_values[0]
        try:
            idx_reg = header.index('登録日時')
            idx_shop = header.index('店名')
            idx_date = header.index('イベント日付')
            idx_name = header.index('イベント名')
            idx_rank = header.index('イベントランク') if 'イベントランク' in header else -1
        except: return False
        
        target_date_str = old_event_date.strftime('%Y-%m-%d')
        new_date_str = new_event_date.strftime('%Y-%m-%d')
        timestamp = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        
        for i, row in enumerate(all_values[1:], start=2):
            if len(row) <= max(idx_shop, idx_date, idx_name): continue
            r_date = row[idx_date]
            is_date_match = (r_date == target_date_str)
            if not is_date_match:
                try: 
                    if pd.to_datetime(r_date).strftime('%Y-%m-%d') == target_date_str: is_date_match = True
                except: pass
            if row[idx_shop] == old_shop_name and row[idx_name] == old_event_name and is_date_match:
                worksheet.update_cell(i, idx_reg + 1, timestamp)
                worksheet.update_cell(i, idx_shop + 1, new_shop_name)
                worksheet.update_cell(i, idx_date + 1, new_date_str)
                worksheet.update_cell(i, idx_name + 1, new_event_name)
                if idx_rank != -1:
                    worksheet.update_cell(i, idx_rank + 1, new_event_rank)
                return True
        return False
    except Exception as e:
        st.error(f"更新エラー: {e}")
        return False

def delete_shop_event(shop_name, event_date, event_name):
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('shop_events')
        all_values = worksheet.get_all_values()
        if not all_values: return False
        header = all_values[0]
        try:
            idx_shop = header.index('店名')
            idx_date = header.index('イベント日付')
            idx_name = header.index('イベント名')
        except: return False
        
        target_date_str = event_date.strftime('%Y-%m-%d')
        for i, row in enumerate(all_values[1:], start=2):
            if len(row) <= max(idx_shop, idx_date, idx_name): continue
            r_date = row[idx_date]
            is_date_match = (r_date == target_date_str)
            if not is_date_match:
                try: 
                    if pd.to_datetime(r_date).strftime('%Y-%m-%d') == target_date_str: is_date_match = True
                except: pass
            if row[idx_shop] == shop_name and row[idx_name] == event_name and is_date_match:
                worksheet.delete_rows(i)
                return True
        return False
    except Exception as e:
        st.error(f"削除エラー: {e}")
        return False

# --- 島マスター管理機能 ---
@st.cache_data(ttl=3600)
def load_island_master():
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('island_master')
        return pd.DataFrame(worksheet.get_all_records())
    except: return pd.DataFrame()

def save_island_master(shop, island_name, rule_str):
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        sheet_name = 'island_master'
        try: 
            worksheet = sh.worksheet(sheet_name)
            header = worksheet.row_values(1)
            if '台番号ルール' not in header:
                worksheet.update_cell(1, len(header) + 1, '台番号ルール')
                header.append('台番号ルール')
        except: 
            worksheet = sh.add_worksheet(title=sheet_name, rows="1000", cols="6")
            worksheet.append_row(['登録日時', '店名', '島名', '開始台番号', '終了台番号', '台番号ルール'])
            header = ['登録日時', '店名', '島名', '開始台番号', '終了台番号', '台番号ルール']
        
        timestamp = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        row_data = [timestamp, shop, island_name, "", "", rule_str]
        
        while len(row_data) < len(header):
            row_data.append("")
            
        if '台番号ルール' in header:
            idx = header.index('台番号ルール')
            row_data[idx] = rule_str
            
        worksheet.append_row(row_data, value_input_option='RAW')
        return True
    except Exception as e:
        st.error(f"島マスター保存エラー: {e}")
        return False

def delete_island_master(target_timestamp):
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('island_master')
        cell = worksheet.find(str(target_timestamp), in_column=1)
        if cell:
            worksheet.delete_rows(cell.row)
            return True
        return False
    except Exception:
        return False

# --- マイ収支管理機能 ---
@st.cache_data(ttl=3600)
def load_my_balance():
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('my_balance')
        df = pd.DataFrame(worksheet.get_all_records())
        if not df.empty and '日付' in df.columns:
            df['日付'] = pd.to_datetime(df['日付'])
        return df
    except: return pd.DataFrame()

def save_my_balance(date_obj, shop, machine, number, invest, recovery, memo):
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        
        sheet_name = 'my_balance'
        try: 
            worksheet = sh.worksheet(sheet_name)
            existing_data = worksheet.get_all_values()
            if existing_data:
                header = existing_data[0]
            else:
                header = ['登録日時', '日付', '店名', '台番号', '機種名', '投資', '回収', '収支', 'メモ']
                existing_data = []
        except: 
            worksheet = sh.add_worksheet(title=sheet_name, rows="1000", cols="10")
            header = ['登録日時', '日付', '店名', '台番号', '機種名', '投資', '回収', '収支', 'メモ']
            existing_data = []
        
        timestamp = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        date_str = date_obj.strftime('%Y-%m-%d')
        balance = int(recovery) - int(invest)
        
        new_row_dict = {
            '登録日時': timestamp, '日付': date_str, '店名': shop,
            '台番号': str(number), '機種名': machine, '投資': str(invest),
            '回収': str(recovery), '収支': str(balance), 'メモ': memo
        }
        
        final_data = [header]
        replaced = False
        
        if existing_data and len(existing_data) > 1:
            try:
                idx_date = header.index('日付')
                idx_shop = header.index('店名')
                idx_num = header.index('台番号')
                
                for row in existing_data[1:]:
                    row_padded = row + [''] * (len(header) - len(row))
                    # 同日・同店舗・同台番号のデータがあれば上書きする
                    if len(row_padded) > max(idx_date, idx_shop, idx_num) and \
                       row_padded[idx_date] == date_str and \
                       row_padded[idx_shop] == shop and \
                       str(row_padded[idx_num]) == str(number):
                        new_row = [new_row_dict.get(col, '') for col in header]
                        final_data.append(new_row)
                        replaced = True
                    else:
                        final_data.append(row_padded)
            except ValueError:
                for row in existing_data[1:]:
                    final_data.append(row + [''] * (len(header) - len(row)))
        
        if not replaced:
            new_row = [new_row_dict.get(col, '') for col in header]
            final_data.append(new_row)
            
        # シートをクリアして一括更新
        worksheet.clear()
        try:
            worksheet.update(values=final_data, range_name='A1')
        except TypeError:
            try: worksheet.update('A1', final_data)
            except Exception: worksheet.update(final_data)
            
        return True
    except Exception as e:
        st.error(f"収支保存エラー: {e}")
        return False

def update_my_balance(old_timestamp, date_obj, shop, machine, number, invest, recovery, memo):
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('my_balance')
        all_values = worksheet.get_all_values()
        if not all_values: return False
        
        header = all_values[0]
        try:
            idx_reg = header.index('登録日時')
            idx_date = header.index('日付')
            idx_shop = header.index('店名')
            idx_num = header.index('台番号')
            idx_mac = header.index('機種名')
            idx_inv = header.index('投資')
            idx_rec = header.index('回収')
            idx_bal = header.index('収支')
            idx_memo = header.index('メモ')
        except: return False
        
        date_str = date_obj.strftime('%Y-%m-%d')
        balance = int(recovery) - int(invest)
        
        for i, row in enumerate(all_values[1:], start=2):
            if len(row) <= idx_reg: continue
            if row[idx_reg] == str(old_timestamp):
                worksheet.update_cell(i, idx_date + 1, date_str)
                worksheet.update_cell(i, idx_shop + 1, shop)
                worksheet.update_cell(i, idx_num + 1, number)
                worksheet.update_cell(i, idx_mac + 1, machine)
                worksheet.update_cell(i, idx_inv + 1, invest)
                worksheet.update_cell(i, idx_rec + 1, recovery)
                worksheet.update_cell(i, idx_bal + 1, balance)
                worksheet.update_cell(i, idx_memo + 1, memo)
                return True
        return False
    except Exception as e:
        st.error(f"収支更新エラー: {e}")
        return False

def delete_my_balance(target_timestamp):
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('my_balance')
        
        cell = worksheet.find(str(target_timestamp), in_column=1)
        if cell:
            worksheet.delete_rows(cell.row)
            return True
        return False
    except Exception as e:
        st.error(f"収支削除エラー: {e}")
        return False

# --- 店舗別AI設定の永続化 ---
@st.cache_data(ttl=3600)
def load_shop_ai_settings():
    """店舗別のAI設定をスプレッドシートから読み込む"""
    default_settings = {
        "デフォルト": {'train_months': 3, 'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50}
    }
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('shop_ai_settings')
        records = worksheet.get_all_records()
        if not records:
            return default_settings
            
        settings_dict = {}
        for record in records:
            shop_name = record.get('店名')
            if not shop_name:
                continue
            
            try:
                settings_dict[shop_name] = {
                    'train_months': int(record.get('train_months')),
                    'n_estimators': int(record.get('n_estimators')),
                    'learning_rate': float(record.get('learning_rate')),
                    'num_leaves': int(record.get('num_leaves')),
                    'max_depth': int(record.get('max_depth')),
                    'min_child_samples': int(record.get('min_child_samples')),
                }
            except (ValueError, TypeError):
                continue
        
        if "デフォルト" not in settings_dict:
            settings_dict["デフォルト"] = default_settings["デフォルト"]
            
        return settings_dict
        
    except gspread.exceptions.WorksheetNotFound:
        return default_settings
    except Exception as e:
        st.warning(f"AI設定の読み込みに失敗しました: {e}。デフォルト設定を使用します。")
        return default_settings

def save_shop_ai_settings(shop_hyperparams):
    """店舗別のAI設定をスプレッドシートに保存する"""
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        sheet_name = 'shop_ai_settings'
        try: worksheet = sh.worksheet(sheet_name)
        except gspread.exceptions.WorksheetNotFound: worksheet = sh.add_worksheet(title=sheet_name, rows="100", cols="10")
        header = ['店名', 'train_months', 'n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'min_child_samples']
        data_to_write = [header] + [[shop_name] + [params.get(k) for k in header[1:]] for shop_name, params in shop_hyperparams.items()]
        worksheet.clear(); worksheet.update('A1', data_to_write)
        return True
    except Exception as e:
        st.error(f"AI設定の保存に失敗しました: {e}")
        return False

# --- 内部関数: 特徴量作成 ---
def _generate_features(df, df_events, df_island, target_date):
    if target_date is not None:
        target_ts = pd.to_datetime(target_date)
        df = df[df['対象日付'] < target_ts].copy()

    if df.empty: return df, []

    shop_col = '店名' if '店名' in df.columns else ('店舗名' if '店舗名' in df.columns else None)
    
    if '機種名' in df.columns: df['machine_code'] = df['機種名'].astype('category').cat.codes
    if shop_col: df['shop_code'] = df[shop_col].astype('category').cat.codes

    if 'REG' in df.columns and 'BIG' in df.columns:
        df['reg_ratio'] = df['REG'] / (df['BIG'] + df['REG'] + 1)

    if shop_col and '機種名' in df.columns and '台番号' in df.columns:
        grp = df.groupby([shop_col, '機種名'])['台番号']
        df['is_corner'] = ((df['台番号'] == grp.transform('min')) | (df['台番号'] == grp.transform('max'))).astype(int)
        df['island_id'] = "Unknown"

        # 島マスターの適用
        if df_island is not None and not df_island.empty:
            unique_machines = df[[shop_col, '台番号']].drop_duplicates()
            island_mapping = []
            
            parsed_islands = []
            for _, i_row in df_island.iterrows():
                s_name = i_row.get('店名')
                i_name = i_row.get('島名')
                machines = []
                
                # 旧仕様(開始〜終了)の互換性維持
                try:
                    s = int(i_row.get('開始台番号', 0))
                    e = int(i_row.get('終了台番号', 0))
                    if s > 0 and e >= s: machines.extend(range(s, e + 1))
                except: pass
                
                # 新仕様(柔軟なルール指定)の解析
                rule = str(i_row.get('台番号ルール', ''))
                if rule and rule.strip() != '' and rule != 'nan':
                    for part in rule.split(','):
                        part = part.strip()
                        if not part: continue
                        if '-' in part:
                            try:
                                s_str, e_str = part.split('-', 1)
                                machines.extend(range(int(s_str), int(e_str) + 1))
                            except: pass
                        else:
                            try: machines.append(int(part))
                            except: pass
                            
                machines = sorted(list(set(machines)))
                if machines:
                    parsed_islands.append({
                        'shop': s_name, 'island_id': f"{s_name}_{i_name}",
                        'machines': machines, 'corner_min': min(machines), 'corner_max': max(machines)
                    })
                    
            for _, row in unique_machines.iterrows():
                s_name = row[shop_col]
                m_num = row['台番号']
                i_id = "Unknown"
                is_cor = 0
                for pi in parsed_islands:
                    if pi['shop'] == s_name and m_num in pi['machines']:
                        i_id = pi['island_id']
                        if m_num == pi['corner_min'] or m_num == pi['corner_max']: is_cor = 1
                        break
                island_mapping.append({shop_col: s_name, '台番号': m_num, 'master_island_id': i_id, 'master_is_corner': is_cor})
            mapping_df = pd.DataFrame(island_mapping)
            df = pd.merge(df, mapping_df, on=[shop_col, '台番号'], how='left')
            df.loc[df['master_island_id'] != "Unknown", 'island_id'] = df['master_island_id']
            df.loc[df['master_island_id'] != "Unknown", 'is_corner'] = df['master_is_corner']

    if shop_col and '台番号' in df.columns and '対象日付' in df.columns:
        if 'island_id' in df.columns:
            # 同じ島IDごとにまとめてからソートすることで、関係ない台が間に挟まるのを防ぐ
            df = df.sort_values([shop_col, '対象日付', 'island_id', '台番号'])
            
            prev_shop = df[shop_col].shift(1)
            prev_date = df['対象日付'].shift(1)
            prev_island = df['island_id'].shift(1)
            prev_no = df['台番号'].shift(1)
            prev_diff = df['差枚'].shift(1)
            
            next_shop = df[shop_col].shift(-1)
            next_date = df['対象日付'].shift(-1)
            next_island = df['island_id'].shift(-1)
            next_no = df['台番号'].shift(-1)
            next_diff = df['差枚'].shift(-1)
            
            is_prev = (df[shop_col] == prev_shop) & (df['対象日付'] == prev_date) & (
                ((df['island_id'] != "Unknown") & (df['island_id'] == prev_island)) |
                ((df['island_id'] == "Unknown") & (df['island_id'] == prev_island) & ((df['台番号'] - prev_no).between(1, 3)))
            )
            is_next = (df[shop_col] == next_shop) & (df['対象日付'] == next_date) & (
                ((df['island_id'] != "Unknown") & (df['island_id'] == next_island)) |
                ((df['island_id'] == "Unknown") & (df['island_id'] == next_island) & ((next_no - df['台番号']).between(1, 3)))
            )
            
            p_val = np.where(is_prev, prev_diff, np.nan)
            n_val = np.where(is_next, next_diff, np.nan)
            df['neighbor_avg_diff'] = pd.DataFrame({'p': p_val, 'n': n_val}).mean(axis=1).fillna(0)
            
            # 元の並び順に戻す
            df = df.sort_values([shop_col, '対象日付', '台番号']).reset_index(drop=True)
        else:
            df = df.sort_values([shop_col, '対象日付', '台番号'])
            prev_shop = df[shop_col].shift(1)
            prev_date = df['対象日付'].shift(1)
            prev_no = df['台番号'].shift(1)
            prev_diff = df['差枚'].shift(1)
            next_shop = df[shop_col].shift(-1)
            next_date = df['対象日付'].shift(-1)
            next_no = df['台番号'].shift(-1)
            next_diff = df['差枚'].shift(-1)
            
            is_prev = (df[shop_col] == prev_shop) & (df['対象日付'] == prev_date) & ((df['台番号'] - prev_no).between(1, 3))
            is_next = (df[shop_col] == next_shop) & (df['対象日付'] == next_date) & ((next_no - df['台番号']).between(1, 3))
        
            p_val = np.where(is_prev, prev_diff, np.nan)
            n_val = np.where(is_next, next_diff, np.nan)
            df['neighbor_avg_diff'] = pd.DataFrame({'p': p_val, 'n': n_val}).mean(axis=1).fillna(0)

    sort_keys = [shop_col, '台番号', '対象日付'] if shop_col else ['台番号', '対象日付']
    group_keys = [shop_col, '台番号'] if shop_col else ['台番号']
    df = df.sort_values(sort_keys).reset_index(drop=True)
    
    for col in ['差枚', 'REG確率', '累計ゲーム', '最終ゲーム']:
        if col in df.columns: df[f'prev_{col}'] = df.groupby(group_keys)[col].shift(1)
    
    df['next_diff'] = df.groupby(group_keys)['差枚'].shift(-1)
    if 'BIG' in df.columns: df['next_BIG'] = df.groupby(group_keys)['BIG'].shift(-1)
    if 'REG' in df.columns: df['next_REG'] = df.groupby(group_keys)['REG'].shift(-1)
    if '累計ゲーム' in df.columns: df['next_累計ゲーム'] = df.groupby(group_keys)['累計ゲーム'].shift(-1)
    
    # --- ターゲットを「翌日の高設定挙動(機種別の設定5基準：REGまたは合算)」に変更 ---
    df['next_reg_prob'] = df['next_REG'] / df['next_累計ゲーム'].replace(0, np.nan)
    df['next_total_prob'] = (df['next_BIG'].fillna(0) + df['next_REG'].fillna(0)) / df['next_累計ゲーム'].replace(0, np.nan)
    specs = get_machine_specs()
    
    # 【高速化】機種スペックのマッピングをapplyから辞書マッピングに変更
    unique_machines = df['機種名'].unique()
    reg_map = {m: 1.0 / specs[get_matched_spec_key(m, specs)].get('設定5', {"REG": 260.0})["REG"] for m in unique_machines}
    tot_map = {m: 1.0 / specs[get_matched_spec_key(m, specs)].get('設定5', {"合算": 128.0})["合算"] for m in unique_machines}
    reg3_map = {m: 1.0 / specs[get_matched_spec_key(m, specs)].get('設定3', {"REG": 300.0})["REG"] for m in unique_machines}
    spec_reg = df['機種名'].map(reg_map)
    spec_tot = df['機種名'].map(tot_map)
    spec_reg3 = df['機種名'].map(reg3_map)
    
    # ターゲット判定時も「翌日3000G以上回ったか」を条件に入れ、上振れノイズを学習しないようにする
    df['target'] = (
        (df['next_累計ゲーム'] >= 3000) & 
        (
            (df['next_reg_prob'] >= spec_reg) | 
            ((df['next_total_prob'] >= spec_tot) & (df['next_reg_prob'] >= spec_reg3))
        )
    ).astype(int)
    
    # --- 予測対象日の情報（未来のカンニングではなく、予測日の日付・曜日・イベント属性） ---
    df['next_date'] = df.groupby(group_keys)['対象日付'].shift(-1)
    if target_date is not None:
        target_ts = pd.to_datetime(target_date)
        df.loc[df['next_diff'].isna(), 'next_date'] = target_ts
    else:
        df['next_date'] = df['next_date'].fillna(df['対象日付'] + pd.Timedelta(days=1))
        
    df['target_weekday'] = df['next_date'].dt.dayofweek
    df['target_date_end_digit'] = df['next_date'].dt.day % 10

    if df_events is not None and not df_events.empty and shop_col:
        events_unique = df_events.drop_duplicates(subset=['店名', 'イベント日付'], keep='last').copy()
        merge_cols = ['店名', 'イベント日付', 'イベント名']
        if 'イベントランク' in events_unique.columns: merge_cols.append('イベントランク')

        # 対象日付ではなく、予測対象日（next_date）のイベントを結合する
        df = pd.merge(df, events_unique[merge_cols], left_on=[shop_col, 'next_date'], right_on=['店名', 'イベント日付'], how='left')
        df = df.drop(columns=['店名_y', 'イベント日付_y', 'イベント日付'], errors='ignore')
        if '店名_x' in df.columns: df = df.rename(columns={'店名_x': '店名'})
        
        df['イベント名'] = df['イベント名'].fillna('通常')
        df['event_code'] = df['イベント名'].astype('category').cat.codes
        if 'イベントランク' in df.columns:
            rank_map = {'S': 5, 'A': 4, 'B': 3, 'C': 2}
            df['event_rank_score'] = df['イベントランク'].map(rank_map).fillna(0)

    # 【高速化】遅い transform(lambda...) を groupby.rolling() に変更
    df = df.sort_values('対象日付').reset_index(drop=True)
    df['weekday'] = df['対象日付'].dt.dayofweek
    df['shifted_diff_wd'] = df.groupby('weekday')['差枚'].shift(1)
    df['weekday_avg_diff'] = df.groupby('weekday')['shifted_diff_wd'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    
    if '日付要素' in df.columns:
        df['shifted_diff_ev'] = df.groupby('日付要素')['差枚'].shift(1)
        df['event_avg_diff'] = df.groupby('日付要素')['shifted_diff_ev'].expanding().mean().reset_index(level=0, drop=True).fillna(0)

    df = df.sort_values(sort_keys).reset_index(drop=True)
    df['shifted_diff'] = df.groupby(group_keys)['差枚'].shift(1)
    group_levels = list(range(len(group_keys))) # インデックス操作用
    
    df['mean_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_14days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=14, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_30days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=30, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)

    # --- 勝率安定度（一撃ノイズ排除用） ---
    df['total_prob'] = (df['BIG'].fillna(0) + df['REG'].fillna(0)) / df['累計ゲーム'].replace(0, np.nan)
    # 高設定フラグにも「3000G以上回っている」条件を加え、信頼性の高い実績だけを評価する
    df['is_win'] = (
        (df['累計ゲーム'] >= 3000) & 
        (
            (df['REG確率'] >= spec_reg) | 
            ((df['total_prob'] >= spec_tot) & (df['REG確率'] >= spec_reg3))
        )
    ).astype(int)
    df['shifted_is_win'] = df.groupby(group_keys)['is_win'].shift(1)
    df['win_rate_7days'] = df.groupby(group_keys)['shifted_is_win'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    
    # 一時的に作成した不要な列を削除
    df = df.drop(columns=['shifted_diff_wd', 'shifted_diff_ev', 'shifted_diff', 'shifted_is_win'], errors='ignore')

    # 現在のソート順を保持（mergeによる順序崩れ防止）
    df['original_order'] = np.arange(len(df))

    if shop_col:
        df['shop_avg_diff'] = df.groupby([shop_col, '対象日付'])['差枚'].transform('mean').fillna(0)
        # 店舗の平均稼働に対する自台の稼働割合（相対的な粘られ度）
        df['shop_avg_games'] = df.groupby([shop_col, '対象日付'])['累計ゲーム'].transform('mean').replace(0, np.nan)
        df['relative_games_ratio'] = (df['累計ゲーム'] / df['shop_avg_games']).fillna(1.0)
        df = df.drop(columns=['shop_avg_games'])
        
        # --- ①店舗全体の回収・還元モード指標 (直近7日間の店舗全体の平均差枚) ---
        shop_daily_avg = df.groupby([shop_col, '対象日付'])['差枚'].mean().reset_index(name='shop_daily_avg_diff')
        shop_daily_avg = shop_daily_avg.sort_values([shop_col, '対象日付'])
        shop_daily_avg['shop_7days_avg_diff'] = shop_daily_avg.groupby(shop_col)['shop_daily_avg_diff'].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()).fillna(0)
        df = pd.merge(df, shop_daily_avg[[shop_col, '対象日付', 'shop_7days_avg_diff']], on=[shop_col, '対象日付'], how='left')

    if shop_col and '機種名' in df.columns:
        # --- ②機種ごとの扱い指標 (過去30日間のその機種の平均差枚) ---
        machine_daily_avg = df.groupby([shop_col, '機種名', '対象日付'])['差枚'].mean().reset_index(name='machine_daily_avg_diff')
        machine_daily_avg = machine_daily_avg.sort_values([shop_col, '機種名', '対象日付'])
        machine_daily_avg['machine_30days_avg_diff'] = machine_daily_avg.groupby([shop_col, '機種名'])['machine_daily_avg_diff'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
        df = pd.merge(df, machine_daily_avg[[shop_col, '機種名', '対象日付', 'machine_30days_avg_diff']], on=[shop_col, '機種名', '対象日付'], how='left')

    # ソート順を元に戻す
    df = df.sort_values('original_order').drop(columns=['original_order']).reset_index(drop=True)

    if 'island_id' in df.columns:
        df['island_avg_diff'] = df.groupby(['island_id', '対象日付'])['差枚'].transform('mean').fillna(0)

    # --- 上げリセット（設定変更）検知用の特徴量 ---
    # 差枚が+500枚以上になったら「設定が入り放出された」とみなしリセット
    df['is_released'] = (df['差枚'] >= 500).astype(int)
    df['temp_reset_group'] = df.groupby(group_keys)['is_released'].cumsum()
    
    # 差枚が+500枚未満の日はすべて「実質マイナス（回収・不発）」としてカウントを継続
    df['is_not_released'] = (df['差枚'] < 500).astype(int)
    df['連続マイナス日数'] = df.groupby(group_keys + ['temp_reset_group'])['is_not_released'].cumsum()
    df = df.drop(columns=['temp_reset_group', 'is_released', 'is_not_released'])

    # --- 連続低稼働日数のカウント（テコ入れ狙い） ---
    UTILIZATION_THRESHOLD = 1500
    df['is_low_utilization'] = (df['累計ゲーム'] < UTILIZATION_THRESHOLD).astype(int)
    df['is_active'] = (df['累計ゲーム'] >= UTILIZATION_THRESHOLD).astype(int)
    df['low_util_reset_group'] = df.groupby(group_keys)['is_active'].cumsum()
    
    df['連続低稼働日数'] = df.groupby(group_keys + ['low_util_reset_group'])['is_low_utilization'].cumsum()
    df = df.drop(columns=['is_low_utilization', 'is_active', 'low_util_reset_group'])

    # 台ごとの過去データ件数（履歴の長さ）を計算し、信頼度の指標とする
    df['history_count'] = df.groupby(group_keys).cumcount() + 1

    # --- 新台・配置変更の正確な検知 ---
    if shop_col:
        # その日時点での、その店舗の最大データ蓄積日数を取得
        df['shop_date_max_history'] = df.groupby([shop_col, '対象日付'])['history_count'].transform('max')
        # 店舗として14日以上データがある状態（取得開始直後ではない）で、その台の履歴が7日以下のものを新台とみなす
        df['is_new_machine'] = ((df['shop_date_max_history'] >= 14) & (df['history_count'] <= 7)).astype(int)
        df = df.drop(columns=['shop_date_max_history'])
    else:
        df['is_new_machine'] = 0

    features = ['累計ゲーム', 'REG確率', 'BIG確率', '差枚', '末尾番号', 'target_weekday', 'target_date_end_digit', 'mean_7days_diff', 'win_rate_7days', '連続マイナス日数', '連続低稼働日数', 'is_new_machine']
    for f in ['machine_code', 'shop_code', 'reg_ratio', 'is_corner', 'neighbor_avg_diff', 'event_avg_diff', 'event_code', 'event_rank_score', 'prev_差枚', 'prev_REG確率', 'prev_累計ゲーム', 'shop_avg_diff', 'island_avg_diff', 'relative_games_ratio', 'shop_7days_avg_diff', 'machine_30days_avg_diff']:
        if f in df.columns: features.append(f)

    return df, features

# --- 内部関数: モデル学習 ---
def _train_models(train_df, predict_df, features, shop_hyperparams):
    shop_col = '店名' if '店名' in train_df.columns else ('店舗名' if '店舗名' in train_df.columns else None)
    default_hp = shop_hyperparams.get("デフォルト", {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50})
    
    X, y = train_df[features], train_df['target']
    sample_weights = None
    if '対象日付' in train_df.columns:
        max_date = train_df['対象日付'].max()
        days_diff = (max_date - train_df['対象日付']).dt.days
        sample_weights = 0.995 ** days_diff
    
    n_est = default_hp.get('n_estimators', 300)
    lr = default_hp.get('learning_rate', 0.03)
    nl = default_hp.get('num_leaves', 15)
    md = default_hp.get('max_depth', 4)
    mcs = default_hp.get('min_child_samples', 50)

    # カテゴリ変数として扱う特徴量のリストを定義
    cat_features = [f for f in ['machine_code', 'shop_code', 'event_code', 'target_weekday', 'target_date_end_digit'] if f in features]

    # --- 全店舗共通モデルの学習と推論 ---
    model = lgb.LGBMClassifier(
        objective='binary', random_state=42, verbose=-1, 
        n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8
    )
    model.fit(X, y, sample_weight=sample_weights, categorical_feature=cat_features)
    reg_model = lgb.LGBMRegressor(
        random_state=42, verbose=-1, 
        n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8
    )
    reg_model.fit(X, train_df['next_diff'], sample_weight=sample_weights, categorical_feature=cat_features)
    
    if not predict_df.empty:
        predict_df['prediction_score'] = model.predict_proba(predict_df[features])[:, 1]
        predict_df['予測差枚数'] = reg_model.predict(predict_df[features]).astype(int)
        predict_df['ai_version'] = "v2.2(共通)"
    if not train_df.empty:
        train_df['prediction_score'] = model.predict_proba(train_df[features])[:, 1]
        train_df['予測差枚数'] = reg_model.predict(train_df[features]).astype(int)
    
    feature_importances_list = []
    feature_importances_list.append(pd.DataFrame({
        'shop_name': '全店舗',
        'category': '全体',
        'feature': features,
        'importance': model.feature_importances_
    }))
    
    # --- 店舗個別モデルの学習と推論の上書き ---
    if shop_col:
        for shop in train_df[shop_col].unique():
            shop_hp = shop_hyperparams.get(shop, default_hp)
            s_n_est = shop_hp.get('n_estimators', 300)
            s_lr = shop_hp.get('learning_rate', 0.03)
            s_nl = shop_hp.get('num_leaves', 15)
            s_md = shop_hp.get('max_depth', 4)
            s_mcs = shop_hp.get('min_child_samples', 50)
            
            shop_train = train_df[train_df[shop_col] == shop]
            if len(shop_train) >= 150: # ノイズ過学習防止のため、最低サンプル数を引き上げ
                X_shop = shop_train[features]
                y_shop = shop_train['target']
                sw_shop = sample_weights.loc[shop_train.index] if sample_weights is not None else None
                
                shop_model = lgb.LGBMClassifier(
                    objective='binary', random_state=42, verbose=-1, 
                    n_estimators=s_n_est, learning_rate=s_lr, num_leaves=s_nl, max_depth=s_md, min_child_samples=s_mcs,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8
                )
                shop_reg = lgb.LGBMRegressor(
                    random_state=42, verbose=-1, 
                    n_estimators=s_n_est, learning_rate=s_lr, num_leaves=s_nl, max_depth=s_md, min_child_samples=s_mcs,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8
                )
                
                try:
                    shop_model.fit(X_shop, y_shop, sample_weight=sw_shop, categorical_feature=cat_features)
                    shop_reg.fit(X_shop, shop_train['next_diff'], sample_weight=sw_shop, categorical_feature=cat_features)
                    feature_importances_list.append(pd.DataFrame({
                        'shop_name': shop,
                        'category': '店舗',
                        'feature': features,
                        'importance': shop_model.feature_importances_
                    }))
                    
                    # その店舗の推論結果を専用モデルで上書きする
                    if not predict_df.empty:
                        shop_pred_idx = predict_df[predict_df[shop_col] == shop].index
                        if len(shop_pred_idx) > 0:
                            predict_df.loc[shop_pred_idx, 'prediction_score'] = shop_model.predict_proba(predict_df.loc[shop_pred_idx, features])[:, 1]
                            predict_df.loc[shop_pred_idx, '予測差枚数'] = shop_reg.predict(predict_df.loc[shop_pred_idx, features]).astype(int)
                            predict_df.loc[shop_pred_idx, 'ai_version'] = f"v2.2(m{shop_hp.get('train_months',3)}_n{s_n_est}_l{s_nl}_lr{s_lr}_d{s_md}_c{s_mcs})"
                    if not train_df.empty:
                        shop_train_idx = train_df[train_df[shop_col] == shop].index
                        if len(shop_train_idx) > 0:
                            train_df.loc[shop_train_idx, 'prediction_score'] = shop_model.predict_proba(X_shop)[:, 1]
                            train_df.loc[shop_train_idx, '予測差枚数'] = shop_reg.predict(X_shop).astype(int)
                except: pass
                    
    # --- 曜日別モデルの学習 ---
    weekdays_map = {0: '月曜', 1: '火曜', 2: '水曜', 3: '木曜', 4: '金曜', 5: '土曜', 6: '日曜'}
    if 'target_weekday' in train_df.columns:
        for wd in sorted(train_df['target_weekday'].unique()):
            wd_train = train_df[train_df['target_weekday'] == wd]
            if len(wd_train) >= 150:
                X_wd = wd_train[features]
                y_wd = wd_train['target']
                sw_wd = sample_weights.loc[wd_train.index] if sample_weights is not None else None
                
                wd_model = lgb.LGBMClassifier(
                    objective='binary', random_state=42, verbose=-1, 
                    n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8
                )
                try:
                    wd_model.fit(X_wd, y_wd, sample_weight=sw_wd, categorical_feature=cat_features)
                    feature_importances_list.append(pd.DataFrame({
                        'shop_name': weekdays_map.get(wd, f"曜日{wd}"),
                        'category': '曜日',
                        'feature': features,
                        'importance': wd_model.feature_importances_
                    }))
                except: pass
                
    # --- イベント有無別モデルの学習 ---
    if 'イベント名' in train_df.columns:
        train_df_ev = train_df.copy()
        train_df_ev['is_event'] = train_df_ev['イベント名'].apply(lambda x: '通常日' if x == '通常' else 'イベント日')
        for ev_type in ['通常日', 'イベント日']:
            ev_train = train_df_ev[train_df_ev['is_event'] == ev_type]
            if len(ev_train) >= 150:
                X_ev = ev_train[features]
                y_ev = ev_train['target']
                sw_ev = sample_weights.loc[ev_train.index] if sample_weights is not None else None
                
                ev_model = lgb.LGBMClassifier(
                    objective='binary', random_state=42, verbose=-1, 
                    n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8
                )
                try:
                    ev_model.fit(X_ev, y_ev, sample_weight=sw_ev, categorical_feature=cat_features)
                    feature_importances_list.append(pd.DataFrame({
                        'shop_name': ev_type,
                        'category': 'イベント',
                        'feature': features,
                        'importance': ev_model.feature_importances_
                    }))
                except: pass

    feature_importances = pd.concat(feature_importances_list, ignore_index=True) if feature_importances_list else pd.DataFrame()
    
    return predict_df, train_df, feature_importances

# --- 内部関数: 予測の後処理 ---
def _postprocess_predictions(predict_df, train_df):
    specs = get_machine_specs()
    
    def apply_setting5_boost(row):
        score = row.get('prediction_score', 0)
        machine_name = row.get('機種名', '')
        reg_prob = row.get('REG確率', 0)
        games = row.get('累計ゲーム', 0)
        
        if reg_prob <= 0 or games < 3000:
            return score
            
        matched_spec_key = get_matched_spec_key(machine_name, specs)
                    
        if matched_spec_key and "設定5" in specs[matched_spec_key]:
            set5_reg_prob = 1.0 / specs[matched_spec_key]["設定5"]["REG"]
            if reg_prob >= set5_reg_prob:
                score = min(1.0, score + 0.3) # 設定5以上なら大幅にスコアを加算
        return score

    if not predict_df.empty: predict_df['prediction_score'] = predict_df.apply(apply_setting5_boost, axis=1)
    if not train_df.empty: train_df['prediction_score'] = train_df.apply(apply_setting5_boost, axis=1)

    def apply_reliability_penalty(row):
        score = row.get('prediction_score', 0)
        hc = row.get('history_count', 1)
        # 過去データが少ない場合は予測のブレが大きいためスコアを割り引く
        if hc < 14: return score * 0.8
        elif hc < 30: return score * 0.95
        return score
        
    def get_reliability_mark(row):
        hc = row.get('history_count', 1)
        if hc < 14: return "🔻低"
        elif hc < 30: return "🔸中"
        return "🔼高"

    if not predict_df.empty: 
        predict_df['prediction_score'] = predict_df.apply(apply_reliability_penalty, axis=1)
        predict_df['予測信頼度'] = predict_df.apply(get_reliability_mark, axis=1)
    if not train_df.empty: 
        train_df['予測信頼度'] = train_df.apply(get_reliability_mark, axis=1)

    def get_rating(score):
        if score >= 0.85: return 'A'
        elif score >= 0.70: return 'B'
        elif score >= 0.50: return 'C'
        elif score >= 0.30: return 'D'
        else: return 'E'

    def get_reason(row):
        comments, reasons = [], []
        score = row.get('prediction_score', 0)
        if score > 0.8: comments.append("【激アツ】AIの自信度が非常に高いです。")
        
        mean_7d = row.get('mean_7days_diff', 0)
        win_rate_7d = row.get('win_rate_7days', 0)
        diff = row.get('差枚', 0)
        reg_prob = row.get('REG確率', 0)
        is_win_flag = row.get('is_win', 0)
        games = row.get('累計ゲーム', 0)

        if mean_7d < -300:
            if diff <= -1000:
                if games >= 7000:
                    reasons.append(f"直近1週間(平均{int(mean_7d)}枚)不調な上、前日は{int(games)}Gもタコ粘りされて大凹みしており、強烈な**「お詫び(反発)」**が期待できます。")
                else:
                    reasons.append(f"直近1週間(平均{int(mean_7d)}枚)と前日が大きく凹んでおり、**「不調台の反発」**の可能性が高いです。")
            else: reasons.append(f"週間成績は不調(平均{int(mean_7d)}枚)ですが、AIは**「底打ち上昇」**を予測しています。")
        elif mean_7d > 500:
            if win_rate_7d >= 0.5 and is_win_flag == 1:
                reasons.append(f"直近1週間(平均+{int(mean_7d)}枚, 高設定率{win_rate_7d*100:.0f}%)と好調かつ、REG確率(1/{int(1/reg_prob) if reg_prob > 0 else '-'})も優秀で、**「高設定の据え置き」**が期待できます。")
            elif win_rate_7d >= 0.5:
                reasons.append(f"直近1週間(平均+{int(mean_7d)}枚, 高設定率{win_rate_7d*100:.0f}%)と安定して高設定が使われています。")
            elif diff >= 2000:
                reasons.append(f"週間平均はプラスですが、直近の一撃(+{int(diff)}枚)による影響が大きいです。一撃後の回収に警戒が必要です。")
        
        # --- 特殊パターンの検証結果を根拠に反映 ---
        prev2_diff = row.get('prev_差枚')
        if pd.notna(prev2_diff):
            if prev2_diff <= -1000 and diff <= -1000:
                reasons.append("【波・推移】2日連続の大凹み(-1000枚以下)で、グラフ底からの強烈な反発(底上げ)サインが点灯しています。")
            elif prev2_diff < 0 and diff >= 0:
                reasons.append("【波・推移】前々日のマイナスから前日プラスへV字反発しており、右肩上がりの好調ウェーブ続伸に期待できます。")
            elif prev2_diff >= 1000 and diff >= 1000:
                if reg_prob >= (1/300):
                    reasons.append("【波・推移】2日連続の大勝(+1000枚以上)かつREG確率も優秀です。高設定の据え置きによる綺麗な右肩上がりグラフに期待できます。")
                else:
                    reasons.append("【波・推移】2日連続の大勝(+1000枚以上)ですが、REG確率が伴っていません。一撃の波が終わる可能性があり警戒が必要です。")

        # 連続マイナスのリセット狙い
        cons_minus = row.get('連続マイナス日数', 0)
        if cons_minus >= 3:
            reasons.append(f"【特殊】現在{int(cons_minus)}日連続マイナス中です。店舗の「上げリセット(底上げ)」ターゲットになる可能性が高いです。")

        # 連続低稼働のテコ入れ狙い
        cons_low_util = row.get('連続低稼働日数', 0)
        if cons_low_util >= 3:
            reasons.append(f"【特殊】現在{int(cons_low_util)}日連続で放置(1500G未満)されています。店側の「稼働喚起のテコ入れ(見せ台)」のターゲットになる可能性があります。")

        shop_7d = row.get('shop_7days_avg_diff', 0)
        if shop_7d < -150:
            reasons.append(f"【店舗状況】店舗全体が直近1週間回収モード(平均{int(shop_7d)}枚)ですが、あえてこの台を推奨しています。")
        elif shop_7d > 150:
            reasons.append(f"【店舗状況】店舗全体が直近1週間還元モード(平均+{int(shop_7d)}枚)で、全体のベースアップに期待できます。")
            
        mac_30d = row.get('machine_30days_avg_diff', 0)
        if mac_30d > 150:
            reasons.append(f"【機種優遇】過去30日間、この機種(平均+{int(mac_30d)}枚)は店舗から甘く使われている傾向があります。")
        elif mac_30d < -150:
            reasons.append(f"【機種冷遇】過去30日間、この機種(平均{int(mac_30d)}枚)は冷遇気味ですが、この台単体は評価されています。")

        if row.get('is_new_machine', 0) == 1:
            reasons.append("【新台/移動】新台導入または配置変更から1週間以内のため、店側のアピール(高設定投入)が期待できます。")

        big = row.get('BIG', 0)
        reg = row.get('REG', 0)
        
        # 機種固有の設定5基準を判定
        machine_name = row.get('機種名', '')
        matched_spec_key = get_matched_spec_key(machine_name, specs)
        
        is_setting5_over = False
        if matched_spec_key and "設定5" in specs[matched_spec_key] and reg_prob > 0:
            set5_reg_prob_threshold = 1.0 / specs[matched_spec_key]["設定5"]["REG"]
            if games >= 3000 and reg_prob >= set5_reg_prob_threshold:
                is_setting5_over = True

        spec_reg_5 = 1.0 / specs[matched_spec_key].get("設定5", {"REG": 260.0})["REG"] if matched_spec_key else 1/260.0
        spec_big_5 = 1.0 / specs[matched_spec_key].get("設定5", {"BIG": 260.0})["BIG"] if matched_spec_key else 1/260.0

        # BIG確率の計算
        big_prob = big / games if games > 0 else 0
        big_denom = 1 / big_prob if big_prob > 0 else 9999

        if is_setting5_over:
            reasons.append(f"【🌟高設定挙動】前日のREG確率が1/{int(1/reg_prob)}で、機種スペックの**「設定5以上」**の基準を満たしており、強く推奨されます。")
        elif reg > big and reg_prob >= spec_reg_5:            
            if big_denom >= 400:
                if diff <= 0:
                    reasons.append(f"【超不発】BIG確率が1/{int(big_denom)}と極端に欠損していますが、REG確率は設定5以上(1/{int(1/reg_prob)})をキープしている超・狙い目台です。")
                else:
                    reasons.append(f"【特殊】BIGが極端に引けていませんが(1/{int(big_denom)})、REG確率は設定5以上(1/{int(1/reg_prob)})の高設定挙動です。")
            else:
                if diff <= 0:
                    reasons.append(f"【特殊】REG先行(BB欠損)で差枚が沈んでいる、狙い目の「高設定 不発台」です。(REG 1/{int(1/reg_prob)})")
                else:
                    reasons.append(f"【特殊】REG先行かつREG確率が設定5以上(1/{int(1/reg_prob)})の「高設定台」です。")
        elif big >= reg and big_prob >= spec_big_5:
            reasons.append(f"【特殊】BIG先行(1/{int(big_denom)})でBIG確率が設定5以上をキープしています。BIGヒキ強台の据え置き狙いとして期待できます。")
        else:
            if reg_prob > (1/280): reasons.append(f"前日のREG確率が**1/{int(1/reg_prob)}**と高設定水準です。")
            elif reg_prob > (1/350): reasons.append(f"REG確率(1/{int(1/reg_prob)})が悪くなく、粘る価値があります。")
        
        e_avg = row.get('event_avg_diff', 0)
        if e_avg > 150: reasons.append(f"今日はイベント特定日(平均+{int(e_avg)}枚)のため期待値が高いです。")

        evt_name = row.get('イベント名', '通常')
        if evt_name != '通常' and pd.notna(evt_name):
            evt_rank = row.get('イベントランク', '')
            rank_str = f"(ランク{evt_rank})" if evt_rank else ""
            reasons.append(f"店舗イベント「{evt_name}」{rank_str}対象日です。")

        w_avg = row.get('weekday_avg_diff', 0)
        if w_avg > 150:
            wd_name = ['月', '火', '水', '木', '金', '土', '日'][int(row['target_weekday'])] if 'target_weekday' in row and 0 <= row['target_weekday'] <= 6 else ''
            reasons.append(f"{wd_name}曜日はこの店の得意日(平均+{int(w_avg)}枚)です。")

        if row.get('is_corner', 0) == 1: reasons.append("角台（設定優遇枠）のため期待大です。")
        n_avg = row.get('neighbor_avg_diff', 0)
        if n_avg > 300: reasons.append(f"両隣が好調(平均+{int(n_avg)}枚)で、並びや全台系の可能性があります。")
        i_avg = row.get('island_avg_diff', 0)
        if i_avg > 400: reasons.append(f"所属する島全体が好調(平均+{int(i_avg)}枚)で、塊対象の可能性があります。")
        
        if reasons: comments.append(" ".join(reasons))
        else:
            if score > 0.6: comments.append("目立った特徴はありませんが、全体バランスからAIが高く評価しました。")
            else: comments.append("特筆すべき強い根拠はありません。")
        return " ".join(comments)

    if not predict_df.empty:
        predict_df['おすすめ度'] = predict_df['prediction_score'].apply(get_rating)
        if '店名' in predict_df.columns:
            shop_mean = predict_df.groupby('店名')['prediction_score'].transform('mean')
            predict_df['店舗期待度'] = shop_mean.apply(get_rating)
        predict_df['根拠'] = predict_df.apply(get_reason, axis=1)
        
    if not train_df.empty:
        train_df['おすすめ度'] = train_df['prediction_score'].apply(get_rating)
    
    return predict_df, train_df

# ---------------------------------------------------------
# 分析・予測ロジック (メイン関数)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=2, ttl=3600)
def run_analysis(df, df_events=None, df_island=None, shop_hyperparams=None, target_date=None):
    if df.empty: return df, pd.DataFrame(), pd.DataFrame()

    if shop_hyperparams is None:
        shop_hyperparams = {"デフォルト": {'train_months': 3, 'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50}}

    # 1. 特徴量エンジニアリング
    df, features = _generate_features(df, df_events, df_island, target_date)
    if df.empty: return df, pd.DataFrame(), pd.DataFrame()

    train_df = df.dropna(subset=['next_diff']).copy()
    predict_df = df[df['next_diff'].isna()].copy()
    
    # 2. 学習データの期間絞り込み (店舗ごとに適用)
    shop_col = '店名' if '店名' in train_df.columns else ('店舗名' if '店舗名' in train_df.columns else None)
    if shop_col and not train_df.empty:
        filtered_train_dfs = []
        for shop in train_df[shop_col].unique():
            shop_hp = shop_hyperparams.get(shop, shop_hyperparams.get("デフォルト", {}))
            t_m = shop_hp.get('train_months', 3)
            shop_df = train_df[train_df[shop_col] == shop]
            if not shop_df.empty and '対象日付' in shop_df.columns:
                max_d = shop_df['対象日付'].max()
                cutoff = max_d - pd.DateOffset(months=t_m)
                filtered_train_dfs.append(shop_df[shop_df['対象日付'] >= cutoff])
        if filtered_train_dfs:
            train_df = pd.concat(filtered_train_dfs, ignore_index=True)

    # 3. 予測データを最新日に絞り込み
    if '対象日付' in predict_df.columns and not predict_df.empty:
        if shop_col:
            latest_dates = predict_df.groupby(shop_col)['対象日付'].transform('max')
            predict_df = predict_df[predict_df['対象日付'] == latest_dates]
        else:
            max_date = predict_df['対象日付'].max()
            predict_df = predict_df[predict_df['対象日付'] == max_date]
        
    if len(train_df) < 10 or len(predict_df) == 0:
        return predict_df, pd.DataFrame(), pd.DataFrame()

    # 4 & 5. モデル学習と推論 (店舗ごとのパラメータで独立して実行される)
    predict_df, train_df, feature_importances = _train_models(train_df, predict_df, features, shop_hyperparams)

    # 6. 後処理 (スコア補正、根拠の自然言語生成)
    predict_df, train_df = _postprocess_predictions(predict_df, train_df)

    return predict_df, train_df, feature_importances
