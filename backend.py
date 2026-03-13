import os
import pandas as pd
import numpy as np
import lightgbm as lgb # type: ignore
import streamlit as st # type: ignore
import gspread
from google.oauth2.service_account import Credentials

# 定数定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_FILE = os.path.join(BASE_DIR, 'service_account.json')
SPREADSHEET_KEY = '1ylt9mdIkKKk6YRcZh4O05O7fPF4d2BU6VXzboP_vs5s'
SHEET_NAME = 'juggler_raw'

# ---------------------------------------------------------
# データ読み込み・保存関数 (Model / Logic)
# ---------------------------------------------------------
def _get_gspread_client():
    """認証クライアントを取得する共通関数"""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    
    # 1. Streamlit CloudのSecrets機能を確認
    if "gcp_service_account" in st.secrets:
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
        return gspread.authorize(creds)
    
    # 2. ローカルのJSONファイルを確認
    elif os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
        return gspread.authorize(creds)
    
    else:
        raise FileNotFoundError("認証情報が見つかりません。st.secrets または service_account.json を設定してください。")

@st.cache_data(ttl=600)
def load_data():
    """Googleスプレッドシートから生の稼働データを読み込む"""
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet(SHEET_NAME)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        if df.empty: return pd.DataFrame()

        # 前処理
        df.columns = [str(c).strip() for c in df.columns]
        rename_map = {'REG回数': 'REG', 'BIG回数': 'BIG', '店舗名': '店名'}
        df = df.rename(columns=rename_map)

        def convert_prob(val):
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
            if col in df.columns:
                df[col] = df[col].apply(convert_prob)

        num_cols = ['台番号', '累計ゲーム', 'BIG', 'REG', '差枚', '末尾番号', '最終ゲーム']
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if '対象日付' in df.columns:
            df['対象日付'] = pd.to_datetime(df['対象日付'])
            
        return df
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
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
        return
    if 'prediction_score' in df.columns:
        df = df.sort_values('prediction_score', ascending=False).head(5)
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        log_sheet_name = 'prediction_log'
        try: worksheet = sh.worksheet(log_sheet_name)
        except: 
            worksheet = sh.add_worksheet(title=log_sheet_name, rows="1000", cols="20")
            worksheet.append_row(['実行日時', '対象日付', '店名', '台番号', '機種名', 'prediction_score', 'おすすめ度', '予測差枚数', '根拠'])
            
        save_df = df.copy()
        save_df['実行日時'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        target_cols = ['実行日時', '対象日付', '店名', '台番号', '機種名', 'prediction_score', 'おすすめ度', '予測差枚数', '根拠']
        valid_cols = [c for c in target_cols if c in save_df.columns]
        save_df = save_df[valid_cols]
        for col in save_df.columns:
            if pd.api.types.is_datetime64_any_dtype(save_df[col]):
                save_df[col] = save_df[col].dt.strftime('%Y-%m-%d')
        save_df = save_df.fillna('')
        worksheet.append_rows(save_df.values.tolist())
        st.success(f"予測結果（Top 5）を '{log_sheet_name}' シートに保存しました！")
    except Exception as e: st.error(f"保存エラー: {e}")

@st.cache_data(ttl=600)
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
        
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        date_str = event_date.strftime('%Y-%m-%d')
        worksheet.append_row([timestamp, shop_name, date_str, event_name, '', event_rank])
        return True
    except Exception as e:
        st.error(f"イベント保存エラー: {e}")
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

# --- マイ収支管理機能 ---
@st.cache_data(ttl=600)
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
        try: worksheet = sh.worksheet(sheet_name)
        except: 
            worksheet = sh.add_worksheet(title=sheet_name, rows="1000", cols="10")
            worksheet.append_row(['登録日時', '日付', '店名', '台番号', '機種名', '投資', '回収', '収支', 'メモ'])
        
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        date_str = date_obj.strftime('%Y-%m-%d')
        balance = int(recovery) - int(invest)
        
        worksheet.append_row([timestamp, date_str, shop, number, machine, invest, recovery, balance, memo])
        return True
    except Exception as e:
        st.error(f"収支保存エラー: {e}")
        return False

# ---------------------------------------------------------
# 分析・予測ロジック
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_analysis(df, df_events=None, hyperparams=None):
    if df.empty: return df, pd.DataFrame()

    shop_col = '店名' if '店名' in df.columns else ('店舗名' if '店舗名' in df.columns else None)
    
    if '機種名' in df.columns: df['machine_code'] = df['機種名'].astype('category').cat.codes
    if shop_col: df['shop_code'] = df[shop_col].astype('category').cat.codes

    if df_events is not None and not df_events.empty and shop_col:
        events_unique = df_events.drop_duplicates(subset=['店名', 'イベント日付'], keep='last').copy()
        merge_cols = ['店名', 'イベント日付', 'イベント名']
        if 'イベントランク' in events_unique.columns: merge_cols.append('イベントランク')

        df = pd.merge(df, events_unique[merge_cols], left_on=[shop_col, '対象日付'], right_on=['店名', 'イベント日付'], how='left')
        df = df.drop(columns=['店名_y', 'イベント日付_y'], errors='ignore')
        if '店名_x' in df.columns: df = df.rename(columns={'店名_x': '店名'})
        
        df['イベント名'] = df['イベント名'].fillna('通常')
        df['event_code'] = df['イベント名'].astype('category').cat.codes
        if 'イベントランク' in df.columns:
            rank_map = {'S': 5, 'A': 4, 'B': 3, 'C': 2}
            df['event_rank_score'] = df['イベントランク'].map(rank_map).fillna(0)

    if 'REG' in df.columns and 'BIG' in df.columns:
        df['reg_ratio'] = df['REG'] / (df['BIG'] + df['REG'] + 1)

    if shop_col and '機種名' in df.columns and '台番号' in df.columns:
        grp = df.groupby([shop_col, '機種名'])['台番号']
        df['is_corner'] = ((df['台番号'] == grp.transform('min')) | (df['台番号'] == grp.transform('max'))).astype(int)

    if shop_col and '台番号' in df.columns and '対象日付' in df.columns:
        df = df.sort_values([shop_col, '対象日付', '台番号'])
        prev_shop = df[shop_col].shift(1)
        prev_date = df['対象日付'].shift(1)
        prev_no = df['台番号'].shift(1)
        prev_diff = df['差枚'].shift(1)
        next_shop = df[shop_col].shift(-1)
        next_date = df['対象日付'].shift(-1)
        next_no = df['台番号'].shift(-1)
        next_diff = df['差枚'].shift(-1)
        
        is_prev = (df[shop_col] == prev_shop) & (df['対象日付'] == prev_date) & ((df['台番号'] - prev_no) == 1)
        is_next = (df[shop_col] == next_shop) & (df['対象日付'] == next_date) & ((next_no - df['台番号']) == 1)
        
        p_val = np.where(is_prev, prev_diff, np.nan)
        n_val = np.where(is_next, next_diff, np.nan)
        df['neighbor_avg_diff'] = pd.DataFrame({'p': p_val, 'n': n_val}).mean(axis=1).fillna(0)

    sort_keys = [shop_col, '台番号', '対象日付'] if shop_col else ['台番号', '対象日付']
    group_keys = [shop_col, '台番号'] if shop_col else ['台番号']
    df = df.sort_values(sort_keys).reset_index(drop=True)
    
    for col in ['差枚', 'REG確率', '累計ゲーム', '最終ゲーム']:
        if col in df.columns: df[f'prev_{col}'] = df.groupby(group_keys)[col].shift(1)
    
    df['next_diff'] = df.groupby(group_keys)['差枚'].shift(-1)
    df['target'] = (df['next_diff'] > 0).astype(int)
    
    df = df.sort_values('対象日付')
    df['weekday'] = df['対象日付'].dt.dayofweek
    df['weekday_avg_diff'] = df.groupby('weekday')['差枚'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    if '日付要素' in df.columns:
        df['event_avg_diff'] = df.groupby('日付要素')['差枚'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)

    df = df.sort_values(sort_keys)
    df['mean_7days_diff'] = df.groupby(group_keys)['差枚'].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()).fillna(0)

    features = ['累計ゲーム', 'REG確率', 'BIG確率', '差枚', '末尾番号', 'weekday', 'weekday_avg_diff', 'mean_7days_diff']
    for f in ['machine_code', 'shop_code', 'reg_ratio', 'is_corner', 'neighbor_avg_diff', 'event_avg_diff', 'prev_最終ゲーム', 'event_code', 'event_rank_score', 'prev_差枚', 'prev_REG確率', 'prev_累計ゲーム']:
        if f in df.columns: features.append(f)

    train_df = df.dropna(subset=['next_diff'])
    predict_df = df[df['next_diff'].isna()].copy()
    
    if '対象日付' in predict_df.columns and not predict_df.empty:
        if shop_col:
            latest_dates = predict_df.groupby(shop_col)['対象日付'].transform('max')
            predict_df = predict_df[predict_df['対象日付'] == latest_dates]
        else:
            max_date = predict_df['対象日付'].max()
            predict_df = predict_df[predict_df['対象日付'] == max_date]
        
    if len(train_df) < 10 or len(predict_df) == 0:
        return predict_df, pd.DataFrame()

    X, y = train_df[features], train_df['target']
    sample_weights = None
    if '対象日付' in train_df.columns:
        max_date = train_df['対象日付'].max()
        days_diff = (max_date - train_df['対象日付']).dt.days
        sample_weights = 0.995 ** days_diff
    
    if hyperparams is None:
        hyperparams = {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'max_depth': -1}

    # LightGBMの初期化エラーを防ぐため、パラメータを明示的に渡す
    n_est = hyperparams.get('n_estimators', 100)
    lr = hyperparams.get('learning_rate', 0.1)
    nl = hyperparams.get('num_leaves', 31)
    md = hyperparams.get('max_depth', -1)

    model = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1, n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md)
    model.fit(X, y, sample_weight=sample_weights)
    
    predict_df['prediction_score'] = model.predict_proba(predict_df[features])[:, 1]
    
    reg_model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md)
    reg_model.fit(X, train_df['next_diff'], sample_weight=sample_weights)
    predict_df['予測差枚数'] = reg_model.predict(predict_df[features]).astype(int)

    train_df['prediction_score'] = model.predict_proba(train_df[features])[:, 1]
    train_df['予測差枚数'] = reg_model.predict(train_df[features]).astype(int)

    def get_rating(score):
        if score >= 0.8: return 'A'
        elif score >= 0.6: return 'B'
        elif score >= 0.4: return 'C'
        elif score >= 0.2: return 'D'
        else: return 'E'

    def get_reason(row):
        comments, reasons = [], []
        score = row.get('prediction_score', 0)
        if score > 0.8: comments.append("【激アツ】AIの自信度が非常に高いです。")
        
        mean_7d = row.get('mean_7days_diff', 0)
        diff = row.get('差枚', 0)
        if mean_7d < -300:
            if diff < -1000: reasons.append(f"直近1週間(平均{int(mean_7d)}枚)と前日が大きく凹んでおり、**「不調台の反発」**の可能性が高いです。")
            else: reasons.append(f"週間成績は不調(平均{int(mean_7d)}枚)ですが、AIは**「底打ち上昇」**を予測しています。")
        elif mean_7d > 500:
            reasons.append(f"直近1週間(平均+{int(mean_7d)}枚)と好調を維持しており、**「据え置き」**が期待できます。")
        
        reg_prob = row.get('REG確率', 0)
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
            wd_name = ['月', '火', '水', '木', '金', '土', '日'][int(row['weekday'])] if 0 <= row['weekday'] <= 6 else ''
            reasons.append(f"{wd_name}曜日はこの店の得意日(平均+{int(w_avg)}枚)です。")

        if row.get('is_corner', 0) == 1: reasons.append("角台（設定優遇枠）の期待大です。")
        n_avg = row.get('neighbor_avg_diff', 0)
        if n_avg > 300: reasons.append(f"両隣が好調(平均+{int(n_avg)}枚)で、並びや全台系の可能性があります。")
        
        if reasons: comments.append(" ".join(reasons))
        else:
            if score > 0.6: comments.append("目立った特徴はありませんが、全体バランスからAIが高く評価しました。")
            else: comments.append("特筆すべき強い根拠はありません。")
        return " ".join(comments)

    predict_df['おすすめ度'] = predict_df['prediction_score'].apply(get_rating)
    train_df['おすすめ度'] = train_df['prediction_score'].apply(get_rating)
    if '店名' in predict_df.columns:
        shop_mean = predict_df.groupby('店名')['prediction_score'].transform('mean')
        predict_df['店舗期待度'] = shop_mean.apply(get_rating)
    predict_df['根拠'] = predict_df.apply(get_reason, axis=1)
    
    return predict_df, train_df
