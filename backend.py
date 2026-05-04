import os
import pandas as pd
import numpy as np
import lightgbm as lgb # type: ignore
import streamlit as st # type: ignore
import gspread
import unicodedata
import hashlib
import pickle
import glob
import re
from google.oauth2.service_account import Credentials
import time
from utils import get_valid_play_mask, get_confidence_indicator, get_matched_spec_key, classify_shop_eval, calculate_high_setting_mask
from config import BASE_FEATURES, FEATURE_NAME_MAP, MACHINE_SPECS
from model_trainer import train_models
from shop_trends import calculate_shop_trends, apply_trends_to_row, analyze_sueoki_and_change_triggers, diagnose_allocation_types, evaluate_sueoki_premise
from postprocessor import postprocess_predictions

try:
    import jpholiday
except ImportError:
    jpholiday = None
try:
    from lstm_feature_extractor import add_lstm_features
except ImportError:
    add_lstm_features = None

# 定数定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_FILE = os.path.join(BASE_DIR, 'service_account.json')
SPREADSHEET_KEY = '1ylt9mdIkKKk6YRcZh4O05O7fPF4d2BU6VXzboP_vs5s'
SHEET_NAME = 'juggler_raw'
HISTORY_CACHE_FILE = os.path.join(BASE_DIR, 'history_cache.parquet')

# 🚨【重要】プログラム（計算式や特徴量など）を変更した際は、必ずここのバージョン番号をカウントアップしてください！
# （「予測の実績検証」ページで、新旧ロジックの成績比較ができるようになります）
APP_VERSION = "v4.64.0" 

def analyze_sueoki_and_change_triggers(df_train, shop_name, shop_col='店名'):
    from shop_trends import analyze_sueoki_and_change_triggers as _analyze
    return _analyze(df_train, shop_name, shop_col)

# ---------------------------------------------------------
# 機種スペック情報
# ---------------------------------------------------------
def get_machine_specs():
    return MACHINE_SPECS

def calculate_setting_score(g, act_b, act_r, machine_name, diff=None, shop_avg_g=4000, 
                            penalty_reg=15, penalty_big=5, low_g_penalty=30, 
                            use_strict_scoring=True, return_details=False):
    """
    稼働データからベイズ推定（事後確率）とZスコアを用いて「設定5近似度（100点満点）」を計算する
    """
    import math
    if pd.isna(g) or g <= 0:
        if return_details:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return np.nan

    specs = get_machine_specs()
    matched_spec = get_matched_spec_key(machine_name, specs)
    
    ms = specs.get(matched_spec, specs['ジャグラー（デフォルト）']) if matched_spec else specs['ジャグラー（デフォルト）']
    
    s1 = ms.get("設定1", {"BIG": 280.0, "REG": 400.0})
    s4 = ms.get("設定4", {"BIG": 260.0, "REG": 300.0})
    s5 = ms.get("設定5", s4)
    s6 = ms.get("設定6", s5)
    
    full_specs = {1: s1, 4: s4, 5: s5, 6: s6}
    for s in [2, 3]:
        full_specs[s] = {}
        for k in ["BIG", "REG"]:
            p1 = 1.0 / s1.get(k, 300.0)
            p4 = 1.0 / s4.get(k, 300.0)
            p_s = p1 + (p4 - p1) * (s - 1) / 3.0
            full_specs[s][k] = 1.0 / p_s if p_s > 0 else 999.0

    # 1. ベイズ推定による事後確率の計算
    # 事前確率はフラット（均等）にして、純粋にデータからの尤度を評価する
    log_likelihoods = []
    
    # ユーザーが設定した「甘め/標準/辛め」のペナルティ値に応じて、BIGの評価ウェイトを変える
    # penalty_regが大きい(辛め)ほど、REGを重視しBIGのヒキを軽視する
    big_weight = 0.5
    if penalty_reg >= 20: big_weight = 0.3 # 辛め：REG超重視
    elif penalty_reg <= 10: big_weight = 0.8 # 甘め：BIGも評価
        
    for i in range(1, 7):
        p_b = 1.0 / full_specs[i]["BIG"]
        p_r = 1.0 / full_specs[i]["REG"]
        exp_b = g * p_b
        exp_r = g * p_r
        
        # ポアソン分布の対数尤度近似 (定数項の階乗は比較時に相殺されるため省略)
        ll_b = (act_b * math.log(exp_b) - exp_b) * big_weight if exp_b > 0 else 0
        ll_r = act_r * math.log(exp_r) - exp_r if exp_r > 0 else 0
        log_likelihoods.append(ll_b + ll_r)
        
    max_ll = max(log_likelihoods)
    # オーバーフロー防止
    posteriors_unnormalized = [math.exp(max(-700, ll - max_ll)) for ll in log_likelihoods]
    sum_post = sum(posteriors_unnormalized)
    posteriors = [p / sum_post for p in posteriors_unnormalized]
    
    # ベーススコアの算出: 設定1〜6の事後確率を重み付け加算
    # 設定6なら100点、設定5なら90点、設定4なら60点、設定3なら30点、設定2なら10点相当
    bayes_score = (posteriors[5] * 100) + (posteriors[4] * 90) + (posteriors[3] * 60) + (posteriors[2] * 30) + (posteriors[1] * 10)
    
    # 2. Zスコア（設定1の否定度）による補正
    p_r1 = 1.0 / full_specs[1]["REG"]
    exp_r1 = g * p_r1
    std_r1 = math.sqrt(g * p_r1 * (1.0 - p_r1)) if g > 0 else 0
    z_score_reg = (act_r - exp_r1) / std_r1 if std_r1 > 0 else 0
    
    # Zスコアが1.64（上位5%）以上なら、稼働が少なく事後確率がバラけていてもベーススコアを底上げする
    if z_score_reg >= 1.64:
        # Z=1.64で最低60点、Z=3.0で最低90点に近づくカーブ
        z_bonus = min(90.0, 50.0 + (z_score_reg - 1.64) * 20.0)
        bayes_score = max(bayes_score, z_bonus)

    # 3. 稼働ゲーム数(G数)によるスコアのスケーリング（期待値が未収束なデータへのペナルティ）
    total_score = bayes_score
    if use_strict_scoring:
        if g <= 3000:
            g_factor = g / 4000.0  # 3000Gでも0.75倍のペナルティがかかる
        else:
            g_factor = 0.75 + ((g - 3000) / 8000.0)
            
        g_factor = min(1.0, max(0.1, g_factor))
        
        # 1000G未満の低稼働はさらにペナルティを重くする
        if g < 1000:
            g_factor *= (g / 1000.0)
            
        # 3000G未満で大勝ち（+1500枚以上）している台への救済（スコア底上げ）
        if g < 3000 and diff is not None and diff >= 1500:
            win_bonus = min(0.4, (diff - 1000) / 2500.0)
            g_factor = min(1.0, g_factor + win_bonus)
            
        total_score *= g_factor
        
        # タコ粘り(7000G以上)で確率がついてきている台はボーナス加点
        if g >= 7000 and z_score_reg >= 0:
            bonus = min(10.0, (g - 7000) / 300.0)
            total_score = min(100.0, total_score + bonus)
    else:
        # 旧ロジックに合わせたマイルドなペナルティ
        discount_target_g = max(2500, min(5000, shop_avg_g))
        if g < discount_target_g:
            multiplier = 0.90 + (g / float(discount_target_g)) * 0.10
            total_score *= multiplier
        if g < 1000:
            total_score *= (1 - ((1000 - g) / 1000.0) * (low_g_penalty / 100.0))
            
    # 見切り台（ノーボナ等）の減点
    is_abandoned = False
    tot_b_r = act_b + act_r
    if g >= 500 and tot_b_r == 0: is_abandoned = True
    elif g >= 1000 and tot_b_r > 0 and (g / tot_b_r) >= 400: is_abandoned = True
    elif g >= 1500 and tot_b_r > 0 and (g / tot_b_r) >= 300: is_abandoned = True
    
    if is_abandoned:
        total_score *= 0.5
        
    final_score = max(0.0, min(100.0, total_score))
    
    if return_details:
        exp_b5 = g * (1.0 / full_specs[5]["BIG"])
        exp_r5 = g * (1.0 / full_specs[5]["REG"])
        diff_b = act_b - exp_b5
        diff_r = act_r - exp_r5
        return final_score, exp_b, exp_r, diff_b, diff_r
    return final_score

def get_setting_score_from_row(row, shop_avg_g=4000, g_col='累計ゲーム', b_col='BIG', r_col='REG', m_col='機種名', d_col='差枚', return_details=False, use_strict_scoring=True):
    """
    DataFrameの各行(row)から設定5近似度を計算するヘルパー関数
    """
    g = pd.to_numeric(row.get(g_col, 0), errors='coerce')
    act_b = pd.to_numeric(row.get(b_col, 0), errors='coerce')
    act_r = pd.to_numeric(row.get(r_col, 0), errors='coerce')
    diff = pd.to_numeric(row.get(d_col, 0), errors='coerce')
    machine = row.get(m_col, '')
    
    penalty_reg = st.session_state.get('penalty_reg', 15)
    penalty_big = st.session_state.get('penalty_big', 5)
    low_g_penalty = st.session_state.get('low_g_penalty', 30)
    
    return calculate_setting_score(
        g=g, act_b=act_b, act_r=act_r, machine_name=machine, diff=diff,
        shop_avg_g=shop_avg_g, penalty_reg=penalty_reg, penalty_big=penalty_big,
        low_g_penalty=low_g_penalty, use_strict_scoring=use_strict_scoring, return_details=return_details
    )

# ---------------------------------------------------------
# データ読み込み・保存関数 (Model / Logic)
# ---------------------------------------------------------
def clear_local_cache():
    """ローカルの履歴キャッシュファイルを削除して全件読み直しを強制する"""
    if os.path.exists(HISTORY_CACHE_FILE):
        try:
            os.remove(HISTORY_CACHE_FILE)
        except Exception:
            pass
    for f_path in glob.glob(os.path.join(BASE_DIR, 'ai_cache_*.pkl')):
        try:
            os.remove(f_path)
        except Exception:
            pass
    return True

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
        
        # 500 Internal Error 等の一時的なAPIエラー対策としてリトライ処理を追加
        max_retries = 3
        data = None
        for attempt in range(max_retries):
            try:
                sh = gc.open_by_key(SPREADSHEET_KEY)
                worksheet = sh.worksheet(SHEET_NAME)
                data = worksheet.get_all_values()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt) # 1秒、2秒と待機して再試行
                    continue
                else:
                    raise e
                    
        if not data or len(data) < 2: return pd.DataFrame()
        raw_df = pd.DataFrame(data[1:], columns=data[0])
        
        if raw_df.empty: return pd.DataFrame()

        # 前処理
        raw_df.columns = [str(c).strip() for c in raw_df.columns]
        raw_df.columns = [re.sub(r'[",\[\]{}:]', '', c) for c in raw_df.columns]
        raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]
        date_col = '対象日付'
        if date_col not in raw_df.columns: return pd.DataFrame()
        
        # --- 列名の揺れを吸収（最も早い段階で統一） ---
        rename_map = {'REG回数': 'REG', 'BIG回数': 'BIG', '店舗名': '店名'}
        raw_df = raw_df.rename(columns=rename_map)

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
                # キャッシュデータの列名も最新仕様に統一
                history_df = history_df.rename(columns=rename_map)
                history_df.columns = [re.sub(r'[",\[\]{}:]', '', str(c)) for c in history_df.columns]
                history_df = history_df.loc[:, ~history_df.columns.duplicated()]
                # キャッシュ内のデータが古すぎる/新しすぎる場合を考慮し、確定済み範囲のみ残す
                if date_col in history_df.columns:
                    history_df = history_df[history_df[date_col] < freeze_threshold]
                else:
                    history_df = pd.DataFrame()
            except:
                history_df = pd.DataFrame()

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
        
        # --- 子役（ぶどう）確率の逆算 ---
        status_col = next((c for c in target_raw_df.columns if 'ステータス' in c or 'OK' in c.upper()), None)
        if status_col:
            ok_mask = target_raw_df[status_col].astype(str).str.strip().str.upper() == 'OK'
            g_arr = target_raw_df['累計ゲーム'].astype(float)
            b_arr = target_raw_df['BIG'].astype(float)
            r_arr = target_raw_df['REG'].astype(float)
            diff_arr = target_raw_df['差枚'].astype(float)
            
            valid_mask = ok_mask & (g_arr >= 3000) # ブドウはブレが大きく設定差も小さいため3000G以上で計算
            
            specs = get_machine_specs()
            b_p_list, r_p_list, g_p_list = [], [], []
            for m in target_raw_df['機種名']:
                sk = get_matched_spec_key(m, specs)
                ms = specs.get(sk, specs['ジャグラー（デフォルト）'])
                b_p_list.append(ms.get('BIG獲得', 252))
                r_p_list.append(ms.get('REG獲得', 96))
                g_p_list.append(ms.get('ぶどう獲得', 7))
                
            in_tokens = g_arr * 3
            out_tokens = in_tokens + diff_arr
            bonus_out = b_arr * np.array(b_p_list) + r_arr * np.array(r_p_list)
            
            # --- 他小役の概算OUTを厳しめに設定 (ぶどう確率が甘く出すぎるのを防ぐ) ---
            # リプレイ: 1/7.298 (3枚) ≒ 0.411 OUT
            # チェリー: 1/33 (2枚) 取得率95%想定 ≒ 0.058 OUT
            # ベル＆ピエロ: 1/1000 (14枚＆10枚) 取得も考慮 ≒ 0.016 OUT
            # さらにボーナス成立後のロス等を加味して厳しく設定
            other_out = g_arr * 0.4900
            
            grape_out = out_tokens - bonus_out - other_out
            grape_count = grape_out / np.array(g_p_list)
            
            calc_prob = np.where(valid_mask & (grape_count > 0), g_arr / grape_count, np.nan)
            # 異常値(極端なブレやエラー)は除外
            target_raw_df['推定ぶどう確率'] = np.where((calc_prob > 4.5) & (calc_prob < 8.0), calc_prob, np.nan)

        # --- キャッシュと新規データを結合 ---
        if not history_df.empty:
            df = pd.concat([history_df, target_raw_df], ignore_index=True)
        else:
            df = target_raw_df

        # --- 重複データの排除 (スプレッドシート上の二重登録を防ぐ) ---
        shop_col_name = '店名' if '店名' in df.columns else ('店舗名' if '店舗名' in df.columns else None)
        dup_subset = [shop_col_name, '台番号', date_col] if shop_col_name else ['台番号', date_col]
        df = df.drop_duplicates(subset=dup_subset, keep='last').reset_index(drop=True)

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
        
        # get_all_records はヘッダー異常でエラーになるため、get_all_values を使用
        data = worksheet.get_all_values()
        if not data or len(data) < 2: return pd.DataFrame()
        df = pd.DataFrame(data[1:], columns=data[0])
        
        df.columns = [str(c).strip() for c in df.columns]
        df.columns = [re.sub(r'[",\[\]{}:]', '', c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 古いカラム名「予想設定5以上確率」との互換性維持
        if '予想設定5以上確率' in df.columns and 'prediction_score' not in df.columns:
            df['prediction_score'] = pd.to_numeric(df['予想設定5以上確率'], errors='coerce')
        if '変更期待度' in df.columns and 'prediction_score' not in df.columns:
            df['prediction_score'] = pd.to_numeric(df['変更期待度'], errors='coerce')
        if '据え置き期待度' in df.columns and 'sueoki_score' not in df.columns:
            df['sueoki_score'] = pd.to_numeric(df['据え置き期待度'], errors='coerce')
            if df['prediction_score'].max() > 1.0:
                df['prediction_score'] = df['prediction_score'] / 100.0
                
        # 空文字などが混入して画面側で計算エラーになるのを防ぐため、確実に数値型に変換
        if 'prediction_score' in df.columns:
            df['prediction_score'] = pd.to_numeric(df['prediction_score'], errors='coerce')
        if 'sueoki_score' in df.columns:
            df['sueoki_score'] = pd.to_numeric(df['sueoki_score'], errors='coerce')
        return df
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_daily_shop_scores():
    """店舗全体の日別平均期待度を読み込む"""
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('daily_shop_scores')
        
        data = worksheet.get_all_values()
        if not data or len(data) < 2: return pd.DataFrame()
        df = pd.DataFrame(data[1:], columns=data[0])
        
        if '店舗平均期待度' in df.columns:
            df['店舗平均期待度'] = pd.to_numeric(df['店舗平均期待度'], errors='coerce')
        if '予測平均差枚' in df.columns:
            df['予測平均差枚'] = pd.to_numeric(df['予測平均差枚'], errors='coerce')
        if '変更平均期待度' in df.columns:
            df['変更平均期待度'] = pd.to_numeric(df['変更平均期待度'], errors='coerce')
        if '据え置き平均期待度' in df.columns:
            df['据え置き平均期待度'] = pd.to_numeric(df['据え置き平均期待度'], errors='coerce')
        if '店舗台数' in df.columns:
            df['店舗台数'] = pd.to_numeric(df['店舗台数'], errors='coerce')
        return df
    except: return pd.DataFrame()

def save_prediction_log(df):
    if df.empty:
        st.warning("保存するデータがありません。")
        return False
    
    save_df_initial = df.copy()
    
    # --- 1. 店舗全体の平均期待度を全台データから計算して別シートに保存 ---
    if 'prediction_score' in save_df_initial.columns:
        shop_col_for_score = '店名' if '店名' in save_df_initial.columns else ('店舗名' if '店舗名' in save_df_initial.columns else None)
        if shop_col_for_score:
            try:
                gc = _get_gspread_client()
                sh = gc.open_by_key(SPREADSHEET_KEY)
                score_sheet_name = 'daily_shop_scores'
                try: 
                    score_ws = sh.worksheet(score_sheet_name)
                    existing_score_data = score_ws.get_all_values()
                except gspread.exceptions.WorksheetNotFound: 
                    score_ws = sh.add_worksheet(title=score_sheet_name, rows="1000", cols="6")
                    existing_score_data = []

                SCORE_HEADER = ['実行日時', '予測対象日', '店名', '店舗平均期待度', '予測平均差枚', '店舗台数', '変更平均期待度', '据え置き平均期待度']
                if existing_score_data and len(existing_score_data) > 1:
                    df_score_existing = pd.DataFrame(existing_score_data[1:], columns=existing_score_data[0])
                    for c in SCORE_HEADER:
                        if c not in df_score_existing.columns: df_score_existing[c] = ''
                    df_score_existing = df_score_existing[SCORE_HEADER]
                else:
                    df_score_existing = pd.DataFrame(columns=SCORE_HEADER)

                # 新しいスコアの計算 (全台の平均)
                temp_df = save_df_initial.copy()
                if 'next_date' in temp_df.columns:
                    temp_df['予測対象日'] = temp_df['next_date']
                else:
                    temp_df['予測対象日'] = temp_df['対象日付'] + pd.Timedelta(days=1)
                    
                if '予測差枚数' not in temp_df.columns:
                    temp_df['予測差枚数'] = np.nan
                    
                if 'sueoki_score' in temp_df.columns:
                    temp_df['max_score'] = temp_df[['prediction_score', 'sueoki_score']].max(axis=1)
                else:
                    temp_df['max_score'] = temp_df['prediction_score']
                    
                if 'sueoki_score' not in temp_df.columns:
                    temp_df['sueoki_score'] = 0.0
                    
                df_score_new = temp_df.groupby([shop_col_for_score, '予測対象日']).agg(
                    店舗平均期待度=('max_score', 'mean'),
                    予測平均差枚=('予測差枚数', 'mean'),
                    変更平均期待度=('prediction_score', 'mean'),
                    据え置き平均期待度=('sueoki_score', 'mean'),
                    店舗台数=('台番号', 'nunique')
                ).reset_index()
                df_score_new = df_score_new.rename(columns={shop_col_for_score: '店名'})
                df_score_new['実行日時'] = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
                df_score_new['予測対象日'] = pd.to_datetime(df_score_new['予測対象日']).dt.strftime('%Y-%m-%d')
                df_score_new = df_score_new[SCORE_HEADER]
                
                # 既存データの該当部分を削除
                if not df_score_existing.empty:
                    keys_to_remove = set(zip(df_score_new['予測対象日'].astype(str), df_score_new['店名'].astype(str)))
                    mask = df_score_existing.apply(lambda x: (str(x.get('予測対象日')), str(x.get('店名'))) in keys_to_remove, axis=1)
                    df_score_existing = df_score_existing[~mask]

                df_score_combined = pd.concat([df_score_existing, df_score_new], ignore_index=True)
                df_score_combined = df_score_combined[SCORE_HEADER].fillna('')
                final_score_data = [SCORE_HEADER] + df_score_combined.values.tolist()

                score_ws.clear()
                try: score_ws.update(values=final_score_data, range_name='A1')
                except TypeError:
                    try: score_ws.update('A1', final_score_data)
                    except Exception: score_ws.update(final_score_data)
            except Exception as e:
                print(f"店舗平均期待度の保存エラー: {e}")
    
    # --- 2. 保存する前に、各店舗の上位10%（最低3台）に絞り込む ---
    if 'prediction_score' in save_df_initial.columns:
        shop_col = '店名' if '店名' in save_df_initial.columns else ('店舗名' if '店舗名' in save_df_initial.columns else None)
        has_sueoki = 'sueoki_score' in save_df_initial.columns
        
        if shop_col:
            df_list = []
            for shop_name, group in save_df_initial.groupby(shop_col):
                limit = max(3, int(len(group) * 0.10))
                if has_sueoki:
                    # 変更期待度の上位と据え置き期待度の上位を完全に独立して抽出
                    change_top = group.sort_values('prediction_score', ascending=False).head(limit)
                    sueoki_top = group.sort_values('sueoki_score', ascending=False).head(limit)
                    # 結合して重複を排除（両方でランクインした台は1つにまとまる）
                    combined_top = pd.concat([change_top, sueoki_top]).drop_duplicates(subset=['台番号'])
                    df_list.append(combined_top)
                else:
                    df_list.append(group.sort_values('prediction_score', ascending=False).head(limit))
                    
            if df_list:
                save_df_initial = pd.concat(df_list, ignore_index=True)
            else:
                save_df_initial = pd.DataFrame(columns=save_df_initial.columns)
        else:
            limit = max(3, int(len(save_df_initial) * 0.10))
            if has_sueoki:
                change_top = save_df_initial.sort_values('prediction_score', ascending=False).head(limit)
                sueoki_top = save_df_initial.sort_values('sueoki_score', ascending=False).head(limit)
                save_df_initial = pd.concat([change_top, sueoki_top]).drop_duplicates(subset=['台番号'])
            else:
                save_df_initial = save_df_initial.sort_values('prediction_score', ascending=False).head(limit)
            
        if save_df_initial.empty:
            st.warning("保存する推奨台がありません。")
            return False

    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        log_sheet_name = 'prediction_log'
        
        # スプレッドシートのヘッダーが手動操作で壊れた場合（重複など）にも耐えられるように、保存する列を厳格に固定する
        STANDARD_HEADER = ['実行日時', '予測対象日', '対象日付', '店名', '台番号', '機種名', '変更期待度', '据え置き期待度', '予測信頼度', 'おすすめ度', '予測差枚数', '根拠', 'ai_version', 'app_version']
        
        try: 
            worksheet = sh.worksheet(log_sheet_name)
            existing_data = worksheet.get_all_values()
        except gspread.exceptions.WorksheetNotFound: 
            worksheet = sh.add_worksheet(title=log_sheet_name, rows="1000", cols="15")
            existing_data = []

        # 既存データのパース (ユーザー操作による重複列や不要な列の混入を綺麗に掃除する)
        if existing_data and len(existing_data) > 1:
            raw_header = existing_data[0]
            raw_header = ['変更期待度' if c in ['予想設定5以上確率', 'prediction_score'] else c for c in raw_header]
            raw_header = ['据え置き期待度' if c == 'sueoki_score' else c for c in raw_header]
            
            df_existing = pd.DataFrame(existing_data[1:], columns=raw_header)
            
            # 列名が重複している場合（スプレッドシート上でドラッグコピーしてしまった等）、最初の列だけ残して他は捨てる
            df_existing = df_existing.loc[:, ~df_existing.columns.duplicated()]
            
            # 旧データの「店舗名」が消滅するのを防ぐため「店名」に引き継ぐ
            if '店名' not in df_existing.columns and '店舗名' in df_existing.columns:
                df_existing['店名'] = df_existing['店舗名']
            
            for c in STANDARD_HEADER:
                if c not in df_existing.columns:
                    df_existing[c] = ''
            df_existing = df_existing[STANDARD_HEADER]
            
            if '変更期待度' in df_existing.columns:
                df_existing['変更期待度'] = pd.to_numeric(df_existing['変更期待度'], errors='coerce')
            if '据え置き期待度' in df_existing.columns:
                df_existing['据え置き期待度'] = pd.to_numeric(df_existing['据え置き期待度'], errors='coerce')
        else:
            df_existing = pd.DataFrame(columns=STANDARD_HEADER)
            
        save_df = save_df_initial.copy()
        if 'prediction_score' in save_df.columns:
            save_df = save_df.rename(columns={'prediction_score': '変更期待度'})
        if 'sueoki_score' in save_df.columns:
            save_df = save_df.rename(columns={'sueoki_score': '据え置き期待度'})
            
        save_df['実行日時'] = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        if 'ai_version' not in save_df.columns:
            save_df['ai_version'] = "不明"
        save_df['app_version'] = APP_VERSION
            
        if '店名' not in save_df.columns and '店舗名' in save_df.columns:
            save_df['店名'] = save_df['店舗名']
            
        if 'next_date' in save_df.columns:
            save_df['予測対象日'] = save_df['next_date']
        else:
            # 確実に予測対象日を生成する
            max_date_in_df = pd.to_datetime(save_df['対象日付']).max()
            target_date = max_date_in_df + pd.Timedelta(days=1)
            save_df['予測対象日'] = target_date
            
        for col in save_df.columns:
            if pd.api.types.is_datetime64_any_dtype(save_df[col]):
                save_df[col] = save_df[col].dt.strftime('%Y-%m-%d')

        # 保存データも標準ヘッダーの形にカッチリ揃える
        for c in STANDARD_HEADER:
            if c not in save_df.columns:
                save_df[c] = ''
        save_df = save_df[STANDARD_HEADER].fillna('')
            
        # --- 今回保存する「対象日」と「店舗」の既存データをあらかじめ削除する ---
        if not df_existing.empty:
            save_dates = save_df['予測対象日'].astype(str).unique()
            
            # 既存データと新規保存データの店舗カラム名をそれぞれ正しく取得
            existing_shop_col = '店名' if '店名' in df_existing.columns else ('店舗名' if '店舗名' in df_existing.columns else None)
            save_shop_col = '店名' if '店名' in save_df.columns else ('店舗名' if '店舗名' in save_df.columns else None)
            
            if existing_shop_col and save_shop_col:
                keys_to_remove = set(zip(save_df['予測対象日'].astype(str), save_df[save_shop_col].astype(str)))
                mask = df_existing.apply(lambda x: (str(x.get('予測対象日')), str(x.get(existing_shop_col))) in keys_to_remove, axis=1)
                df_existing = df_existing[~mask]
            else:
                mask = df_existing['予測対象日'].astype(str).isin(save_dates)
                df_existing = df_existing[~mask]

        # 新旧データを結合
        df_combined = pd.concat([df_existing, save_df], ignore_index=True)
        
        # 保存用に台番号のフォーマットを綺麗に統一する
        if '台番号' in df_combined.columns:
            df_combined['台番号'] = df_combined['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
            
        # 一時カラムを除外し、順番を固定してリスト化
        df_combined = df_combined[STANDARD_HEADER].fillna('')
        final_data = [STANDARD_HEADER] + df_combined.values.tolist()
        
        # シートをクリアして一括更新
        worksheet.clear()
        try:
            worksheet.update(values=final_data, range_name='A1')
        except TypeError:
            try:
                worksheet.update('A1', final_data)
            except Exception:
                worksheet.update(final_data)
                
        st.success(f"予測結果（各店舗上位10%）を '{log_sheet_name}' シートに保存（上書き）しました！")
        return True
    except Exception as e: 
        st.error(f"保存エラー: {e}")
        return False
        
def delete_old_prediction_logs(months):
    """指定した月数より古い予測ログを削除する（0の場合は全削除）"""
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('prediction_log')
        
        if months == 0:
            data = worksheet.get_all_values()
            if data:
                header = data[0]
                worksheet.clear()
                try: worksheet.update(values=[header], range_name='A1')
                except TypeError: worksheet.update('A1', [header])
                # --- 店舗平均期待度ログも全削除 ---
                try:
                    score_ws = sh.worksheet('daily_shop_scores')
                    s_data = score_ws.get_all_values()
                    if s_data:
                        score_ws.clear()
                        try: score_ws.update(values=[s_data[0]], range_name='A1')
                        except TypeError: score_ws.update('A1', [s_data[0]])
                except: pass
                return len(data) - 1
            return 0
            
        data = worksheet.get_all_values()
        if len(data) <= 1: return 0
            
        header = data[0]
        if '実行日時' not in header: return 0
            
        df = pd.DataFrame(data[1:], columns=header)
        df['実行日時'] = pd.to_datetime(df['実行日時'], errors='coerce')
        
        cutoff_date = (pd.Timestamp.now(tz='Asia/Tokyo') - pd.DateOffset(months=months)).tz_localize(None)
        df_keep = df[df['実行日時'] >= cutoff_date].copy()
        
        if len(df) == len(df_keep): return 0
            
        deleted_count = len(df) - len(df_keep)
        for col in df_keep.columns:
            if pd.api.types.is_datetime64_any_dtype(df_keep[col]): df_keep[col] = df_keep[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
        final_data = [header] + df_keep.fillna('').values.tolist()
        worksheet.clear()
        try: worksheet.update(values=final_data, range_name='A1')
        except TypeError:
            try: worksheet.update('A1', final_data)
            except Exception: worksheet.update(final_data)

        # --- 店舗平均期待度ログも古いものを削除 ---
        try:
            score_ws = sh.worksheet('daily_shop_scores')
            s_data = score_ws.get_all_values()
            if len(s_data) > 1:
                s_df = pd.DataFrame(s_data[1:], columns=s_data[0])
                if '実行日時' in s_df.columns:
                    s_df['実行日時'] = pd.to_datetime(s_df['実行日時'], errors='coerce')
                    s_df_keep = s_df[s_df['実行日時'] >= cutoff_date].copy()
                    for col in s_df_keep.columns:
                        if pd.api.types.is_datetime64_any_dtype(s_df_keep[col]): s_df_keep[col] = s_df_keep[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    s_final_data = [s_data[0]] + s_df_keep.fillna('').values.tolist()
                    score_ws.clear()
                    try: score_ws.update(values=s_final_data, range_name='A1')
                    except TypeError:
                        try: score_ws.update('A1', s_final_data)
                        except Exception: score_ws.update(s_final_data)
        except Exception: pass
            
        return deleted_count
    except Exception as e:
        st.error(f"予測ログの削除中にエラーが発生しました: {e}")
        return -1

@st.cache_data(ttl=3600)
def load_shop_events():
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('shop_events')
        
        data = worksheet.get_all_values()
        if not data or len(data) < 2: return pd.DataFrame()
        df = pd.DataFrame(data[1:], columns=data[0])
        
        if not df.empty and 'イベント日付' in df.columns:
            df['イベント日付'] = pd.to_datetime(df['イベント日付'])
        return df
    except: return pd.DataFrame()

def save_shop_event(shop_name, event_date, event_name, event_rank, event_type="全体", target_machine="指定なし"):
    if not shop_name or not event_name:
        st.warning("店舗名とイベント名を入力してください。")
        return False
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        try: 
            worksheet = sh.worksheet('shop_events')
            header = worksheet.row_values(1)
            for col_name in ['イベントランク', 'イベント種別', '対象機種']:
                if col_name not in header:
                    if worksheet.col_count < len(header) + 1:
                        worksheet.add_cols((len(header) + 1) - worksheet.col_count)
                    worksheet.update_cell(1, len(header) + 1, col_name)
                    header.append(col_name)
        except gspread.exceptions.WorksheetNotFound: 
            worksheet = sh.add_worksheet(title='shop_events', rows="1000", cols="8")
            worksheet.append_row(['登録日時', '店名', 'イベント日付', 'イベント名', '備考', 'イベントランク', 'イベント種別', '対象機種'])
            header = ['登録日時', '店名', 'イベント日付', 'イベント名', '備考', 'イベントランク', 'イベント種別', '対象機種']
        
        timestamp = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        date_str = event_date.strftime('%Y-%m-%d')
        row_data = [timestamp, shop_name, date_str, event_name, '', event_rank]
        while len(row_data) < len(header): row_data.append("")
        if 'イベント種別' in header: row_data[header.index('イベント種別')] = event_type
        if '対象機種' in header: row_data[header.index('対象機種')] = target_machine
        worksheet.append_row(row_data, value_input_option='RAW')
        return True
    except Exception as e:
        st.error(f"イベント保存エラー: {e}")
        return False

def update_shop_event(old_shop_name, old_event_date, old_event_name, new_shop_name, new_event_date, new_event_name, new_event_rank, new_event_type="全体", new_target_machine="指定なし"):
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
            idx_type = header.index('イベント種別') if 'イベント種別' in header else -1
            idx_target = header.index('対象機種') if '対象機種' in header else -1
        except: return False
        
        target_date_str = pd.to_datetime(old_event_date).strftime('%Y-%m-%d')
        new_date_str = pd.to_datetime(new_event_date).strftime('%Y-%m-%d')
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
                if idx_type != -1:
                    worksheet.update_cell(i, idx_type + 1, new_event_type)
                if idx_target != -1:
                    worksheet.update_cell(i, idx_target + 1, new_target_machine)
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
        
        target_date_str = pd.to_datetime(event_date).strftime('%Y-%m-%d')
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
        
        data = worksheet.get_all_values()
        if not data or len(data) < 2: return pd.DataFrame()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    except: return pd.DataFrame()

def save_island_master(shop, island_name, rule_str, main_corner="指定なし", island_type="普通"):
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        sheet_name = 'island_master'
        try: 
            worksheet = sh.worksheet(sheet_name)
            header = worksheet.row_values(1)
            if '台番号ルール' not in header:
                if worksheet.col_count < len(header) + 1:
                    worksheet.add_cols((len(header) + 1) - worksheet.col_count)
                worksheet.update_cell(1, len(header) + 1, '台番号ルール')
                header.append('台番号ルール')
            if 'メイン角番' not in header:
                if worksheet.col_count < len(header) + 1:
                    worksheet.add_cols((len(header) + 1) - worksheet.col_count)
                worksheet.update_cell(1, len(header) + 1, 'メイン角番')
                header.append('メイン角番')
            if '島属性' not in header:
                if worksheet.col_count < len(header) + 1:
                    worksheet.add_cols((len(header) + 1) - worksheet.col_count)
                worksheet.update_cell(1, len(header) + 1, '島属性')
                header.append('島属性')
        except gspread.exceptions.WorksheetNotFound: 
            worksheet = sh.add_worksheet(title=sheet_name, rows="1000", cols="8")
            worksheet.append_row(['登録日時', '店名', '島名', '開始台番号', '終了台番号', '台番号ルール', 'メイン角番', '島属性'])
            header = ['登録日時', '店名', '島名', '開始台番号', '終了台番号', '台番号ルール', 'メイン角番', '島属性']
        
        timestamp = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        row_data = [timestamp, shop, island_name, "", "", rule_str]
        
        while len(row_data) < len(header):
            row_data.append("")
            
        if '台番号ルール' in header:
            row_data[header.index('台番号ルール')] = rule_str
        if 'メイン角番' in header:
            row_data[header.index('メイン角番')] = "" if main_corner == "指定なし" else str(main_corner)
        if '島属性' in header:
            row_data[header.index('島属性')] = island_type
            
        worksheet.append_row(row_data, value_input_option='RAW')
        return True
    except Exception as e:
        st.error(f"島マスター保存エラー: {e}")
        return False

def update_island_master(target_timestamp, shop, island_name, rule_str, main_corner="指定なし", island_type="普通"):
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        worksheet = sh.worksheet('island_master')
        all_values = worksheet.get_all_values()
        if not all_values: return False
        
        header = all_values[0]
        try:
            idx_reg = header.index('登録日時')
            idx_shop = header.index('店名')
            idx_name = header.index('島名')
            idx_rule = header.index('台番号ルール')
            idx_corner = header.index('メイン角番') if 'メイン角番' in header else -1
            idx_type = header.index('島属性') if '島属性' in header else -1
        except: return False
        
        for i, row in enumerate(all_values[1:], start=2):
            if len(row) <= idx_reg: continue
            if row[idx_reg] == str(target_timestamp):
                worksheet.update_cell(i, idx_shop + 1, shop)
                worksheet.update_cell(i, idx_name + 1, island_name)
                worksheet.update_cell(i, idx_rule + 1, rule_str)
                if idx_corner != -1:
                    worksheet.update_cell(i, idx_corner + 1, "" if main_corner == "指定なし" else str(main_corner))
                if idx_type != -1:
                    worksheet.update_cell(i, idx_type + 1, island_type)
                return True
        return False
    except Exception as e:
        st.error(f"島マスター更新エラー: {e}")
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
        
        data = worksheet.get_all_values()
        if not data or len(data) < 2: return pd.DataFrame()
        df = pd.DataFrame(data[1:], columns=data[0])
        
        if not df.empty:
            if '日付' in df.columns:
                df['日付'] = pd.to_datetime(df['日付'])
            for col in ['投資', '回収', '収支']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            if '稼働時間' in df.columns:
                df['稼働時間'] = pd.to_numeric(df['稼働時間'], errors='coerce').fillna(0.0)
        return df
    except: return pd.DataFrame()

def save_my_balance(date_obj, shop, machine, number, invest, recovery, hours, memo):
    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        
        sheet_name = 'my_balance'
        try: 
            worksheet = sh.worksheet(sheet_name)
            existing_data = worksheet.get_all_values()
            if existing_data:
                header = existing_data[0]
                # 既存シートに「稼働時間」列がなければ追加する
                if '稼働時間' not in header:
                    header.insert(-1, '稼働時間')
                    idx = header.index('稼働時間')
                    for i in range(1, len(existing_data)):
                        existing_data[i].insert(idx, '')
            else:
                header = ['登録日時', '日付', '店名', '台番号', '機種名', '投資', '回収', '収支', '稼働時間', 'メモ']
                existing_data = []
        except gspread.exceptions.WorksheetNotFound: 
            worksheet = sh.add_worksheet(title=sheet_name, rows="1000", cols="11")
            header = ['登録日時', '日付', '店名', '台番号', '機種名', '投資', '回収', '収支', '稼働時間', 'メモ']
            existing_data = []
        
        timestamp = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        date_str = date_obj.strftime('%Y-%m-%d')
        balance = int(recovery) - int(invest)
        
        new_row_dict = {
            '登録日時': timestamp, '日付': date_str, '店名': shop,
            '台番号': str(number), '機種名': machine, '投資': str(invest),
            '回収': str(recovery), '収支': str(balance), '稼働時間': str(hours), 'メモ': memo
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
        if worksheet.col_count < len(header):
            worksheet.add_cols(len(header) - worksheet.col_count)
        try:
            worksheet.update(values=final_data, range_name='A1')
        except TypeError:
            try: worksheet.update('A1', final_data)
            except Exception: worksheet.update(final_data)
            
        return True
    except Exception as e:
        st.error(f"収支保存エラー: {e}")
        return False

def update_my_balance(old_timestamp, date_obj, shop, machine, number, invest, recovery, hours, memo):
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
            idx_hours = header.index('稼働時間') if '稼働時間' in header else -1
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
                if idx_hours != -1:
                    worksheet.update_cell(i, idx_hours + 1, hours)
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
        "デフォルト": {'train_months': 3, 'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 
                   'k_train_months': 6, 'k_n_estimators': 300, 'k_learning_rate': 0.03, 'k_num_leaves': 15, 'k_max_depth': 4, 'k_min_child_samples': 50, 'k_reg_alpha': 0.0, 'k_reg_lambda': 0.0,
                   'lstm_hidden_size': 64, 'lstm_lr': 0.001, 'lstm_epochs': 20}
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
                    'reg_alpha': float(record.get('reg_alpha', 0.0)),
                    'reg_lambda': float(record.get('reg_lambda', 0.0)),
                    'k_train_months': int(record.get('k_train_months', record.get('train_months', 6))),
                    'k_n_estimators': int(record.get('k_n_estimators', record.get('n_estimators', 300))),
                    'k_learning_rate': float(record.get('k_learning_rate', record.get('learning_rate', 0.03))),
                    'k_num_leaves': int(record.get('k_num_leaves', record.get('num_leaves', 15))),
                    'k_max_depth': int(record.get('k_max_depth', record.get('max_depth', 4))),
                    'k_min_child_samples': int(record.get('k_min_child_samples', record.get('min_child_samples', 50))),
                    'k_reg_alpha': float(record.get('k_reg_alpha', record.get('reg_alpha', 0.0))),
                    'k_reg_lambda': float(record.get('k_reg_lambda', record.get('reg_lambda', 0.0))),
                    'lstm_hidden_size': int(record.get('lstm_hidden_size', 64)),
                    'lstm_lr': float(record.get('lstm_lr', 0.001)),
                    'lstm_epochs': int(record.get('lstm_epochs', 20)),
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
        except gspread.exceptions.WorksheetNotFound: worksheet = sh.add_worksheet(title=sheet_name, rows="100", cols="20")
        header = ['店名', 'train_months', 'n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'min_child_samples', 'reg_alpha', 'reg_lambda', 'k_train_months', 'k_n_estimators', 'k_learning_rate', 'k_num_leaves', 'k_max_depth', 'k_min_child_samples', 'k_reg_alpha', 'k_reg_lambda', 'lstm_hidden_size', 'lstm_lr', 'lstm_epochs']
        data_to_write = [header] + [[shop_name] + [params.get(k) for k in header[1:]] for shop_name, params in shop_hyperparams.items()]
        worksheet.clear(); worksheet.update('A1', data_to_write)
        return True
    except Exception as e:
        st.error(f"AI設定の保存に失敗しました: {e}")
        return False

# --- 内部関数: 特徴量作成サブモジュール ---
def _apply_island_features(df, df_island, shop_col):
    """島・配置特徴量の追加"""
    if not shop_col or '機種名' not in df.columns or '台番号' not in df.columns:
        return df

    grp = df.groupby([shop_col, '機種名'])['台番号']
    df['is_corner'] = ((df['台番号'] == grp.transform('min')) | (df['台番号'] == grp.transform('max'))).astype(int)
    df['island_id'] = "Unknown"

    if df_island is not None and not df_island.empty:
        unique_machines = df[[shop_col, '台番号']].drop_duplicates()
        island_mapping = []
        
        parsed_islands = []
        for _, i_row in df_island.iterrows():
            s_name = i_row.get('店名')
            i_name = i_row.get('島名')
            machines = []
            
            try:
                s = int(i_row.get('開始台番号', 0))
                e = int(i_row.get('終了台番号', 0))
                if s > 0 and e >= s: machines.extend(range(s, e + 1))
            except: pass
            
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
                        
            main_corner = str(i_row.get('メイン角番', '')).strip()
            island_type = str(i_row.get('島属性', '普通'))
            
            machines = sorted(list(set(machines)))
            if machines:
                parsed_islands.append({
                    'shop': s_name, 'island_id': f"{s_name}_{i_name}",
                    'machines': machines, 'corner_min': min(machines), 'corner_max': max(machines),
                    'main_corner': main_corner, 'island_type': island_type
                })
                
        for _, row in unique_machines.iterrows():
            s_name = row[shop_col]
            m_num = row['台番号']
            i_id = "Unknown"
            is_cor = 0
            is_main_cor = 0
            is_main_isl = 0
            is_wall_isl = 0
            
            try:
                m_num_int = int(float(m_num))
            except Exception:
                m_num_int = None
                
            for pi in parsed_islands:
                if pi['shop'] == s_name and (m_num in pi['machines'] or m_num_int in pi['machines']):
                    i_id = pi['island_id']
                    if m_num == pi['corner_min'] or m_num == pi['corner_max'] or m_num_int == pi['corner_min'] or m_num_int == pi['corner_max']: is_cor = 1
                    if str(m_num) == pi['main_corner'] or (m_num_int is not None and str(m_num_int) == pi['main_corner']): is_main_cor = 1
                    if pi['island_type'] == 'メイン通路沿い (目立つ)': is_main_isl = 1
                    elif pi['island_type'] == '壁側・奥 (目立たない)': is_wall_isl = 1
                    break
            island_mapping.append({
                shop_col: s_name, '台番号': m_num, 
                'master_island_id': i_id, 'master_is_corner': is_cor,
                'master_is_main_corner': is_main_cor,
                'master_is_main_island': is_main_isl,
                'master_is_wall_island': is_wall_isl
            })
        mapping_df = pd.DataFrame(island_mapping)
        df = pd.merge(df, mapping_df, on=[shop_col, '台番号'], how='left')
        df.loc[df['master_island_id'] != "Unknown", 'island_id'] = df['master_island_id']
        df.loc[df['master_island_id'] != "Unknown", 'is_corner'] = df['master_is_corner']
        df.loc[df['master_island_id'] != "Unknown", 'is_main_corner'] = df['master_is_main_corner']
        df.loc[df['master_island_id'] != "Unknown", 'is_main_island'] = df['master_is_main_island']
        df.loc[df['master_island_id'] != "Unknown", 'is_wall_island'] = df['master_is_wall_island']
        df = df.drop(columns=['master_island_id', 'master_is_corner', 'master_is_main_corner', 'master_is_main_island', 'master_is_wall_island'])

    return df

def _generate_neighbor_features(df, shop_col):
    """両隣の稼働データに基づく特徴量の追加"""
    if not shop_col or '台番号' not in df.columns or '対象日付' not in df.columns:
        return df

    if 'island_id' in df.columns:
        # 同じ島IDごとにまとめてからソートすることで、関係ない台が間に挟まるのを防ぐ
        df = df.sort_values([shop_col, '対象日付', 'island_id', '台番号'])
        
        prev_shop = df[shop_col].shift(1)
        prev_date = df['対象日付'].shift(1)
        prev_island = df['island_id'].shift(1)
        prev_no = df['台番号'].shift(1)
        
        next_shop = df[shop_col].shift(-1)
        next_date = df['対象日付'].shift(-1)
        next_island = df['island_id'].shift(-1)
        next_no = df['台番号'].shift(-1)
        
        is_prev = (df[shop_col] == prev_shop) & (df['対象日付'] == prev_date) & (
            ((df['island_id'] != "Unknown") & (df['island_id'] == prev_island)) |
            ((df['island_id'] == "Unknown") & (df['island_id'] == prev_island) & ((df['台番号'] - prev_no).between(1, 3)))
        )
        is_next = (df[shop_col] == next_shop) & (df['対象日付'] == next_date) & (
            ((df['island_id'] != "Unknown") & (df['island_id'] == next_island)) |
            ((df['island_id'] == "Unknown") & (df['island_id'] == next_island) & ((next_no - df['台番号']).between(1, 3)))
        )
    else:
        df = df.sort_values([shop_col, '対象日付', '台番号'])
        prev_shop = df[shop_col].shift(1)
        prev_date = df['対象日付'].shift(1)
        prev_no = df['台番号'].shift(1)
        next_shop = df[shop_col].shift(-1)
        next_date = df['対象日付'].shift(-1)
        next_no = df['台番号'].shift(-1)
        
        is_prev = (df[shop_col] == prev_shop) & (df['対象日付'] == prev_date) & ((df['台番号'] - prev_no).between(1, 3))
        is_next = (df[shop_col] == next_shop) & (df['対象日付'] == next_date) & ((next_no - df['台番号']).between(1, 3))
    
    prev_diff = df['差枚'].shift(1)
    next_diff = df['差枚'].shift(-1)
    
    p_val = np.where(is_prev, prev_diff, np.nan)
    n_val = np.where(is_next, next_diff, np.nan)
    # 両隣の爆発に引っ張られないようクリップ処理を適用 (-2000枚 ～ +2000枚)
    p_val_clip = np.clip(p_val, -2000, 2000)
    n_val_clip = np.clip(n_val, -2000, 2000)
    df['neighbor_avg_diff'] = pd.DataFrame({'p': p_val_clip, 'n': n_val_clip}).mean(axis=1).fillna(0)
    
    df['left_diff'] = np.where(is_prev, prev_diff, 0)
    df['right_diff'] = np.where(is_next, next_diff, 0)
    df['neighbor_positive_count'] = (np.where(is_prev & (prev_diff > 0), 1, 0) + np.where(is_next & (next_diff > 0), 1, 0))
    
    # 両隣(並び3台)の合算REG確率を計算
    prev_reg = df['REG'].shift(1)
    next_reg = df['REG'].shift(-1)
    prev_g = df['累計ゲーム'].shift(1)
    next_g = df['累計ゲーム'].shift(-1)
        
    neighbor_reg_sum_3 = df['REG'] + np.where(is_prev, prev_reg, 0) + np.where(is_next, next_reg, 0)
    neighbor_g_sum_3 = df['累計ゲーム'] + np.where(is_prev, prev_g, 0) + np.where(is_next, next_g, 0)
    df['neighbor_reg_prob'] = np.where(neighbor_g_sum_3 > 0, neighbor_reg_sum_3 / neighbor_g_sum_3, 0)

    # 両隣のみのデータに基づく特徴量
    neighbor_reg_sum = np.where(is_prev, prev_reg, 0) + np.where(is_next, next_reg, 0)
    neighbor_g_sum = np.where(is_prev, prev_g, 0) + np.where(is_next, next_g, 0)
    neighbor_count = np.where(is_prev, 1, 0) + np.where(is_next, 1, 0)
    
    df['neighbor_only_reg_prob'] = np.where(neighbor_g_sum > 0, neighbor_reg_sum / neighbor_g_sum, 0)
    df['neighbor_only_avg_g'] = np.where(neighbor_count > 0, neighbor_g_sum / neighbor_count, 0)
    
    is_prev_high = is_prev & (prev_g >= 3000) & ((prev_reg / prev_g.replace(0, np.nan)) >= 1.0/260.0)
    is_next_high = is_next & (next_g >= 3000) & ((next_reg / next_g.replace(0, np.nan)) >= 1.0/260.0)
    df['neighbor_high_setting_count'] = np.where(is_prev_high, 1, 0) + np.where(is_next_high, 1, 0)
    
    df['is_neighbor_high_reg'] = (df['neighbor_only_reg_prob'] >= 1.0/260.0) & (neighbor_g_sum >= 4000)
    df['neighbor_reg_reliability_score'] = np.clip(df['neighbor_only_reg_prob'] / (1.0/260.0), 0, 2.0) * (df['neighbor_only_avg_g'] / 1000.0)

    if 'island_id' in df.columns:
        # 元の並び順に戻す
        df = df.sort_values([shop_col, '対象日付', '台番号']).reset_index(drop=True)
        
    return df

def _apply_event_features(df, df_events, shop_col):
    """イベント情報の適用と特徴量追加"""
    if df_events is None or df_events.empty or not shop_col:
        df['イベント名'] = '通常'
        df['event_rank_score'] = 0
        df['イベントランク'] = ''
        df['event_code'] = 0
        df['prev_event_rank_score'] = 0
        return df

    valid_events = df_events.copy()
    
    # --- 周年イベントを毎年の同じ月日に自動展開 ---
    anniversary_events = valid_events[valid_events['イベント名'].astype(str).str.contains('周年')].copy()
    if not anniversary_events.empty:
        target_years = df['next_date'].dt.year.dropna().unique() if 'next_date' in df.columns else [pd.Timestamp.now(tz='Asia/Tokyo').year]
        expanded_events = []
        for y in target_years:
            temp_ev = anniversary_events.copy()
            def replace_year(d, tgt_y):
                if pd.isna(d): return d
                try: return d.replace(year=int(tgt_y))
                except ValueError: return d + pd.offsets.DateOffset(years=(int(tgt_y) - d.year))
            temp_ev['イベント日付'] = temp_ev['イベント日付'].apply(lambda d: replace_year(d, y))
            expanded_events.append(temp_ev)
        if expanded_events:
            valid_events = pd.concat([valid_events] + expanded_events, ignore_index=True)
            valid_events = valid_events.drop_duplicates(subset=['店名', 'イベント日付', 'イベント名'])

    # 「対象外(無効)」イベントのみAIの学習から除外
    if 'イベント種別' in valid_events.columns:
        cond_exclude = valid_events['イベント種別'] == '対象外(無効)'
        cond_ss = (valid_events['イベントランク'] == 'SS (周年)') | (valid_events['イベント名'].astype(str).str.contains('周年|リニューアル|グランド'))
        valid_events = valid_events[~(cond_exclude & ~cond_ss)].copy()

    def calc_single_score(row):
        rank = str(row.get('イベントランク', ''))
        t_mac = str(row.get('対象機種', '指定なし'))
        my_mac = str(row.get('機種名', ''))
        e_type = str(row.get('イベント種別', '全体')).replace('スロット/全体', '全体')
        
        rank_map = {'SS (周年)': 6, 'S': 5, 'A': 4, 'B': 3, 'C': 2}
        score = rank_map.get(rank, 0)
        
        is_super_event = (rank == 'SS (周年)') or ('周年' in str(row.get('イベント名', ''))) or ('リニューアル' in str(row.get('イベント名', ''))) or ('グランド' in str(row.get('イベント名', '')))
        if is_super_event and score < 6:
            score = max(score, 5) # 特大イベントは最低でもSランク相当のパワーを保証

        # パチンコ専用イベントの処理
        if e_type == 'パチンコ専用':
            if is_super_event:
                return 3 # スロットへの波及効果として中間スコア(Bランク相当)を与える
            else:
                return -score if score > 0 else -1 # ランクが高いほどマイナス(回収)スコアを大きくする
                
        # 対象外(無効)だが「周年」で例外的に残ったデータの処理
        if e_type == '対象外(無効)':
            if is_super_event:
                return 3
            else:
                return 0

        # 特定機種が指定されている場合、関係ない機種はイベントの恩恵（スコア）をマイナス化
        if score > 0 and t_mac not in ['指定なし', 'スロット全体', 'ジャグラー全体', '全体', 'nan', 'None']:
            if t_mac == 'ジャグラー以外 (パチスロ他機種)':
                if is_super_event: score = 3
                else: score = -score # ランクが高いほどマイナススコアを大きくする
            elif my_mac not in t_mac and t_mac not in my_mac:
                if is_super_event: score = 3 # 特大イベントなら対象外機種でもおこぼれ(ベースアップ)としてスコアを残す
                else: score = -score # イベント対象機種やパチンコへの還元のシワ寄せで回収されるとみなしてマイナス評価
        return score

    unique_combinations = df[['店名', 'next_date', '機種名']].drop_duplicates()
    merged_ev = pd.merge(unique_combinations, valid_events, left_on=['店名', 'next_date'], right_on=['店名', 'イベント日付'], how='inner')
    
    if not merged_ev.empty:
        merged_ev['single_score'] = merged_ev.apply(calc_single_score, axis=1)
        
        # 同日・同店舗の複数イベントのスコアと名前を合算・結合する
        ev_summary = merged_ev.groupby(['店名', 'next_date', '機種名']).agg(
            event_rank_score=('single_score', 'sum'),
            イベント名=('イベント名', lambda x: ' + '.join(x.astype(str))),
            イベントランク=('イベントランク', lambda x: ' + '.join(x.astype(str)))
        ).reset_index()
        
        df = pd.merge(df, ev_summary, on=['店名', 'next_date', '機種名'], how='left')
        df['イベント名'] = df['イベント名'].fillna('通常')
        df['event_rank_score'] = df['event_rank_score'].fillna(0)
        df['イベントランク'] = df['イベントランク'].fillna('')
        df['event_code'] = df['イベント名'].astype('category').cat.codes
    else:
        df['イベント名'] = '通常'
        df['event_rank_score'] = 0
        df['イベントランク'] = ''
        df['event_code'] = 0

    # --- 予測日に対する「前日（今日の対象日付）」がイベントだったかのスコア（特日翌日の扱い学習用） ---
    unique_combinations_today = df[['店名', '対象日付', '機種名']].drop_duplicates()
    merged_ev_today = pd.merge(unique_combinations_today, valid_events, left_on=['店名', '対象日付'], right_on=['店名', 'イベント日付'], how='inner')
    
    if not merged_ev_today.empty:
        merged_ev_today['single_score'] = merged_ev_today.apply(calc_single_score, axis=1)
        ev_summary_today = merged_ev_today.groupby(['店名', '対象日付', '機種名']).agg(
            prev_event_rank_score=('single_score', 'sum')
        ).reset_index()
        df = pd.merge(df, ev_summary_today, on=['店名', '対象日付', '機種名'], how='left')
        df['prev_event_rank_score'] = df['prev_event_rank_score'].fillna(0)
    else:
        df['prev_event_rank_score'] = 0

    return df

# --- 内部関数: 特徴量作成 ---
def _generate_features(df, df_events, df_island, df_daily_scores, target_date):
    if target_date is not None:
        target_ts = pd.to_datetime(target_date)
        df = df[df['対象日付'] < target_ts].copy()

    if df.empty: return df, []

    # 店舗名のカラムを「店名」に統一し、以降の処理をシンプルにする
    # 現在のソート順を保持（mergeによる順序崩れ防止）
    df['original_order'] = np.arange(len(df))
    if '店舗名' in df.columns and '店名' not in df.columns:
        df = df.rename(columns={'店舗名': '店名'})

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
                            
                main_corner = str(i_row.get('メイン角番', '')).strip()
                island_type = str(i_row.get('島属性', '普通'))
                
                machines = sorted(list(set(machines)))
                if machines:
                    parsed_islands.append({
                        'shop': s_name, 'island_id': f"{s_name}_{i_name}",
                        'machines': machines, 'corner_min': min(machines), 'corner_max': max(machines),
                        'main_corner': main_corner, 'island_type': island_type
                    })
                    
            for _, row in unique_machines.iterrows():
                s_name = row[shop_col]
                m_num = row['台番号']
                i_id = "Unknown"
                is_cor = 0
                is_main_cor = 0
                is_main_isl = 0
                is_wall_isl = 0
                for pi in parsed_islands:
                    if pi['shop'] == s_name and m_num in pi['machines']:
                        i_id = pi['island_id']
                        if m_num == pi['corner_min'] or m_num == pi['corner_max']: is_cor = 1
                        if str(m_num) == pi['main_corner']: is_main_cor = 1
                        if pi['island_type'] == 'メイン通路沿い (目立つ)': is_main_isl = 1
                        elif pi['island_type'] == '壁側・奥 (目立たない)': is_wall_isl = 1
                        break
                island_mapping.append({
                    shop_col: s_name, '台番号': m_num, 
                    'master_island_id': i_id, 'master_is_corner': is_cor,
                    'master_is_main_corner': is_main_cor,
                    'master_is_main_island': is_main_isl,
                    'master_is_wall_island': is_wall_isl
                })
            mapping_df = pd.DataFrame(island_mapping)
            df = pd.merge(df, mapping_df, on=[shop_col, '台番号'], how='left')
            df.loc[df['master_island_id'] != "Unknown", 'island_id'] = df['master_island_id']
            df.loc[df['master_island_id'] != "Unknown", 'is_corner'] = df['master_is_corner']
            df.loc[df['master_island_id'] != "Unknown", 'is_main_corner'] = df['master_is_main_corner']
            df.loc[df['master_island_id'] != "Unknown", 'is_main_island'] = df['master_is_main_island']
            df.loc[df['master_island_id'] != "Unknown", 'is_wall_island'] = df['master_is_wall_island']

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
        else:
            df = df.sort_values([shop_col, '対象日付', '台番号'])
            prev_shop = df[shop_col].shift(1)
            prev_date = df['対象日付'].shift(1)
            prev_no = df['台番号'].shift(1)
            next_shop = df[shop_col].shift(-1)
            next_date = df['対象日付'].shift(-1)
            next_no = df['台番号'].shift(-1)
            
            is_prev = (df[shop_col] == prev_shop) & (df['対象日付'] == prev_date) & ((df['台番号'] - prev_no).between(1, 3))
            is_next = (df[shop_col] == next_shop) & (df['対象日付'] == next_date) & ((next_no - df['台番号']).between(1, 3))
        
        prev_diff = df['差枚'].shift(1)
        next_diff = df['差枚'].shift(-1)
        
        p_val = np.where(is_prev, prev_diff, np.nan)
        n_val = np.where(is_next, next_diff, np.nan)
        # 両隣の爆発に引っ張られないようクリップ処理を適用 (-2000枚 ～ +2000枚)
        p_val_clip = np.clip(p_val, -2000, 2000)
        n_val_clip = np.clip(n_val, -2000, 2000)
        df['neighbor_avg_diff'] = pd.DataFrame({'p': p_val_clip, 'n': n_val_clip}).mean(axis=1).fillna(0)
        
        df['left_diff'] = np.where(is_prev, prev_diff, 0)
        df['right_diff'] = np.where(is_next, next_diff, 0)
        df['neighbor_positive_count'] = (np.where(is_prev & (prev_diff > 0), 1, 0) + np.where(is_next & (next_diff > 0), 1, 0))
        
        # 両隣(並び3台)の合算REG確率を計算
        prev_reg = df['REG'].shift(1)
        next_reg = df['REG'].shift(-1)
        prev_g = df['累計ゲーム'].shift(1)
        next_g = df['累計ゲーム'].shift(-1)
            
        neighbor_reg_sum_3 = df['REG'] + np.where(is_prev, prev_reg, 0) + np.where(is_next, next_reg, 0)
        neighbor_g_sum_3 = df['累計ゲーム'] + np.where(is_prev, prev_g, 0) + np.where(is_next, next_g, 0)
        df['neighbor_reg_prob'] = np.where(neighbor_g_sum_3 > 0, neighbor_reg_sum_3 / neighbor_g_sum_3, 0)

        # --- 新規: 両隣のみのデータに基づく特徴量 ---
        neighbor_reg_sum = np.where(is_prev, prev_reg, 0) + np.where(is_next, next_reg, 0)
        neighbor_g_sum = np.where(is_prev, prev_g, 0) + np.where(is_next, next_g, 0)
        neighbor_count = np.where(is_prev, 1, 0) + np.where(is_next, 1, 0)
        
        df['neighbor_only_reg_prob'] = np.where(neighbor_g_sum > 0, neighbor_reg_sum / neighbor_g_sum, 0)
        df['neighbor_only_avg_g'] = np.where(neighbor_count > 0, neighbor_g_sum / neighbor_count, 0)
        
        is_prev_high = is_prev & (prev_g >= 3000) & ((prev_reg / prev_g) >= 1.0/260.0)
        is_next_high = is_next & (next_g >= 3000) & ((next_reg / next_g) >= 1.0/260.0)
        df['neighbor_high_setting_count'] = np.where(is_prev_high, 1, 0) + np.where(is_next_high, 1, 0)
        df['is_sandwich_target'] = (is_prev_high & is_next_high).astype(int)
        
        df['is_neighbor_high_reg'] = ((df['neighbor_only_reg_prob'] >= 1.0/260.0) & (neighbor_g_sum >= 4000)).astype(int)
        df['neighbor_reg_reliability_score'] = np.clip(df['neighbor_only_reg_prob'] / (1.0/260.0), 0, 2.0) * (df['neighbor_only_avg_g'] / 1000.0)

        if 'island_id' in df.columns:
            # 元の並び順に戻す
            df = df.sort_values([shop_col, '対象日付', '台番号']).reset_index(drop=True)
            
    if shop_col and '末尾番号' in df.columns and '対象日付' in df.columns:
        df['end_digit_total_g'] = df.groupby([shop_col, '対象日付', '末尾番号'])['累計ゲーム'].transform('sum')
        df['end_digit_total_reg'] = df.groupby([shop_col, '対象日付', '末尾番号'])['REG'].transform('sum')
        df['end_digit_reg_prob'] = np.where(df['end_digit_total_g'] > 0, df['end_digit_total_reg'] / df['end_digit_total_g'], 0)

    sort_keys = [shop_col, '台番号', '対象日付'] if shop_col else ['台番号', '対象日付']
    group_keys = [shop_col, '台番号'] if shop_col else ['台番号']
    df = df.sort_values(sort_keys).reset_index(drop=True)
    
    for col in ['差枚', 'REG確率', '累計ゲーム', '最終ゲーム', 'BIG', 'REG']:
        if col in df.columns: df[f'prev_{col}'] = df.groupby(group_keys)[col].shift(1)
        
    # 2日前、3日前のデータをシフトして作成
    for col in ['差枚', 'REG確率', '累計ゲーム']:
        if col in df.columns:
            df[f'prev2_{col}'] = df.groupby(group_keys)[col].shift(2)
            df[f'prev3_{col}'] = df.groupby(group_keys)[col].shift(3)

    for col in ['neighbor_reg_prob', 'end_digit_reg_prob']:
        if col in df.columns: df[f'prev_{col}'] = df.groupby(group_keys)[col].shift(1)
        
    for col in ['neighbor_high_setting_count', 'is_neighbor_high_reg', 'neighbor_reg_reliability_score']:
        if col in df.columns:
            df[col] = df.groupby(group_keys)[col].shift(1).fillna(0)
    
    if '推定ぶどう確率' in df.columns:
        df['prev_推定ぶどう確率_raw'] = df.groupby(group_keys)['推定ぶどう確率'].shift(1)
        
        # --- ①〜③ ぶどうを「良台の裏取り(補助指標)」に限定するフィルター ---
        cond_reg = df['prev_REG確率'] >= (1/280.0)
        cond_diff = df['prev_差枚'] > 0
        cond_games_high = df['prev_累計ゲーム'] >= 4000
        cond_games_mid = (df['prev_累計ゲーム'] >= 3000) & (df['prev_累計ゲーム'] < 4000)
        
        is_good_base = cond_reg & cond_diff
        
        # ノイズ対策: 細かい計算誤差をAIが「必勝の法則」と誤認しないよう、0.1刻み（5.7, 5.8等）に丸め、
        # 極端な外れ値に引っ張られないように 5.5 〜 6.5 の範囲にクリップする
        val_high = (df['prev_推定ぶどう確率_raw'] * 10).round() / 10.0
        val_high = val_high.clip(5.5, 6.5)
        
        # 3000〜4000Gの場合はブレが大きいため、平均値(6.0)側に引っ張ってから丸める
        val_mid = ((df['prev_推定ぶどう確率_raw'] * 0.5 + 6.0 * 0.5) * 10).round() / 10.0
        val_mid = val_mid.clip(5.5, 6.5)
        
        # 条件を満たす「裏取り」の状況のみAIに数値を渡し、それ以外はNaN(無効化)してノイズを防ぐ
        df['prev_推定ぶどう確率'] = np.where(
            is_good_base & cond_games_high, val_high,
            np.where(
                is_good_base & cond_games_mid, val_mid,
                np.nan
            )
        )

    # 【高速化】機種スペックのマッピングを上に移動して is_win を先に計算
    unique_machines = df['機種名'].unique()
    specs = get_machine_specs()
    reg_map = {m: 1.0 / specs[get_matched_spec_key(m, specs)].get('設定5', {"REG": 260.0})["REG"] for m in unique_machines}
    tot_map = {m: 1.0 / specs[get_matched_spec_key(m, specs)].get('設定5', {"合算": 128.0})["合算"] for m in unique_machines}
    reg3_map = {m: 1.0 / specs[get_matched_spec_key(m, specs)].get('設定3', {"REG": 300.0})["REG"] for m in unique_machines}
    b6_den_map = {m: specs[get_matched_spec_key(m, specs)].get('設定6', {"BIG": 260.0})["BIG"] for m in unique_machines}
    reg1_map = {m: 1.0 / specs[get_matched_spec_key(m, specs)].get('設定1', {"REG": 400.0})["REG"] for m in unique_machines}
    
    df['spec_reg'] = df['機種名'].map(reg_map)
    df['spec_tot'] = df['機種名'].map(tot_map)
    df['spec_reg3'] = df['機種名'].map(reg3_map)
    df['spec_b6_den'] = df['機種名'].map(b6_den_map)
    df['spec_reg1'] = df['機種名'].map(reg1_map)
    
    b5_den_map = {m: specs[get_matched_spec_key(m, specs)].get('設定5', {"BIG": 260.0})["BIG"] for m in unique_machines}
    df['spec_b5_den'] = df['機種名'].map(b5_den_map)
    if 'prev_BIG' in df.columns and 'prev_REG' in df.columns:
        df['prev_bb_reg_ratio'] = df['prev_BIG'] / df['prev_REG'].replace(0, 1)
        df['spec_bb_reg_ratio'] = df['spec_reg'] / df['spec_b5_den'].replace(0, 1)
        df['bb_reg_ratio_diff'] = df['prev_bb_reg_ratio'] - df['spec_bb_reg_ratio']

    df['is_win'] = (
        (df['累計ゲーム'] >= 3000) & 
        calculate_high_setting_mask(df, specs, include_bb_filter=True)
    ).astype(int)

    # --- 新規追加: 高設定率の計算において未稼働台(3000G未満)を分母から除外するためのフラグ ---
    # 3000G未満の台は np.nan にすることで、mean() 集計時に分母に含まれなくなる
    df['valid_is_win'] = np.where(df['累計ゲーム'] >= 3000, df['is_win'], np.nan)

    # --- 新規追加: ローテーション型対策（最終高設定からの経過日数と投入優先度ランク） ---
    df['high_setting_date'] = df['対象日付'].where(df['valid_is_win'] == 1)
    df['last_high_setting_date'] = df.groupby(group_keys)['high_setting_date'].ffill().shift(1)
    df['days_since_last_high'] = (df['対象日付'] - df['last_high_setting_date']).dt.days
    df['days_since_last_high'] = df['days_since_last_high'].fillna(999)

    if shop_col and 'island_id' in df.columns:
        island_rank = df[df['island_id'] != "Unknown"].groupby([shop_col, '対象日付', 'island_id'])['days_since_last_high'].rank(method='min', ascending=False)
        df['rotation_priority_rank_island'] = island_rank.fillna(999)
    else:
        df['rotation_priority_rank_island'] = 999

    if shop_col and '機種名' in df.columns:
        df['rotation_priority_rank_mac'] = df.groupby([shop_col, '対象日付', '機種名'])['days_since_last_high'].rank(method='min', ascending=False).fillna(999)
    else:
        df['rotation_priority_rank_mac'] = 999
        
    df['rotation_priority_rank'] = df[['rotation_priority_rank_island', 'rotation_priority_rank_mac']].min(axis=1)
    df = df.drop(columns=['high_setting_date', 'last_high_setting_date', 'rotation_priority_rank_island', 'rotation_priority_rank_mac'])

    # --- 新規追加: REG確率評価の厳格化と特徴量エンジニアリング ---
    df['is_prev_high_reg'] = (
        (df['累計ゲーム'] >= 3000) & 
        calculate_high_setting_mask(df, specs, include_bb_filter=False)
    ).astype(int)
    df['is_high_reg_plus_diff'] = (df['is_prev_high_reg'] == 1) & (df['差枚'] > 0)
    df['is_low_reg_plus_diff'] = (df['is_prev_high_reg'] == 0) & (df['差枚'] > 0)
    df['is_high_reg_minus_diff'] = ((df['is_prev_high_reg'] == 1) & (df['差枚'] <= 0)).astype(int)
    df['reg_diff_ratio'] = (df['差枚'] / df['REG'].replace(0, np.nan)).fillna(0)

    # --- 新規追加: 時間軸と複合条件に基づく特徴量エンジニアリング ---
    if 'prev2_累計ゲーム' in df.columns and 'prev3_累計ゲーム' in df.columns:
        # X日前REG高確率フラグ (Zスコア救済を含む is_prev_high_reg をシフトして生成)
        df['reg_rate_2_days_ago_high'] = df.groupby(group_keys)['is_prev_high_reg'].shift(1).fillna(0).astype(int)
        df['reg_rate_3_days_ago_high'] = df.groupby(group_keys)['is_prev_high_reg'].shift(2).fillna(0).astype(int)
        
        # X日前差枚低めフラグ
        df['delta_2_days_ago_modest'] = df['prev2_差枚'].fillna(0).between(-1000, 1000).astype(int)
        df['delta_3_days_ago_modest'] = df['prev3_差枚'].fillna(0).between(-1000, 1000).astype(int)

        # X日後低稼働フラグ (3日前の台から見て1日後、2日後)
        df['low_play_1_day_after'] = (df['prev2_累計ゲーム'].fillna(0) < 1000).astype(int)
        df['low_play_2_day_after'] = (df['prev_累計ゲーム'].fillna(0) < 1000).astype(int)

        # 隠れ高設定パターン
        df['hidden_high_setting_pattern'] = ((df['reg_rate_3_days_ago_high'] == 1) & (df['delta_3_days_ago_modest'] == 1) & (df['low_play_1_day_after'] == 1) & (df['low_play_2_day_after'] == 1)).astype(int)

    df['next_diff'] = df.groupby(group_keys)['差枚'].shift(-1)
    if 'BIG' in df.columns: df['next_BIG'] = df.groupby(group_keys)['BIG'].shift(-1)
    if 'REG' in df.columns: df['next_REG'] = df.groupby(group_keys)['REG'].shift(-1)
    if '累計ゲーム' in df.columns: df['next_累計ゲーム'] = df.groupby(group_keys)['累計ゲーム'].shift(-1)
    
    # --- ターゲットを「翌日の高設定挙動(機種別の設定5基準：REGまたは合算)」に設定 ---
    df['target'] = (
        (df['next_累計ゲーム'] >= 3000) & 
        calculate_high_setting_mask(df, specs, g_col='next_累計ゲーム', b_col='next_BIG', r_col='next_REG', include_bb_filter=True)
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
        
    # 月初(還元されやすい)と月末(回収されやすい)のフラグ
    if 'next_date' in df.columns:
        df['is_beginning_of_month'] = (df['next_date'].dt.day <= 7).astype(int)
        df['is_end_of_month'] = (df['next_date'].dt.day >= 25).astype(int)
        # 年金支給日(15日近辺)フラグの追加：高齢者が多いためジャグラーが回収されやすい
        df['is_pension_day'] = (df['next_date'].dt.day.between(14, 16)).astype(int)
        
        # --- 新規追加: 大型連休（GW、お盆、年末年始）フラグ ---
        n_m = df['next_date'].dt.month
        n_d = df['next_date'].dt.day
        df['is_gw'] = (((n_m == 4) & (n_d >= 29)) | ((n_m == 5) & (n_d <= 6))).astype(int)
        df['is_obon'] = ((n_m == 8) & (n_d >= 10) & (n_d <= 16)).astype(int)
        df['is_year_end_start'] = (((n_m == 12) & (n_d >= 28)) | ((n_m == 1) & (n_d <= 5))).astype(int)
        
        df['is_gw_before'] = ((n_m == 4) & (n_d >= 25) & (n_d <= 28)).astype(int)
        df['is_gw_after'] = ((n_m == 5) & (n_d >= 7) & (n_d <= 9)).astype(int)
        df['is_obon_before'] = ((n_m == 8) & (n_d >= 7) & (n_d <= 9)).astype(int)
        df['is_obon_after'] = ((n_m == 8) & (n_d >= 17) & (n_d <= 19)).astype(int)
        df['is_year_end_before'] = ((n_m == 12) & (n_d >= 25) & (n_d <= 27)).astype(int)
        df['is_year_end_after'] = ((n_m == 1) & (n_d >= 6) & (n_d <= 8)).astype(int)

        if jpholiday is not None:
            unique_dates = df['next_date'].dt.date.dropna().unique()
            holiday_map = {d: jpholiday.is_holiday(d) for d in unique_dates}
            df['is_holiday'] = df['next_date'].dt.date.map(holiday_map).fillna(False).astype(int)
        else:
            df['is_holiday'] = 0

    if shop_col:
        shop_daily_avg2 = df.groupby([shop_col, '対象日付'])['差枚'].mean().reset_index(name='shop_daily_avg_diff')
        shop_daily_avg2['wd'] = shop_daily_avg2['対象日付'].dt.dayofweek
        shop_daily_avg2['digit'] = shop_daily_avg2['対象日付'].dt.day % 10
        wd_stats = shop_daily_avg2.groupby([shop_col, 'wd'])['shop_daily_avg_diff'].mean().reset_index(name='past_wd_avg')
        digit_stats = shop_daily_avg2.groupby([shop_col, 'digit'])['shop_daily_avg_diff'].mean().reset_index(name='past_digit_avg')
        wd_digit_stats = shop_daily_avg2.groupby([shop_col, 'wd', 'digit'])['shop_daily_avg_diff'].mean().reset_index(name='past_wd_digit_avg')
        
        df = pd.merge(df, wd_stats, left_on=[shop_col, 'target_weekday'], right_on=[shop_col, 'wd'], how='left').drop(columns=['wd'])
        df = pd.merge(df, digit_stats, left_on=[shop_col, 'target_date_end_digit'], right_on=[shop_col, 'digit'], how='left').drop(columns=['digit'])
        df = pd.merge(df, wd_digit_stats, left_on=[shop_col, 'target_weekday', 'target_date_end_digit'], right_on=[shop_col, 'wd', 'digit'], how='left').drop(columns=['wd', 'digit'])
        
        # 曜日と特定日、より強い方(還元要素)をベースにする
        base_target_diff = df[['past_wd_avg', 'past_digit_avg']].max(axis=1)
        
        # ただし、「過去に同じ曜日×同じ末尾の実績」が存在し、それがベース期待値より悪い場合のみ、その悪いリアルな実績を採用して予測をシビアにする
        df['target_date_type_avg_diff'] = np.where(
            df['past_wd_digit_avg'].notna() & (df['past_wd_digit_avg'] < base_target_diff),
            df['past_wd_digit_avg'],
            base_target_diff
        )
        df['target_date_type_avg_diff'] = df['target_date_type_avg_diff'].fillna(0)
        df['past_wd_avg'] = df['past_wd_avg'].fillna(0)
        df['past_digit_avg'] = df['past_digit_avg'].fillna(0)
        df = df.drop(columns=['past_wd_digit_avg'])

        # --- 新規: 過去の曜日ごと・末尾ごとの合算REG確率 ---
        shop_daily_sum = df.groupby([shop_col, '対象日付'])[['累計ゲーム', 'REG']].sum().reset_index()
        shop_daily_sum['wd'] = shop_daily_sum['対象日付'].dt.dayofweek
        shop_daily_sum['digit'] = shop_daily_sum['対象日付'].dt.day % 10
        
        wd_reg_stats = shop_daily_sum.groupby([shop_col, 'wd']).agg(
            wd_total_g=('累計ゲーム', 'sum'),
            wd_total_reg=('REG', 'sum')
        ).reset_index()
        wd_reg_stats['past_wd_reg_prob'] = np.where(wd_reg_stats['wd_total_g'] > 0, wd_reg_stats['wd_total_reg'] / wd_reg_stats['wd_total_g'], 0)
        
        digit_reg_stats = shop_daily_sum.groupby([shop_col, 'digit']).agg(
            digit_total_g=('累計ゲーム', 'sum'),
            digit_total_reg=('REG', 'sum')
        ).reset_index()
        digit_reg_stats['past_digit_reg_prob'] = np.where(digit_reg_stats['digit_total_g'] > 0, digit_reg_stats['digit_total_reg'] / digit_reg_stats['digit_total_g'], 0)

        df = pd.merge(df, wd_reg_stats[[shop_col, 'wd', 'past_wd_reg_prob']], left_on=[shop_col, 'target_weekday'], right_on=[shop_col, 'wd'], how='left').drop(columns=['wd'])
        df = pd.merge(df, digit_reg_stats[[shop_col, 'digit', 'past_digit_reg_prob']], left_on=[shop_col, 'target_date_end_digit'], right_on=[shop_col, 'digit'], how='left').drop(columns=['digit'])
        df['past_wd_reg_prob'] = df['past_wd_reg_prob'].fillna(0)
        df['past_digit_reg_prob'] = df['past_digit_reg_prob'].fillna(0)

    df = _apply_event_features(df, df_events, shop_col)

    if shop_col and df_events is not None and not df_events.empty:
        # --- 新規: 過去のイベント名ごとの合算REG確率 ---
        ev_for_past = df_events.copy()
        ev_shop_col = '店名' if '店名' in ev_for_past.columns else ('店舗名' if '店舗名' in ev_for_past.columns else None)
        if ev_shop_col:
            ev_for_past = ev_for_past.rename(columns={ev_shop_col: shop_col})
            ev_for_past['イベント日付'] = pd.to_datetime(ev_for_past['イベント日付'], errors='coerce')
            ev_for_past = ev_for_past.dropna(subset=['イベント日付'])
            
            ev_past_grouped = ev_for_past.groupby([shop_col, 'イベント日付']).agg(
                past_event_name=('イベント名', lambda x: ' + '.join(x.astype(str)))
            ).reset_index()
            
            shop_daily_sum_ev = df.groupby([shop_col, '対象日付'])[['累計ゲーム', 'REG']].sum().reset_index()
            shop_daily_sum_ev = pd.merge(shop_daily_sum_ev, ev_past_grouped, left_on=[shop_col, '対象日付'], right_on=[shop_col, 'イベント日付'], how='inner')
            
            ev_reg_stats = shop_daily_sum_ev.groupby([shop_col, 'past_event_name']).agg(
                ev_total_g=('累計ゲーム', 'sum'),
                ev_total_reg=('REG', 'sum')
            ).reset_index()
            ev_reg_stats['past_event_reg_prob'] = np.where(ev_reg_stats['ev_total_g'] > 0, ev_reg_stats['ev_total_reg'] / ev_reg_stats['ev_total_g'], 0)
            
            df = pd.merge(df, ev_reg_stats[[shop_col, 'past_event_name', 'past_event_reg_prob']], left_on=[shop_col, 'イベント名'], right_on=[shop_col, 'past_event_name'], how='left').drop(columns=['past_event_name'])
            
    df['past_event_reg_prob'] = df['past_event_reg_prob'].fillna(0) if 'past_event_reg_prob' in df.columns else 0

    # 【高速化】遅い transform(lambda...) を groupby.rolling() に変更
    df = df.sort_values('対象日付').reset_index(drop=True)
    df['weekday'] = df['対象日付'].dt.dayofweek
    df['shifted_diff_wd'] = df.groupby('weekday')['差枚'].shift(1)
    df['weekday_avg_diff'] = df.groupby('weekday')['shifted_diff_wd'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
    
    if '日付要素' in df.columns:
        df['shifted_diff_ev'] = df.groupby('日付要素')['差枚'].shift(1)
        df['event_avg_diff'] = df.groupby('日付要素')['shifted_diff_ev'].expanding().mean().reset_index(level=0, drop=True).fillna(0)
        
    if shop_col and 'イベント名' in df.columns:
        if '機種名' in df.columns:
            df['shifted_diff_ev_mac'] = df.groupby([shop_col, 'イベント名', '機種名'])['差枚'].shift(1)
            df['event_x_machine_avg_diff'] = df.groupby([shop_col, 'イベント名', '機種名'])['shifted_diff_ev_mac'].expanding().mean().reset_index(level=[0,1,2], drop=True).fillna(0)
            
        if '末尾番号' in df.columns:
            df['shifted_diff_ev_end'] = df.groupby([shop_col, 'イベント名', '末尾番号'])['差枚'].shift(1)
            df['event_x_end_digit_avg_diff'] = df.groupby([shop_col, 'イベント名', '末尾番号'])['shifted_diff_ev_end'].expanding().mean().reset_index(level=[0,1,2], drop=True).fillna(0)

    if shop_col and '末尾番号' in df.columns:
        df['shifted_diff_pure_tail'] = df.groupby([shop_col, '末尾番号'])['差枚'].shift(1)
        df['pure_tail_number_avg_diff'] = df.groupby([shop_col, '末尾番号'])['shifted_diff_pure_tail'].expanding().mean().reset_index(level=[0,1], drop=True).fillna(0)

    df = df.sort_values(sort_keys).reset_index(drop=True)
    df['shifted_diff'] = df.groupby(group_keys)['差枚'].shift(1)
    df['shifted_g'] = df.groupby(group_keys)['累計ゲーム'].shift(1)
    df['shifted_reg'] = df.groupby(group_keys)['REG'].shift(1)
    group_levels = list(range(len(group_keys))) # インデックス操作用
    
    df['mean_3days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=3, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['median_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).median().reset_index(level=group_levels, drop=True).fillna(0)
    df['std_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).std().reset_index(level=group_levels, drop=True).fillna(0)
    df['min_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).min().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_14days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=14, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_30days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=30, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)

    df['mean_3days_games'] = df.groupby(group_keys)['shifted_g'].rolling(window=3, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_7days_games'] = df.groupby(group_keys)['shifted_g'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['is_prev_no_play'] = (df['shifted_g'] == 0).astype(int)

    # --- 新規: 週間REG確率 ---
    # 7日間の合計REG回数と合計ゲーム数から計算し、ノイズを低減
    df['sum_7d_reg'] = df.groupby(group_keys)['shifted_reg'].rolling(window=7, min_periods=1).sum().reset_index(level=group_levels, drop=True).fillna(0).astype(float)
    df['sum_7d_g'] = df.groupby(group_keys)['shifted_g'].rolling(window=7, min_periods=1).sum().reset_index(level=group_levels, drop=True).fillna(0).astype(float)
    df['mean_7days_reg_prob'] = np.where(df['sum_7d_g'] > 0, df['sum_7d_reg'] / df['sum_7d_g'], 0)
    
    df = df.drop(columns=['sum_7d_reg', 'sum_7d_g'], errors='ignore')

    # --- 勝率安定度（一撃ノイズ排除用） ---
    df['shifted_valid_is_win'] = df.groupby(group_keys)['valid_is_win'].shift(1)
    df['win_rate_7days'] = df.groupby(group_keys)['shifted_valid_is_win'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    
    # --- 単純な差枚プラス勝率 ---
    df['is_plus_diff'] = (df['差枚'] > 0).astype(int)
    df['shifted_is_plus_diff'] = df.groupby(group_keys)['is_plus_diff'].shift(1)
    df['plus_rate_7days'] = df.groupby(group_keys)['shifted_is_plus_diff'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)

    # --- 大負け(-1000枚以下)フラグ ---
    df['is_heavy_lose'] = (df['差枚'] <= -1000).astype(int)
    # --- 遊べる台(±500枚以内)フラグ ---
    df['is_play_machine'] = ((df['差枚'] >= -500) & (df['差枚'] <= 500)).astype(int)

    # 一時的に作成した不要な列を削除
    df = df.drop(columns=['shifted_diff_wd', 'shifted_diff_ev', 'shifted_diff', 'shifted_is_win', 'shifted_is_plus_diff', 'shifted_diff_ev_mac', 'shifted_diff_ev_end', 'shifted_diff_pure_tail', 'prev_推定ぶどう確率_adj', 'is_plus_diff'], errors='ignore')

    if shop_col:
        df['shop_avg_diff'] = df.groupby([shop_col, '対象日付'])['差枚'].transform('mean').fillna(0)
        df['shop_median_diff'] = df.groupby([shop_col, '対象日付'])['差枚'].transform('median').fillna(0)
        df['shop_high_rate'] = df.groupby([shop_col, '対象日付'])['valid_is_win'].transform('mean').fillna(0)
        df['shop_heavy_lose_rate'] = df.groupby([shop_col, '対象日付'])['is_heavy_lose'].transform('mean').fillna(0)
        df['shop_play_rate'] = df.groupby([shop_col, '対象日付'])['is_play_machine'].transform('mean').fillna(0)
        # 店舗の平均稼働に対する自台の稼働割合（相対的な粘られ度）
        df['shop_avg_games'] = df.groupby([shop_col, '対象日付'])['累計ゲーム'].transform('mean').fillna(0)
        df['relative_games_ratio'] = (df['累計ゲーム'] / df['shop_avg_games'].replace(0, np.nan)).fillna(1.0)
        
        # --- 新規: 回収日に甘い機種フラグ ---
        if '機種名' in df.columns:
            df['is_recovery_day'] = (df['shop_avg_diff'] <= -100).astype(int)
            df_recovery = df[df['is_recovery_day'] == 1].copy()
            if not df_recovery.empty:
                df_recovery = df_recovery.sort_values('対象日付')
                df_recovery['shifted_rec_diff'] = df_recovery.groupby([shop_col, '機種名'])['差枚'].shift(1)
                df_recovery['recovery_mac_avg'] = df_recovery.groupby([shop_col, '機種名'])['shifted_rec_diff'].expanding().mean().reset_index(level=[0,1], drop=True)
                # 過去の回収日平均差枚がプラスの機種にフラグを立てる
                df_recovery['recovery_day_sweet_machine'] = (df_recovery['recovery_mac_avg'] > 0).astype(int)
                rec_map = df_recovery.dropna(subset=['recovery_day_sweet_machine']).drop_duplicates(subset=[shop_col, '機種名', '対象日付'], keep='last')
                df = pd.merge(df, rec_map[[shop_col, '機種名', '対象日付', 'recovery_day_sweet_machine']], on=[shop_col, '機種名', '対象日付'], how='left')
                df['recovery_day_sweet_machine'] = df.groupby([shop_col, '機種名'])['recovery_day_sweet_machine'].ffill().fillna(0).astype(int)
            else:
                df['recovery_day_sweet_machine'] = 0
            df = df.drop(columns=['is_recovery_day'], errors='ignore')
        
        # 店舗の見切りスピード（2000G未満で放置された台の割合）
        df['is_abandoned_machine'] = (df['累計ゲーム'] < 2000).astype(int)
        df['shop_abandon_rate'] = df.groupby([shop_col, '対象日付'])['is_abandoned_machine'].transform('mean').fillna(0)
        df = df.drop(columns=['is_abandoned_machine'])
        
        # --- ①店舗全体の回収・還元モード指標 (直近7日間の店舗全体の平均差枚) ---
        shop_daily_avg = df.groupby([shop_col, '対象日付'])['差枚'].mean().reset_index(name='shop_daily_avg_diff')
        shop_daily_avg = shop_daily_avg.sort_values([shop_col, '対象日付'])
        shop_daily_avg['shop_7days_avg_diff'] = shop_daily_avg.groupby(shop_col)['shop_daily_avg_diff'].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()).fillna(0)
        
        # --- 店舗フェイク率 (REGは引けているが差枚マイナスの台の割合) ---
        df['is_fake_high'] = ((df['累計ゲーム'] >= 1000) & (df['REG確率'] >= (1/300)) & (df['差枚'] <= 0)).astype(int)
        shop_daily_fake = df.groupby([shop_col, '対象日付'])['is_fake_high'].mean().reset_index(name='daily_shop_fake_rate')
        shop_daily_fake = shop_daily_fake.sort_values([shop_col, '対象日付'])
        shop_daily_fake['prev_shop_fake_rate'] = shop_daily_fake.groupby(shop_col)['daily_shop_fake_rate'].shift(1).fillna(0)
        df = pd.merge(df, shop_daily_fake[[shop_col, '対象日付', 'prev_shop_fake_rate']], on=[shop_col, '対象日付'], how='left')
        df = df.drop(columns=['is_fake_high'])

        # --- 新規追加: タコ粘りフェイク（稼働マジック）対策 ---
        if 'shop_avg_games' in df.columns:
            df['heavy_play_fake_penalty'] = ((df['累計ゲーム'] >= df['shop_avg_games'] * 1.5) & (df['差枚'] <= 0)).astype(int)
        else:
            df['heavy_play_fake_penalty'] = 0

        # --- 前日の店舗平均差枚 (日次ノルマのショート/オーバーによる緊急回収の確認用) ---
        shop_daily_avg['prev_shop_daily_avg_diff'] = shop_daily_avg.groupby(shop_col)['shop_daily_avg_diff'].shift(1).fillna(0)
        df = pd.merge(df, shop_daily_avg[[shop_col, '対象日付', 'shop_7days_avg_diff', 'prev_shop_daily_avg_diff']], on=[shop_col, '対象日付'], how='left')

        # --- 月間ノルマ進捗指標 (その月の前日までの店舗累計差枚) ---
        shop_daily_total = df.groupby([shop_col, '対象日付'])['差枚'].sum().reset_index(name='shop_daily_total_diff')
        shop_daily_total['年月'] = shop_daily_total['対象日付'].dt.to_period('M')
        shop_daily_total = shop_daily_total.sort_values([shop_col, '対象日付'])
        shop_daily_total['shop_monthly_cumulative_diff'] = shop_daily_total.groupby([shop_col, '年月'])['shop_daily_total_diff'].transform(lambda x: x.shift(1).cumsum()).fillna(0)
        df = pd.merge(df, shop_daily_total[[shop_col, '対象日付', 'shop_monthly_cumulative_diff']], on=[shop_col, '対象日付'], how='left')

        # --- 過去のAI予測（店舗全体の予測平均差枚）を読み込み、直近1週間の移動平均を特徴量に ---
        if df_daily_scores is not None and not df_daily_scores.empty:
            scores_to_merge = df_daily_scores[['店名', '予測対象日', '予測平均差枚']].copy()
            scores_to_merge = scores_to_merge.rename(columns={'予測対象日': '対象日付', '予測平均差枚': 'prev_day_shop_predicted_avg_diff'})
            scores_to_merge['対象日付'] = pd.to_datetime(scores_to_merge['対象日付'], errors='coerce')
            
            scores_to_merge = scores_to_merge.drop_duplicates(subset=['店名', '対象日付'], keep='last')
            
            df = pd.merge(df, scores_to_merge, on=['店名', '対象日付'], how='left')
            
            df = df.sort_values([shop_col, '対象日付'])
            df['shop_pred_diff_7d_avg'] = df.groupby(shop_col)['prev_day_shop_predicted_avg_diff'].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()).fillna(0)
            df = df.sort_values('original_order').reset_index(drop=True)
        else:
            df['shop_pred_diff_7d_avg'] = 0

        # --- 新規追加: 特日翌日の据え置き信頼度 ---
        if df_events is not None and not df_events.empty:
            ev_dates = df_events[['店名', 'イベント日付']].drop_duplicates().copy()
            ev_dates = ev_dates.rename(columns={'店名': shop_col})
            ev_dates['翌日'] = ev_dates['イベント日付'] + pd.Timedelta(days=1)
            
            df = pd.merge(df, ev_dates[[shop_col, '翌日']].assign(is_post_event=1), left_on=[shop_col, '対象日付'], right_on=[shop_col, '翌日'], how='left').drop(columns=['翌日'])
            df['is_post_event'] = df['is_post_event'].fillna(0)
            
            post_ev_df = df[df['is_post_event'] == 1].copy()
            if not post_ev_df.empty and 'valid_is_win' in post_ev_df.columns:
                post_sue_stats = post_ev_df[post_ev_df['is_prev_high_reg'] == 1].groupby(shop_col).agg(
                    post_ev_sueoki_trust=('valid_is_win', 'mean')
                ).reset_index()
                df = pd.merge(df, post_sue_stats, on=shop_col, how='left')
                df['post_ev_sueoki_trust'] = df['post_ev_sueoki_trust'].fillna(0)
            else:
                df['post_ev_sueoki_trust'] = 0
        else:
            df['post_ev_sueoki_trust'] = 0

    if shop_col and '機種名' in df.columns:
        # --- 新規: 過去の機種ごとの合算REG確率 ---
        mac_reg_stats = df.groupby([shop_col, '機種名']).agg(
            mac_total_g=('累計ゲーム', 'sum'),
            mac_total_reg=('REG', 'sum')
        ).reset_index()
        mac_reg_stats['past_mac_reg_prob'] = np.where(mac_reg_stats['mac_total_g'] > 0, mac_reg_stats['mac_total_reg'] / mac_reg_stats['mac_total_g'], 0)
        df = pd.merge(df, mac_reg_stats[[shop_col, '機種名', 'past_mac_reg_prob']], on=[shop_col, '機種名'], how='left')
        df['past_mac_reg_prob'] = df['past_mac_reg_prob'].fillna(0)

        # --- 新規: 過去の特定日(末尾)×機種ごとの合算REG確率 ---
        mac_digit_sum = df.groupby([shop_col, '対象日付', '機種名'])[['累計ゲーム', 'REG']].sum().reset_index()
        mac_digit_sum['digit'] = mac_digit_sum['対象日付'].dt.day % 10
        
        digit_mac_reg_stats = mac_digit_sum.groupby([shop_col, 'digit', '機種名']).agg(
            digit_mac_total_g=('累計ゲーム', 'sum'),
            digit_mac_total_reg=('REG', 'sum')
        ).reset_index()
        digit_mac_reg_stats['past_digit_mac_reg_prob'] = np.where(digit_mac_reg_stats['digit_mac_total_g'] > 0, digit_mac_reg_stats['digit_mac_total_reg'] / digit_mac_reg_stats['digit_mac_total_g'], 0)
        
        df = pd.merge(df, digit_mac_reg_stats[[shop_col, 'digit', '機種名', 'past_digit_mac_reg_prob']], left_on=[shop_col, 'target_date_end_digit', '機種名'], right_on=[shop_col, 'digit', '機種名'], how='left').drop(columns=['digit'])
        df['past_digit_mac_reg_prob'] = df['past_digit_mac_reg_prob'].fillna(0)

        # --- 新規: 過去の曜日×機種ごとの合算REG確率 ---
        mac_wd_sum = df.groupby([shop_col, '対象日付', '機種名'])[['累計ゲーム', 'REG']].sum().reset_index()
        mac_wd_sum['wd'] = mac_wd_sum['対象日付'].dt.dayofweek
        
        wd_mac_reg_stats = mac_wd_sum.groupby([shop_col, 'wd', '機種名']).agg(
            wd_mac_total_g=('累計ゲーム', 'sum'),
            wd_mac_total_reg=('REG', 'sum')
        ).reset_index()
        wd_mac_reg_stats['past_wd_mac_reg_prob'] = np.where(wd_mac_reg_stats['wd_mac_total_g'] > 0, wd_mac_reg_stats['wd_mac_total_reg'] / wd_mac_reg_stats['wd_mac_total_g'], 0)
        
        df = pd.merge(df, wd_mac_reg_stats[[shop_col, 'wd', '機種名', 'past_wd_mac_reg_prob']], left_on=[shop_col, 'target_weekday', '機種名'], right_on=[shop_col, 'wd', '機種名'], how='left').drop(columns=['wd'])
        df['past_wd_mac_reg_prob'] = df['past_wd_mac_reg_prob'].fillna(0)

    if shop_col and 'island_id' in df.columns:
        # --- 新規: 過去の島・列ごとの合算REG確率 ---
        island_reg_stats = df[df['island_id'] != "Unknown"].groupby([shop_col, 'island_id']).agg(
            isl_total_g=('累計ゲーム', 'sum'),
            isl_total_reg=('REG', 'sum')
        ).reset_index()
        island_reg_stats['past_island_reg_prob'] = np.where(island_reg_stats['isl_total_g'] > 0, island_reg_stats['isl_total_reg'] / island_reg_stats['isl_total_g'], 0)
        df = pd.merge(df, island_reg_stats[[shop_col, 'island_id', 'past_island_reg_prob']], on=[shop_col, 'island_id'], how='left')
        df['past_island_reg_prob'] = df['past_island_reg_prob'].fillna(0)

        # --- 新規追加: 過去の島のフェイク率 ---
        df['is_fake_high_tmp'] = ((df['累計ゲーム'] >= 1000) & (df['REG確率'] >= (1/300)) & (df['差枚'] <= 0)).astype(int)
        island_fake_stats = df[df['island_id'] != "Unknown"].groupby([shop_col, 'island_id'])['is_fake_high_tmp'].mean().reset_index(name='past_island_fake_rate')
        df = pd.merge(df, island_fake_stats, on=[shop_col, 'island_id'], how='left')
        df['past_island_fake_rate'] = df['past_island_fake_rate'].fillna(0)
        df = df.drop(columns=['is_fake_high_tmp'])

        # ※島ごとの高設定割合
        df['island_high_setting_ratio'] = df.groupby(['island_id', '対象日付'])['valid_is_win'].transform('mean').fillna(0)

        # --- ②機種ごとの扱い指標 (過去30日間のその機種の平均差枚・高設定率) ---
        machine_daily_avg = df.groupby([shop_col, '機種名', '対象日付']).agg(
            machine_daily_avg_diff=('差枚', 'mean'),
            machine_daily_high_rate=('valid_is_win', 'mean'),
            machine_daily_games=('累計ゲーム', 'mean')
        ).reset_index()
        machine_daily_avg = machine_daily_avg.sort_values([shop_col, '機種名', '対象日付'])
        machine_daily_avg['machine_30days_avg_diff'] = machine_daily_avg.groupby([shop_col, '機種名'])['machine_daily_avg_diff'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
        machine_daily_avg['machine_30days_high_rate'] = machine_daily_avg.groupby([shop_col, '機種名'])['machine_daily_high_rate'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
        
        machine_daily_avg['machine_3days_avg_diff'] = machine_daily_avg.groupby([shop_col, '機種名'])['machine_daily_avg_diff'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()).fillna(0)
        machine_daily_avg['machine_3days_high_setting_ratio'] = machine_daily_avg.groupby([shop_col, '機種名'])['machine_daily_high_rate'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()).fillna(0)
        machine_daily_avg['machine_prev_avg_games'] = machine_daily_avg.groupby([shop_col, '機種名'])['machine_daily_games'].transform(lambda x: x.shift(1)).fillna(0)

        df = pd.merge(df, machine_daily_avg[[shop_col, '機種名', '対象日付', 'machine_30days_avg_diff', 'machine_30days_high_rate', 'machine_3days_avg_diff', 'machine_3days_high_setting_ratio', 'machine_prev_avg_games']], on=[shop_col, '機種名', '対象日付'], how='left')
        
        df['machine_avg_diff'] = df.groupby([shop_col, '機種名', '対象日付'])['差枚'].transform('mean').fillna(0)
        df['machine_median_diff'] = df.groupby([shop_col, '機種名', '対象日付'])['差枚'].transform('median').fillna(0)
        df['machine_high_rate'] = df.groupby([shop_col, '機種名', '対象日付'])['valid_is_win'].transform('mean').fillna(0)
        df['machine_heavy_lose_rate'] = df.groupby([shop_col, '機種名', '対象日付'])['is_heavy_lose'].transform('mean').fillna(0)
        df['machine_play_rate'] = df.groupby([shop_col, '機種名', '対象日付'])['is_play_machine'].transform('mean').fillna(0)

    if shop_col:
        # --- ③台番号（場所）ごとの扱い指標 (過去30日間のその台番号の平均差枚・高設定率) ---
        machine_no_daily_avg = df.groupby([shop_col, '台番号', '対象日付']).agg(
            machine_no_daily_avg_diff=('差枚', 'mean'),
            machine_no_daily_high_rate=('valid_is_win', 'mean')
        ).reset_index()
        machine_no_daily_avg = machine_no_daily_avg.sort_values([shop_col, '台番号', '対象日付'])
        machine_no_daily_avg['machine_no_30days_avg_diff'] = machine_no_daily_avg.groupby([shop_col, '台番号'])['machine_no_daily_avg_diff'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
        machine_no_daily_avg['machine_no_30days_high_rate'] = machine_no_daily_avg.groupby([shop_col, '台番号'])['machine_no_daily_high_rate'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
        df = pd.merge(df, machine_no_daily_avg[[shop_col, '台番号', '対象日付', 'machine_no_30days_avg_diff', 'machine_no_30days_high_rate']], on=[shop_col, '台番号', '対象日付'], how='left')

    if shop_col and '機種名' in df.columns and 'prev_累計ゲーム' in df.columns:
        df['machine_play_volume_rank'] = df.groupby([shop_col, '対象日付', '機種名'])['prev_累計ゲーム'].rank(pct=True).fillna(0.5)

    # ソート順を元に戻す
    df = df.sort_values('original_order').drop(columns=['original_order']).reset_index(drop=True)
    
    df['prev_bonus_balance'] = df['REG'] - df['BIG']
    df['prev_unlucky_gap'] = (df['REG'] * 200) - df['差枚']

    # --- 新規: 低稼働・高設定据え置き特化の特徴量 ---
    if 'prev_累計ゲーム' in df.columns and 'prev_REG確率' in df.columns:
        df['is_low_play_high_reg'] = ((df['prev_累計ゲーム'] >= 1000) & (df['prev_累計ゲーム'] < 3000) & (df['prev_REG確率'] >= df['spec_reg'])).astype(int)
        
    # --- 新規: 週間吸い込みからの還元曜日狙い 特化の特徴量 ---
    if 'mean_7days_diff' in df.columns and 'weekday_avg_diff' in df.columns:
        df['is_hot_wd_and_heavy_lose'] = ((df['weekday_avg_diff'] >= 100) & (df['mean_7days_diff'] <= -500)).astype(int)
        
    # --- 新規: 前日の据え置き狙い複合特徴量 ---
    if 'prev_差枚' in df.columns and 'prev_REG確率' in df.columns:
        df['is_prev_up_trend_and_high_reg'] = ((df['prev_差枚'] > 0) & (df['prev_REG確率'] >= df['spec_reg'])).astype(int)
        df['is_prev_low_reg_and_good_diff'] = ((df['prev_REG確率'] < df['spec_reg']) & (df['prev_差枚'] > 0)).astype(int)
        df['prev_reg_reliability_score'] = np.where(df['spec_reg'] > 0, np.clip(df['prev_REG確率'] / df['spec_reg'], 0, 2.0) * (df['prev_累計ゲーム'] / 1000.0), 0)

    # --- 新規追加: 2日間の波(トレンド) 複合特徴量 ---
    if 'prev_差枚' in df.columns and '差枚' in df.columns:
        df['trend_v_recovery'] = ((df['prev_差枚'] < 0) & (df['差枚'] > 0)).astype(int)
        df['trend_cont_lose'] = ((df['prev_差枚'] < 0) & (df['差枚'] < 0)).astype(int)
        df['trend_cont_win'] = ((df['prev_差枚'] > 0) & (df['差枚'] > 0)).astype(int)
        df['trend_down_rebound'] = ((df['prev_差枚'] > 0) & (df['差枚'] < 0)).astype(int)

    # --- 新規追加: 誤認誘導型対策（REGと差枚のバランス） ---
    safe_g = df['累計ゲーム'].replace(0, np.nan)
    df['reg_diff_interaction'] = (df['REG'] / safe_g) * (df['差枚'] / safe_g)
    df['reg_diff_interaction'] = df['reg_diff_interaction'].fillna(0)
    
    df['big_reg_ratio_gap'] = (df['BIG'] / safe_g) - (df['REG'] / safe_g)
    df['big_reg_ratio_gap'] = df['big_reg_ratio_gap'].fillna(0)
    
    if 'mean_7days_games' in df.columns and 'mean_7days_diff' in df.columns:
        safe_7d_g = df['mean_7days_games'].replace(0, np.nan)
        efficiency = (df['mean_7days_diff'] / safe_7d_g).fillna(0)
        df['reg_efficiency_penalty'] = (df['REG'] / safe_g).fillna(0) * efficiency
    else:
        df['reg_efficiency_penalty'] = 0.0

    # --- ノイズ対策: 低稼働データの確率系特徴量を無効化 ---
    # 以前は2000G未満を0に丸めていたが、低稼働台のポテンシャルを見抜くため1000G未満に緩和
    low_kado_mask = df['累計ゲーム'] < 1000
    if 'REG確率' in df.columns:
        df.loc[low_kado_mask, 'REG確率'] = 0.0
    if 'BIG確率' in df.columns:
        df.loc[low_kado_mask, 'BIG確率'] = 0.0
    if 'reg_ratio' in df.columns:
        df.loc[low_kado_mask, 'reg_ratio'] = 0.0
    df.loc[low_kado_mask, 'prev_bonus_balance'] = 0.0
    df.loc[low_kado_mask, 'prev_unlucky_gap'] = 0.0
    
    if 'prev_累計ゲーム' in df.columns and 'prev_REG確率' in df.columns:
        prev_low_kado_mask = df['prev_累計ゲーム'] < 1000
        df.loc[prev_low_kado_mask, 'prev_REG確率'] = 0.0

    if 'island_id' in df.columns:
        df['island_avg_diff'] = df.groupby(['island_id', '対象日付'])['差枚'].transform('mean').fillna(0)
        df['island_high_rate'] = df.groupby(['island_id', '対象日付'])['valid_is_win'].transform('mean').fillna(0)
        df['island_total_g'] = df.groupby(['island_id', '対象日付'])['累計ゲーム'].transform('sum')
        df['island_total_reg'] = df.groupby(['island_id', '対象日付'])['REG'].transform('sum')
        df['island_reg_prob'] = np.where(df['island_total_g'] > 0, df['island_total_reg'] / df['island_total_g'], 0)
        df['prev_island_reg_prob'] = df.groupby(group_keys)['island_reg_prob'].shift(1).fillna(0)

        # --- 新規追加: 面配分のフェイク見極め用特徴量 ---
        df['is_plus_tmp'] = (df['差枚'] > 0).astype(int)
        df['island_win_rate'] = df.groupby(['island_id', '対象日付'])['is_plus_tmp'].transform('mean').fillna(0)
        df = df.drop(columns=['is_plus_tmp'])
        
        df['is_fake_high_tmp'] = ((df['累計ゲーム'] >= 1000) & (df['REG確率'] >= (1/300)) & (df['差枚'] <= 0)).astype(int)
        df['island_fake_ratio'] = df.groupby(['island_id', '対象日付'])['is_fake_high_tmp'].transform('mean').fillna(0)
        df = df.drop(columns=['is_fake_high_tmp'])

        # --- 新規追加: 見せ台（角台）フェイクフラグ ---
        inner_df = df[df['is_corner'] == 0].copy()
        if not inner_df.empty:
            inner_avg = inner_df.groupby(['island_id', '対象日付'])['差枚'].mean().reset_index(name='island_inner_avg_diff')
            df = pd.merge(df, inner_avg, on=['island_id', '対象日付'], how='left')
            df['island_inner_avg_diff'] = df['island_inner_avg_diff'].fillna(0)
            df['is_corner_showpiece'] = ((df['is_corner'] == 1) & (df['REG確率'] >= (1/300)) & (df['island_inner_avg_diff'] < 0)).astype(int)
        else:
            df['is_corner_showpiece'] = 0

        # --- 新規追加: 面配分（島型）の未発掘お宝台フラグ ---
        df['island_unexplored_flag'] = ((df['island_id'] != "Unknown") & (df['island_avg_diff'] >= 500) & (df['累計ゲーム'] < 2000)).astype(int)

    # --- 上げリセット（設定変更）検知用の特徴量 ---
    # 差枚が+500枚以上になったら「設定が入り放出された」とみなしリセット
    df['is_released'] = (df['差枚'] >= 500).astype(int)
    df['temp_reset_group'] = df.groupby(group_keys)['is_released'].cumsum()
    
    # 差枚が+500枚未満かつ稼働した日のみ「実質マイナス」としてカウント (0G放置はノーカウントで連続日数を維持)
    df['is_not_released'] = ((df['差枚'] < 500) & (df['累計ゲーム'] > 0)).astype(int)
    df['連続マイナス日数'] = df.groupby(group_keys + ['temp_reset_group'])['is_not_released'].cumsum()
    df['cons_minus_total_diff'] = df.groupby(group_keys + ['temp_reset_group'])['差枚'].cumsum()
    df = df.drop(columns=['temp_reset_group', 'is_released', 'is_not_released'])

    # --- 据え置き（高設定継続）検知用の特徴量 ---
    # 差枚が0以下かつ稼働した日のみカウントリセット (0G放置はリセットせず保留)
    df['is_minus_or_zero'] = ((df['差枚'] <= 0) & (df['累計ゲーム'] > 0)).astype(int)
    df['plus_reset_group'] = df.groupby(group_keys)['is_minus_or_zero'].cumsum()
    
    df['is_plus'] = (df['差枚'] > 0).astype(int)
    df['連続プラス日数'] = df.groupby(group_keys + ['plus_reset_group'])['is_plus'].cumsum()
    df = df.drop(columns=['is_minus_or_zero', 'plus_reset_group', 'is_plus'])

    # --- 連続低稼働日数のカウント（テコ入れ狙い） ---
    UTILIZATION_THRESHOLD = 1500
    df['is_low_utilization'] = (df['累計ゲーム'] < UTILIZATION_THRESHOLD).astype(int)
    df['is_active'] = (df['累計ゲーム'] >= UTILIZATION_THRESHOLD).astype(int)
    df['low_util_reset_group'] = df.groupby(group_keys)['is_active'].cumsum()
    
    df['連続低稼働日数'] = df.groupby(group_keys + ['low_util_reset_group'])['is_low_utilization'].cumsum()
    df = df.drop(columns=['is_low_utilization', 'is_active', 'low_util_reset_group'])

    # --- 連続高REG日数のカウント（高設定の据え置き狙い） ---
    # Zスコア等の救済を含んだ is_prev_high_reg をそのまま高設定挙動の基準として使用する
    # しっかり回された(3000G以上)上で高REGでなかった日のみリセット (稼働不足の日はリセットせず保留)
    df['is_not_high_reg_day'] = ((df['is_prev_high_reg'] == 0) & (df['prev_累計ゲーム'] >= 3000)).astype(int)
    df['high_reg_reset_group'] = df.groupby(group_keys)['is_not_high_reg_day'].cumsum()
    df['cons_high_reg_days'] = df.groupby(group_keys + ['high_reg_reset_group'])['is_prev_high_reg'].cumsum()
    df = df.drop(columns=['is_not_high_reg_day', 'high_reg_reset_group'])

    # 台ごとの過去データ件数（履歴の長さ）を計算し、信頼度の指標とする
    df['history_count'] = df.groupby(group_keys).cumcount() + 1

    if '機種名' in df.columns:
        mac_group_keys = group_keys + ['機種名']
        df['machine_history_count'] = df.groupby(mac_group_keys).cumcount() + 1
    else:
        df['machine_history_count'] = df['history_count']

    # --- 新台・配置変更の正確な検知 ---
    if shop_col:
        # その日時点での、その店舗の最大データ蓄積日数を取得
        df['shop_date_max_history'] = df.groupby([shop_col, '対象日付'])['history_count'].transform('max')
        is_recent_machine = (df['shop_date_max_history'] >= 14) & (df['machine_history_count'] <= 7)
        
        df['first_active_date'] = df.groupby([shop_col, '台番号', '機種名'])['対象日付'].transform('min')
        
        if df_events is not None and not df_events.empty:
            valid_ev = df_events.copy()
            if 'イベント種別' in valid_ev.columns:
                valid_ev = valid_ev[~valid_ev['イベント種別'].isin(['パチンコ専用', '対象外(無効)'])]
            
            ev_for_new = valid_ev[['店名', 'イベント日付', '対象機種', 'イベント名']].copy()
            ev_for_new = ev_for_new.rename(columns={'店名': shop_col})
            
            first_dates = df.loc[is_recent_machine, [shop_col, '台番号', '機種名', 'first_active_date']].drop_duplicates()
            merged_first = pd.merge(first_dates, ev_for_new, left_on=[shop_col, 'first_active_date'], right_on=[shop_col, 'イベント日付'], how='left')
            
            def check_new_machine(row):
                if pd.isna(row['イベント日付']): return 0
                t_mac = str(row['対象機種'])
                my_mac = str(row['機種名'])
                ev_name = str(row['イベント名'])
                
                is_shindai_event = '新台' in ev_name or '入替' in ev_name or '導入' in ev_name
                
                if t_mac in ['指定なし', 'スロット全体', 'ジャグラー全体', '全体', 'nan', 'None']:
                    return 1 if is_shindai_event else 0
                elif my_mac in t_mac or t_mac in my_mac:
                    return 1
                elif t_mac == 'ジャグラー以外 (パチスロ他機種)':
                    return 0
                return 0
                
            merged_first['is_real_new'] = merged_first.apply(check_new_machine, axis=1)
            real_new_map = merged_first.groupby([shop_col, '台番号', '機種名'])['is_real_new'].max().reset_index()
            
            df = pd.merge(df, real_new_map, on=[shop_col, '台番号', '機種名'], how='left')
            df['is_real_new'] = df['is_real_new'].fillna(0)
            
            df['is_new_machine'] = (is_recent_machine & (df['is_real_new'] == 1)).astype(int)
            df['is_moved_machine'] = (is_recent_machine & (df['is_real_new'] == 0)).astype(int)
            df = df.drop(columns=['shop_date_max_history', 'machine_history_count', 'first_active_date', 'is_real_new'])
        else:
            df['is_new_machine'] = 0
            df['is_moved_machine'] = is_recent_machine.astype(int)
            df = df.drop(columns=['shop_date_max_history', 'machine_history_count', 'first_active_date'])
    else:
        df['is_new_machine'] = 0
        df['is_moved_machine'] = 0

    # 一時的に作成したフラグは削除
    df = df.drop(columns=['is_heavy_lose', 'is_play_machine', 'shifted_g', 'shifted_reg', 'shifted_diff_wd', 'shifted_is_win_wd', 'shifted_diff_ev', 'shifted_is_win_ev', 'shifted_diff_ev_mac', 'shifted_is_win_ev_mac', 'shifted_diff_ev_end', 'spec_reg', 'spec_tot', 'spec_reg3', 'spec_b6_den', 'spec_reg1', 'BIG分母', 'total_prob', 'z_score_reg', 'next_z_score_reg'], errors='ignore')

    df = df.rename(columns=lambda x: re.sub(r'[",\[\]{}:]', '', str(x)))
    df = df.loc[:, ~df.columns.duplicated()]

    features = [f for f in BASE_FEATURES if f in df.columns]

    return df, features

# ---------------------------------------------------------
# 分析・予測ロジック (メイン関数)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=2, ttl=3600)
def run_analysis(df, _df_events=None, _df_island=None, shop_hyperparams=None, target_date=None):
    df_raw_for_eval = df.copy() # 据え置き前提判定用の生データを確保
    if df.empty: return df, pd.DataFrame(), pd.DataFrame()

    # --- AI処理のローカルキャッシュ機構 ---
    # 生データの状態（行数や最新日付）、予測対象日、AI設定から一意なハッシュを作成
    data_status = f"{len(df)}_{df['対象日付'].max()}_{target_date}_{APP_VERSION}_{str(shop_hyperparams)}"
    cache_key = hashlib.md5(data_status.encode()).hexdigest()
    cache_file = os.path.join(BASE_DIR, f"ai_cache_{cache_key}.pkl")
    
    # すでに同じデータ状態で計算済みのキャッシュがあれば、それをロードして一瞬で返す
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data['df'], cached_data['df_verify'], cached_data['df_importance']
        except Exception:
            pass # 読み込みエラー時はそのまま通常の分析へ進む

    # 過去の店舗別予測スコアを読み込む
    _df_daily_scores = load_daily_shop_scores()

    if shop_hyperparams is None:
        shop_hyperparams = {"デフォルト": {'train_months': 3, 'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50}}

    # 1. 特徴量エンジニアリング
    df, features = _generate_features(df, _df_events, _df_island, _df_daily_scores, target_date)
    if df.empty: return df, pd.DataFrame(), pd.DataFrame()

    train_df = df.dropna(subset=['next_diff']).copy()
    
    # --- ノイズデータ（設定が判別できない放置台）を学習母数から除外 ---
    # 3000G以上回った台、または3000G未満でも±750枚以上動いて見切られた(または誤爆した)台のみを学習対象とする
    # これにより、0G放置台などが分母から消え、純粋な「稼働した際の本物の高設定確率」が算出される
    train_df['valid_play_mask'] = get_valid_play_mask(train_df['next_累計ゲーム'], train_df['next_diff'])
    
    predict_df = df[df['next_diff'].isna()].copy()
    
    shop_col = '店名' if '店名' in train_df.columns else ('店舗名' if '店舗名' in train_df.columns else None)

    # 3. 予測データを最新日に絞り込み
    if '対象日付' in predict_df.columns and not predict_df.empty:
        if shop_col:
            latest_dates = predict_df.groupby(shop_col)['対象日付'].transform('max')
            predict_df = predict_df[predict_df['対象日付'] == latest_dates]
        else:
            max_date = predict_df['対象日付'].max()
            predict_df = predict_df[predict_df['対象日付'] == max_date]
        
    # --- LSTMによる時系列「波」特徴量の追加 ---
    if add_lstm_features is not None:
        try:
            lstm_hp = shop_hyperparams.get("デフォルト", {})
            l_hidden = lstm_hp.get('lstm_hidden_size', 64)
            l_lr = lstm_hp.get('lstm_lr', 0.001)
            l_epochs = lstm_hp.get('lstm_epochs', 20)
            train_df = add_lstm_features(train_df, shop_col=shop_col, seq_length=7, hidden_size=l_hidden, lr=l_lr, epochs=l_epochs)
            predict_df = add_lstm_features(predict_df, shop_col=shop_col, seq_length=7, hidden_size=l_hidden, lr=l_lr, epochs=l_epochs)
            if 'lstm_wave_score' not in features:
                features.append('lstm_wave_score')
        except Exception as e:
            print(f"LSTM特徴量の追加に失敗しました: {e}")

    if len(train_df) < 10:
        return predict_df, train_df, pd.DataFrame()

    # 4 & 5. モデル学習と推論 (店舗ごとのパラメータで独立して実行される)
    predict_df, train_df, feature_importances = train_models(train_df, predict_df, features, shop_hyperparams)

    # 6. 後処理 (スコア補正、根拠の自然言語生成)
    predict_df, train_df = postprocess_predictions(predict_df, train_df)

    if 'valid_play_mask' in train_df.columns:
        train_df = train_df.drop(columns=['valid_play_mask'], errors='ignore')
        
    if 'valid_play_mask' in predict_df.columns:
        predict_df = predict_df.drop(columns=['valid_play_mask'], errors='ignore')

    # --- 7. 据え置き前提判定の根本適用 (NOの日は sueoki_score を強制リセット) ---
    if shop_col:
        def apply_sueoki_premise_to_df(df_target):
            if df_target.empty or 'sueoki_score' not in df_target.columns: return df_target
            date_col_for_grp = 'next_date' if 'next_date' in df_target.columns else '対象日付'
            
            for (shop, tgt_date), group in df_target.groupby([shop_col, date_col_for_grp]):
                # その日より前の生データで判定
                past_raw = df_raw_for_eval[(df_raw_for_eval[shop_col] == shop) & (df_raw_for_eval['対象日付'] < tgt_date)].copy()
                premise, reason = evaluate_sueoki_premise(past_raw, tgt_date, _df_events)
                
                if premise == "NO":
                    df_target.loc[group.index, 'sueoki_score'] = 0.0
                    def add_no_sue_reason(r):
                        orig = str(r.get('根拠', ''))
                        if orig == 'nan': orig = ''
                        new_r = f"【据え置き無効】{reason}"
                        if orig and orig != '-': return orig + " " + new_r
                        return new_r
                    df_target.loc[group.index, '根拠'] = df_target.loc[group.index].apply(add_no_sue_reason, axis=1)
            return df_target

        predict_df = apply_sueoki_premise_to_df(predict_df)
        train_df = apply_sueoki_premise_to_df(train_df)

    # --- 処理の最後に計算結果をキャッシュとして保存 ---
    try:
        # 古いAIキャッシュファイルを掃除
        for f_path in glob.glob(os.path.join(BASE_DIR, 'ai_cache_*.pkl')):
            try: os.remove(f_path)
            except Exception: pass
        # 今回の結果を保存
        with open(cache_file, 'wb') as f:
            pickle.dump({'df': predict_df, 'df_verify': train_df, 'df_importance': feature_importances}, f)
    except Exception:
        pass

    return predict_df, train_df, feature_importances

def get_island_prediction_ranking(df):
    """島（列）別の予測期待度ランキングを集計する"""
    if 'island_id' not in df.columns:
        return pd.DataFrame()
        
    island_pred_df = df[df['island_id'] != "Unknown"].copy()
    if island_pred_df.empty:
        return pd.DataFrame()
    
    island_pred_df['島名'] = island_pred_df['island_id'].apply(lambda x: str(x).split('_', 1)[1] if '_' in str(x) else str(x))
    
    if 'sueoki_score' in island_pred_df.columns:
        island_pred_df['max_score'] = island_pred_df[['prediction_score', 'sueoki_score']].max(axis=1)
    else:
        island_pred_df['max_score'] = island_pred_df['prediction_score']
        
    isl_stats = island_pred_df.groupby('島名').agg(
        平均期待度=('max_score', 'mean'),
        激アツ台数=('max_score', lambda x: (x >= 0.30).sum()),
        全台数=('台番号', 'count')
    ).reset_index().sort_values('平均期待度', ascending=False)
    
    if '予測差枚数' in island_pred_df.columns:
        diff_stats = island_pred_df.groupby('島名')['予測差枚数'].mean().reset_index().rename(columns={'予測差枚数': '予測平均差枚'})
        isl_stats = pd.merge(isl_stats, diff_stats, on='島名', how='left')
    else:
        isl_stats['予測平均差枚'] = np.nan
        
    isl_stats['営業予測'] = isl_stats.apply(
        lambda row: classify_shop_eval(row.get('予測平均差枚'), row.get('全台数', 20), is_prediction=False).replace('営業', '島').replace('日', '島'), axis=1
    )
    isl_stats['平均期待度'] = isl_stats['平均期待度'] * 100
    return isl_stats

def get_shop_prediction_ranking(df, df_raw, df_pred_log, specs, eval_period, shop_col):
    """店舗別の予測期待度ランキングおよび実績勝率を集計する"""
    temp_df = df.copy()
    if 'sueoki_score' in temp_df.columns:
        temp_df['max_score'] = temp_df[['prediction_score', 'sueoki_score']].max(axis=1)
    else:
        temp_df['max_score'] = temp_df['prediction_score']
        
    shop_stats = temp_df.groupby(shop_col).agg(
        平均スコア=('max_score', 'mean'),
        推奨台数=('max_score', lambda x: (x >= 0.30).sum()),
        全台数=('台番号', 'nunique')
    ).reset_index()
    
    if '予測差枚数' in df.columns:
        diff_stats = df.groupby(shop_col).agg(予測平均差枚=('予測差枚数', 'mean')).reset_index()
        shop_stats = pd.merge(shop_stats, diff_stats, on=shop_col, how='left')
    else:
        shop_stats['予測平均差枚'] = np.nan

    # --- 収集日数の計算 ---
    shop_days_map = {}
    if not df_raw.empty and shop_col in df_raw.columns and '対象日付' in df_raw.columns:
        days_stats = df_raw.groupby(shop_col)['対象日付'].nunique().reset_index()
        shop_days_map = dict(zip(days_stats[shop_col], days_stats['対象日付']))
    shop_stats['収集日数'] = shop_stats[shop_col].map(shop_days_map).fillna(0).astype(int)
    
    # --- ガチ予測ログベースのAI正答率・勝率計算 ---
    ai_accuracy_map, ai_win_rate_map = {}, {}
    ai_acc_str_map, ai_win_str_map = {}, {}

    c_ai_win_rate_map, s_ai_win_rate_map = {}, {}
    c_ai_win_str_map, s_ai_win_str_map = {}, {}
    
    if df_pred_log is not None and not df_pred_log.empty and not df_raw.empty:
        df_pred_log_temp = df_pred_log.copy()
        if '予測対象日' in df_pred_log_temp.columns:
            df_pred_log_temp['予測対象日'] = pd.to_datetime(df_pred_log_temp['予測対象日'], errors='coerce')
        df_pred_log_temp['対象日付'] = pd.to_datetime(df_pred_log_temp['対象日付'], errors='coerce')
        
        if '予測対象日' in df_pred_log_temp.columns:
            df_pred_log_temp['予測対象日_merge'] = df_pred_log_temp['予測対象日'].fillna(df_pred_log_temp['対象日付'] + pd.Timedelta(days=1))
        else:
            df_pred_log_temp['予測対象日_merge'] = df_pred_log_temp['対象日付'] + pd.Timedelta(days=1)
            
        shop_col_pred = '店名' if '店名' in df_pred_log_temp.columns else '店舗名'
        if shop_col != shop_col_pred:
            df_pred_log_temp = df_pred_log_temp.rename(columns={shop_col_pred: shop_col})
            
        df_pred_log_temp['台番号'] = df_pred_log_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
        if '実行日時' in df_pred_log_temp.columns:
            df_pred_log_temp = df_pred_log_temp.sort_values('実行日時', ascending=False).drop_duplicates(
                subset=['予測対象日_merge', shop_col, '台番号'], keep='first'
            )

        df_raw_temp = df_raw.copy()
        df_raw_temp['台番号'] = df_raw_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_raw_temp['対象日付'] = pd.to_datetime(df_raw_temp['対象日付'], errors='coerce')
        
        merged = pd.merge(df_pred_log_temp, df_raw_temp, left_on=['予測対象日_merge', shop_col, '台番号'], right_on=['対象日付', shop_col, '台番号'], how='inner', suffixes=('_pred', '_raw'))
        
        if not merged.empty and 'prediction_score' in merged.columns:
            if eval_period != "全期間":
                days = 7 if eval_period == "直近1週間" else 30
                merged = merged[merged['予測対象日_merge'] > (merged['予測対象日_merge'].max() - pd.Timedelta(days=days))].copy()
            
            merged['prediction_score'] = pd.to_numeric(merged['prediction_score'], errors='coerce')
            if 'sueoki_score' not in merged.columns:
                merged['sueoki_score'] = 0.0
            merged['sueoki_score'] = pd.to_numeric(merged['sueoki_score'], errors='coerce')
            
            shop_machine_counts = df.groupby(shop_col)['台番号'].nunique().to_dict()
            merged['top_k_threshold'] = merged[shop_col].apply(lambda x: max(3, min(10, int(shop_machine_counts.get(x, 50) * 0.10))))
            
            merged['c_daily_rank'] = merged.groupby(['予測対象日_merge', shop_col])['prediction_score'].rank(method='first', ascending=False)
            merged['s_daily_rank'] = merged.groupby(['予測対象日_merge', shop_col])['sueoki_score'].rank(method='first', ascending=False)
            
            act_b = pd.to_numeric(merged['BIG'], errors='coerce').fillna(0)
            act_r = pd.to_numeric(merged['REG'], errors='coerce').fillna(0)
            act_g = pd.to_numeric(merged['累計ゲーム'], errors='coerce').fillna(0)
            act_diff = pd.to_numeric(merged['差枚'], errors='coerce').fillna(0)
            
            merged['valid_high_play'] = (act_g >= 3000)
            merged['is_high_setting'] = calculate_high_setting_mask(merged, specs, mac_col='機種名_raw').astype(int)
            
            merged['valid_play'] = get_valid_play_mask(act_g, act_diff)
            merged['valid_win'] = merged['valid_play'] & (act_diff > 0)
            merged['valid_high'] = merged['valid_high_play'] & (merged['is_high_setting'] == 1)

            c_high_expect_df = merged[merged['c_daily_rank'] <= merged['top_k_threshold']].copy()
            s_high_expect_df = merged[merged['s_daily_rank'] <= merged['top_k_threshold']].copy()
            
            if not c_high_expect_df.empty:
                c_acc_stats = c_high_expect_df.groupby(shop_col).agg(
                    c_正答数=('valid_high', 'sum'), c_高設定有効数=('valid_high_play', 'sum'), c_有効稼働数=('valid_play', 'sum'), c_勝数=('valid_win', 'sum')
                ).reset_index()
                c_acc_stats['c_勝率'] = np.where(c_acc_stats['c_有効稼働数'] > 0, c_acc_stats['c_勝数'] / c_acc_stats['c_有効稼働数'], 0.0)
                
                c_ai_win_rate_map = dict(zip(c_acc_stats[shop_col], c_acc_stats['c_勝率']))
                c_ai_win_str_map = {}
                for _, r in c_acc_stats.iterrows():
                    c_ai_win_str_map[r[shop_col]] = f"{r['c_勝率']*100:.1f}% ({int(r['c_勝数'])}/{int(r['c_有効稼働数'])}台)"
            else:
                c_ai_win_rate_map = {}
                c_ai_win_str_map = {}

            if not s_high_expect_df.empty:
                s_acc_stats = s_high_expect_df.groupby(shop_col).agg(
                    s_正答数=('valid_high', 'sum'), s_高設定有効数=('valid_high_play', 'sum'), s_有効稼働数=('valid_play', 'sum'), s_勝数=('valid_win', 'sum')
                ).reset_index()
                s_acc_stats['s_勝率'] = np.where(s_acc_stats['s_有効稼働数'] > 0, s_acc_stats['s_勝数'] / s_acc_stats['s_有効稼働数'], 0.0)
                
                s_ai_win_rate_map = dict(zip(s_acc_stats[shop_col], s_acc_stats['s_勝率']))
                s_ai_win_str_map = {}
                for _, r in s_acc_stats.iterrows():
                    s_ai_win_str_map[r[shop_col]] = f"{r['s_勝率']*100:.1f}% ({int(r['s_勝数'])}/{int(r['s_有効稼働数'])}台)"
            else:
                s_ai_win_rate_map = {}
                s_ai_win_str_map = {}
        
    shop_stats['変更勝率_数値'] = shop_stats[shop_col].map(c_ai_win_rate_map).fillna(0.0) * 100
    shop_stats['据え置き勝率_数値'] = shop_stats[shop_col].map(s_ai_win_rate_map).fillna(0.0) * 100
    shop_stats['変更勝率'] = shop_stats[shop_col].map(c_ai_win_str_map).fillna("- (0/0台)")
    shop_stats['据え置き勝率'] = shop_stats[shop_col].map(s_ai_win_str_map).fillna("- (0/0台)")
    
    def calc_sort_score(row):
        score = row['平均スコア']
        sample_count_c = int(str(row.get('変更勝率', '')).split('/')[1].split('台)')[0]) if '/' in str(row.get('変更勝率', '')) else 0
        sample_count_s = int(str(row.get('据え置き勝率', '')).split('/')[1].split('台)')[0]) if '/' in str(row.get('据え置き勝率', '')) else 0
        
        if sample_count_c >= 5 and row['変更勝率_数値'] > 0:
            if row['変更勝率_数値'] < 30: score -= 0.15
            elif row['変更勝率_数値'] < 40: score -= 0.05
        if sample_count_s >= 5 and row['据え置き勝率_数値'] > 0:
            if row['据え置き勝率_数値'] < 30: score -= 0.15
            elif row['据え置き勝率_数値'] < 40: score -= 0.05
            
        return score

    shop_stats['ソート用スコア'] = shop_stats.apply(calc_sort_score, axis=1)
    shop_stats['営業予測'] = shop_stats.apply(lambda row: classify_shop_eval(row.get('予測平均差枚'), row.get('全台数', 50), is_prediction=True).replace('営業', '').replace('日', ''), axis=1)
    return shop_stats.sort_values('ソート用スコア', ascending=False)

def get_daily_machine_stats(df_raw_shop, machine_name):
    """指定した機種の日別の全台合算成績を集計する"""
    daily_mac_df = df_raw_shop[df_raw_shop['機種名'] == machine_name].copy()
    if daily_mac_df.empty:
        return pd.DataFrame()
        
    daily_mac_stats = daily_mac_df.groupby('対象日付').agg(
        設置台数=('台番号', 'nunique'),
        総回転=('累計ゲーム', 'sum'),
        BIG=('BIG', 'sum'),
        REG=('REG', 'sum'),
        平均差枚=('差枚', 'mean'),
        合計差枚=('差枚', 'sum')
    ).reset_index().sort_values('対象日付', ascending=False)
    
    daily_mac_df['valid_play'] = get_valid_play_mask(daily_mac_df['累計ゲーム'], daily_mac_df['差枚'])
    daily_mac_df['is_win'] = daily_mac_df['valid_play'] & (daily_mac_df['差枚'] > 0)
    
    win_stats = daily_mac_df.groupby('対象日付').agg(
        有効稼働=('valid_play', 'sum'), 勝台数=('is_win', 'sum')
    ).reset_index()
    
    daily_mac_stats = pd.merge(daily_mac_stats, win_stats, on='対象日付', how='left')
    daily_mac_stats['勝率'] = np.where(daily_mac_stats['有効稼働'] > 0, (daily_mac_stats['勝台数'] / daily_mac_stats['有効稼働']) * 100, 0.0)
    daily_mac_stats['合算確率分母'] = np.where((daily_mac_stats['BIG'] + daily_mac_stats['REG']) > 0, daily_mac_stats['総回転'] / (daily_mac_stats['BIG'] + daily_mac_stats['REG']), 0)
    daily_mac_stats['REG確率分母'] = np.where(daily_mac_stats['REG'] > 0, daily_mac_stats['総回転'] / daily_mac_stats['REG'], 0)
    daily_mac_stats['対象日付_str'] = daily_mac_stats['対象日付'].dt.strftime('%Y-%m-%d')
    daily_mac_stats['合算確率'] = daily_mac_stats['合算確率分母'].apply(lambda x: f"1/{x:.1f}" if x > 0 else "-")
    daily_mac_stats['REG確率'] = daily_mac_stats['REG確率分母'].apply(lambda x: f"1/{x:.1f}" if x > 0 else "-")
    
    return daily_mac_stats

def get_machine_basic_stats(df_raw_shop, specs):
    """機種別の基本成績（設定5近似度、高設定率など）を集計する"""
    if df_raw_shop.empty:
        return pd.DataFrame()
        
    mac_df = df_raw_shop.copy()
    mac_df['REG確率'] = mac_df['REG'] / mac_df['累計ゲーム'].replace(0, np.nan)
    mac_df['valid_play'] = get_valid_play_mask(mac_df['累計ゲーム'], mac_df['差枚'])
    
    shop_avg_g = mac_df['累計ゲーム'].mean() if not mac_df.empty else 4000
    
    mac_df['設定5近似度'] = mac_df.apply(lambda row: get_setting_score_from_row(row, shop_avg_g=shop_avg_g), axis=1)
    mac_df['REG確率_val'] = np.where(mac_df['累計ゲーム'] > 0, mac_df['REG'] / mac_df['累計ゲーム'], 0)
    
    mac_df['高設定挙動'] = (
        (mac_df['累計ゲーム'] >= 3000) & 
        calculate_high_setting_mask(mac_df, specs)
    ).astype(int)
    mac_df['高設定率'] = np.where(mac_df['valid_play'], mac_df['高設定挙動'], np.nan) * 100
    
    mac_df['valid_差枚'] = np.where(mac_df['valid_play'], mac_df['差枚'], np.nan)
    mac_df['valid_設定5近似度'] = np.where(mac_df['valid_play'], mac_df['設定5近似度'], np.nan)
    mac_df['valid_REG確率'] = np.where(mac_df['valid_play'], mac_df['REG確率_val'], np.nan)
    mac_df['valid_累計ゲーム'] = np.where(mac_df['valid_play'], mac_df['累計ゲーム'], np.nan)

    mac_stats = mac_df.groupby('機種名').agg(
        平均差枚=('valid_差枚', 'mean'),
        設定5近似度=('valid_設定5近似度', 'mean'),
        高設定率=('高設定率', 'mean'),
        平均REG確率=('valid_REG確率', 'mean'),
        平均回転数=('valid_累計ゲーム', 'mean'),
        サンプル数=('台番号', 'count')
    ).reset_index().sort_values('設定5近似度', ascending=False)
    
    mac_stats['信頼度'] = mac_stats['サンプル数'].apply(get_confidence_indicator)
    mac_stats['REG確率'] = mac_stats['平均REG確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
    
    return mac_stats
