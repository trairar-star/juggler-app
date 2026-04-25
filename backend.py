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
from utils import get_valid_play_mask, get_confidence_indicator, get_matched_spec_key, classify_shop_eval
from config import BASE_FEATURES, FEATURE_NAME_MAP, MACHINE_SPECS
from model_trainer import train_models
from shop_trends import calculate_shop_trends, apply_trends_to_row
from postprocessor import postprocess_predictions

# 定数定義
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_FILE = os.path.join(BASE_DIR, 'service_account.json')
SPREADSHEET_KEY = '1ylt9mdIkKKk6YRcZh4O05O7fPF4d2BU6VXzboP_vs5s'
SHEET_NAME = 'juggler_raw'
HISTORY_CACHE_FILE = os.path.join(BASE_DIR, 'history_cache.parquet')

# 🚨【重要】プログラム（計算式や特徴量など）を変更した際は、必ずここのバージョン番号をカウントアップしてください！
# （「予測の実績検証」ページで、新旧ロジックの成績比較ができるようになります）
APP_VERSION = "v4.43.0" 

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
        
        # 古いカラム名「予想設定5以上確率」との互換性維持
        if '予想設定5以上確率' in df.columns and 'prediction_score' not in df.columns:
            df['prediction_score'] = pd.to_numeric(df['予想設定5以上確率'], errors='coerce')
            if df['prediction_score'].max() > 1.0:
                df['prediction_score'] = df['prediction_score'] / 100.0
                
        # 空文字などが混入して画面側で計算エラーになるのを防ぐため、確実に数値型に変換
        if 'prediction_score' in df.columns:
            df['prediction_score'] = pd.to_numeric(df['prediction_score'], errors='coerce')
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

                SCORE_HEADER = ['実行日時', '予測対象日', '店名', '店舗平均期待度', '予測平均差枚', '店舗台数']
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
                    
                df_score_new = temp_df.groupby([shop_col_for_score, '予測対象日']).agg(
                    店舗平均期待度=('prediction_score', 'mean'),
                    予測平均差枚=('予測差枚数', 'mean'),
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
        if shop_col:
            df_list = []
            for shop_name, group in save_df_initial.groupby(shop_col):
                df_list.append(group.sort_values('prediction_score', ascending=False).head(max(3, int(len(group) * 0.10))))
            if df_list:
                save_df_initial = pd.concat(df_list, ignore_index=True)
            else:
                save_df_initial = pd.DataFrame(columns=save_df_initial.columns)
        else:
            save_df_initial = save_df_initial.sort_values('prediction_score', ascending=False).head(max(3, int(len(save_df_initial) * 0.10)))
            
        if save_df_initial.empty:
            st.warning("保存する推奨台がありません。")
            return False

    try:
        gc = _get_gspread_client()
        sh = gc.open_by_key(SPREADSHEET_KEY)
        log_sheet_name = 'prediction_log'
        
        # スプレッドシートのヘッダーが手動操作で壊れた場合（重複など）にも耐えられるように、保存する列を厳格に固定する
        STANDARD_HEADER = ['実行日時', '予測対象日', '対象日付', '店名', '台番号', '機種名', 'prediction_score', '予測信頼度', 'おすすめ度', '予測差枚数', '根拠', 'ai_version', 'app_version']
        
        try: 
            worksheet = sh.worksheet(log_sheet_name)
            existing_data = worksheet.get_all_values()
        except gspread.exceptions.WorksheetNotFound: 
            worksheet = sh.add_worksheet(title=log_sheet_name, rows="1000", cols="15")
            existing_data = []

        # 既存データのパース (ユーザー操作による重複列や不要な列の混入を綺麗に掃除する)
        if existing_data and len(existing_data) > 1:
            raw_header = existing_data[0]
            raw_header = ['prediction_score' if c == '予想設定5以上確率' else c for c in raw_header]
            
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
            
            if 'prediction_score' in df_existing.columns:
                df_existing['prediction_score'] = pd.to_numeric(df_existing['prediction_score'], errors='coerce')
        else:
            df_existing = pd.DataFrame(columns=STANDARD_HEADER)
            
        save_df = save_df_initial.copy()
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
        "デフォルト": {'train_months': 3, 'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50, 'reg_alpha': 0.0, 'reg_lambda': 0.0}
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
        header = ['店名', 'train_months', 'n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'min_child_samples', 'reg_alpha', 'reg_lambda']
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
    
    df['is_neighbor_high_reg'] = ((df['neighbor_only_reg_prob'] >= 1.0/260.0) & (neighbor_g_sum >= 4000)).astype(int)
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

def _generate_rolling_features(df, shop_col, group_keys, df_daily_scores):
    """各種移動平均（ローリング）や集計系特徴量の生成"""
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

    sort_keys = [shop_col, '台番号', '対象日付'] if shop_col else ['台番号', '対象日付']
    df = df.sort_values(sort_keys).reset_index(drop=True)
    df['shifted_diff'] = df.groupby(group_keys)['差枚'].shift(1)
    df['shifted_g'] = df.groupby(group_keys)['累計ゲーム'].shift(1)
    df['shifted_reg'] = df.groupby(group_keys)['REG'].shift(1)
    group_levels = list(range(len(group_keys))) # インデックス操作用
    
    df['mean_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['median_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).median().reset_index(level=group_levels, drop=True).fillna(0)
    df['std_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).std().reset_index(level=group_levels, drop=True).fillna(0)
    df['min_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).min().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_14days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=14, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_30days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=30, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)

    df['mean_7days_games'] = df.groupby(group_keys)['shifted_g'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['is_prev_no_play'] = (df['shifted_g'] == 0).astype(int)

    # --- 新規: 週間REG確率 ---
    rolling_7d_reg_g = df.groupby(group_keys)[['shifted_reg', 'shifted_g']].rolling(window=7, min_periods=1)
    sum_7d_reg = rolling_7d_reg_g['shifted_reg'].sum().reset_index(level=group_levels, drop=True).fillna(0)
    sum_7d_g = rolling_7d_reg_g['shifted_g'].sum().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_7days_reg_prob'] = np.where(sum_7d_g > 0, sum_7d_reg / sum_7d_g, 0)

    # --- 勝率安定度（一撃ノイズ排除用） ---
    if 'is_win' in df.columns:
        df['shifted_is_win'] = df.groupby(group_keys)['is_win'].shift(1)
        df['win_rate_7days'] = df.groupby(group_keys)['shifted_is_win'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    else:
        df['win_rate_7days'] = 0

    # --- 単純な差枚プラス勝率 ---
    df['is_plus_diff'] = (df['差枚'] > 0).astype(int)
    df['shifted_is_plus_diff'] = df.groupby(group_keys)['is_plus_diff'].shift(1)
    df['plus_rate_7days'] = df.groupby(group_keys)['shifted_is_plus_diff'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)

    # --- 大負け(-1000枚以下)フラグ ---
    df['is_heavy_lose'] = (df['差枚'] <= -1000).astype(int)
    # --- 遊べる台(±500枚以内)フラグ ---
    df['is_play_machine'] = ((df['差枚'] >= -500) & (df['差枚'] <= 500)).astype(int)

    # 一時的に作成した不要な列を削除
    df = df.drop(columns=['shifted_diff_wd', 'shifted_diff_ev', 'shifted_diff', 'shifted_is_win', 'shifted_is_plus_diff', 'shifted_diff_ev_mac', 'shifted_diff_ev_end', 'shifted_diff_pure_tail', 'is_plus_diff'], errors='ignore')

    if shop_col:
        df['shop_avg_diff'] = df.groupby([shop_col, '対象日付'])['差枚'].transform('mean').fillna(0)
        df['shop_median_diff'] = df.groupby([shop_col, '対象日付'])['差枚'].transform('median').fillna(0)
        
        if 'is_win' in df.columns:
            df['shop_high_rate'] = df.groupby([shop_col, '対象日付'])['is_win'].transform('mean').fillna(0)
        else:
            df['shop_high_rate'] = 0
            
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
        else:
            df['shop_pred_diff_7d_avg'] = 0

    if shop_col and '機種名' in df.columns:
        # --- ②機種ごとの扱い指標 (過去30日間のその機種の平均差枚・高設定率) ---
        if 'is_win' in df.columns:
            machine_daily_avg = df.groupby([shop_col, '機種名', '対象日付']).agg(
                machine_daily_avg_diff=('差枚', 'mean'),
                machine_daily_high_rate=('is_win', 'mean')
            ).reset_index()
        else:
            machine_daily_avg = df.groupby([shop_col, '機種名', '対象日付']).agg(
                machine_daily_avg_diff=('差枚', 'mean')
            ).reset_index()
            machine_daily_avg['machine_daily_high_rate'] = 0
            
        machine_daily_avg = machine_daily_avg.sort_values([shop_col, '機種名', '対象日付'])
        machine_daily_avg['machine_30days_avg_diff'] = machine_daily_avg.groupby([shop_col, '機種名'])['machine_daily_avg_diff'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
        machine_daily_avg['machine_30days_high_rate'] = machine_daily_avg.groupby([shop_col, '機種名'])['machine_daily_high_rate'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
        df = pd.merge(df, machine_daily_avg[[shop_col, '機種名', '対象日付', 'machine_30days_avg_diff', 'machine_30days_high_rate']], on=[shop_col, '機種名', '対象日付'], how='left')
        
        df['machine_avg_diff'] = df.groupby([shop_col, '機種名', '対象日付'])['差枚'].transform('mean').fillna(0)
        df['machine_median_diff'] = df.groupby([shop_col, '機種名', '対象日付'])['差枚'].transform('median').fillna(0)
        if 'is_win' in df.columns:
            df['machine_high_rate'] = df.groupby([shop_col, '機種名', '対象日付'])['is_win'].transform('mean').fillna(0)
        else:
            df['machine_high_rate'] = 0
            
        df['machine_heavy_lose_rate'] = df.groupby([shop_col, '機種名', '対象日付'])['is_heavy_lose'].transform('mean').fillna(0)
        df['machine_play_rate'] = df.groupby([shop_col, '機種名', '対象日付'])['is_play_machine'].transform('mean').fillna(0)

    if shop_col:
        # --- ③台番号（場所）ごとの扱い指標 (過去30日間のその台番号の平均差枚・高設定率) ---
        if 'is_win' in df.columns:
            machine_no_daily_avg = df.groupby([shop_col, '台番号', '対象日付']).agg(
                machine_no_daily_avg_diff=('差枚', 'mean'),
                machine_no_daily_high_rate=('is_win', 'mean')
            ).reset_index()
        else:
            machine_no_daily_avg = df.groupby([shop_col, '台番号', '対象日付']).agg(
                machine_no_daily_avg_diff=('差枚', 'mean')
            ).reset_index()
            machine_no_daily_avg['machine_no_daily_high_rate'] = 0
            
        machine_no_daily_avg = machine_no_daily_avg.sort_values([shop_col, '台番号', '対象日付'])
        machine_no_daily_avg['machine_no_30days_avg_diff'] = machine_no_daily_avg.groupby([shop_col, '台番号'])['machine_no_daily_avg_diff'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
        machine_no_daily_avg['machine_no_30days_high_rate'] = machine_no_daily_avg.groupby([shop_col, '台番号'])['machine_no_daily_high_rate'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
        df = pd.merge(df, machine_no_daily_avg[[shop_col, '台番号', '対象日付', 'machine_no_30days_avg_diff', 'machine_no_30days_high_rate']], on=[shop_col, '台番号', '対象日付'], how='left')

    if shop_col and '機種名' in df.columns and 'prev_累計ゲーム' in df.columns:
        df['machine_play_volume_rank'] = df.groupby([shop_col, '対象日付', '機種名'])['prev_累計ゲーム'].rank(pct=True).fillna(0.5)

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

    # 先に当日の高設定挙動フラグ(is_win)を計算
    df['total_prob'] = (df['BIG'].fillna(0) + df['REG'].fillna(0)) / df['累計ゲーム'].replace(0, np.nan)
    df['BIG分母'] = np.where(df['BIG'].fillna(0) > 0, df['累計ゲーム'] / df['BIG'], 9999)
    
    # Zスコア（設定1の否定度）の計算
    exp_reg1 = df['累計ゲーム'] * df['spec_reg1']
    std_reg1 = np.sqrt(df['累計ゲーム'] * df['spec_reg1'] * (1.0 - df['spec_reg1']))
    df['z_score_reg'] = np.where(std_reg1 > 0, (df['REG'].fillna(0) - exp_reg1) / std_reg1, 0)
    
    df['is_win'] = (
        (df['累計ゲーム'] >= 3000) & 
        (
            (df['REG確率'] >= df['spec_reg']) | 
            ((df['total_prob'] >= df['spec_tot']) & (df['REG確率'] >= df['spec_reg3'])) |
            (df['z_score_reg'] >= 1.64)
        ) &
        (df['BIG分母'] <= df['spec_b6_den'] + 100)
    ).astype(int)

    # --- 新規追加: REG確率評価の厳格化と特徴量エンジニアリング ---
    df['is_prev_high_reg'] = ((df['累計ゲーム'] >= 3000) & (df['REG確率'] >= df['spec_reg'])).astype(int)
    df['is_high_reg_plus_diff'] = ((df['is_prev_high_reg'] == 1) & (df['差枚'] > 0)).astype(int)
    df['is_low_reg_plus_diff'] = ((df['is_prev_high_reg'] == 0) & (df['差枚'] > 0)).astype(int)

    # --- 新規追加: 時間軸と複合条件に基づく特徴量エンジニアリング ---
    if 'prev2_累計ゲーム' in df.columns and 'prev3_累計ゲーム' in df.columns:
        # X日前REG高確率フラグ
        df['reg_rate_2_days_ago_high'] = ((df['prev2_累計ゲーム'].fillna(0) >= 3000) & (df['prev2_REG確率'].fillna(0) >= df['spec_reg'])).astype(int)
        df['reg_rate_3_days_ago_high'] = ((df['prev3_累計ゲーム'].fillna(0) >= 3000) & (df['prev3_REG確率'].fillna(0) >= df['spec_reg'])).astype(int)
        
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
    
    # --- ターゲットを「翌日の高設定挙動(機種別の設定5基準：REGまたは合算)」に変更 ---
    df['next_reg_prob'] = df['next_REG'] / df['next_累計ゲーム'].replace(0, np.nan)
    df['next_total_prob'] = (df['next_BIG'].fillna(0) + df['next_REG'].fillna(0)) / df['next_累計ゲーム'].replace(0, np.nan)
    df['next_big_den'] = np.where(df['next_BIG'].fillna(0) > 0, df['next_累計ゲーム'] / df['next_BIG'], 9999)
    
    next_exp_reg1 = df['next_累計ゲーム'].fillna(0) * df['spec_reg1']
    next_std_reg1 = np.sqrt(df['next_累計ゲーム'].fillna(0) * df['spec_reg1'] * (1.0 - df['spec_reg1']))
    df['next_z_score_reg'] = np.where(next_std_reg1 > 0, (df['next_REG'].fillna(0) - next_exp_reg1) / next_std_reg1, 0)
    
    # --- ターゲットを「翌日の高設定挙動(機種別の設定5基準：REGまたは合算)」に設定 ---
    df['target'] = (
        (df['next_累計ゲーム'] >= 3000) & 
        (
            (df['next_reg_prob'] >= df['spec_reg']) | 
            ((df['next_total_prob'] >= df['spec_tot']) & (df['next_reg_prob'] >= df['spec_reg3'])) |
            (df['next_z_score_reg'] >= 1.64)
        ) &
        (df['next_big_den'] <= df['spec_b6_den'] + 100)
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

    if shop_col:
        shop_daily_avg2 = df.groupby([shop_col, '対象日付'])['差枚'].mean().reset_index(name='shop_daily_avg_diff')
        shop_daily_avg2['wd'] = shop_daily_avg2['対象日付'].dt.dayofweek
        shop_daily_avg2['digit'] = shop_daily_avg2['対象日付'].dt.day % 10
        wd_stats = shop_daily_avg2.groupby([shop_col, 'wd'])['shop_daily_avg_diff'].mean().reset_index(name='past_wd_avg')
        digit_stats = shop_daily_avg2.groupby([shop_col, 'digit'])['shop_daily_avg_diff'].mean().reset_index(name='past_digit_avg')
        df = pd.merge(df, wd_stats, left_on=[shop_col, 'target_weekday'], right_on=[shop_col, 'wd'], how='left').drop(columns=['wd'])
        df = pd.merge(df, digit_stats, left_on=[shop_col, 'target_date_end_digit'], right_on=[shop_col, 'digit'], how='left').drop(columns=['digit'])
        df['target_date_type_avg_diff'] = df[['past_wd_avg', 'past_digit_avg']].mean(axis=1).fillna(0)
        df['past_wd_avg'] = df['past_wd_avg'].fillna(0)
        df['past_digit_avg'] = df['past_digit_avg'].fillna(0)

    df = _apply_event_features(df, df_events, shop_col)

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
    
    df['mean_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['median_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).median().reset_index(level=group_levels, drop=True).fillna(0)
    df['std_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).std().reset_index(level=group_levels, drop=True).fillna(0)
    df['min_7days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=7, min_periods=1).min().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_14days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=14, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_30days_diff'] = df.groupby(group_keys)['shifted_diff'].rolling(window=30, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)

    df['mean_7days_games'] = df.groupby(group_keys)['shifted_g'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    df['is_prev_no_play'] = (df['shifted_g'] == 0).astype(int)

    # --- 新規: 週間REG確率 ---
    # 7日間の合計REG回数と合計ゲーム数から計算し、ノイズを低減
    rolling_7d_reg_g = df.groupby(group_keys)[['shifted_reg', 'shifted_g']].rolling(window=7, min_periods=1)
    sum_7d_reg = rolling_7d_reg_g['shifted_reg'].sum().reset_index(level=group_levels, drop=True).fillna(0)
    sum_7d_g = rolling_7d_reg_g['shifted_g'].sum().reset_index(level=group_levels, drop=True).fillna(0)
    df['mean_7days_reg_prob'] = np.where(sum_7d_g > 0, sum_7d_reg / sum_7d_g, 0)

    # --- 勝率安定度（一撃ノイズ排除用） ---
    df['shifted_is_win'] = df.groupby(group_keys)['is_win'].shift(1)
    df['win_rate_7days'] = df.groupby(group_keys)['shifted_is_win'].rolling(window=7, min_periods=1).mean().reset_index(level=group_levels, drop=True).fillna(0)
    
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
        df['shop_high_rate'] = df.groupby([shop_col, '対象日付'])['is_win'].transform('mean').fillna(0)
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


    if shop_col and '機種名' in df.columns:
        # --- ②機種ごとの扱い指標 (過去30日間のその機種の平均差枚・高設定率) ---
        machine_daily_avg = df.groupby([shop_col, '機種名', '対象日付']).agg(
            machine_daily_avg_diff=('差枚', 'mean'),
            machine_daily_high_rate=('is_win', 'mean')
        ).reset_index()
        machine_daily_avg = machine_daily_avg.sort_values([shop_col, '機種名', '対象日付'])
        machine_daily_avg['machine_30days_avg_diff'] = machine_daily_avg.groupby([shop_col, '機種名'])['machine_daily_avg_diff'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
        machine_daily_avg['machine_30days_high_rate'] = machine_daily_avg.groupby([shop_col, '機種名'])['machine_daily_high_rate'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()).fillna(0)
        df = pd.merge(df, machine_daily_avg[[shop_col, '機種名', '対象日付', 'machine_30days_avg_diff', 'machine_30days_high_rate']], on=[shop_col, '機種名', '対象日付'], how='left')
        
        df['machine_avg_diff'] = df.groupby([shop_col, '機種名', '対象日付'])['差枚'].transform('mean').fillna(0)
        df['machine_median_diff'] = df.groupby([shop_col, '機種名', '対象日付'])['差枚'].transform('median').fillna(0)
        df['machine_high_rate'] = df.groupby([shop_col, '機種名', '対象日付'])['is_win'].transform('mean').fillna(0)
        df['machine_heavy_lose_rate'] = df.groupby([shop_col, '機種名', '対象日付'])['is_heavy_lose'].transform('mean').fillna(0)
        df['machine_play_rate'] = df.groupby([shop_col, '機種名', '対象日付'])['is_play_machine'].transform('mean').fillna(0)

    if shop_col:
        # --- ③台番号（場所）ごとの扱い指標 (過去30日間のその台番号の平均差枚・高設定率) ---
        machine_no_daily_avg = df.groupby([shop_col, '台番号', '対象日付']).agg(
            machine_no_daily_avg_diff=('差枚', 'mean'),
            machine_no_daily_high_rate=('is_win', 'mean')
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
        df['is_low_play_high_reg'] = ((df['prev_累計ゲーム'] >= 1000) & (df['prev_累計ゲーム'] < 3000) & (df['prev_REG確率'] >= 1.0/260.0)).astype(int)
        
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
        df['island_high_rate'] = df.groupby(['island_id', '対象日付'])['is_win'].transform('mean').fillna(0)
        df['island_total_g'] = df.groupby(['island_id', '対象日付'])['累計ゲーム'].transform('sum')
        df['island_total_reg'] = df.groupby(['island_id', '対象日付'])['REG'].transform('sum')
        df['island_reg_prob'] = np.where(df['island_total_g'] > 0, df['island_total_reg'] / df['island_total_g'], 0)
        df['prev_island_reg_prob'] = df.groupby(group_keys)['island_reg_prob'].shift(1).fillna(0)

    # --- 上げリセット（設定変更）検知用の特徴量 ---
    # 差枚が+500枚以上になったら「設定が入り放出された」とみなしリセット
    df['is_released'] = (df['差枚'] >= 500).astype(int)
    df['temp_reset_group'] = df.groupby(group_keys)['is_released'].cumsum()
    
    # 差枚が+500枚未満の日はすべて「実質マイナス（回収・不発）」としてカウントを継続
    df['is_not_released'] = (df['差枚'] < 500).astype(int)
    df['連続マイナス日数'] = df.groupby(group_keys + ['temp_reset_group'])['is_not_released'].cumsum()
    df['cons_minus_total_diff'] = df.groupby(group_keys + ['temp_reset_group'])['差枚'].cumsum()
    df = df.drop(columns=['temp_reset_group', 'is_released', 'is_not_released'])

    # --- 据え置き（高設定継続）検知用の特徴量 ---
    # 差枚が0以下の日は「実質マイナス・通常」としてカウントをリセット
    df['is_minus_or_zero'] = (df['差枚'] <= 0).astype(int)
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
    df['is_high_reg_day'] = ((df['累計ゲーム'] >= 3000) & (df['REG確率'] >= df['spec_reg'])).astype(int)
    df['is_not_high_reg_day'] = (df['is_high_reg_day'] == 0).astype(int)
    df['high_reg_reset_group'] = df.groupby(group_keys)['is_not_high_reg_day'].cumsum()
    df['cons_high_reg_days'] = df.groupby(group_keys + ['high_reg_reset_group'])['is_high_reg_day'].cumsum()
    df = df.drop(columns=['is_high_reg_day', 'is_not_high_reg_day', 'high_reg_reset_group'])

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

    features = [f for f in BASE_FEATURES if f in df.columns]

    return df, features
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
                score = score + (1.0 - score) * 0.15 # 設定5以上なら残りの伸びしろの15%を加算
        return score

    if not predict_df.empty: predict_df['prediction_score'] = predict_df.apply(apply_setting5_boost, axis=1)
    if not train_df.empty: train_df['prediction_score'] = train_df.apply(apply_setting5_boost, axis=1)

    def apply_reliability_penalty(row):
        score = row.get('prediction_score', 0)
        hc = row.get('history_count', 1)
        games = row.get('累計ゲーム', 0)
        reg_prob = row.get('REG確率', 0)
        
        # 過去データが少ない場合は予測のブレが大きいためスコアを割り引く
        if hc < 14: score *= 0.8
        elif hc < 30: score *= 0.95
        
        # 前日の稼働が少ない場合の減点（ただし、高設定挙動を示している場合は減点を緩和する）
        is_good_reg = reg_prob >= (1.0 / 280.0) if reg_prob > 0 else False
        
        if games < 1000: score *= (0.85 if is_good_reg else 0.70)
        elif games < 2000: score *= (0.95 if is_good_reg else 0.85)
        elif games < 3000: score *= (1.0 if is_good_reg else 0.95)
        
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
        
    # --- 営業状態（還元/通常/回収）の事前付与と店癖の適用 ---
    shop_col = '店名' if '店名' in train_df.columns else ('店舗名' if '店舗名' in train_df.columns else None)
    if shop_col and not train_df.empty:
        def _add_shop_status(df_target, is_actual=False):
            if df_target.empty: return df_target
            diff_col = 'next_diff' if is_actual and 'next_diff' in df_target.columns else '予測差枚数'
            if diff_col not in df_target.columns:
                df_target['営業状態'] = "⚖️ 通常営業"
                return df_target
                
            shop_daily = df_target.groupby([shop_col, 'next_date']).agg(
                avg_diff=(diff_col, 'mean'),
                count=('台番号', 'nunique')
            ).reset_index()
            
            shop_daily['営業状態'] = shop_daily.apply(lambda r: classify_shop_eval(r['avg_diff'], r['count'], is_prediction=False), axis=1)
            return pd.merge(df_target, shop_daily[[shop_col, 'next_date', '営業状態']], on=[shop_col, 'next_date'], how='left')

        train_df = _add_shop_status(train_df, is_actual=True)
        if not predict_df.empty:
            predict_df = _add_shop_status(predict_df, is_actual=False)

        all_trends_dict = calculate_shop_trends(train_df, shop_col, specs)
        if not predict_df.empty:
            predict_df = predict_df.apply(lambda row: apply_trends_to_row(row, all_trends_dict, shop_col, specs), axis=1)
        if not train_df.empty:
            train_df = train_df.apply(lambda row: apply_trends_to_row(row, all_trends_dict, shop_col, specs), axis=1)

    # --- ダメ台ペナルティとメリハリ補正の適用 ---
    def apply_hopeless_penalty(row):
        score = row.get('prediction_score', 0)
        games = row.get('累計ゲーム', 0)
        reg_prob = row.get('REG確率', 0)
        diff = row.get('差枚', 0)
        win_rate_7d = row.get('win_rate_7days', 0)
        
        penalty_factor = 1.0
        reasons = []

        if games >= 3000 and reg_prob > 0 and (1.0 / reg_prob) >= 400 and diff < 0:
            penalty_factor *= 0.60
            reasons.append("【🔻大幅減点】前日しっかり回された上でREGが絶望的(1/400以下)かつマイナスです。低設定の放置台の可能性が高く、危険です。")
            
        if diff >= 2000 and reg_prob > 0 and (1.0 / reg_prob) >= 350:
            penalty_factor *= 0.60
            reasons.append("【🔻大幅減点】前日大勝していますがREG確率が悪く、低設定のまぐれ吹きの可能性が高いです。本日の反動(回収)に警戒してください。")
            
        if win_rate_7d == 0 and diff < 0 and games >= 1000:
            penalty_factor *= 0.70
            reasons.append("【🔻減点】過去1週間で高設定挙動がなく、店側が全く設定を入れていない(見捨てられている)可能性が高いです。")
            
        if games < 1000:
            is_good_reg = reg_prob >= (1.0 / 280.0) if reg_prob > 0 else False
            if is_good_reg:
                reasons.append("【💎期待】前日の回転数は少ないですが、高設定挙動を示しているためポテンシャルを評価しています。")
            else:
                reasons.append("【🔻減点】前日の総回転数が極端に少なく、データ不足のため期待度を少し割り引いています。")

        score *= penalty_factor
        
        if reasons:
            existing_reason = str(row.get('根拠', ''))
            if existing_reason == 'nan': existing_reason = ''
            new_reason = " ".join(reasons)
            if existing_reason and existing_reason != '-':
                row['根拠'] = (existing_reason + " " + new_reason).strip()
            else:
                row['根拠'] = new_reason
                
        row['prediction_score'] = score
        return row

    if not predict_df.empty:
        predict_df = predict_df.apply(apply_hopeless_penalty, axis=1)
    if not train_df.empty:
        train_df = train_df.apply(apply_hopeless_penalty, axis=1)

    # --- 回収日の「完全ベタピン店」判定の事前計算 ---
    shop_col_for_betapin = '店名' if '店名' in train_df.columns else ('店舗名' if '店舗名' in train_df.columns else None)
    betapin_shops = set()
    if shop_col_for_betapin and not train_df.empty:
        if '予測差枚数' in train_df.columns:
            shop_daily = train_df.groupby([shop_col_for_betapin, 'next_date']).agg(予測差枚数=('予測差枚数', 'mean'), 台数=('台番号', 'nunique')).reset_index()
            shop_daily['営業予測'] = shop_daily.apply(lambda r: classify_shop_eval(r['予測差枚数'], r['台数'], is_prediction=True), axis=1)
            cold_days_dates = shop_daily[shop_daily['営業予測'] == "🥶 回収日予測"]
            cold_days = pd.merge(train_df, cold_days_dates[[shop_col_for_betapin, 'next_date']], on=[shop_col_for_betapin, 'next_date'])
            if not cold_days.empty:
                cold_stats = cold_days.groupby(shop_col_for_betapin).agg(高設定率=('target', 'mean'), サンプル数=('target', 'count')).reset_index()
                betapin_shops = set(cold_stats[(cold_stats['サンプル数'] >= 100) & (cold_stats['高設定率'] < 0.015)][shop_col_for_betapin])

    # --- 店舗全体の空気感による最終補正とメッセージ付与 ---
    def apply_shop_mood_correction(df_target):
        shop_col = '店名' if '店名' in df_target.columns else ('店舗名' if '店舗名' in df_target.columns else None)
        if df_target.empty or not shop_col: return df_target
            
        if '予測差枚数' in df_target.columns:
            df_target['temp_shop_diff'] = df_target.groupby([shop_col, 'next_date'])['予測差枚数'].transform('mean')
        else:
            df_target['temp_shop_diff'] = np.nan
        
        def _correct(row):
            score = row.get('prediction_score', 0)
            s_raw_avg = row.get('temp_shop_avg', 0.15)
            s_diff = row.get('temp_shop_diff', 0)
            s_name = row.get(shop_col)
            reasons = []
            
            if s_raw_avg < 0.10 and (pd.isna(s_diff) or s_diff < 0):
                is_betapin = s_name in betapin_shops
                if is_betapin:
                    reasons.append(f"【⚠️絶対回収】過去のデータから、この店舗は回収日に高設定をほぼ100%使わない「完全ベタピン」の傾向が確認されています。")
                else:
                    if score >= 0.30:
                        reasons.append(f"【💎一点突破】店舗全体は回収傾向ですが、AIはこの台に確かな高設定の根拠(見せ台)があると確信して強く推奨しています。")
                    else:
                        reasons.append("【🚨フェイク警戒】店舗全体が回収傾向です。普段なら強い根拠がある台ですが、今日はフェイク（罠）として使われるリスクが高いため慎重に判断してください。")

            if reasons:
                existing_reason = str(row.get('根拠', ''))
                if existing_reason == 'nan': existing_reason = ''
                new_reason = " ".join(reasons)
                row['根拠'] = (existing_reason + " " + new_reason).strip() if existing_reason and existing_reason != '-' else new_reason
                
            return row
            
        res_df = df_target.apply(_correct, axis=1)
        res_df = res_df.drop(columns=['temp_shop_avg', 'temp_shop_diff'], errors='ignore')
        return res_df

    predict_df = apply_shop_mood_correction(predict_df)
    train_df = apply_shop_mood_correction(train_df)

    # --- 予測スコアから「前日の自己評価スコア(past_prediction_score)」を作成 ---
    # 最終的な予測スコアを使って「前日も推奨していたか」を判定する
    all_df = pd.concat([train_df, predict_df], ignore_index=True)
    if shop_col:
        all_df = all_df.sort_values([shop_col, '台番号', '対象日付']).reset_index(drop=True)
        all_df['past_prediction_score'] = all_df.groupby([shop_col, '台番号'])['prediction_score'].shift(1).fillna(0.0)
    else:
        all_df = all_df.sort_values(['台番号', '対象日付']).reset_index(drop=True)
        all_df['past_prediction_score'] = all_df.groupby('台番号')['prediction_score'].shift(1).fillna(0.0)
        
    train_df = all_df[all_df['next_diff'].notna()].copy()
    predict_df = all_df[all_df['next_diff'].isna()].copy()

    def get_rating(score):
        if score >= 0.70: return 'A'
        elif score >= 0.55: return 'B'
        elif score >= 0.40: return 'C'
        elif score >= 0.25: return 'D'
        else: return 'E'

    def get_reason(row):
        comments, reasons = [], []
        score = row.get('prediction_score', 0)
        if score > 0.50: comments.append("【激アツ】AIの自信度が非常に高い(生確率で50%超え)です。")
        
        past_score = row.get('past_prediction_score', 0)
        diff = row.get('差枚', 0)
        
        if past_score >= 0.30:
            if diff <= -500:
                reasons.append(f"【AIリベンジ狙い】前日もAIが推奨(期待度{past_score*100:.0f}%)していましたが不発でした。高設定据え置きのリベンジが期待できます。")
            elif diff > 0:
                reasons.append(f"【AI推奨継続】前日もAIが推奨(期待度{past_score*100:.0f}%)しており、好調のまま今日も強い根拠を維持しています。")
        
        mean_7d = row.get('mean_7days_diff', 0)
        win_rate_7d = row.get('win_rate_7days', 0)
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

        if row.get('is_prev_no_play', 0) == 1:
            reasons.append("【特殊】前日は全く稼働しておらず(0G)、店側のアピール(設定変更・テコ入れ)のターゲットになる可能性があります。")

        shop_7d = row.get('shop_7days_avg_diff', 0)
        if shop_7d < -150:
            reasons.append(f"【店舗状況】店舗全体が直近1週間回収モード(平均{int(shop_7d)}枚)ですが、あえてこの台を推奨しています。")
        elif shop_7d > 150:
            reasons.append(f"【店舗状況】店舗全体が直近1週間還元モード(平均+{int(shop_7d)}枚)で、全体のベースアップに期待できます。")
            
        # 機種情報の取得（ぶどう確率の判定などより前に実行）
        machine_name = row.get('機種名', '')
        matched_spec_key = get_matched_spec_key(machine_name, specs)

        # ぶどう確率の根拠追加 (生の推測値を使用)
        prev_grape_raw = row.get('prev_推定ぶどう確率_raw', row.get('prev_推定ぶどう確率'))
        prev_games = row.get('prev_累計ゲーム', 0)
        if pd.notna(prev_grape_raw) and prev_grape_raw > 0 and prev_games >= 4000:
            spec_grape_5 = specs[matched_spec_key].get('設定5', {}).get('ぶどう', 5.9)
            # 店補正済みの値が裏付けフィルターを通過している（NaNでない）場合のみ根拠として採用
            if pd.notna(row.get('prev_推定ぶどう確率')) and prev_grape_raw <= spec_grape_5:
                reasons.append(f"【🍇小役優秀】前日は{int(prev_games)}G稼働で推定ぶどう確率が1/{prev_grape_raw:.2f}と優秀です。REG確率・差枚も伴っており、高設定の強い裏付け（裏取り）となっています。")

        mac_30d = row.get('machine_30days_avg_diff', 0)
        if mac_30d > 150:
            reasons.append(f"【機種優遇】過去30日間、この機種(平均+{int(mac_30d)}枚)は店舗から甘く使われている傾向があります。")
        elif mac_30d < -150:
            reasons.append(f"【機種冷遇】過去30日間、この機種(平均{int(mac_30d)}枚)は冷遇気味ですが、この台単体は評価されています。")

        if row.get('is_new_machine', 0) == 1:
            reasons.append("【新台】新台導入から1週間以内のため、店側のアピール(高設定投入)が期待できます。")
        if row.get('is_moved_machine', 0) == 1:
            reasons.append("【配置変更】配置変更(移動)から1週間以内のため、扱いが変化している可能性があります。")
            
        if row.get('is_low_play_high_reg', 0) == 1:
            reasons.append("【💎お宝台候補】前日はあまり回されていませんが、高設定挙動を示しており、そのまま据え置かれる(隠れ高設定)ポテンシャルがあります。")
        if row.get('is_hot_wd_and_heavy_lose', 0) == 1:
            reasons.append("【📈還元曜日×凹み反発】この店舗の強い曜日(還元日)と、直近1週間大きく凹んでいる条件が重なっており、絶好の上げリセット狙い目です。")

        if row.get('is_prev_up_trend_and_high_reg', 0) == 1:
            reasons.append("【📈右肩上がり据え置き】前日は差枚がプラスでREG確率も高設定水準です。優秀台の完全な据え置きが期待できます。")
        if row.get('is_prev_low_reg_and_good_diff', 0) == 1:
            reasons.append("【⚠️低REG・差枚プラス】前日は差枚がプラスですがREG確率が伴っていません。低設定の誤爆による回収リスクと、据え置きの両面をAIが評価しています。")

        big = row.get('BIG', 0)
        reg = row.get('REG', 0)
        
        is_setting5_over = False
        if matched_spec_key and "設定5" in specs[matched_spec_key] and reg_prob > 0:
            set5_reg_prob_threshold = 1.0 / specs[matched_spec_key]["設定5"]["REG"]
            if games >= 5000 and reg_prob >= set5_reg_prob_threshold:
                is_setting5_over = True

        spec_reg_5 = 1.0 / specs[matched_spec_key].get("設定5", {"REG": 260.0})["REG"] if matched_spec_key else 1/260.0
        spec_big_5 = 1.0 / specs[matched_spec_key].get("設定5", {"BIG": 260.0})["BIG"] if matched_spec_key else 1/260.0

        # BIG確率の計算
        big_prob = big / games if games > 0 else 0
        big_denom = 1 / big_prob if big_prob > 0 else 9999

        if is_setting5_over:
            reasons.append(f"【🌟高設定挙動】5000G以上回って前日のREG確率が1/{int(1/reg_prob)}で、機種スペックの**「設定5以上」**の基準を満たしており、強く推奨されます。")
        elif reg > big and reg_prob >= spec_reg_5 and games >= 5000:            
            if big_denom >= 400:
                if diff <= 0:
                    reasons.append(f"【超不発】5000G以上回ってBIG確率が1/{int(big_denom)}と極端に欠損していますが、REG確率は設定5以上(1/{int(1/reg_prob)})をキープしている超・狙い目台です。")
                else:
                    reasons.append(f"【特殊】5000G以上回ってBIGが極端に引けていませんが(1/{int(big_denom)})、REG確率は設定5以上(1/{int(1/reg_prob)})の高設定挙動です。")
            else:
                if diff <= 0:
                    reasons.append(f"【特殊】5000G以上回ってREG先行(BB欠損)で差枚が沈んでいる、狙い目の「高設定 不発台」です。(REG 1/{int(1/reg_prob)})")
                else:
                    reasons.append(f"【特殊】5000G以上回ってREG先行かつREG確率が設定5以上(1/{int(1/reg_prob)})の「高設定台」です。")
        elif big >= reg and big_prob >= spec_big_5 and games >= 5000:
            reasons.append(f"【特殊】5000G以上回ってBIG先行(1/{int(big_denom)})でBIG確率が設定5以上をキープしています。BIGヒキ強台の据え置き狙いとして期待できます。")
        else:
            if reg_prob > (1/280) and games >= 5000: reasons.append(f"5000G以上回って前日のREG確率が**1/{int(1/reg_prob)}**と高設定水準です。")
            elif reg_prob > (1/350) and games >= 5000: reasons.append(f"5000G以上回ってREG確率(1/{int(1/reg_prob)})が悪くなく、粘る価値があります。")
            elif reg_prob > (1/280) and games >= 3000: reasons.append(f"3000G以上回って前日のREG確率が**1/{int(1/reg_prob)}**と高設定水準です。")
            elif reg_prob > (1/350) and games >= 3000: reasons.append(f"3000G以上回ってREG確率(1/{int(1/reg_prob)})が悪くなく、粘る価値があります。")
        
        e_avg = row.get('event_avg_diff', 0)
        if e_avg > 150: reasons.append(f"今日はイベント特定日(平均+{int(e_avg)}枚)のため期待値が高いです。")

        evt_name = row.get('イベント名', '通常')
        evt_score = row.get('event_rank_score', 0)
        if evt_name != '通常' and pd.notna(evt_name):
            evt_rank = row.get('イベントランク', '')
            rank_str = f"(ランク{evt_rank})" if evt_rank else ""
            if evt_score <= -5:
                reasons.append(f"【🚨極悪回収警戒】本日は複合イベント「{evt_name}」ですが、他機種やパチンコへの還元によるシワ寄せで、極めて強い回収(期待度スコア: {evt_score})が予測されます。")
            elif evt_score < 0:
                reasons.append(f"【⚠️回収警戒】本日は複合イベント「{evt_name}」ですが、対象外機種であるため回収に回される危険性が高いです。")
            elif evt_score > 0:
                reasons.append(f"店舗イベント「{evt_name}」{rank_str}対象日です(複合スコア: {evt_score})。")

        prev_evt_score = row.get('prev_event_rank_score', 0)
        if evt_score <= 0 and prev_evt_score > 0 and score >= 0.60:
            reasons.append("【特日翌日】前日は特日でしたが、AIは本日の「据え置き」または「入れ直し」を有力視しています。")

        w_avg = row.get('weekday_avg_diff', 0)
        if w_avg > 150:
            wd_name = ['月', '火', '水', '木', '金', '土', '日'][int(row['target_weekday'])] if 'target_weekday' in row and 0 <= row['target_weekday'] <= 6 else ''
            reasons.append(f"{wd_name}曜日はこの店の得意日(平均+{int(w_avg)}枚)です。")

        # --- クロス分析 (イベント×機種 / 末尾) の根拠追加 ---
        ev_label = f"イベント「{evt_name}」" if evt_name != '通常' and pd.notna(evt_name) else "通常営業日"
        
        evt_mac_avg = row.get('event_x_machine_avg_diff', 0)
        if evt_mac_avg > 200:
            reasons.append(f"【特効機種】過去の{ev_label}において、この機種は非常に甘く使われています(平均+{int(evt_mac_avg)}枚)。")
        elif evt_mac_avg < -300:
            reasons.append(f"【警戒機種】過去の{ev_label}において、この機種は回収傾向(平均{int(evt_mac_avg)}枚)ですが、AIはこの台単体を評価しています。")
            
        evt_end_avg = row.get('event_x_end_digit_avg_diff', 0)
        if evt_end_avg > 200:
            end_digit = int(row.get('末尾番号', -1))
            if end_digit != -1:
                reasons.append(f"【当たり末尾】過去の{ev_label}において、末尾『{end_digit}』は対象になりやすい強い傾向があります(平均+{int(evt_end_avg)}枚)。")

        if row.get('is_corner', 0) == 1: reasons.append("角台（設定優遇枠）のため期待大です。")
        
        neighbor_high_count = row.get('neighbor_high_setting_count', 0)
        if neighbor_high_count > 0:
            reasons.append(f"【🤝並び・塊】隣台のうち{int(neighbor_high_count)}台が高設定挙動を示しており、並びの対象になっている可能性が高いです。")
        elif row.get('is_neighbor_high_reg', 0) == 1:
            reasons.append("【🤝並び・塊】隣台の合算REG確率が高設定水準であり、強い「並び」の根拠となっています。")
            
        i_avg = row.get('island_avg_diff', 0)
        if i_avg > 400: reasons.append(f"所属する島全体が好調(平均+{int(i_avg)}枚)で、塊対象の可能性があります。")
        
        shop_reason = str(row.get('根拠', '')).strip()
        if shop_reason == 'nan': shop_reason = ''
        has_strong_reason = bool(reasons) or (shop_reason and shop_reason != '-')
        
        if reasons: 
            comments.append(" ".join(reasons))
            
        if not has_strong_reason:
            if score >= 0.40: 
                comments.append("目立った特徴（特定の店癖など）には合致していませんが、様々なデータの全体バランスからAIが非常に高く評価しています。")
            elif score >= 0.20:
                comments.append("目立った強い根拠はありませんが、複数の細かな要因（周りの状況や過去のわずかな傾向）の組み合わせにより、消去法的に期待度が底上げされています。")
            else: 
                comments.append("特筆すべき強い根拠はありません。")
            
        base_reason = " ".join(comments)
        if shop_reason and shop_reason != '-':
            return f"{base_reason} {shop_reason}".strip()
        return base_reason

    if not predict_df.empty:
        predict_df['根拠'] = predict_df.apply(get_reason, axis=1)
        predict_df['おすすめ度'] = predict_df['prediction_score'].apply(get_rating)
        if '店名' in predict_df.columns:
            shop_mean = predict_df.groupby('店名')['prediction_score'].transform('mean')
            predict_df['店舗期待度'] = shop_mean.apply(get_rating)
        
    if not train_df.empty:
        train_df['根拠'] = train_df.apply(get_reason, axis=1)
        train_df['おすすめ度'] = train_df['prediction_score'].apply(get_rating)

    def highlight_reasons(text):
        if not isinstance(text, str): return text
        # 1. まず全ての 【...】 を一律で太字にする
        text = re.sub(r'(【[^】]+】)', r'**\1**', text)
        
        # 2. 特に強調したいタグをStreamlitのカラー構文で色付けする
        text = text.replace('**【🎯激熱】**', '**:red[【🎯激熱】]**')
        text = text.replace('**【激アツ】**', '**:red[【激アツ】]**')
        text = text.replace('**【🚨極悪回収警戒】**', '**:blue[【🚨極悪回収警戒】]**')
        text = text.replace('**【⚠️絶対回収】**', '**:blue[【⚠️絶対回収】]**')
        text = text.replace('**【🔻大幅減点】**', '**:blue[【🔻大幅減点】]**')
        text = text.replace('**【💎お宝台候補】**', '**:orange[【💎お宝台候補】]**')
        text = text.replace('**【💎一点突破】**', '**:orange[【💎一点突破】]**')
        text = text.replace('**【🌟高設定挙動】**', '**:orange[【🌟高設定挙動】]**')
        text = text.replace('**【超不発】**', '**:orange[【超不発】]**')
        text = text.replace('**【波・推移】**', '**:green[【波・推移】]**')
        text = text.replace('**【AIリベンジ狙い】**', '**:orange[【AIリベンジ狙い】]**')
        return text

    if not predict_df.empty and '根拠' in predict_df.columns:
        predict_df['根拠'] = predict_df['根拠'].apply(highlight_reasons)
    if not train_df.empty and '根拠' in train_df.columns:
        train_df['根拠'] = train_df['根拠'].apply(highlight_reasons)

    return predict_df, train_df

# ---------------------------------------------------------
# 分析・予測ロジック (メイン関数)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=2, ttl=3600)
def run_analysis(df, _df_events=None, _df_island=None, shop_hyperparams=None, target_date=None):
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
        
    if len(train_df) < 10 or len(predict_df) == 0:
        return predict_df, pd.DataFrame(), pd.DataFrame()

    # 4 & 5. モデル学習と推論 (店舗ごとのパラメータで独立して実行される)
    predict_df, train_df, feature_importances = train_models(train_df, predict_df, features, shop_hyperparams)

    # 6. 後処理 (スコア補正、根拠の自然言語生成)
    predict_df, train_df = postprocess_predictions(predict_df, train_df)

    if 'valid_play_mask' in train_df.columns:
        train_df = train_df.drop(columns=['valid_play_mask'], errors='ignore')
        
    if 'valid_play_mask' in predict_df.columns:
        predict_df = predict_df.drop(columns=['valid_play_mask'], errors='ignore')

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
    isl_stats = island_pred_df.groupby('島名').agg(
        平均期待度=('prediction_score', 'mean'),
        激アツ台数=('prediction_score', lambda x: (x >= 0.30).sum()),
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
    shop_stats = df.groupby(shop_col).agg(
        平均スコア=('prediction_score', 'mean'),
        推奨台数=('prediction_score', lambda x: (x >= 0.30).sum()),
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
            shop_machine_counts = df.groupby(shop_col)['台番号'].nunique().to_dict()
            merged['top_k_threshold'] = merged[shop_col].apply(lambda x: max(3, min(10, int(shop_machine_counts.get(x, 50) * 0.10))))
            
            merged['daily_rank'] = merged.groupby(['予測対象日_merge', shop_col])['prediction_score'].rank(method='first', ascending=False)
            high_expect_df = merged[merged['daily_rank'] <= merged['top_k_threshold']].copy()
            
            if not high_expect_df.empty:
                act_b = pd.to_numeric(high_expect_df['BIG'], errors='coerce').fillna(0)
                act_r = pd.to_numeric(high_expect_df['REG'], errors='coerce').fillna(0)
                act_g = pd.to_numeric(high_expect_df['累計ゲーム'], errors='coerce').fillna(0)
                
                spec_reg_val = high_expect_df['機種名_raw'].apply(lambda x: specs[get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
                spec_tot_val = high_expect_df['機種名_raw'].apply(lambda x: specs[get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
                
                reg_prob_den = np.where(act_r > 0, act_g / act_r, 0)
                tot_prob_den = np.where((act_b + act_r) > 0, act_g / (act_b + act_r), 0)
                
                high_expect_df['valid_high_play'] = (act_g >= 3000)
                high_expect_df['is_high_setting'] = ((((reg_prob_den > 0) & (reg_prob_den <= spec_reg_val)) | ((tot_prob_den > 0) & (tot_prob_den <= spec_tot_val)))).astype(int)
                
                act_diff = pd.to_numeric(high_expect_df['差枚'], errors='coerce').fillna(0)
                high_expect_df['valid_play'] = get_valid_play_mask(act_g, act_diff)
                high_expect_df['valid_win'] = high_expect_df['valid_play'] & (act_diff > 0)
                high_expect_df['valid_high'] = high_expect_df['valid_high_play'] & (high_expect_df['is_high_setting'] == 1)

                acc_stats = high_expect_df.groupby(shop_col).agg(
                    正答数=('valid_high', 'sum'), 高設定有効数=('valid_high_play', 'sum'), 有効稼働数=('valid_play', 'sum'), 勝数=('valid_win', 'sum'), サンプル数=('台番号', 'count')
                ).reset_index()
                acc_stats['勝率'] = np.where(acc_stats['有効稼働数'] > 0, acc_stats['勝数'] / acc_stats['有効稼働数'], 0.0)
                acc_stats['正答率'] = np.where(acc_stats['高設定有効数'] > 0, acc_stats['正答数'] / acc_stats['高設定有効数'], 0.0)
                
                ai_accuracy_map = dict(zip(acc_stats[shop_col], acc_stats['正答率']))
                ai_win_rate_map = dict(zip(acc_stats[shop_col], acc_stats['勝率']))
                for _, r in acc_stats.iterrows():
                    shop = r[shop_col]
                    ai_acc_str_map[shop] = f"{r['正答率']*100:.1f}% ({int(r['正答数'])}/{int(r['高設定有効数'])}台)"
                    ai_win_str_map[shop] = f"{r['勝率']*100:.1f}% ({int(r['勝数'])}/{int(r['有効稼働数'])}台)"
        
    shop_stats['AI正答率_数値'] = shop_stats[shop_col].map(ai_accuracy_map).fillna(0.0) * 100
    shop_stats['AI推奨台勝率_数値'] = shop_stats[shop_col].map(ai_win_rate_map).fillna(0.0) * 100
    shop_stats['AI正答率'] = shop_stats[shop_col].map(ai_acc_str_map).fillna("- (0/0台)")
    shop_stats['AI推奨台勝率'] = shop_stats[shop_col].map(ai_win_str_map).fillna("- (0/0台)")
    
    def calc_sort_score(row):
        score = row['平均スコア']
        sample_count = int(str(row.get('AI正答率', '')).split('/')[1].split('台)')[0]) if '/' in str(row.get('AI正答率', '')) else 0
        if sample_count >= 5:
            if row['AI正答率_数値'] > 0:
                if row['AI正答率_数値'] < 30: score -= 0.15
                elif row['AI正答率_数値'] < 40: score -= 0.05
            if row['AI推奨台勝率_数値'] > 0:
                if row['AI推奨台勝率_数値'] < 30: score -= 0.15
                elif row['AI推奨台勝率_数値'] < 40: score -= 0.05
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
    
    mac_df['合算確率'] = (mac_df['BIG'] + mac_df['REG']) / mac_df['累計ゲーム'].replace(0, np.nan)
    spec_reg = mac_df['機種名'].apply(lambda x: 1.0 / specs[get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
    spec_tot = mac_df['機種名'].apply(lambda x: 1.0 / specs[get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
    spec_reg3 = mac_df['機種名'].apply(lambda x: 1.0 / specs[get_matched_spec_key(x, specs)].get('設定3', {"REG": 300.0})["REG"])
    
    spec_reg1 = mac_df['機種名'].apply(lambda x: 1.0 / specs[get_matched_spec_key(x, specs)].get('設定1', {"REG": 400.0})["REG"])
    exp_reg1 = mac_df['累計ゲーム'] * spec_reg1
    std_reg1 = np.sqrt(mac_df['累計ゲーム'] * spec_reg1 * (1.0 - spec_reg1))
    z_score_reg = np.where(std_reg1 > 0, (mac_df['REG'].fillna(0) - exp_reg1) / std_reg1, 0)
    
    mac_df['高設定挙動'] = (
        (mac_df['累計ゲーム'] >= 3000) & 
        ((mac_df['REG確率'] >= spec_reg) | ((mac_df['合算確率'] >= spec_tot) & (mac_df['REG確率'] >= spec_reg3)) | (z_score_reg >= 1.64))
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
