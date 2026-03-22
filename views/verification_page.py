import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import altair as alt # type: ignore
import math

import backend
from utils import get_confidence_indicator

def render_verification_page(df_pred_log, df_verify, df_predict, df_raw):
    st.header("✅ 精度検証 (各店舗AI設定)")
    st.caption("あなたが保存した過去の「予測結果ログ」と、実際の結果データを照合して、当時のAIの精度を検証します。")

    selected_version = 'すべて'

    # --- 検証・評価設定 ---
    with st.expander("🎯 検証・評価設定", expanded=False):
        st.caption("「設定5近似度」を計算する際の減点ルールを調整できます。よく分からなければ「標準」のままがおすすめです。")
        
        c1, c2, c3 = st.columns(3)
        if c1.button("🍭 甘め", help="BIGのヒキ強も評価する設定"):
            st.session_state.penalty_reg = 10
            st.session_state.penalty_big = 2
            st.session_state.low_g_penalty = 20
            st.session_state.min_g_filter = 0
        if c2.button("⚖️ 標準", help="REGを重視するバランス型（推奨）"):
            st.session_state.penalty_reg = 15
            st.session_state.penalty_big = 5
            st.session_state.low_g_penalty = 30
            st.session_state.min_g_filter = 0
        if c3.button("🌶️ 辛め", help="REG不足に厳しく、低稼働を許さない設定"):
            st.session_state.penalty_reg = 20
            st.session_state.penalty_big = 8
            st.session_state.low_g_penalty = 50
            st.session_state.min_g_filter = 1000
            
        penalty_reg = st.slider("REG 1回不足ごとの減点", 0, 50, st.session_state.get('penalty_reg', 15), 1, key="penalty_reg")
        penalty_big = st.slider("BIG 1回不足ごとの減点", 0, 50, st.session_state.get('penalty_big', 5), 1, key="penalty_big")
        low_g_penalty = st.slider("低稼働(1000G未満)の最大減点率(%)", 0, 100, st.session_state.get('low_g_penalty', 30), 5, key="low_g_penalty")
        min_g_filter = st.slider("検証から除外する最低ゲーム数", 0, 3000, st.session_state.get('min_g_filter', 0), 100, key="min_g_filter", help="客層に合わせて、指定したG数未満しか回されなかった台を採点対象から除外します。※ただし、確率的に高設定が極めて薄い（500Gノーボナ等）と判断できる「明らかな見切り台」はハズレとして減点対象に残ります。")

    if df_pred_log.empty:
        st.warning("保存された予測結果ログがありません。日々の予測結果は、ログイン時に自動で保存されます。")
        return

    # --- 予測ログの実行日時フィルター ---
    if '実行日時' in df_pred_log.columns:
        df_pred_log['実行日時'] = pd.to_datetime(df_pred_log['実行日時'], errors='coerce')
        valid_dates = df_pred_log['実行日時'].dropna()
        if not valid_dates.empty:
            min_log_date = valid_dates.min().date()
            max_log_date = valid_dates.max().date()
            
            with st.expander("⚙️ AI設定・バージョンで絞り込み", expanded=True):
                st.caption("モデルの設定（学習回数や葉の数など）ごとに成績を分けて確認・比較できます。")
                
                if 'ai_version' in df_pred_log.columns:
                    df_pred_log['ai_version'] = df_pred_log['ai_version'].replace('', 'v1.0 (記録なし)').fillna('v1.0 (記録なし)')
                    # 実行日時が一番新しいログのバージョンをデフォルトにする
                    latest_version = df_pred_log.sort_values('実行日時', ascending=False)['ai_version'].iloc[0] if not df_pred_log.empty else 'すべて'
                    versions = ['すべて'] + sorted(list(df_pred_log['ai_version'].astype(str).unique()))
                    default_idx = versions.index(latest_version) if latest_version in versions else 0
                    selected_version = st.selectbox("AIバージョンで絞り込み", versions, index=default_idx)
                    if selected_version != 'すべて':
                        df_pred_log = df_pred_log[df_pred_log['ai_version'] == selected_version]

                date_range = st.date_input(
                    "保存日の範囲",
                    value=(min_log_date, max_log_date),
                    min_value=min_log_date,
                    max_value=max_log_date
                )
                if len(date_range) == 2:
                    start_d, end_d = date_range
                    df_pred_log = df_pred_log[(df_pred_log['実行日時'].dt.date >= start_d) & (df_pred_log['実行日時'].dt.date <= end_d)]
                    
            if df_pred_log.empty:
                st.warning("指定された期間の予測結果ログがありません。")
                return

    if df_verify.empty or 'next_diff' not in df_verify.columns:
        st.warning("分析可能な実データ（結果）がありません。")
        return

    # --- データ準備 ---
    if '予測対象日' in df_pred_log.columns:
        df_pred_log['予測対象日'] = pd.to_datetime(df_pred_log['予測対象日'], errors='coerce')
    df_pred_log['対象日付'] = pd.to_datetime(df_pred_log['対象日付'], errors='coerce')
    
    if '予測対象日' in df_pred_log.columns:
        df_pred_log['予測対象日_merge'] = df_pred_log['予測対象日'].fillna(df_pred_log['対象日付'] + pd.Timedelta(days=1))
    else:
        df_pred_log['予測対象日_merge'] = df_pred_log['対象日付'] + pd.Timedelta(days=1)
        
    # 店舗カラム名の統一
    shop_col = '店名' if '店名' in df_verify.columns else ('店舗名' if '店舗名' in df_verify.columns else '店名')
    shop_col_pred = '店名' if '店名' in df_pred_log.columns else '店舗名'
    if shop_col != shop_col_pred:
        df_pred_log = df_pred_log.rename(columns={shop_col_pred: shop_col})
        
    # 台番号を文字列化して型を統一（結合ミス防止）
    df_pred_log['台番号'] = df_pred_log['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # 重複して保存された予測データへの対策（最新の実行日時のデータを優先して残す）
    if '実行日時' in df_pred_log.columns:
        df_pred_log = df_pred_log.sort_values('実行日時', ascending=False).drop_duplicates(
            subset=['予測対象日_merge', shop_col, '台番号'], keep='first'
        )

    # df_verify と df_predict を結合して全台の特徴量ベースを作る
    full_feature_df = pd.concat([df_verify, df_predict], ignore_index=True)
    if '台番号' in full_feature_df.columns:
        full_feature_df['台番号'] = full_feature_df['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)

    if 'next_date' in full_feature_df.columns:
        full_feature_df['予測対象日_merge'] = pd.to_datetime(full_feature_df['next_date'], errors='coerce')
    else:
        full_feature_df['予測対象日_merge'] = pd.to_datetime(full_feature_df['対象日付'], errors='coerce') + pd.Timedelta(days=1)

    # 保存ログをベースに左外部結合（データ落ち防止）
    base_df = pd.merge(
        df_pred_log,
        full_feature_df,
        on=['予測対象日_merge', shop_col, '台番号'],
        how='left',
        suffixes=('_saved', '_current')
    )

    if base_df.empty:
        st.info("保存された予測に対して、まだ結果データ（翌日の稼働データ）が登録されていないか、一致するデータがありません。")
        return
        
    # 古いログのフォールバック用に元の対象日付を保存
    if '対象日付_saved' in base_df.columns:
        base_df['予測ベース日'] = base_df['対象日付_saved']
    elif '対象日付' in base_df.columns:
        base_df['予測ベース日'] = base_df['対象日付']
        
    # 以降の集計や表示のために対象日付を予測対象日に上書きする
    base_df['対象日付'] = base_df['予測対象日_merge']

    # 保存当時のスコアと予測結果を使用
    if 'prediction_score_saved' in base_df.columns:
        base_df['prediction_score'] = base_df['prediction_score_saved']
    if '予測差枚数_saved' in base_df.columns:
        base_df['予測差枚数'] = base_df['予測差枚数_saved']
    if '機種名_saved' in base_df.columns:
        base_df['機種名'] = base_df['機種名_saved']
    if '予測信頼度_saved' in base_df.columns:
        base_df['予測信頼度'] = base_df['予測信頼度_saved']
    elif '予測信頼度_current' in base_df.columns:
        base_df['予測信頼度'] = base_df['予測信頼度_current']
    else:
        base_df['予測信頼度'] = "-"
        
    # --- 実際の成績を df_raw から直接取得 (未稼働台のデータ落ちを防ぐ) ---
    if '台番号' in df_raw.columns:
        df_raw_temp = df_raw.copy()
        df_raw_temp['台番号'] = df_raw_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_raw_temp['対象日付'] = pd.to_datetime(df_raw_temp['対象日付'], errors='coerce')
    else:
        df_raw_temp = pd.DataFrame()

    shop_operating_dates = {}
    if not df_raw_temp.empty and shop_col in df_raw_temp.columns:
        for shop in df_raw_temp[shop_col].unique():
            shop_dates = sorted(df_raw_temp[df_raw_temp[shop_col] == shop]['対象日付'].dropna().unique())
            shop_operating_dates[shop] = shop_dates

    def get_actual_result(row):
        shop = row.get(shop_col)
        target_date = row.get('対象日付') # 予測対象日に上書き済み
        base_date = row.get('予測ベース日') # 古いログのフォールバック用
        machine_no = row.get('台番号')
        
        actual_date = pd.NaT
        if pd.notna(target_date) and shop in shop_operating_dates:
            for d in shop_operating_dates[shop]:
                if d >= target_date:
                    actual_date = d
                    break
        elif pd.notna(base_date) and shop in shop_operating_dates:
            for d in shop_operating_dates[shop]:
                if d > base_date:
                    actual_date = d
                    break
                    
        if pd.isna(actual_date) or df_raw_temp.empty:
            return pd.Series([np.nan, np.nan, np.nan, np.nan, pd.NaT])
            
        target_row = df_raw_temp[
            (df_raw_temp[shop_col] == shop) & 
            (df_raw_temp['対象日付'] == actual_date) & 
            (df_raw_temp['台番号'] == machine_no)
        ]
        
        if not target_row.empty:
            tr = target_row.iloc[0]
            return pd.Series([tr.get('差枚', 0), tr.get('BIG', 0), tr.get('REG', 0), tr.get('累計ゲーム', 0), actual_date])
        else:
            # 営業日だが該当台のデータがない = 未稼働(0G)とみなす
            return pd.Series([0, 0, 0, 0, actual_date])
            
    results = base_df.apply(get_actual_result, axis=1)
    base_df[['差枚_actual', '結果_BIG', '結果_REG', '結果_累計ゲーム', '実際の稼働日']] = results
    
    # --- BIG/REG確率分母の計算 ---
    cum_g = pd.to_numeric(base_df['結果_累計ゲーム'], errors='coerce').fillna(0)
    big_c = pd.to_numeric(base_df['結果_BIG'], errors='coerce').fillna(0)
    reg_c = pd.to_numeric(base_df['結果_REG'], errors='coerce').fillna(0)
    base_df['結果_BIG確率分母'] = np.where(big_c > 0, cum_g / big_c, 0)
    base_df['結果_REG確率分母'] = np.where(reg_c > 0, cum_g / reg_c, 0)

    # 特徴量の名前を _current から元に戻す（弱点分析などで使用するため）
    for col in base_df.columns.copy():
        if col.endswith('_current'):
            orig_col = col.replace('_current', '')
            if orig_col not in ['prediction_score', '予測差枚数', '機種名', '予測信頼度', 'おすすめ度', '根拠', 'next_diff', 'next_BIG', 'next_REG', 'next_累計ゲーム']:
                base_df[orig_col] = base_df[col]

    base_df = base_df.dropna(subset=['差枚_actual', 'prediction_score']).copy()
    
    if base_df.empty:
        st.info("まだ結果が判明している予測がありません。")
        return

    # --- 設定5近似度の算出 (ボーナス回数の精査) ---
    specs = backend.get_machine_specs()
    def evaluate_setting5(row):
        g = row.get('結果_累計ゲーム', 0)
        act_b = row.get('結果_BIG', 0)
        act_r = row.get('結果_REG', 0)
        if pd.isna(g) or g <= 0: return pd.Series([0, 0, 0, 0, 0, 0])
        machine = row.get('機種名', '')
        matched_spec = backend.get_matched_spec_key(machine, specs)
        p_b, p_r = 1/259.0, 1/255.0 # デフォルト
        if matched_spec and "設定5" in specs[matched_spec]:
            s5 = specs[matched_spec]["設定5"]
            if "BIG" in s5: p_b = 1.0 / s5["BIG"]
            if "REG" in s5: p_r = 1.0 / s5["REG"]
        exp_b, exp_r = g * p_b, g * p_r
        diff_b, diff_r = act_b - exp_b, act_r - exp_r
        
        # --- 1. 真の確率（ベイズ推定）による結果ベース設定5以上確率の計算 ---
        prob_5_over = 0.0
        if matched_spec:
            ms = specs[matched_spec]
            s1 = ms.get("設定1", {"BIG": 280.0, "REG": 400.0})
            s4 = ms.get("設定4", {"BIG": 260.0, "REG": 300.0})
            s5 = ms.get("設定5", s4)
            s6 = ms.get("設定6", s5)
            
            full_specs = {1: s1, 4: s4, 5: s5, 6: s6}
            for s in [2, 3]:
                full_specs[s] = {}
                for k in ["BIG", "REG"]:
                    p1_val = 1.0 / s1.get(k, 300.0)
                    p4_val = 1.0 / s4.get(k, 300.0)
                    p_s = p1_val + (p4_val - p1_val) * (s - 1) / 3.0
                    full_specs[s][k] = 1.0 / p_s if p_s > 0 else 999.0
            
            log_L = []
            for i in range(1, 7):
                p_big_i = 1.0 / full_specs[i].get("BIG", 300.0)
                p_reg_i = 1.0 / full_specs[i].get("REG", 300.0)
                exp_big_i = g * p_big_i
                exp_reg_i = g * p_reg_i
                ll_big = act_b * math.log(exp_big_i) - exp_big_i if exp_big_i > 0 else 0
                ll_reg = act_r * math.log(exp_reg_i) - exp_reg_i if exp_reg_i > 0 else 0
                log_L.append(ll_big + ll_reg)
                
            max_ll = max(log_L)
            likelihoods = [math.exp(ll - max_ll) for ll in log_L]
            sum_L = sum(likelihoods)
            if sum_L > 0:
                prob_5_over = (likelihoods[4] + likelihoods[5]) / sum_L
                
        prob_5_over_pct = int(prob_5_over * 100)
        
        # --- 2. 点数（設定5近似度）の確率論的な調整 ---
        # 標準偏差(σ)を計算し、「確率的なブレ（0.5σ）」の範囲内ならペナルティを免除する
        sigma_r = math.sqrt(g * p_r * (1.0 - p_r)) if g > 0 else 0
        sigma_b = math.sqrt(g * p_b * (1.0 - p_b)) if g > 0 else 0
        
        deficit_r = max(0, exp_r - act_r)
        adjusted_deficit_r = max(0, deficit_r - (sigma_r * 0.5)) # 0.5σ分の下振れは許容
        
        deficit_b = max(0, exp_b - act_b)
        adjusted_deficit_b = max(0, deficit_b - (sigma_b * 0.5)) # 0.5σ分の下振れは許容
        
        score_r = max(0, 80 - (adjusted_deficit_r * penalty_reg))
        score_b = max(0, 20 - (adjusted_deficit_b * penalty_big))
        
        total_score = score_r + score_b
        
        # --- 回転数（ゲーム数）による信頼度補正（いい塩梅の調整） ---
        # 1. 稼働が十分でない（5000G未満）場合：まぐれ上振れの可能性を考慮し、最大20%の範囲で徐々に割り引く
        if g < 5000:
            # 例: 1000Gで約0.84倍、3000Gで約0.92倍、5000Gで1.0倍
            multiplier = 0.80 + (g / 5000.0) * 0.20
            total_score *= multiplier
            
        # 2. 極端な低稼働（1000G未満）：サイドバーの設定（low_g_penalty）をさらに強力に適用
        if g < 1000:
            total_score *= (1 - ((1000 - g) / 1000.0) * (low_g_penalty / 100.0))
            
        # 3. 超高稼働（7000G以上）：本物の高設定の証拠として、不足が少ない優秀台にボーナス加点
        if g >= 7000 and adjusted_deficit_r <= 0: # 確率的に全く不足していない場合
            # 7000Gで+0点、8000Gで+2点、9500G以上で最大+5点のボーナス
            bonus = min(5.0, (g - 7000) / 500.0)
            total_score = min(100.0, total_score + bonus)
            
        # 4. 明らかな「見切り台（確率的に高設定が極めて薄い）」への追加ペナルティ
        is_abandoned = False
        tot_b_r = act_b + act_r
        if g >= 500 and tot_b_r == 0: is_abandoned = True
        elif g >= 1000 and tot_b_r > 0 and (g / tot_b_r) >= 400: is_abandoned = True
        elif g >= 1500 and tot_b_r > 0 and (g / tot_b_r) >= 300: is_abandoned = True
        
        if is_abandoned:
            total_score *= 0.5 # 高設定の可能性が著しく低いため点数半減
            
        return pd.Series([total_score, exp_b, exp_r, diff_b, diff_r, prob_5_over_pct])
        
    eval_df = base_df.apply(evaluate_setting5, axis=1)
    base_df[['設定5近似度', '期待BIG', '期待REG', 'BIG不足分', 'REG不足分', '結果_設定5以上確率']] = eval_df

    # --- 低稼働台のフィルタリング (ノーカウント処理) ---
    if min_g_filter > 0:
        # 指定G数以上回されている台
        cond_g_enough = base_df['結果_累計ゲーム'] >= min_g_filter
        
        # 指定G数未満だが、確率的に高設定が極めて薄い「明らかな見切り台」
        g_val = base_df['結果_累計ゲーム']
        p_val = base_df['結果_合算確率分母']
        
        cond_500 = (g_val >= 500) & (p_val == 0)
        cond_1000 = (g_val >= 1000) & (p_val >= 400)
        cond_1500 = (g_val >= 1500) & (p_val >= 300)
        cond_abandoned = cond_500 | cond_1000 | cond_1500
        
        base_df = base_df[cond_g_enough | cond_abandoned].copy()
        if base_df.empty:
            st.warning(f"検証対象となるデータがありません。（指定G数以上の台、または明らかな見切り台が存在しません）")
            return

    # --- 店舗フィルター ---
    if shop_col not in base_df.columns:
        st.warning("店舗データがありません。")
        return

    shops = ["店舗を選択してください"] + sorted(list(base_df[shop_col].unique()))
    
    default_index = 0
    saved_shop = st.session_state.get("global_selected_shop", "店舗を選択してください")
    if saved_shop in shops:
        default_index = shops.index(saved_shop)
        
    selected_shop = st.selectbox("分析対象の店舗を選択", shops, index=default_index, key="verification_shop")
    
    if selected_shop != "店舗を選択してください":
        st.session_state["global_selected_shop"] = selected_shop

    if selected_shop == "店舗を選択してください":
        st.info("👆 分析対象の店舗を選択してください。店舗ごとの精度やバックテスト成績を表示します。")
        return

    merged_df = base_df[base_df[shop_col] == selected_shop].copy()
    st.subheader(f"📊 AIモデル バックテスト通算成績 ({selected_shop} / {selected_version})")

    if merged_df.empty:
        st.info("選択された店舗の分析データがありません。")
        return

    # --- バージョン別成績比較表 (「すべて」選択時のみ表示) ---
    if selected_version == 'すべて' and 'ai_version' in merged_df.columns:
        st.markdown("##### 🔍 バージョン別 成績比較")
        st.caption("過去に試したAI設定ごとの成績一覧です。どの設定が最も優秀だったかを比較できます。")
        ver_stats = merged_df.groupby('ai_version').agg(
            検証台数=('台番号', 'count'),
            高設定率=('is_high_setting', 'mean'),
            勝率=('差枚_actual', lambda x: (x > 0).mean()),
            平均差枚=('差枚_actual', 'mean'),
            平均事後確率=('結果_設定5以上確率', 'mean'),
            設定5近似度=('設定5近似度', 'mean')
        ).reset_index().sort_values('設定5近似度', ascending=False)
        
        st.dataframe(
            ver_stats,
            column_config={
                "ai_version": st.column_config.TextColumn("バージョン設定"),
                "検証台数": st.column_config.NumberColumn("台数", format="%d台"),
                "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=1),
                "勝率": st.column_config.ProgressColumn("勝率(差枚)", format="%.1f%%", min_value=0, max_value=1),
                "平均差枚": st.column_config.NumberColumn("平均差枚", format="%+d枚"),
                "平均事後確率": st.column_config.NumberColumn("平均事後確率", format="%.1f%%", help="結果から算出した設定5以上確率の平均"),
                "設定5近似度": st.column_config.NumberColumn("平均5近似度", format="%.1f点")
            },
            hide_index=True,
            width="stretch"
        )

    # --- 1. 全体成績 (KPI) & 円グラフ ---
    specs = backend.get_machine_specs()
    spec_reg_val = merged_df['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
    spec_tot_val = merged_df['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
    
    m_cum_g = pd.to_numeric(merged_df['結果_累計ゲーム'], errors='coerce').fillna(0)
    m_big_c = pd.to_numeric(merged_df['結果_BIG'], errors='coerce').fillna(0)
    m_reg_c = pd.to_numeric(merged_df['結果_REG'], errors='coerce').fillna(0)
    merged_df['結果_合算確率分母'] = np.where((m_big_c + m_reg_c) > 0, m_cum_g / (m_big_c + m_reg_c), 0)
    merged_df['is_high_setting'] = (((merged_df['結果_REG確率分母'] > 0) & (merged_df['結果_REG確率分母'] <= spec_reg_val)) | ((merged_df['結果_合算確率分母'] > 0) & (merged_df['結果_合算確率分母'] <= spec_tot_val))).astype(int)

    total_count = len(merged_df)
    high_set_count = merged_df['is_high_setting'].sum()
    low_set_count = total_count - high_set_count
    high_setting_rate = high_set_count / total_count if total_count > 0 else 0
    win_count = (merged_df['差枚_actual'] > 0).sum()
    win_rate = win_count / total_count if total_count > 0 else 0
    avg_diff = merged_df['差枚_actual'].mean()
    total_diff = merged_df['差枚_actual'].sum()
    
    col_kpi, col_pie = st.columns([2, 1])
    
    with col_kpi:
        k1, k2, k5 = st.columns(3)
        k1.metric("検証台数", f"{total_count} 台")
        k2.metric("高設定率", f"{high_setting_rate:.1%}")
        k5.metric("勝率(差枚)", f"{win_rate:.1%}")
        
        k3, k4 = st.columns(2)
        k3.metric("平均差枚", f"{int(avg_diff):+d} 枚")
        k4.metric("合計収支", f"{int(total_diff):+d} 枚")
        
    with col_pie:
        # 勝敗円グラフ (Altair)
        pie_data = pd.DataFrame({
            'Category': ['高設定', '低設定'],
            'Count': [high_set_count, low_set_count]
        })
        
        pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=35).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Category", type="nominal", 
                            scale=alt.Scale(domain=['高設定', '低設定'], range=['#FF4B4B', '#4B4BFF']),
                            legend=alt.Legend(title="設定挙動", orient="bottom")),
            tooltip=['Category', 'Count']
        ).properties(height=200)
        
        st.altair_chart(pie_chart, width="stretch")

    # 日別推移データをAI評価より先に計算する
    daily_stats = merged_df.groupby('対象日付').agg(
        high_setting_rate=('is_high_setting', 'mean'),
        total_profit=('差枚_actual', 'sum'),
        avg_s5_score=('設定5近似度', 'mean'),
        count=('台番号', 'count')
    ).reset_index().sort_values('対象日付')

    # --- AI振り返りレポート ---
    st.divider()
    st.subheader("🤖 AIの振り返りレポート (過去の自分との比較)")
    st.caption("最新の予測結果（設定5近似度）を過去の平均的なパフォーマンスと比較し、AIが自身の成長や調子を分析します。")
    
    avg_s5_score = merged_df['設定5近似度'].mean()
    avg_g = merged_df['結果_累計ゲーム'].mean()
    avg_diff_r = merged_df['REG不足分'].mean()
    avg_diff_b = merged_df['BIG不足分'].mean()
    
    # データ不足の考慮 (検証台数不足 or 学習データ不足)
    low_rel_count = (merged_df['予測信頼度'] == '🔻低').sum() if '予測信頼度' in merged_df.columns else 0
    mid_rel_count = (merged_df['予測信頼度'] == '🔸中').sum() if '予測信頼度' in merged_df.columns else 0
    low_rel_rate = low_rel_count / total_count if total_count > 0 else 0
    mid_rel_rate = mid_rel_count / total_count if total_count > 0 else 0
    comment_prefix = ""
    if total_count < 5:
        comment_prefix = f"⚠️ **【検証台数不足】** 今回対象となった推奨台が **{total_count}台** と少ないため、たまたまのヒキの影響を強く受けています。点数は参考程度にご覧ください。\n\n"
    elif low_rel_rate > 0.3:
        comment_prefix = f"⚠️ **【台ごとの履歴データ不足】** 対象台の多く（{low_rel_rate:.0%}）が、過去履歴が14日未満の「信頼度:低」状態です。AIが各台のクセを正確に把握して期待度を高く出すには、1台あたり約30日分のデータが必要です。あと**約2〜3週間**ほど日々のデータ取得を続けると、予測スコアと精度が劇的に安定します。\n\n"
    elif low_rel_rate + mid_rel_rate > 0.5:
        comment_prefix = f"💡 **【データ蓄積の途中】** 対象台の半数以上が過去履歴30日未満です。AIのポテンシャルを完全に引き出すため、あと**約1〜2週間**ほど毎日のデータ取得（ログ蓄積）を継続することをおすすめします。\n\n"

    # 全体評価（期間全体の総評）
    eval_mode_str = "標準"
    if penalty_reg <= 10:
        eval_mode_str = "甘め"
    elif penalty_reg >= 20:
        eval_mode_str = "辛め"
        
    if pd.isna(avg_g) or avg_g < 2000:
        overall_comment = f"全体として推奨台の平均回転数が **{int(avg_g if not pd.isna(avg_g) else 0)}G** と少なく、試行回数不足です。もう少し稼働がある状況で検証したいですね。"
        mood = "🤔"
    elif avg_s5_score >= 80:
        if eval_mode_str == "甘め":
            overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と大成功レベルです！（※甘め評価のためBIGのヒキ強も含まれます）"
        elif eval_mode_str == "辛め":
            overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と大成功レベルです！辛め評価でもこの点数は、文句なしの**本物の高設定**を的確に見抜けています！"
        else:
            overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と大成功レベルです！本物の高設定を的確に見抜けています！"
        mood = "🌟"
    elif avg_s5_score >= 60:
        if avg_diff_r < 0:
            if eval_mode_str == "甘め":
                overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** でまずまずです。REGが平均 {abs(avg_diff_r):.1f}回 不足していますが、甘め評価（出玉重視）としては合格点です。"
            else:
                overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** でまずまずですが、REGが平均 {abs(avg_diff_r):.1f}回 不足しています。低〜中間設定の上振れに助けられている部分もありそうです。"
        else:
            if eval_mode_str == "辛め":
                overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** です。辛め評価の中では健闘しており、中身もしっかり高設定挙動を示しています。"
            else:
                overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** で優秀です！中身もしっかり高設定挙動を示しています。"
        mood = "👍"
    elif avg_s5_score >= 40:
        if avg_g < 4000:
            overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と惜しい結果です。平均回転数が {int(avg_g)}G と少なめなので、下振れの可能性もあります。"
        else:
            if eval_mode_str == "甘め":
                overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と反省点が残ります。甘め評価でこの点数なので、低設定の誤爆だった可能性が高いです。"
            elif eval_mode_str == "辛め":
                overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** です。辛め評価なので点数が低く出やすくなっていますが、もう少しREGが欲しいところです。"
            else:
                overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と反省点が残ります。低〜中間設定が混ざっている可能性が高いです。"
        mood = "💦"
    else:
        if eval_mode_str == "辛め":
             overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と惨敗です…。辛め評価であることを考慮しても厳しい結果です。設定状況が変わった可能性があります。"
        else:
             overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と惨敗です…。過去の傾向が変わった可能性があるので、最近のデータで学習し直すか、店選びを見直す余地があります。"
        mood = "😭"

    # 過去の自分との比較ロジック
    has_comparison = False
    if len(daily_stats) >= 2:
        latest_date = daily_stats['対象日付'].max()
        latest_stat = daily_stats.iloc[-1]
        past_stats = daily_stats.iloc[:-1]
        
        # 直近の過去（最大14日分）を「最近の実力」とする
        recent_past = past_stats.tail(14)
        
        # 台数（サンプル数）を考慮した加重平均を計算する
        total_past_score = (recent_past['avg_s5_score'] * recent_past['count']).sum()
        total_past_count = recent_past['count'].sum()
        past_avg_score = total_past_score / total_past_count if total_past_count > 0 else 0
        
        latest_score = latest_stat['avg_s5_score']
        score_diff = latest_score - past_avg_score
        
        has_comparison = True
        latest_date_str = latest_date.strftime('%m/%d')
        
        if total_past_count < 10:
             comparison_comment = f"最新 ({latest_date_str}) の設定5近似度は **{latest_score:.1f}点** でした。比較対象となる過去の検証台数が少なく（計{int(total_past_count)}台）、AIの成長や調子を正しく評価するにはまだデータが不足しています。"
             mood_cmp = "🤔"
        elif latest_stat['count'] < 5:
             comparison_comment = f"最新 ({latest_date_str}) の設定5近似度は **{latest_score:.1f}点** でした。直近の平均 ({past_avg_score:.1f}点) と比較したいところですが、今回の検証台数が{int(latest_stat['count'])}台と少ないため、たまたまのブレが大きい可能性があります。"
             mood_cmp = "🤔"
        elif score_diff >= 5:
             comparison_comment = f"最新 ({latest_date_str}) の設定5近似度は **{latest_score:.1f}点** でした！直近の平均 ({past_avg_score:.1f}点) より **{score_diff:+.1f}点** も向上しており、予測精度が上がっています！日々学習して賢くなっているのを感じますね！"
             mood_cmp = "📈"
        elif score_diff <= -5:
             comparison_comment = f"最新 ({latest_date_str}) の設定5近似度は **{latest_score:.1f}点** でした…。直近の平均 ({past_avg_score:.1f}点) より **{score_diff:+.1f}点** 下がっています。少し調子を落としているか、お店の設定配分のクセが変わった（フェイクが増えた等）可能性があります。"
             mood_cmp = "📉"
        else:
             comparison_comment = f"最新 ({latest_date_str}) の設定5近似度は **{latest_score:.1f}点** でした。直近の平均 ({past_avg_score:.1f}点) とほぼ同水準をキープしており、安定した予測ができています。"
             mood_cmp = "⚖️"

    if has_comparison:
        final_comment = f"{mood_cmp} **最近の調子 (過去の自分との比較):**\n{comparison_comment}\n\n{mood} **全体の総評:**\n{overall_comment}"
    else:
        final_comment = f"{mood} **全体の総評:**\n{overall_comment}\n\n※比較対象となる過去の推移データが不足しています。"

    st.info(f"{comment_prefix}{final_comment}\n\n※全体平均 REG過不足: **{avg_diff_r:+.1f}回** / BIG過不足: **{avg_diff_b:+.1f}回**")
    
    # --- 特に優秀だった台トップ3 ---
    if not merged_df.empty:
        top3_df = merged_df[merged_df['設定5近似度'] > 0].sort_values('設定5近似度', ascending=False).head(3)
        if not top3_df.empty:
            st.markdown("##### 🌟 特に高設定挙動だった推奨台 トップ3")
            for _, row in top3_df.iterrows():
                shop = row.get(shop_col, '')
                m_num = row.get('台番号', '')
                m_name = row.get('機種名', '')
                score = row.get('設定5近似度', 0)
                diff = row.get('差枚_actual', 0)
                g = row.get('結果_累計ゲーム', 0)
                r = row.get('結果_REG', 0)
                b = row.get('結果_BIG', 0)
                post_prob = int(row.get('結果_設定5以上確率', 0))
                date_str = row['対象日付'].strftime('%m/%d') if pd.notna(row.get('対象日付')) else ""
                st.markdown(f"- **{score:.1f}点 (事後確率: {post_prob}%)** : {date_str} {shop} #{m_num} {m_name} ({int(g)}G BIG{int(b)} REG{int(r)} / {int(diff):+d}枚)")

    # --- 2. 時系列推移 (勝率 & 収支) ---
    st.subheader("📈 日別の推移")
    
    if not daily_stats.empty:
        daily_stats['date_str'] = daily_stats['対象日付'].dt.strftime('%m/%d')
        
        tab_prof, tab_s5, tab_ver = st.tabs(["💰 高設定率・収支推移", "🎯 設定5近似度 推移", "📅 予測保存日ごとの精度推移"])
        with tab_prof:
            base_chart = alt.Chart(daily_stats).encode(x=alt.X('date_str', title='予測対象日', sort=None))
            bar_chart = base_chart.mark_bar(opacity=0.6).encode(
                y=alt.Y('total_profit', title='日別収支 (枚)'), color=alt.condition(alt.datum.total_profit > 0, alt.value("#FF4B4B"), alt.value("#4B4BFF")), tooltip=['date_str', alt.Tooltip('total_profit', format='+d'), 'count']
            )
            line_chart = base_chart.mark_line(point=True, color='#FFA726', strokeWidth=3).encode(
                y=alt.Y('high_setting_rate', title='高設定率 (%)', axis=alt.Axis(format='%')), tooltip=['date_str', alt.Tooltip('high_setting_rate', format='.1%')]
            )
        st.altair_chart(alt.layer(bar_chart, line_chart).resolve_scale(y='independent'), width="stretch")
        with tab_s5:
            base_chart_s5 = alt.Chart(daily_stats).encode(x=alt.X('date_str', title='予測対象日', sort=None))
            line_s5 = base_chart_s5.mark_line(point=True, color='#AB47BC', strokeWidth=3).encode(
                y=alt.Y('avg_s5_score', title='設定5近似度 (平均点)', scale=alt.Scale(domain=[0, 100])), tooltip=['date_str', alt.Tooltip('avg_s5_score', format='.1f', title='設定5近似度'), 'count']
            )
            st.altair_chart(line_s5, width="stretch")
            st.caption("※点数が高いほど、推奨台が実際に設定5以上の確率でBIG/REGを引けていたことを示します。(予測対象日ベース)")
            
        with tab_ver:
            if '実行日時' in merged_df.columns:
                merged_df['実行日時'] = pd.to_datetime(merged_df['実行日時'], errors='coerce')
                merged_df['実行日'] = merged_df['実行日時'].dt.date
                exec_stats = merged_df.groupby('実行日').agg(
                    avg_s5_score=('設定5近似度', 'mean'),
                    high_setting_rate=('is_high_setting', 'mean'),
                    count=('台番号', 'count')
                ).reset_index().dropna(subset=['実行日'])
                
                if not exec_stats.empty:
                    exec_stats['exec_date_str'] = exec_stats['実行日'].apply(lambda x: x.strftime('%m/%d'))
                    
                    base_chart_exec = alt.Chart(exec_stats).encode(x=alt.X('exec_date_str', title='予測保存日 (実行日)', sort=None))
                    
                    # 棒グラフ: 高設定率 (左軸)
                    bar_exec = base_chart_exec.mark_bar(opacity=0.6, color='#42A5F5').encode(
                        y=alt.Y('high_setting_rate', title='高設定率 (%)', axis=alt.Axis(format='%')),
                        tooltip=['exec_date_str', alt.Tooltip('high_setting_rate', format='.1%'), 'count']
                    )
                    
                    # 折れ線: 設定5近似度 (右軸)
                    line_exec = base_chart_exec.mark_line(point=True, color='#AB47BC', strokeWidth=3).encode(
                        y=alt.Y('avg_s5_score', title='設定5近似度 (平均点)', scale=alt.Scale(domain=[0, 100])),
                        tooltip=['exec_date_str', alt.Tooltip('avg_s5_score', format='.1f', title='設定5近似度'), 'count']
                    )
                    
                    st.altair_chart(alt.layer(bar_exec, line_exec).resolve_scale(y='independent'), use_container_width=True)
                    st.caption("※ AIが予測を保存した「実行日」ごとの精度推移です。AIの設定変更(設定4基準→設定5基準など)による成長を確認できます。")
                else:
                    st.info("予測保存日のデータがありません。")
            else:
                st.info("実行日時のデータがありません。")

    # --- 2. 確率帯別 精度分析 ---
    st.subheader("📈 確率帯別 精度分析")
    if 'prediction_score' in merged_df.columns:
        def get_prob_band(score):
            if score >= 0.85: return '85%以上'
            elif score >= 0.70: return '70%〜84%'
            elif score >= 0.50: return '50%〜69%'
            elif score >= 0.30: return '30%〜49%'
            else: return '30%未満'
            
        merged_df['確率帯'] = merged_df['prediction_score'].apply(get_prob_band)
        
        # 期間絞り込みUIの追加
        period_options = {"直近1ヶ月": 30, "直近3ヶ月": 90, "直近半年": 180, "全期間": None}
        selected_period = st.radio("集計対象の期間 (AIの現在の実力を測るため直近絞り込み推奨)", list(period_options.keys()), index=0, horizontal=True)
        
        if period_options[selected_period] is not None:
            latest_date = merged_df['対象日付'].max()
            cutoff_date = latest_date - pd.Timedelta(days=period_options[selected_period])
            prob_analysis_df = merged_df[merged_df['対象日付'] >= cutoff_date].copy()
        else:
            prob_analysis_df = merged_df.copy()
            
        if prob_analysis_df.empty:
            st.warning(f"指定された期間（{selected_period}）のデータがありません。")
        else:
            # --- 期待度スコアのヒストグラム ---
            st.markdown(f"**📊 期待度スコア (予測スコア) の分布と正解率 ({selected_period})**")
            
            # 5%刻みの代表値(左端)に丸めて集計
            prob_analysis_df['score_bin_left'] = (prob_analysis_df['prediction_score'] // 0.05) * 0.05
            hist_stats = prob_analysis_df.groupby('score_bin_left').agg(
                count=('台番号', 'count'),
                high_setting_rate=('is_high_setting', 'mean')
            ).reset_index()
            
            base_hist = alt.Chart(hist_stats).encode(
                x=alt.X('score_bin_left:O', title='AI期待度 (5%刻み)', axis=alt.Axis(format='%'))
            )
            bar_hist = base_hist.mark_bar(color='#42A5F5', opacity=0.8).encode(
                y=alt.Y('count:Q', title='検証台数'),
                tooltip=[alt.Tooltip('score_bin_left:Q', format='.0%', title='スコア帯(下限)'), alt.Tooltip('count:Q', title='台数')]
            )
            line_hist = base_hist.mark_line(color='#FF7043', point=True, strokeWidth=3).encode(
                y=alt.Y('high_setting_rate:Q', title='実際の高設定率', axis=alt.Axis(format='%')),
                tooltip=[alt.Tooltip('score_bin_left:Q', format='.0%', title='スコア帯(下限)'), alt.Tooltip('high_setting_rate:Q', format='.1%', title='実際の高設定率')]
            )
            st.altair_chart(alt.layer(bar_hist, line_hist).resolve_scale(y='independent').properties(height=300), width="stretch")

            # --- 🤖 総合原因分析 (AIの自己診断レポート) ---
            total_eval_count = len(prob_analysis_df)
            period_high_setting_rate = prob_analysis_df['is_high_setting'].mean()
            high_score_df = prob_analysis_df[prob_analysis_df['prediction_score'] >= 0.70]
            high_score_accuracy = high_score_df['is_high_setting'].mean() if not high_score_df.empty else 0
            score_mean = prob_analysis_df['prediction_score'].mean()
            
            avg_g_recent = prob_analysis_df['結果_累計ゲーム'].mean()
            
            low_rel_count = (prob_analysis_df['予測信頼度'] == '🔻低').sum() if '予測信頼度' in prob_analysis_df.columns else 0
            low_rel_rate = low_rel_count / total_eval_count if total_eval_count > 0 else 0
            
            # 要因1: データ量
            diag_data = {"status": "🟢", "title": "データ蓄積量", "msg": "十分なデータが蓄積されています。"}
            if total_eval_count < 30:
                diag_data = {"status": "🔴", "title": "データ蓄積量", "msg": f"検証台数が{total_eval_count}台と少なすぎます。たまたまのヒキによるブレが大きいため、まずはデータ取得を続けてください。"}
            elif low_rel_rate > 0.3:
                diag_data = {"status": "🟡", "title": "データ蓄積量", "msg": f"各台の履歴データが浅い（新台・配置変更など）台が {low_rel_rate:.0%} 含まれています。AIがクセを把握しきるまであと数週間のデータ取得が必要です。"}
                
            # 要因2: 稼働（客層）
            diag_kado = {"status": "🟢", "title": "稼働状況(客層)", "msg": "検証対象の台は全体的にしっかり回されており、結果の信頼度が高いです。"}
            if avg_g_recent < 3000:
                diag_kado = {"status": "🔴", "title": "稼働状況(客層)", "msg": f"平均稼働が{int(avg_g_recent)}Gと低すぎます。客層の見切りが早いため、AIが当てていたとしても「不発・見切り」としてハズレ扱いになっている可能性が高いです。"}
            elif avg_g_recent < 4000:
                diag_kado = {"status": "🟡", "title": "稼働状況(客層)", "msg": f"平均稼働が{int(avg_g_recent)}Gとやや低めです。高設定が埋もれてしまっている可能性があります。"}

            # 要因3: AI設定 (過学習 / 未学習)
            diag_ai = {"status": "🟢", "title": "AI設定(パラメータ)", "msg": "スコア分布と正解率のバランスが良く、現在の設定は良好です。"}
            if total_eval_count >= 30:
                if len(high_score_df) >= 5 and high_score_accuracy < period_high_setting_rate:
                    diag_ai = {"status": "🔴", "title": "AI設定(過学習)", "msg": "AIが高評価した台の勝率が、店全体の平均勝率を下回っています。ノイズ（たまたま出た低設定）を必勝法だと勘違いしている「過学習」の疑いがあります。「深さ制限(max_depth)」を下げるか、「最小データ数」を増やしてください。"}
                elif score_mean > period_high_setting_rate + 0.15:
                    diag_ai = {"status": "🟡", "title": "AI設定(評価甘め)", "msg": "全体的にスコアが高く出すぎています。「葉の数(num_leaves)」を少し下げるか、「学習率」を下げてみてください。"}
                elif score_mean < period_high_setting_rate - 0.15:
                    diag_ai = {"status": "🟡", "title": "AI設定(評価慎重)", "msg": "全体的にスコアが低く出すぎています。「学習回数」を増やすか、「葉の数」を少し上げてみてください。"}

            # 要因4: 特徴量の多さ (次元の呪い)
            diag_feat = {"status": "🟢", "title": "特徴量の多さ", "msg": "学習データに対して適切な条件分岐が行われています。"}
            if total_eval_count >= 50 and len(high_score_df) == 0:
                 diag_feat = {"status": "🟡", "title": "特徴量の多さ(条件厳格化)", "msg": "期待度70%を超える台が1台もありません。AIが多くの特徴量（条件）を同時に満たす完璧な台を探しすぎて、身動きが取れなくなっています。"}

            # 要因5: 店舗の読みにくさ (ランダム・フェイク)
            diag_shop = {"status": "🟢", "title": "店舗の素直さ", "msg": "店舗のクセをある程度捉えられています。"}
            if period_high_setting_rate < 0.05:
                 diag_shop = {"status": "🔴", "title": "店舗の素直さ(ベタピン疑惑)", "msg": "そもそも店舗全体の高設定投入率が極端に低すぎます。AIが予測する以前に、戦うべき店舗（優良店）ではない可能性があります。"}
            elif diag_data["status"] == "🟢" and diag_kado["status"] == "🟢" and diag_ai["status"] in ["🟢", "🟡"]:
                if len(high_score_df) >= 5 and (high_score_accuracy - period_high_setting_rate) < 0.05:
                    diag_shop = {"status": "🔴", "title": "店舗の素直さ(予測困難)", "msg": "データ・稼働・AI設定は悪くないにも関わらず、AI推奨台が結果を出せていません。店長が「完全ランダム」で設定を入れているか、前日の凹み台などを「意図的にフェイクとして使う」など、非常に読みにくい（騙してくる）店舗である可能性が高いです。"}

            with st.expander("🤖 総合原因分析 (AIの自己診断レポート)", expanded=True):
                st.markdown("精度検証の結果から、予測がうまくいっているか、あるいは**何が原因で精度が落ちているか**を5つの観点で総合的に診断します。")
                
                for diag in [diag_data, diag_kado, diag_ai, diag_feat, diag_shop]:
                    st.markdown(f"**{diag['status']} {diag['title']}**: {diag['msg']}")
                
                st.divider()
                
                # --- 最終アプローチ（ネクストアクション）の判定 ---
                if diag_shop["status"] == "🔴":
                    final_action_title = "🚨 【最終結論】この店舗での稼働を見直す（店を変える）"
                    if "ベタピン疑惑" in diag_shop["title"]:
                        final_action_msg = "店舗全体に高設定が全く使われていない可能性が高いです。AIの設定やデータ収集を頑張るよりも、**『より状況の良い別の店舗を開拓する』** ことが最も勝率に直結します。"
                    else:
                        final_action_msg = "店長の配分がランダムすぎるか、フェイク（罠）が多くてAIの予測が通用していません。**『この店での稼働を控える』**か、**『強いイベント日のみに絞る』** ことを強くおすすめします。AIの設定をいじっても改善は難しいです。"
                    st.error(f"**{final_action_title}**\n\n{final_action_msg}")

                elif diag_data["status"] == "🔴":
                    final_action_title = "⏳ 【最終結論】今は情報を集めるまで待つ（データ収集継続）"
                    final_action_msg = "まだAIが傾向を学習・評価するための十分なデータが揃っていません。今AIのパラメータをいじると逆に精度が壊れる（過学習する）危険があります。**まずは毎日のデータ取得を継続し、サンプル数が30台以上に達するのを待ってください。**"
                    st.warning(f"**{final_action_title}**\n\n{final_action_msg}")
                    
                elif diag_kado["status"] == "🔴":
                    final_action_title = "🤔 【最終結論】店舗の客層・稼働状況を再評価する"
                    final_action_msg = "AIの予測云々以前に、客層の見切りが早すぎて「答え合わせ」ができていない状態です。この店で勝つには、**『自分自身でしっかり回して判別する』**覚悟が必要になります。または、もっと稼働が良い店を探すのも一つの手です。"
                    st.warning(f"**{final_action_title}**\n\n{final_action_msg}")

                elif diag_ai["status"] in ["🔴", "🟡"] or diag_feat["status"] in ["🔴", "🟡"]:
                    final_action_title = "⚙️ 【最終結論】アプリ内のAI設定をチューニングする"
                    tuning_hints = []
                    if "過学習" in diag_ai["title"]: tuning_hints.append("「深さ制限(max_depth)」を下げる、または「最小データ数」を増やす")
                    if "評価甘め" in diag_ai["title"]: tuning_hints.append("「葉の数(num_leaves)」を下げる、または「学習率」を下げる")
                    if "評価慎重" in diag_ai["title"]: tuning_hints.append("「学習回数」を増やす、または「葉の数」を上げる")
                    if "条件厳格化" in diag_feat["title"]: tuning_hints.append("「深さ制限」を下げる、または「最小データ数」を増やす")
                    
                    final_action_msg = "データ量も店舗の素直さも悪くありませんが、AIの「クセの覚え方」が現在の店舗の状況とズレています。すぐ下の**【店舗専用 AIモデル設定】**から、以下の調整を試して再分析してください。\n\n"
                    for hint in set(tuning_hints):
                        final_action_msg += f"- {hint}\n"
                    st.info(f"**{final_action_title}**\n\n{final_action_msg}")

                else:
                    final_action_title = "🌟 【最終結論】現状維持でOK（素晴らしい状態です！）"
                    final_action_msg = "店舗の状況、データ量、AIの設定、すべてが完璧に噛み合っています。**今のAI設定のまま、自信を持って日々の立ち回りに活用してください！**"
                    st.success(f"**{final_action_title}**\n\n{final_action_msg}")
                    
            # --- 各店舗専用 AIモデル設定 ---
            st.divider()
            st.subheader(f"⚙️ 【{selected_shop}】専用 AIモデル設定")
            st.caption("上のアドバイスを参考に手動で調整するか、「自動チューニング」を試してください。")
            
            if "shop_hyperparams" not in st.session_state:
                st.session_state["shop_hyperparams"] = {"デフォルト": {'train_months': 3, 'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50}}
                
            default_hp = st.session_state["shop_hyperparams"]["デフォルト"]
            current_hp = st.session_state["shop_hyperparams"].get(selected_shop, default_hp)
            
            with st.form(f"hp_form_{selected_shop}"):
                hp_train_months = st.slider("学習データ期間 (直近〇ヶ月)", 1, 12, current_hp.get('train_months', 3), step=1)
                hp_n_estimators = st.slider("学習回数 (n_estimators)", 50, 1000, current_hp.get('n_estimators', 300), step=50)
                hp_learning_rate = st.slider("学習率 (learning_rate)", 0.01, 0.3, current_hp.get('learning_rate', 0.03), step=0.01)
                hp_num_leaves = st.slider("葉の数 (num_leaves)", 10, 127, current_hp.get('num_leaves', 15), step=1)
                hp_max_depth = st.slider("深さ制限 (max_depth)", -1, 15, current_hp.get('max_depth', 4), step=1)
                hp_min_child_samples = st.slider("最小データ数 (min_child_samples)", 10, 200, current_hp.get('min_child_samples', 50), step=10)
                
                cols = st.columns(3)
                submitted = cols[0].form_submit_button("この店舗の設定を保存して再分析", type="primary")
                reset_btn = cols[1].form_submit_button("全店舗共通設定に戻す")
                auto_tune_btn = cols[2].form_submit_button("✨ 自動チューニング")
                
            if submitted:
                st.session_state["shop_hyperparams"][selected_shop] = {
                    'train_months': hp_train_months, 'n_estimators': hp_n_estimators, 'learning_rate': hp_learning_rate,
                    'num_leaves': hp_num_leaves, 'max_depth': hp_max_depth, 'min_child_samples': hp_min_child_samples
                }
                backend.save_shop_ai_settings(st.session_state["shop_hyperparams"])
                st.cache_data.clear(); st.rerun()
                
            if reset_btn:
                if selected_shop in st.session_state["shop_hyperparams"]:
                    del st.session_state["shop_hyperparams"][selected_shop]
                    backend.save_shop_ai_settings(st.session_state["shop_hyperparams"])
                    st.cache_data.clear(); st.rerun()
                    
            if auto_tune_btn:
                with st.spinner("AIが過去データを分割し、数多くの組み合わせから最適な設定を探索中... (約10〜20秒かかります)"):
                    import lightgbm as lgb
                    # 特徴量リストの復元
                    base_features = ['累計ゲーム', 'REG確率', 'BIG確率', '差枚', '末尾番号', 'target_weekday', 'target_date_end_digit', 'mean_7days_diff', 'win_rate_7days', '連続マイナス日数', '連続低稼働日数', 'is_new_machine', 'history_count', 'machine_code', 'shop_code', 'reg_ratio', 'is_corner', 'neighbor_avg_diff', 'event_avg_diff', 'event_code', 'event_rank_score', 'prev_差枚', 'prev_REG確率', 'prev_累計ゲーム', 'shop_avg_diff', 'island_avg_diff', 'relative_games_ratio', 'shop_7days_avg_diff', 'machine_30days_avg_diff']
                    actual_features = [f for f in base_features if f in df_verify.columns]
                    cat_features = [f for f in ['machine_code', 'shop_code', 'event_code', 'target_weekday', 'target_date_end_digit'] if f in actual_features]
                    
                    shop_df = df_verify[df_verify[shop_col] == selected_shop].copy()
                    
                    if len(shop_df) < 150:
                        st.error("データが少なすぎて自動チューニングを実行できません。150件以上のデータが必要です。")
                    else:
                        # 時系列でソートし、最新20%をテスト、古い80%を学習に分割
                        shop_df = shop_df.sort_values('対象日付')
                        split_idx = int(len(shop_df) * 0.8)
                        train_data = shop_df.iloc[:split_idx]
                        test_data = shop_df.iloc[split_idx:]
                        
                        X_train, y_train = train_data[actual_features], train_data['target']
                        X_test, y_test = test_data[actual_features], test_data['target']
                        
                        # サンプルウェイト（時間減衰）の再現
                        max_date = train_data['対象日付'].max()
                        days_diff = (max_date - train_data['対象日付']).dt.days
                        sample_weights = 0.995 ** days_diff
                        
                        # 探索するパラメータの候補群
                        param_candidates = [
                            {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50}, # デフォルト
                            {'n_estimators': 500, 'learning_rate': 0.01, 'num_leaves': 15, 'max_depth': 3, 'min_child_samples': 80}, # 慎重派 (過学習防止)
                            {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 5, 'min_child_samples': 30}, # 積極派 (複雑な条件)
                            {'n_estimators': 400, 'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6, 'min_child_samples': 50}, # 深読み派
                            {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 10, 'max_depth': 3, 'min_child_samples': 100}, # 超シンプル派
                            {'n_estimators': 600, 'learning_rate': 0.01, 'num_leaves': 20, 'max_depth': 4, 'min_child_samples': 60}, # じっくり学習
                            {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 7, 'min_child_samples': 20}, # 荒波・一点張り
                            {'n_estimators': 100, 'learning_rate': 0.10, 'num_leaves': 15, 'max_depth': 3, 'min_child_samples': 40}, # 浅く広く
                            {'n_estimators': 400, 'learning_rate': 0.02, 'num_leaves': 25, 'max_depth': 5, 'min_child_samples': 40}, # バランス型
                            {'n_estimators': 200, 'learning_rate': 0.04, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 30}, # バランス型2
                        ]
                        
                        best_score = -9999
                        best_params = param_candidates[0]
                        
                        progress_bar = st.progress(0)
                        
                        # 各パラメータで模擬テストを実施
                        for i, params in enumerate(param_candidates):
                            try:
                                model = lgb.LGBMClassifier(
                                    objective='binary', random_state=42, verbose=-1, 
                                    n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], 
                                    num_leaves=params['num_leaves'], max_depth=params['max_depth'], 
                                    min_child_samples=params['min_child_samples'],
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8
                                )
                                model.fit(X_train, y_train, sample_weight=sample_weights, categorical_feature=cat_features)
                                
                                preds = model.predict_proba(X_test)[:, 1]
                                test_eval = test_data.copy()
                                test_eval['pred_score'] = preds
                                
                                # 期待度70%以上を出した台の「実際の成績」でAIを評価する
                                target_df = test_eval[test_eval['pred_score'] >= 0.70]
                                
                                if len(target_df) == 0:
                                    score = -1 # オススメ台を1台も出せないAIは失格
                                else:
                                    precision = target_df['target'].mean() # 推奨台の高設定率
                                    avg_diff = target_df['next_diff'].mean() # 推奨台の平均差枚
                                    coverage = len(target_df) / len(test_eval) # 推奨台の割合(少なすぎを防止)
                                    
                                    # 総合評価スコア (高設定率と差枚を重視)
                                    score = (precision * 100) + (avg_diff / 10) + (coverage * 50)
                                    
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                            except Exception:
                                pass
                            
                            progress_bar.progress((i + 1) / len(param_candidates))
                        
                        # 一番優秀だった設定を適用して保存
                        st.session_state["shop_hyperparams"][selected_shop] = {
                            'train_months': hp_train_months, # データ期間はユーザー指定を維持
                            'n_estimators': best_params['n_estimators'],
                            'learning_rate': best_params['learning_rate'],
                            'num_leaves': best_params['num_leaves'],
                            'max_depth': best_params['max_depth'],
                            'min_child_samples': best_params['min_child_samples']
                        }
                        backend.save_shop_ai_settings(st.session_state["shop_hyperparams"])
                        st.toast("✅ 自動チューニングが完了し、最も優秀だった設定を適用しました！")
                        st.cache_data.clear()
                        st.rerun()
                        
            st.markdown("**🔬 現在の設定でのシミュレーション成績 (最新データ)**")
            st.info("💡 **なぜ日別予測とスコアが違うの？**\nこのシミュレーションは「過去の全データを学習したAI」に「同じ過去のデータ」をテストさせているため、AIが結果を暗記しており期待度が異常に高く（85%以上などに）出ます。一方、「日別確認」は過去に遡って『その前日までのデータだけで未来を予測』する実戦形式のため、現実的なスコアになります。")
            st.caption("現在適用されている設定で過去データを再評価（答え合わせ）した結果です。AIが「どの程度店のクセを表現できるようになったか」の設定調整の参考値としてご利用ください。")
            sim_df = df_verify[df_verify[shop_col] == selected_shop].copy() if not df_verify.empty and shop_col in df_verify.columns else pd.DataFrame()
            if not sim_df.empty and 'prediction_score' in sim_df.columns and 'target' in sim_df.columns and 'next_diff' in sim_df.columns:
                sim_df['確率帯'] = sim_df['prediction_score'].apply(get_prob_band)
                sim_stats = sim_df.groupby('確率帯').agg(
                    台数=('台番号', 'count'),
                    高設定率=('target', lambda x: x.mean() * 100),
                    勝率=('next_diff', lambda x: (x > 0).mean() * 100),
                    平均差枚=('next_diff', 'mean'),
                    合計差枚=('next_diff', 'sum')
                ).reset_index()
                
                rank_order = {'85%以上': 1, '70%〜84%': 2, '50%〜69%': 3, '30%〜49%': 4, '30%未満': 5}
                sim_stats['sort'] = sim_stats['確率帯'].map(rank_order).fillna(99)
                sim_stats = sim_stats.sort_values('sort').drop('sort', axis=1)
                sim_stats['信頼度'] = sim_stats['台数'].apply(get_confidence_indicator)
                
                st.dataframe(
                    sim_stats,
                    column_config={
                        "確率帯": st.column_config.TextColumn("期待度"),
                        "台数": st.column_config.NumberColumn("台数", format="%d台", help="検証数"),
                        "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100),
                        "勝率": st.column_config.ProgressColumn("勝率(差枚)", format="%.1f%%", min_value=0, max_value=100),
                        "平均差枚": st.column_config.NumberColumn("平均", format="%+d枚", help="平均結果(差枚)"),
                        "合計差枚": st.column_config.NumberColumn("合計", format="%+d枚", help="合計収支(差枚)"),
                        "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                    },
                        width="stretch",
                    hide_index=True
                )
                
                with st.expander("🔍 シミュレーション詳細データを確認", expanded=False):
                    st.caption("シミュレーションで各確率帯に分類された台の具体的な日付と結果を確認できます。")
                    band_options_sim = ['すべて', '85%以上', '70%〜84%', '50%〜69%', '30%〜49%', '30%未満']
                    selected_band_sim = st.selectbox("表示する確率帯を選択", band_options_sim, index=1, key="sim_band_select")
                    
                    sim_display_df = sim_df.copy()
                    if selected_band_sim != 'すべて':
                        sim_display_df = sim_display_df[sim_display_df['確率帯'] == selected_band_sim]
                        
                    if sim_display_df.empty:
                        st.info("該当するデータがありません。")
                    else:
                        sim_display_df['予想設定5以上確率'] = (sim_display_df['prediction_score'] * 100).astype(int)
                        if 'next_date' in sim_display_df.columns:
                            sim_display_df['予測対象日'] = pd.to_datetime(sim_display_df['next_date'])
                        else:
                            sim_display_df['予測対象日'] = pd.to_datetime(sim_display_df['対象日付']) + pd.Timedelta(days=1)
                        
                        sim_display_df['高設定挙動'] = sim_display_df['target'].apply(lambda x: '🌟' if x == 1 else '')
                        sim_display_df = sim_display_df.sort_values('予測対象日', ascending=False)
                        
                        st.dataframe(
                            sim_display_df[['予測対象日', '台番号', '機種名', '予想設定5以上確率', '高設定挙動', 'next_diff', 'next_累計ゲーム', 'next_BIG', 'next_REG']],
                            column_config={
                                "予測対象日": st.column_config.DateColumn("予測日", format="MM/DD"),
                                "予想設定5以上確率": st.column_config.NumberColumn("期待度", format="%d%%"),
                                "高設定挙動": st.column_config.TextColumn("挙動", help="設定5以上基準を満たしたか"),
                                "next_diff": st.column_config.NumberColumn("結果差枚", format="%+d"),
                                "next_累計ゲーム": st.column_config.NumberColumn("総G数", format="%dG"),
                                "next_BIG": st.column_config.NumberColumn("BIG", format="%d"),
                                "next_REG": st.column_config.NumberColumn("REG", format="%d"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
            else:
                st.info("シミュレーション用のデータがありません。")

            st.markdown(f"**📝 過去の保存ログベースの成績 ({selected_period})**")
            st.caption("過去に予測結果を保存した時点でのスコアと、実際の結果を照合した成績です。")
            rank_stats = prob_analysis_df.groupby('確率帯').agg(
                台数=('台番号', 'count'),
                高設定率=('is_high_setting', lambda x: x.mean() * 100),
                勝率=('差枚_actual', lambda x: (x > 0).mean() * 100),
                平均差枚=('差枚_actual', 'mean'),
                合計差枚=('差枚_actual', 'sum')
            ).reset_index()
            
            # ソート順序固定
            rank_order = {'85%以上': 1, '70%〜84%': 2, '50%〜69%': 3, '30%〜49%': 4, '30%未満': 5}
            rank_stats['sort'] = rank_stats['確率帯'].map(rank_order).fillna(99)
            rank_stats = rank_stats.sort_values('sort').drop('sort', axis=1)
            
            rank_stats['信頼度'] = rank_stats['台数'].apply(get_confidence_indicator)
            st.dataframe(
                rank_stats,
                column_config={
                    "確率帯": st.column_config.TextColumn("期待度"),
                    "台数": st.column_config.NumberColumn("台数", format="%d台", help="検証数"),
                    "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100),
                    "勝率": st.column_config.ProgressColumn("勝率(差枚)", format="%.1f%%", min_value=0, max_value=100),
                    "平均差枚": st.column_config.NumberColumn("平均", format="%+d枚", help="平均結果(差枚)"),
                    "合計差枚": st.column_config.NumberColumn("合計", format="%+d枚", help="合計収支(差枚)"),
                    "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                },
                width="stretch",
                hide_index=True
            )
        
    # --- 4. AIの弱点分析 (騙された台の共通点) ---
    st.divider()
    st.subheader("🧠 AIの弱点分析 (騙された台の共通点)")
    st.caption(f"【{selected_shop}】において、AIが予測を大きく外した台の特徴を店舗平均と比較して分析します。店舗ごとのAIのクセや弱点を把握するのに役立ちます。")

    display_df = merged_df.copy() # このセクションで使うDF
    bad_pred_df = display_df[(display_df['prediction_score'] >= 0.70) & (display_df['差枚_actual'] <= -1000)].copy()
    missed_df = display_df[(display_df['prediction_score'] <= 0.40) & (display_df['差枚_actual'] >= 2000)].copy()

    if bad_pred_df.empty or missed_df.empty:
        st.info("分析に必要な「期待はずれ台」または「逃したお宝台」のサンプルが不足しています。")
    else:
        # 特徴量の定義
        features_to_analyze = {
            '累計ゲーム': "前日: 累計ゲーム",
            'REG分母': "前日: REG確率(分母)",
            '差枚': "前日: 差枚",
            'mean_7days_diff': "台: 過去7日平均差枚",
            'win_rate_7days': "台: 過去7日高設定率",
            '連続マイナス日数': "台: 連続マイナス日数",
            'neighbor_avg_diff': "配置: 両隣の平均差枚",
            'is_corner': "配置: 角台の割合",
            'event_rank_score': "イベント: ランクスコア"
        }
        
        # REG分母を計算
        for df_ in [display_df, bad_pred_df, missed_df]:
            if 'REG確率' in df_.columns:
                df_['REG分母'] = df_['REG確率'].apply(lambda x: 1/x if x > 0 else 9999)
            else:
                df_['REG分母'] = 9999

        analysis_results = []
        for f_key, f_name in features_to_analyze.items():
            if f_key in display_df.columns:
                # is_cornerは割合(%)、他は平均値
                if f_key == 'is_corner':
                    avg_all = display_df[f_key].mean() * 100
                    avg_bad = bad_pred_df[f_key].mean() * 100
                    avg_missed = missed_df[f_key].mean() * 100
                else:
                    avg_all = display_df[f_key].mean()
                    avg_bad = bad_pred_df[f_key].mean()
                    avg_missed = missed_df[f_key].mean()

                analysis_results.append({
                    "特徴": f_name,
                    "期待はずれ台": avg_bad,
                    "逃したお宝台": avg_missed,
                    "全体平均": avg_all,
                })

        if analysis_results:
            analysis_df = pd.DataFrame(analysis_results)
            
            # --- 解説の生成 ---
            explanation_bad, explanation_missed = [], []
            
            try:
                # 期待はずれ台の解説
                bad_diff = analysis_df[analysis_df['特徴'] == '前日: 差枚']['期待はずれ台'].iloc[0]
                if bad_diff > 300: explanation_bad.append("前日勝っている台の「据え置き」を期待しすぎている")
                bad_games = analysis_df[analysis_df['特徴'] == '前日: 累計ゲーム']['期待はずれ台'].iloc[0]
                all_games = analysis_df[analysis_df['特徴'] == '前日: 累計ゲーム']['全体平均'].iloc[0]
                if bad_games < all_games * 0.9: explanation_bad.append("回転数が少ない台の数値を信用しすぎている")

                # 逃したお宝台の解説
                missed_diff = analysis_df[analysis_df['特徴'] == '前日: 差枚']['逃したお宝台'].iloc[0]
                if missed_diff < -500: explanation_missed.append("前日大きく凹んだ台の「反発」を読み切れていない")
                missed_minus_days = analysis_df[analysis_df['特徴'] == '台: 連続マイナス日数']['逃したお宝台'].iloc[0]
                if missed_minus_days > 2: explanation_missed.append("連続凹み台の「上げリセット」を見逃している")
            except (IndexError, KeyError):
                pass # データが足りない場合はスキップ

            # 表示
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.markdown("##### 📉 期待はずれ台の傾向")
                if explanation_bad:
                    for item in explanation_bad: st.markdown(f"- {item}傾向があります。")
                else: st.markdown("特に目立った傾向はありません。")
            with col_exp2:
                st.markdown("##### 📈 逃したお宝台の傾向")
                if explanation_missed:
                    for item in explanation_missed: st.markdown(f"- {item}傾向があります。")
                else: st.markdown("特に目立った傾向はありません。")

            st.dataframe(
                analysis_df,
                column_config={
                    "期待はずれ台": st.column_config.NumberColumn(format="%.1f"),
                    "逃したお宝台": st.column_config.NumberColumn(format="%.1f"),
                    "全体平均": st.column_config.NumberColumn(format="%.1f"),
                },
                hide_index=True,
            width="stretch"
            )

    # --- 5. 大外れ（ワースト予測）の分析 ---
    st.divider()
    st.subheader("⚠️ AIの予測が大外れした台 (ワーストランキング)")
    st.caption("AIが高く評価したのに大きく負けてしまった台（期待はずれ）や、低評価だったのに大勝ちした台のワーストランキングです。")
    # 表示用に整理
    display_df = merged_df.copy()
    display_df['結果判定'] = display_df['差枚_actual'].apply(lambda x: 'Win 🔴' if x > 0 else 'Lose 🔵')
    display_df = display_df.sort_values('対象日付', ascending=False)
    
    if 'prediction_score' in display_df.columns:
        display_df['予想設定5以上確率'] = (display_df['prediction_score'] * 100).astype(int)
    else:
        display_df['予想設定5以上確率'] = 0

    cols = ['対象日付', shop_col, '台番号', '機種名', '予想設定5以上確率', '結果_設定5以上確率', '設定5近似度', '差枚_actual', '結果_累計ゲーム', '結果_BIG', '結果_BIG確率分母', '結果_REG', '結果_REG確率分母']
    config_dict = {
        "対象日付": st.column_config.DateColumn("予測対象日", format="MM/DD"),
        "予想設定5以上確率": st.column_config.NumberColumn("事前AI期待度", format="%d%%", help="AIが事前に予測した設定5以上の確率"),
        "結果_設定5以上確率": st.column_config.ProgressColumn("結果(事後確率)", format="%d%%", min_value=0, max_value=100, help="実際の結果(BIG/REG回数)から統計的に逆算した、本当に設定5以上だった確率"),
        "設定5近似度": st.column_config.NumberColumn("近似度スコア", format="%d点", help="設定5近似度"),
        "差枚_actual": st.column_config.NumberColumn("差枚", format="%+d"),
        "結果_累計ゲーム": st.column_config.NumberColumn("総G数", format="%dG"),
        "結果_BIG": st.column_config.NumberColumn("BIG", format="%d"),
        "結果_BIG確率分母": st.column_config.NumberColumn("B確率", format="1/%d"),
        "結果_REG": st.column_config.NumberColumn("REG", format="%d"),
        "結果_REG確率分母": st.column_config.NumberColumn("R確率", format="1/%d"),
    }

    if 'prediction_score' in merged_df.columns:
        bad_pred_df = display_df[(display_df['prediction_score'] >= 0.70) & (display_df['差枚_actual'] <= -1000)].copy()
        missed_df = display_df[(display_df['prediction_score'] <= 0.40) & (display_df['差枚_actual'] >= 2000)].copy()
        
        tab_b1, tab_b2 = st.tabs(["📉 期待はずれ (高評価で大負け)", "📈 逃したお宝台 (低評価で大勝ち)"])
        
        with tab_b1:
            if bad_pred_df.empty: st.success("現在、大きく期待を裏切った台はありません！")
            else: st.dataframe(bad_pred_df[cols].sort_values('差枚_actual'), column_config=config_dict, use_container_width=True, hide_index=True)
                
        with tab_b2:
            if missed_df.empty: st.success("現在、大きく見逃した台はありません！")
            else: st.dataframe(missed_df[cols].sort_values('差枚_actual', ascending=False), column_config=config_dict, use_container_width=True, hide_index=True)

    # --- 6. 全履歴データ (バックテスト結果) ---
    st.divider()
    st.subheader("📝 全履歴データ (バックテスト結果)")
    
    band_options = ['すべて', '85%以上', '70%〜84%', '50%〜69%', '30%〜49%', '30%未満']
    selected_band = st.selectbox("表示する期待度（確率帯）を選択", band_options, index=0)
    
    history_display_df = display_df.copy()
    if selected_band != 'すべて':
        if '確率帯' in history_display_df.columns:
            history_display_df = history_display_df[history_display_df['確率帯'] == selected_band]

    st.dataframe(
        history_display_df[cols],
        column_config=config_dict,
        width="stretch",
        hide_index=True
    )