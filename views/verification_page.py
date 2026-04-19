import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import altair as alt # type: ignore
import math

import backend
from utils import get_confidence_indicator

def render_verification_page(df_pred_log, df_verify, df_predict, df_raw):
    st.header("📊 予測の実績検証・AI設定")
    st.caption("過去の予測ログと実際の結果を照合し、同じジャンルの「日別ランキング比較」「通算成績の分析」「AIの設定チューニング」を一括で行えます。")

    shop_col = '店名' if '店名' in df_verify.columns else ('店舗名' if '店舗名' in df_verify.columns else '店名')
    
    # 共通の店舗フィルターをページ上部に配置
    shops = []
    if not df_verify.empty and shop_col in df_verify.columns:
        shops.extend(df_verify[shop_col].dropna().unique().tolist())
    if not df_pred_log.empty:
        shop_col_pred = '店名' if '店名' in df_pred_log.columns else '店舗名'
        if shop_col_pred in df_pred_log.columns:
            shops.extend(df_pred_log[shop_col_pred].dropna().unique().tolist())
            
    shops = ["店舗を選択してください"] + sorted(list(set(shops)))
    default_index = 0
    saved_shop = st.session_state.get("global_selected_shop", "店舗を選択してください")
    if saved_shop in shops:
        default_index = shops.index(saved_shop)
        
    selected_shop = st.selectbox("🏬 分析対象の店舗を選択", shops, index=default_index, key="common_verification_shop")
    
    if selected_shop != "店舗を選択してください":
        st.session_state["global_selected_shop"] = selected_shop

    if selected_shop == "店舗を選択してください":
        st.info("👆 上記のメニューから分析対象の店舗を選択してください。")
        return

    tab_rank, tab_stats, tab_setting = st.tabs([
        "🏆 日別ランキング比較", 
        "📊 AI通算成績・弱点分析", 
        "⚙️ AIモデル設定・チューニング"
    ])

    with tab_rank:
        from views import ranking_comparison_page
        ranking_comparison_page.render_ranking_comparison_page(df_pred_log, df_verify, df_predict, df_raw, selected_shop)

    with tab_stats:
        _render_verification_stats(df_pred_log, df_verify, df_predict, df_raw, tab_setting, selected_shop)

def _render_verification_stats(df_pred_log, df_verify, df_predict, df_raw, tab_setting, selected_shop):

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
        use_strict_scoring = st.checkbox("🔥 厳格な採点ルールを適用（新ロジック）", value=True, help="ONにすると「3000G以上での確率ブレ免除の半減・廃止」や「2000G以上での悪確率に対する強制減点」が適用されます。OFFにすると従来の採点ロジックに戻ります。新旧の点数変動を見比べる際に活用してください。")

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
            
            with st.expander("⚙️ プログラム＆AI設定で絞り込み", expanded=True):
                st.caption("プログラムのバージョンや、モデルの設定ごとに成績を分けて確認・比較できます。")
                
                if 'app_version' not in df_pred_log.columns:
                    df_pred_log['app_version'] = 'v1.0.0 (記録なし)'
                df_pred_log['app_version'] = df_pred_log['app_version'].replace('', 'v1.0.0 (記録なし)').fillna('v1.0.0 (記録なし)')
                
                if 'ai_version' not in df_pred_log.columns:
                    df_pred_log['ai_version'] = 'v1.0 (記録なし)'
                df_pred_log['ai_version'] = df_pred_log['ai_version'].replace('', 'v1.0 (記録なし)').fillna('v1.0 (記録なし)')
                
                c_v1, c_v2 = st.columns(2)
                
                app_versions = ['すべて'] + sorted(list(df_pred_log['app_version'].astype(str).unique()))
                selected_app_version = c_v1.selectbox("アプリバージョンで絞り込み", app_versions, index=0)
                if selected_app_version != 'すべて':
                    df_pred_log = df_pred_log[df_pred_log['app_version'] == selected_app_version]

                ai_versions = ['すべて'] + sorted(list(df_pred_log['ai_version'].astype(str).unique()))
                selected_version = c_v2.selectbox("AI設定(パラメータ)で絞り込み", ai_versions, index=0)
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
    
    if '実行日時' in df_pred_log.columns:
        df_pred_log['実行日時'] = pd.to_datetime(df_pred_log['実行日時'], errors='coerce')
        # 同じタイミング（実行日）で保存された予測は、最も新しい日付の翌日にすべて統一する（日付の散らばり防止）
        max_dates = df_pred_log.groupby(df_pred_log['実行日時'].dt.date)['対象日付'].transform('max')
        fallback_pred_date = max_dates + pd.Timedelta(days=1)
    else:
        fallback_pred_date = df_pred_log['対象日付'] + pd.Timedelta(days=1)
        
    if '予測対象日' in df_pred_log.columns:
        df_pred_log['予測対象日_merge'] = df_pred_log['予測対象日'].fillna(fallback_pred_date)
    else:
        df_pred_log['予測対象日_merge'] = fallback_pred_date
        
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
    if 'prediction_score' in base_df.columns:
        base_df['prediction_score'] = pd.to_numeric(base_df['prediction_score'], errors='coerce')
        
    if '予測差枚数_saved' in base_df.columns:
        base_df['予測差枚数'] = base_df['予測差枚数_saved']
    if '予測差枚数' in base_df.columns:
        base_df['予測差枚数'] = pd.to_numeric(base_df['予測差枚数'], errors='coerce')
    if '機種名_saved' in base_df.columns:
        base_df['機種名'] = base_df['機種名_saved']
    if '予測信頼度_saved' in base_df.columns:
        base_df['予測信頼度'] = base_df['予測信頼度_saved']
    elif '予測信頼度_current' in base_df.columns:
        base_df['予測信頼度'] = base_df['予測信頼度_current']
    else:
        base_df['予測信頼度'] = "-"
        
    if 'ai_version_saved' in base_df.columns:
        base_df['ai_version'] = base_df['ai_version_saved']
    elif 'ai_version_current' in base_df.columns:
        base_df['ai_version'] = base_df['ai_version_current']
        
    if 'app_version_saved' in base_df.columns:
        base_df['app_version'] = base_df['app_version_saved']
    elif 'app_version_current' in base_df.columns:
        base_df['app_version'] = base_df['app_version_current']

    # --- 実際の成績を df_raw から直接取得 (未稼働台のデータ落ちを防ぐ) ---
    if '台番号' in df_raw.columns:
        df_raw_temp = df_raw.copy()
        df_raw_temp['台番号'] = df_raw_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_raw_temp['対象日付'] = pd.to_datetime(df_raw_temp['対象日付'], errors='coerce')
        
        # 実績データ側の店舗カラム名を統一 (紐付け失敗を防ぐ)
        raw_shop_col = '店名' if '店名' in df_raw_temp.columns else ('店舗名' if '店舗名' in df_raw_temp.columns else None)
        if raw_shop_col and raw_shop_col != shop_col:
            df_raw_temp = df_raw_temp.rename(columns={raw_shop_col: shop_col})
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
            if orig_col not in ['prediction_score', '予測差枚数', '機種名', '予測信頼度', 'おすすめ度', '根拠', 'next_diff', 'next_BIG', 'next_REG', 'next_累計ゲーム', 'ai_version', 'app_version']:
                base_df[orig_col] = base_df[col]

    base_df = base_df.dropna(subset=['差枚_actual', 'prediction_score']).copy()
    
    if base_df.empty:
        st.info("まだ結果が判明している予測がありません。")
        return

    shop_avg_g_dict = base_df.groupby(shop_col)['結果_累計ゲーム'].mean().to_dict()

    # --- 設定5近似度の算出 (ボーナス回数の精査) ---
    def evaluate_setting5(row):
        g = row.get('結果_累計ゲーム', 0)
        act_b = row.get('結果_BIG', 0)
        act_r = row.get('結果_REG', 0)
        diff = row.get('差枚_actual', 0)
        machine = row.get('機種名', '')
        
        s_name = row.get(shop_col, '')
        shop_avg_g = shop_avg_g_dict.get(s_name, 4000)
        score, exp_b, exp_r, diff_b, diff_r = backend.calculate_setting_score(
            g=g, act_b=act_b, act_r=act_r, machine_name=machine, diff=diff, shop_avg_g=shop_avg_g,
            penalty_reg=penalty_reg, penalty_big=penalty_big, low_g_penalty=low_g_penalty,
            use_strict_scoring=use_strict_scoring, return_details=True
        )
        return pd.Series([score, exp_b, exp_r, diff_b, diff_r])
        
    eval_df = base_df.apply(evaluate_setting5, axis=1)
    base_df[['設定5近似度', '期待BIG', '期待REG', 'BIG不足分', 'REG不足分']] = eval_df

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

    st.divider() # タブ外のデータ準備からUIに戻る区切り
    if shop_col not in base_df.columns:
        st.warning("店舗データがありません。")
        return

    merged_df = base_df[base_df[shop_col] == selected_shop].copy()
    st.subheader(f"📊 AIモデル バックテスト通算成績 ({selected_shop} / {selected_version})")
    
    # --- 有効稼働フラグの追加 ---
    merged_df['valid_play'] = (pd.to_numeric(merged_df['結果_累計ゲーム'], errors='coerce').fillna(0) >= 3000) | \
                              ((pd.to_numeric(merged_df['結果_累計ゲーム'], errors='coerce').fillna(0) < 3000) & \
                               ((pd.to_numeric(merged_df['差枚_actual'], errors='coerce').fillna(0) <= -750) | (pd.to_numeric(merged_df['差枚_actual'], errors='coerce').fillna(0) >= 750)))
    merged_df['valid_win'] = merged_df['valid_play'] & (pd.to_numeric(merged_df['差枚_actual'], errors='coerce').fillna(0) > 0)
    
    specs = backend.get_machine_specs()
    spec_reg_val = merged_df['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
    spec_tot_val = merged_df['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
    m_cum_g = pd.to_numeric(merged_df['結果_累計ゲーム'], errors='coerce').fillna(0)
    m_big_c = pd.to_numeric(merged_df['結果_BIG'], errors='coerce').fillna(0)
    m_reg_c = pd.to_numeric(merged_df['結果_REG'], errors='coerce').fillna(0)
    merged_df['結果_合算確率分母'] = np.where((m_big_c + m_reg_c) > 0, m_cum_g / (m_big_c + m_reg_c), 0)
    merged_df['is_high_setting'] = (((merged_df['結果_REG確率分母'] > 0) & (merged_df['結果_REG確率分母'] <= spec_reg_val)) | ((merged_df['結果_合算確率分母'] > 0) & (merged_df['結果_合算確率分母'] <= spec_tot_val))).astype(int)
    merged_df['valid_high_play'] = m_cum_g >= 3000
    merged_df['valid_high'] = merged_df['valid_high_play'] & (merged_df['is_high_setting'] == 1)

    merged_df['valid_設定5近似度'] = np.where(merged_df['valid_play'], merged_df['設定5近似度'], np.nan)
    merged_df['valid_差枚_actual'] = np.where(merged_df['valid_play'], merged_df['差枚_actual'], np.nan)

    # REG確率の計算をここで行う
    merged_df['結果_REG確率_val'] = np.where(pd.to_numeric(merged_df['結果_累計ゲーム'], errors='coerce').fillna(0) > 0, pd.to_numeric(merged_df['結果_REG'], errors='coerce').fillna(0) / pd.to_numeric(merged_df['結果_累計ゲーム'], errors='coerce').fillna(0), 0)
    merged_df['valid_REG確率'] = np.where(merged_df['valid_play'], merged_df['結果_REG確率_val'], np.nan)

    # 合算確率の計算を追加
    merged_df['結果_合算確率_val'] = np.where(pd.to_numeric(merged_df['結果_累計ゲーム'], errors='coerce').fillna(0) > 0, (pd.to_numeric(merged_df['結果_BIG'], errors='coerce').fillna(0) + pd.to_numeric(merged_df['結果_REG'], errors='coerce').fillna(0)) / pd.to_numeric(merged_df['結果_累計ゲーム'], errors='coerce').fillna(0), 0)
    merged_df['valid_合算確率'] = np.where(merged_df['valid_play'], merged_df['結果_合算確率_val'], np.nan)

    # 保存されている予測ログはすでに「各店舗の上位10%」に絞られているため、そのまま使用する
    ai_recom_df = merged_df.copy()

    if merged_df.empty:
        st.info("選択された店舗の分析データがありません。")
        return

    # --- バージョン別成績比較表 (「すべて」選択時のみ表示) ---
    if selected_version == 'すべて' and selected_app_version == 'すべて' and 'ai_version' in merged_df.columns:
        st.markdown("##### 🔍 バージョン別 成績比較")
        st.caption("過去に試したプログラムバージョンやAI設定ごとの成績一覧です。どの組み合わせが最も優秀だったかを比較できます。")
        ver_df = ai_recom_df.copy() # 変更を元のDFに影響させない
        ver_df['比較用バージョン'] = ver_df['app_version'].astype(str) + " | " + ver_df['ai_version'].astype(str)
        ver_stats = ver_df.groupby('比較用バージョン').agg(
            検証台数=('台番号', 'count'),
            高設定数=('valid_high', 'sum'),
            高設定有効数=('valid_high_play', 'sum'),
            有効稼働数=('valid_play', 'sum'),
            勝数=('valid_win', 'sum'),
            平均差枚=('valid_差枚_actual', 'mean'),
            設定5近似度=('valid_設定5近似度', 'mean'),
            平均期待度=('prediction_score', 'mean'),
            平均REG確率=('valid_REG確率', 'mean')
        ).reset_index().sort_values('設定5近似度', ascending=False)
        ver_stats['勝率'] = np.where(ver_stats['有効稼働数'] > 0, ver_stats['勝数'] / ver_stats['有効稼働数'] * 100, 0.0)
        ver_stats['高設定率'] = np.where(ver_stats['高設定有効数'] > 0, ver_stats['高設定数'] / ver_stats['高設定有効数'] * 100, 0.0)
        ver_stats['平均期待度'] = ver_stats['平均期待度'] * 100
        ver_stats['REG確率'] = ver_stats['平均REG確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
        
        st.dataframe(
            ver_stats[['比較用バージョン', '平均期待度', '検証台数', '有効稼働数', '高設定率', '勝率', '平均差枚', '設定5近似度', 'REG確率']],
            column_config={
                "比較用バージョン": st.column_config.TextColumn("アプリ | AI設定"),
                "平均期待度": st.column_config.ProgressColumn("平均期待度", format="%.1f%%", min_value=0, max_value=100),
                "検証台数": st.column_config.NumberColumn("台数", format="%d台"),
                "有効稼働数": st.column_config.NumberColumn("有効稼働", format="%d台"),
                "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100),
                "勝率": st.column_config.ProgressColumn("勝率(差枚)", format="%.1f%%", min_value=0, max_value=100),
                "平均差枚": st.column_config.NumberColumn("平均差枚", format="%+d枚"),
                "設定5近似度": st.column_config.NumberColumn("平均5近似度", format="%.1f点"),
                "REG確率": st.column_config.TextColumn("平均REG確率"),
            },
            hide_index=True,
            width="stretch"
        )

    # --- 1. 全体成績 (KPI) & 円グラフ ---
    total_count = len(ai_recom_df)
    valid_count = ai_recom_df['valid_play'].sum()
    high_valid_count = ai_recom_df['valid_high_play'].sum()
    high_set_count = ai_recom_df['valid_high'].sum()
    low_set_count = high_valid_count - high_set_count
    high_setting_rate = high_set_count / high_valid_count if high_valid_count > 0 else 0
    win_count = ai_recom_df['valid_win'].sum()
    win_rate = win_count / valid_count if valid_count > 0 else 0
    avg_diff = ai_recom_df['valid_差枚_actual'].mean()
    total_diff = ai_recom_df['差枚_actual'].sum()
    
    if pd.isna(avg_diff): avg_diff = 0
    if pd.isna(total_diff): total_diff = 0
    
    col_kpi, col_pie = st.columns([2, 1])
    
    with col_kpi:
        k1, k2, k5 = st.columns(3)
        k1.metric("検証台数", f"{total_count} 台")
        k2.metric("高設定率", f"{high_setting_rate:.1%}")
        k5.metric("勝率(有効稼働)", f"{win_rate:.1%}")
        
        k3, k4 = st.columns(2)
        k3.metric("平均差枚", f"{int(avg_diff):+d} 枚")
        k4.metric("合計収支", f"{int(total_diff):+d} 枚")
        
    with col_pie:
        # 勝敗円グラフ (Altair)
        pie_data = pd.DataFrame({
            'Category': ['高設定', '低設定'],
            'Count': [high_set_count, low_set_count]
        })
        
        if valid_count > 0:
            pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=35).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Category", type="nominal", 
                                scale=alt.Scale(domain=['高設定', '低設定'], range=['#FF4B4B', '#4B4BFF']),
                                legend=alt.Legend(title="設定挙動", orient="bottom")),
                tooltip=['Category', 'Count']
            ).properties(height=200)
            st.altair_chart(pie_chart, width="stretch")
        else:
            st.caption("※有効稼働データなし")

    # ユーザーの要望通り、ごちゃごちゃした結合を廃止し、最もシンプルに「予測した日付」の「実績差枚」をぶつける
    ai_recom_df['日付キー'] = pd.to_datetime(ai_recom_df['対象日付']).dt.normalize()

    # --- 1. 店舗全体の実際の実績差枚を取得 ---
    if not df_raw_temp.empty:
        shop_raw_temp = df_raw_temp[df_raw_temp[shop_col] == selected_shop].copy()
        shop_raw_temp['日付キー'] = pd.to_datetime(shop_raw_temp['対象日付']).dt.normalize()
        shop_daily_actual = shop_raw_temp.groupby('日付キー').agg(
            店舗全体平均差枚=('差枚', 'mean')
        ).reset_index()
        ai_recom_df = pd.merge(ai_recom_df, shop_daily_actual, on='日付キー', how='left')
    else:
        ai_recom_df['店舗全体平均差枚'] = np.nan
        
    # --- 2. 実績データがない（未来の）予測ログは検証から除外する ---
    ai_recom_df = ai_recom_df.dropna(subset=['店舗全体平均差枚']).copy()

    # --- 3. 営業区分の判定 (絶対値でシンプルに) ---
    def determine_actual_shop_eval(row):
        actual_diff = row.get('店舗全体平均差枚')
        if pd.isna(actual_diff): return "⚖️ 通常営業"
        if actual_diff > 0: return "🔥 還元日"
        elif actual_diff < 0: return "🥶 回収日"
        else: return "⚖️ 通常営業"

    ai_recom_df['営業区分'] = ai_recom_df.apply(determine_actual_shop_eval, axis=1)
    
    ai_recom_df = ai_recom_df.drop(columns=['日付キー', '予測対象日'], errors='ignore')

    # 日別推移データをAI評価より先に計算する
    daily_stats = ai_recom_df.groupby(['対象日付', '営業区分']).agg(
        high_setting_count=('valid_high', 'sum'),
        high_valid_count=('valid_high_play', 'sum'),
        valid_count=('valid_play', 'sum'),
        total_profit=('差枚_actual', 'sum'),
        avg_s5_score=('valid_設定5近似度', 'mean'),
        count=('台番号', 'count')
    ).reset_index().sort_values('対象日付')
    daily_stats['high_setting_rate'] = np.where(daily_stats['high_valid_count'] > 0, daily_stats['high_setting_count'] / daily_stats['high_valid_count'], 0.0)

    # --- 還元日/回収日 予測別のAI成績 ---
    # --- 還元日/回収日 別の成績 ---
    st.divider()
    st.subheader("🗓️ 還元日 / 回収日 別の成績")
    st.caption("AIが判断した「予測」ベースか、実際の店舗の差枚による「結果」ベースかで、推奨台の成績を振り分けて確認できます。")
    st.caption("実際の店舗全体の平均差枚から「還元日」「通常営業」「回収日」と判定した日ごとの、AI推奨台の成績です。店舗が実際に還元している日にAIの推奨台がどれだけ勝てているかが確認できます。")
    
    eval_base = st.radio("集計の基準", ["🎯 予測ベース (AIが回収日を警戒できているかの確認用)", "📊 結果ベース (実際の還元日にどれだけ勝てたかの確認用)"], index=1, horizontal=True)
    group_col = '予測営業区分' if '予測' in eval_base else '実際営業区分'
    group_col = '実際営業区分'
    
    day_type_stats = ai_recom_df.groupby(group_col).agg(
        検証日数=('対象日付', 'nunique'),
        検証台数=('台番号', 'count'),
        有効稼働数=('valid_play', 'sum'),
        高設定数=('valid_high', 'sum'),
        高設定有効数=('valid_high_play', 'sum'),
        勝数=('valid_win', 'sum'),
        推奨台平均差枚=('valid_差枚_actual', 'mean'),
        平均設定5近似度=('valid_設定5近似度', 'mean'),
        平均REG確率=('valid_REG確率', 'mean')
    ).reset_index()
    day_type_stats['推奨台勝率'] = np.where(day_type_stats['有効稼働数'] > 0, day_type_stats['勝数'] / day_type_stats['有効稼働数'] * 100, 0.0)
    day_type_stats['推奨台高設定率'] = np.where(day_type_stats['高設定有効数'] > 0, day_type_stats['高設定数'] / day_type_stats['高設定有効数'] * 100, 0.0)
    day_type_stats['REG確率'] = day_type_stats['平均REG確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
    
    # 店舗全体の実際の平均差枚を集計して結合 (表示期間の絞り込みを反映するため ai_recom_df から日別にユニーク化して計算)
    day_eval_summary = ai_recom_df[['対象日付', group_col, '店舗全体平均差枚']].drop_duplicates().groupby(group_col).agg(店舗全体平均差枚=('店舗全体平均差枚', 'mean')).reset_index()
    day_type_stats = pd.merge(day_type_stats, day_eval_summary, on=group_col, how='left')
    
    day_order_pred = {"🔥 還元日予測": 1, "⚖️ 通常営業予測": 2, "🥶 回収日予測": 3}
    day_order_act = {"🔥 還元日": 1, "⚖️ 通常営業": 2, "🥶 回収日": 3}
    day_order = day_order_pred if '予測' in eval_base else day_order_act
    
    day_type_stats['sort'] = day_type_stats[group_col].map(day_order).fillna(99)
    day_type_stats = day_type_stats.sort_values('sort').drop('sort', axis=1)
    
    st.dataframe(
        day_type_stats[[group_col, '検証日数', '検証台数', '有効稼働数', '推奨台高設定率', '推奨台勝率', '推奨台平均差枚', '店舗全体平均差枚', '平均設定5近似度', 'REG確率']],
        column_config={
            group_col: st.column_config.TextColumn("営業区分"),
            "検証日数": st.column_config.NumberColumn("日数", format="%d日"),
            "検証台数": st.column_config.NumberColumn("推奨台数", format="%d台"),
            "有効稼働数": st.column_config.NumberColumn("有効稼働", format="%d台"),
            "推奨台高設定率": st.column_config.ProgressColumn("推奨台 高設定率", format="%.1f%%", min_value=0, max_value=100),
            "推奨台勝率": st.column_config.ProgressColumn("推奨台 勝率", format="%.1f%%", min_value=0, max_value=100),
            "推奨台平均差枚": st.column_config.NumberColumn("推奨台 平均差枚", format="%+d枚"),
            "店舗全体平均差枚": st.column_config.NumberColumn("店舗全体 平均差枚", format="%+d枚", help="その日の店舗全体の実際の平均差枚です。"),
            "平均設定5近似度": st.column_config.NumberColumn("平均5近似度", format="%.1f点"),
            "REG確率": st.column_config.TextColumn("平均REG確率")
        },
        hide_index=True,
        width="stretch"
    )

    # --- 推奨台のカテゴリ別成績 (機種・末尾・島) ---
    st.divider()
    st.subheader("🎰 推奨台のカテゴリ別成績 (機種・末尾・島)")
    st.caption("AIが推奨した台（店舗上位約10%）の、機種ごと・末尾ごと・島（列）ごとの実際の成績です。AIがどのカテゴリを的確に予測できているかが分かります。")
    
    if not ai_recom_df.empty:
        ai_recom_df['末尾番号'] = ai_recom_df['台番号'].astype(str).str[-1]
        tab_mac, tab_end, tab_isl = st.tabs(["🎰 機種別", "🔢 末尾番号別", "🏝️ 島(列)別"])
        
        with tab_mac:
            mac_stats = ai_recom_df.groupby('機種名').agg(
                推奨台数=('台番号', 'count'),
                有効稼働数=('valid_play', 'sum'),
                勝数=('valid_win', 'sum'),
                平均差枚=('valid_差枚_actual', 'mean'),
                合計差枚=('差枚_actual', 'sum'),
                平均REG確率=('valid_REG確率', 'mean')
            ).reset_index()
            mac_stats['勝率'] = np.where(mac_stats['有効稼働数'] > 0, mac_stats['勝数'] / mac_stats['有効稼働数'] * 100, 0.0)
            mac_stats['REG確率'] = mac_stats['平均REG確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
            mac_stats = mac_stats.sort_values('合計差枚', ascending=False)
            
            st.dataframe(
                mac_stats[['機種名', '推奨台数', '有効稼働数', '勝率', '平均差枚', '合計差枚', 'REG確率']],
                column_config={
                    "機種名": st.column_config.TextColumn("機種名"),
                    "推奨台数": st.column_config.NumberColumn("推奨台数", format="%d台"),
                    "有効稼働数": st.column_config.NumberColumn("有効稼働", format="%d台"),
                    "勝率": st.column_config.ProgressColumn("勝率(有効稼働)", format="%.1f%%", min_value=0, max_value=100),
                    "平均差枚": st.column_config.NumberColumn("平均差枚", format="%+d枚"),
                    "合計差枚": st.column_config.NumberColumn("合計差枚", format="%+d枚"),
                    "REG確率": st.column_config.TextColumn("平均REG確率"),
                },
                hide_index=True,
                use_container_width=True
            )
            
        with tab_end:
            end_stats = ai_recom_df.groupby('末尾番号').agg(
                推奨台数=('台番号', 'count'),
                有効稼働数=('valid_play', 'sum'),
                勝数=('valid_win', 'sum'),
                平均差枚=('valid_差枚_actual', 'mean'),
                合計差枚=('差枚_actual', 'sum'),
                平均REG確率=('valid_REG確率', 'mean')
            ).reset_index()
            end_stats['勝率'] = np.where(end_stats['有効稼働数'] > 0, end_stats['勝数'] / end_stats['有効稼働数'] * 100, 0.0)
            end_stats['REG確率'] = end_stats['平均REG確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
            end_stats = end_stats.sort_values('末尾番号')
            
            st.dataframe(
                end_stats[['末尾番号', '推奨台数', '有効稼働数', '勝率', '平均差枚', '合計差枚', 'REG確率']],
                column_config={
                    "末尾番号": st.column_config.TextColumn("末尾番号"),
                    "推奨台数": st.column_config.NumberColumn("推奨台数", format="%d台"),
                    "有効稼働数": st.column_config.NumberColumn("有効稼働", format="%d台"),
                    "勝率": st.column_config.ProgressColumn("勝率(有効稼働)", format="%.1f%%", min_value=0, max_value=100),
                    "平均差枚": st.column_config.NumberColumn("平均差枚", format="%+d枚"),
                    "合計差枚": st.column_config.NumberColumn("合計差枚", format="%+d枚"),
                    "REG確率": st.column_config.TextColumn("平均REG確率"),
                },
                hide_index=True,
                use_container_width=True
            )

        with tab_isl:
            if 'island_id' in ai_recom_df.columns:
                isl_df = ai_recom_df[ai_recom_df['island_id'] != "Unknown"].copy()
                if not isl_df.empty:
                    isl_df['島名'] = isl_df['island_id'].apply(lambda x: str(x).split('_', 1)[1] if '_' in str(x) else str(x))
                    isl_stats = isl_df.groupby('島名').agg(
                        推奨台数=('台番号', 'count'),
                        有効稼働数=('valid_play', 'sum'),
                        勝数=('valid_win', 'sum'),
                        平均差枚=('valid_差枚_actual', 'mean'),
                        合計差枚=('差枚_actual', 'sum'),
                        平均REG確率=('valid_REG確率', 'mean')
                    ).reset_index()
                    isl_stats['勝率'] = np.where(isl_stats['有効稼働数'] > 0, isl_stats['勝数'] / isl_stats['有効稼働数'] * 100, 0.0)
                    isl_stats['REG確率'] = isl_stats['平均REG確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
                    isl_stats = isl_stats.sort_values('合計差枚', ascending=False)
                    
                    st.dataframe(
                        isl_stats[['島名', '推奨台数', '有効稼働数', '勝率', '平均差枚', '合計差枚', 'REG確率']],
                        column_config={
                            "島名": st.column_config.TextColumn("島名"),
                            "推奨台数": st.column_config.NumberColumn("推奨台数", format="%d台"),
                            "有効稼働数": st.column_config.NumberColumn("有効稼働", format="%d台"),
                            "勝率": st.column_config.ProgressColumn("勝率(有効稼働)", format="%.1f%%", min_value=0, max_value=100),
                            "平均差枚": st.column_config.NumberColumn("平均差枚", format="%+d枚"),
                            "合計差枚": st.column_config.NumberColumn("合計差枚", format="%+d枚"),
                            "REG確率": st.column_config.TextColumn("平均REG確率"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("島マスターに登録された台の推奨データがありません。")
            else:
                st.info("島データがありません。事前にサイドバーの「島マスター管理」から島を登録してください。")

    # --- AI振り返りレポート ---
    st.divider()
    st.subheader("🤖 AIの振り返りレポート (過去の自分との比較)")
    st.caption("最新の予測結果（設定5近似度）を過去の平均的なパフォーマンスと比較し、AIが自身の成長や調子を分析します。")
    
    avg_s5_score = ai_recom_df['valid_設定5近似度'].mean() if not ai_recom_df.empty else 0
    if pd.isna(avg_s5_score): avg_s5_score = 0
    
    avg_g = ai_recom_df.loc[ai_recom_df['valid_play'], '結果_累計ゲーム'].mean()
    avg_diff_r = ai_recom_df.loc[ai_recom_df['valid_play'], 'REG不足分'].mean()
    avg_diff_b = ai_recom_df.loc[ai_recom_df['valid_play'], 'BIG不足分'].mean()
    
    if pd.isna(avg_diff_r): avg_diff_r = 0
    if pd.isna(avg_diff_b): avg_diff_b = 0
    
    # データ不足の考慮 (検証台数不足 or 学習データ不足)
    low_rel_count = (ai_recom_df['予測信頼度'] == '🔻低').sum() if '予測信頼度' in ai_recom_df.columns else 0
    mid_rel_count = (ai_recom_df['予測信頼度'] == '🔸中').sum() if '予測信頼度' in ai_recom_df.columns else 0
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
        overall_comment = f"期間中の全推奨台の平均回転数が **{int(avg_g if not pd.isna(avg_g) else 0)}G** と少なく、試行回数不足です。もう少し稼働がある状況で検証したいですね。"
        mood = "🤔"
    elif avg_s5_score >= 80:
        if eval_mode_str == "甘め":
            overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** と大成功レベルです！（※甘め評価のためBIGのヒキ強も含まれます）"
        elif eval_mode_str == "辛め":
            overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** と大成功レベルです！辛め評価でもこの点数は、文句なしの**本物の高設定**を的確に見抜けています！"
        else:
            overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** と大成功レベルです！本物の高設定を的確に見抜けています！"
        mood = "🌟"
    elif avg_s5_score >= 60:
        if avg_diff_r < 0:
            if eval_mode_str == "甘め":
                overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** でまずまずです。REGが平均 {abs(avg_diff_r):.1f}回 不足していますが、甘め評価（出玉重視）としては合格点です。"
            else:
                overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** でまずまずですが、REGが平均 {abs(avg_diff_r):.1f}回 不足しています。低〜中間設定の上振れに助けられている部分もありそうです。"
        else:
            if eval_mode_str == "辛め":
                overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** です。辛め評価の中では健闘しており、中身もしっかり高設定挙動を示しています。"
            else:
                overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** で優秀です！中身もしっかり高設定挙動を示しています。"
        mood = "👍"
    elif avg_s5_score >= 40:
        if avg_g < 4000:
            overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** と惜しい結果です。平均回転数が {int(avg_g)}G と少なめなので、下振れの可能性もあります。"
        else:
            if eval_mode_str == "甘め":
                overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** と反省点が残ります。甘め評価でこの点数なので、低設定の誤爆だった可能性が高いです。"
            elif eval_mode_str == "辛め":
                overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** です。辛め評価なので点数が低く出やすくなっていますが、もう少しREGが欲しいところです。"
            else:
                overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** と反省点が残ります。低〜中間設定が混ざっている可能性が高いです。"
        mood = "💦"
    else:
        if eval_mode_str == "辛め":
             overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** と惨敗です…。辛め評価であることを考慮しても厳しい結果です。設定状況が変わった可能性があります。"
        else:
             overall_comment = f"期間中の全推奨台の平均結果点数は **{avg_s5_score:.1f}点 / 100点** と惨敗です…。過去の傾向が変わった可能性があるので、最近のデータで学習し直すか、店選びを見直す余地があります。"
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
             comparison_comment = f"最新の予測日 ({latest_date_str}) の結果点数は **{latest_score:.1f}点 / 100点** でした。比較対象となる過去の検証台数が少なく（計{int(total_past_count)}台）、AIの成長や調子を正しく評価するにはまだデータが不足しています。"
             mood_cmp = "🤔"
        elif latest_stat['count'] < 5:
             comparison_comment = f"最新の予測日 ({latest_date_str}) の結果点数は **{latest_score:.1f}点 / 100点** でした。直近の平均 ({past_avg_score:.1f}点) と比較したいところですが、今回の検証台数が{int(latest_stat['count'])}台と少ないため、たまたまのブレが大きい可能性があります。"
             mood_cmp = "🤔"
        elif score_diff >= 5:
             comparison_comment = f"最新の予測日 ({latest_date_str}) の結果点数は **{latest_score:.1f}点 / 100点** でした！直近の平均 ({past_avg_score:.1f}点) より **{score_diff:+.1f}点** も向上しており、予測精度が上がっています！日々学習して賢くなっているのを感じますね！"
             mood_cmp = "📈"
        elif score_diff <= -5:
             comparison_comment = f"最新の予測日 ({latest_date_str}) の結果点数は **{latest_score:.1f}点 / 100点** でした…。直近の平均 ({past_avg_score:.1f}点) より **{score_diff:+.1f}点** 下がっています。少し調子を落としているか、お店の設定配分のクセが変わった（フェイクが増えた等）可能性があります。"
             mood_cmp = "📉"
        else:
             comparison_comment = f"最新の予測日 ({latest_date_str}) の結果点数は **{latest_score:.1f}点 / 100点** でした。直近の平均 ({past_avg_score:.1f}点) とほぼ同水準をキープしており、安定した予測ができています。"
             mood_cmp = "⚖️"

    if has_comparison:
        final_comment = f"{mood_cmp} **最近の調子 (直近14日間との比較):**\n{comparison_comment}\n\n{mood} **期間全体の総評 (上のKPI表のデータに基づく):**\n{overall_comment}"
    else:
        final_comment = f"{mood} **期間全体の総評 (上のKPI表のデータに基づく):**\n{overall_comment}\n\n※比較対象となる過去の推移データが不足しています。"

    st.info(f"{comment_prefix}{final_comment}\n\n※全体平均 REG過不足: **{avg_diff_r:+.1f}回** / BIG過不足: **{avg_diff_b:+.1f}回**")
    
    # --- 特に優秀だった台トップ3 ---
    if not ai_recom_df.empty:
        top3_df = ai_recom_df[ai_recom_df['設定5近似度'] > 0].sort_values('設定5近似度', ascending=False).head(3)
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
                date_str = row['対象日付'].strftime('%m/%d') if pd.notna(row.get('対象日付')) else ""
                st.markdown(f"- **{score:.1f}点** : {date_str} {shop} #{m_num} {m_name} ({int(g)}G BIG{int(b)} REG{int(r)} / {int(diff):+d}枚)")

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
            if '実行日時' in ai_recom_df.columns:
                ai_recom_df['実行日時'] = pd.to_datetime(ai_recom_df['実行日時'], errors='coerce')
                ai_recom_df['実行日'] = ai_recom_df['実行日時'].dt.date
                exec_stats = ai_recom_df.groupby('実行日').agg(
                    avg_s5_score=('設定5近似度', 'mean'),
                    high_setting_count=('valid_high', 'sum'),
                    valid_count=('valid_play', 'sum'),
                    count=('台番号', 'count')
                ).reset_index().dropna(subset=['実行日'])
                exec_stats['high_setting_rate'] = np.where(exec_stats['valid_count'] > 0, exec_stats['high_setting_count'] / exec_stats['valid_count'], 0.0)
                
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
            if score >= 0.50: return '50%以上'
            elif score >= 0.40: return '40%〜49%'
            elif score >= 0.30: return '30%〜39%'
            elif score >= 0.20: return '20%〜29%'
            elif score >= 0.15: return '15%〜19%'
            elif score >= 0.10: return '10%〜14%'
            else: return '10%未満'
            
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
            # 営業区分の付与
            if '対象日付' in prob_analysis_df.columns and not ai_recom_df.empty and '営業区分' in ai_recom_df.columns:
                prob_analysis_df['対象日付_merge_key'] = pd.to_datetime(prob_analysis_df['対象日付']).dt.normalize()
                day_eval_map = ai_recom_df[['対象日付', '営業区分']].drop_duplicates()
                day_eval_map['対象日付_merge_key'] = pd.to_datetime(day_eval_map['対象日付']).dt.normalize()
                prob_analysis_df = pd.merge(prob_analysis_df, day_eval_map[['対象日付_merge_key', '営業区分']], on='対象日付_merge_key', how='left')
                prob_analysis_df = prob_analysis_df.drop(columns=['対象日付_merge_key'], errors='ignore')
                prob_analysis_df['営業区分'] = prob_analysis_df['営業区分'].fillna("⚖️ 通常営業")
            else:
                prob_analysis_df['営業区分'] = "⚖️ 通常営業"

            # --- 期待度スコアのヒストグラム ---
            st.markdown(f"**📊 期待度スコア (予測スコア) の分布と正解率 ({selected_period})**")
            
            # 5%刻みの代表値(左端)に丸めて集計
            prob_analysis_df['score_bin_left'] = (prob_analysis_df['prediction_score'] // 0.05) * 0.05
            hist_stats = prob_analysis_df.groupby('score_bin_left').agg(
                count=('台番号', 'count'),
                high_setting_count=('valid_high', 'sum'),
                high_valid_count=('valid_high_play', 'sum'),
                valid_count=('valid_play', 'sum')
            ).reset_index()
            hist_stats['high_setting_rate'] = np.where(hist_stats['high_valid_count'] > 0, hist_stats['high_setting_count'] / hist_stats['high_valid_count'], 0.0)
            
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
            
            st.markdown(f"**📝 過去のガチ予測ログベースの成績 ({selected_period})**")
            st.caption("過去にAIが予測結果を保存した時点でのスコアと、実際の結果を照合した『本物の実戦成績』です。")
            
            tabs_1 = st.tabs(["全体", "🔥 還元日", "⚖️ 通常営業", "🥶 回収日"])
            
            def render_rank_stats_top(target_df):
                if target_df.empty:
                    st.info("該当するデータがありません。")
                    return
                r_stats = target_df.groupby('確率帯').agg(
                    台数=('台番号', 'count'),
                    高設定数=('valid_high', 'sum'),
                    高設定有効数=('valid_high_play', 'sum'),
                    有効稼働数=('valid_play', 'sum'),
                    勝数=('valid_win', 'sum'),
                    平均差枚=('valid_差枚_actual', 'mean'),
                    合計差枚=('差枚_actual', 'sum'),
                    平均期待度=('prediction_score', 'mean'),
                    平均REG確率=('valid_REG確率', 'mean'),
                    平均合算確率=('valid_合算確率', 'mean')
                ).reset_index()
                r_stats['勝率'] = np.where(r_stats['有効稼働数'] > 0, r_stats['勝数'] / r_stats['有効稼働数'] * 100, 0.0)
                r_stats['高設定率'] = np.where(r_stats['高設定有効数'] > 0, r_stats['高設定数'] / r_stats['高設定有効数'] * 100, 0.0)
                r_stats['平均期待度'] = r_stats['平均期待度'] * 100
                r_stats['REG確率'] = r_stats['平均REG確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
                r_stats['合算確率'] = r_stats['平均合算確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
                
                rank_order = {'50%以上': 1, '40%〜49%': 2, '30%〜39%': 3, '20%〜29%': 4, '15%〜19%': 5, '10%〜14%': 6, '10%未満': 7}
                r_stats['sort'] = r_stats['確率帯'].map(rank_order).fillna(99)
                r_stats = r_stats.sort_values('sort').drop('sort', axis=1)
                
                r_stats['信頼度'] = r_stats['台数'].apply(get_confidence_indicator)
                st.dataframe(
                    r_stats[['確率帯', '平均期待度', '台数', '有効稼働数', '高設定率', '勝率', '平均差枚', '合計差枚', 'REG確率', '合算確率', '信頼度']],
                    column_config={
                        "確率帯": st.column_config.TextColumn("期待度"),
                        "平均期待度": st.column_config.ProgressColumn("平均期待度", format="%.1f%%", min_value=0, max_value=100),
                        "台数": st.column_config.NumberColumn("台数", format="%d台", help="検証数"),
                        "有効稼働数": st.column_config.NumberColumn("有効稼働", format="%d台"),
                        "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100),
                        "勝率": st.column_config.ProgressColumn("勝率(差枚)", format="%.1f%%", min_value=0, max_value=100),
                        "平均差枚": st.column_config.NumberColumn("平均", format="%+d枚", help="平均結果(差枚)"),
                        "合計差枚": st.column_config.NumberColumn("合計", format="%+d枚", help="合計収支(差枚)"),
                        "REG確率": st.column_config.TextColumn("平均REG確率"),
                        "合算確率": st.column_config.TextColumn("平均合算確率"),
                        "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                    },
                    width="stretch",
                    hide_index=True
                )

            with tabs_1[0]: render_rank_stats_top(prob_analysis_df)
            with tabs_1[1]: render_rank_stats_top(prob_analysis_df[prob_analysis_df['営業区分'] == "🔥 還元日"])
            with tabs_1[2]: render_rank_stats_top(prob_analysis_df[prob_analysis_df['営業区分'] == "⚖️ 通常営業"])
            with tabs_1[3]: render_rank_stats_top(prob_analysis_df[prob_analysis_df['営業区分'] == "🥶 回収日"])

            # --- 🤖 総合原因分析 (AIの自己診断レポート) ---
            total_eval_count = len(prob_analysis_df)
            
            # 回収日を除外した「真っ当な営業日」のデータでAIの真の実力を測る
            fair_play_df = prob_analysis_df[prob_analysis_df['営業区分'] != "🥶 回収日"]
            if fair_play_df.empty: fair_play_df = prob_analysis_df # 全部回収日なら仕方なく全体を使う
            
            period_high_setting_rate = fair_play_df['valid_high'].sum() / fair_play_df['valid_play'].sum() if fair_play_df['valid_play'].sum() > 0 else 0
            high_score_df = fair_play_df[fair_play_df['prediction_score'] >= 0.30]
            high_score_accuracy = high_score_df['valid_high'].sum() / high_score_df['valid_play'].sum() if high_score_df['valid_play'].sum() > 0 else 0
            score_mean = fair_play_df['prediction_score'].mean()
            
            avg_g_recent = prob_analysis_df['結果_累計ゲーム'].mean()
            
            low_rel_count = (prob_analysis_df['予測信頼度'] == '🔻低').sum() if '予測信頼度' in prob_analysis_df.columns else 0
            low_rel_rate = low_rel_count / total_eval_count if total_eval_count > 0 else 0
            
            # 要因1: データ量
            diag_data = {"status": "🟢", "title": "データ蓄積量", "msg": "十分なデータが蓄積されています。"}
            if total_eval_count < 30:
                diag_data = {"status": "🔴", "title": "データ蓄積量", "msg": f"検証台数が{total_eval_count}台と少なすぎます。たまたまのヒキによるブレが大きいため、まだAIの実力は正しく測れません。AIのパラメータは変更せずに、まずはデータ収集を続けてください。"}
            elif low_rel_rate > 0.3:
                diag_data = {"status": "🟡", "title": "データ蓄積量", "msg": f"各台の履歴データが浅い（新台・配置変更など）台が {low_rel_rate:.0%} 含まれています。AIがクセを把握しきるまであと数週間のデータ取得が必要です。"}
                
            # 要因2: 稼働（客層）
            diag_kado = {"status": "🟢", "title": "稼働状況(客層)", "msg": "検証対象の台は全体的にしっかり回されており、結果の信頼度が高いです。"}
            if avg_g_recent < 3000:
                diag_kado = {"status": "🔴", "title": "稼働状況(客層)", "msg": f"推奨台の平均稼働が{int(avg_g_recent)}Gと低すぎます。採点基準は低稼働向けに自動補正されていますが、客層の見切りが早すぎるため、本物の高設定が回されずに埋もれてしまっている可能性が高いです。"}
            elif avg_g_recent < 4000:
                diag_kado = {"status": "🟡", "title": "稼働状況(客層)", "msg": f"推奨台の平均稼働が{int(avg_g_recent)}Gとやや低めです。採点基準は店舗に合わせて自動補正されていますが、高設定が十分に出玉を伸ばしきれていない可能性があります。"}

            # 要因3: AI設定 (過学習 / 未学習)
            diag_ai = {"status": "🟢", "title": "AI設定(パラメータ)", "msg": "還元日・通常営業日におけるスコア分布と正解率のバランスが良く、現在の設定は良好です。"}
            if total_eval_count >= 30:
                if len(high_score_df) >= 5 and high_score_accuracy < period_high_setting_rate:
                    diag_ai = {"status": "🔴", "title": "AI設定(過学習)", "msg": "回収日のフェイクを除外した「真っ当な営業日」であっても、AIが高評価した台の勝率が店全体の平均を下回っています。ノイズを必勝法だと勘違いしている「過学習」の疑いがあります。「深さ制限(max_depth)」を下げるか、「最小データ数」を増やしてください。"}
                elif score_mean > period_high_setting_rate + 0.15:
                    diag_ai = {"status": "🟡", "title": "AI設定(評価甘め)", "msg": "全体的にスコアが高く出すぎています。「葉の数(num_leaves)」を少し下げるか、「学習率」を下げてみてください。"}
                elif score_mean < period_high_setting_rate - 0.15:
                    diag_ai = {"status": "🟡", "title": "AI設定(評価慎重)", "msg": "全体的にスコアが低く出すぎています。「学習回数」を増やすか、「葉の数」を少し上げてみてください。"}

            # 要因4: 特徴量の多さ (次元の呪い)
            diag_feat = {"status": "🟢", "title": "特徴量の多さ", "msg": "学習データに対して適切な条件分岐が行われています。"}
            if total_eval_count >= 50 and len(high_score_df) == 0:
                 diag_feat = {"status": "🟡", "title": "特徴量の多さ(条件厳格化)", "msg": "期待度30%を超える台が1台もありません。AIが多くの特徴量（条件）を同時に満たす完璧な台を探しすぎて、身動きが取れなくなっています。"}

            # 要因5: 店舗の読みにくさ (ランダム・フェイク)
            diag_shop = {"status": "🟢", "title": "店舗の素直さ", "msg": "店舗のクセをある程度捉えられています。"}
            if period_high_setting_rate < 0.05:
                 diag_shop = {"status": "🔴", "title": "店舗の素直さ(ベタピン疑惑)", "msg": "そもそも店舗全体の高設定投入率が極端に低すぎます。AIが予測する以前に、戦うべき店舗（優良店）ではない可能性があります。"}
            elif diag_data["status"] == "🟢" and diag_kado["status"] == "🟢" and diag_ai["status"] in ["🟢", "🟡"]:
                if len(high_score_df) >= 5 and (high_score_accuracy - period_high_setting_rate) < 0.05:
                    diag_shop = {"status": "🔴", "title": "店舗の素直さ(予測困難)", "msg": "データ・稼働・AI設定は悪くないにも関わらず、AI推奨台が結果を出せていません。店長が「完全ランダム」で設定を入れているか、前日の凹み台などを「意図的にフェイクとして使う」など、非常に読みにくい（騙してくる）店舗である可能性が高いです。"}

            st.divider()
            with st.expander("🚀 全店舗一括 自動チューニング", expanded=False):
                st.info("💡 登録されているすべての店舗に対して、自動チューニングを順番に実行します。AIの根本的な計算ロジック（純粋確率化など）がアップデートされた際などに、全店舗の設定を一気に最適化し直すのに便利です。店舗数によっては完了まで数分かかる場合があります。")
                if st.button("⚠️ 全店舗を一括でチューニングする", type="primary"):
                    all_shops = df_verify[shop_col].dropna().unique().tolist()
                    if not all_shops:
                        st.warning("チューニング対象の店舗がありません。")
                    else:
                        try:
                            import optuna
                        except ImportError:
                            st.error("Optunaがインストールされていません。ターミナル等で `pip install optuna` を実行してください。")
                            st.stop()
                        
                        actual_features = [f for f in backend.BASE_FEATURES if f in df_verify.columns]
                        cat_features = [f for f in ['machine_code', 'shop_code', 'event_code', 'target_weekday', 'target_date_end_digit'] if f in actual_features]
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        optuna.logging.set_verbosity(optuna.logging.WARNING)
                        
                        for shop_idx, shop_name in enumerate(all_shops):
                            status_text.text(f"[{shop_idx+1}/{len(all_shops)}] {shop_name} の最適なパラメータをOptunaで探索中...")
                            shop_df = df_verify[df_verify[shop_col] == shop_name].copy()
                            if len(shop_df) < 150:
                                continue
                            
                            shop_df = shop_df.sort_values('対象日付')
                            split_idx = int(len(shop_df) * 0.8)
                            train_data = shop_df.iloc[:split_idx]
                            test_data = shop_df.iloc[split_idx:]
                            
                            X_train, y_train = train_data[actual_features], train_data['target']
                            X_test, y_test = test_data[actual_features], test_data['target']
                            
                            max_date = train_data['対象日付'].max()
                            days_diff = (max_date - train_data['対象日付']).dt.days
                            sample_weights = 0.995 ** days_diff
                            
                            def objective_all(trial):
                                params = {
                                    'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
                                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                                    'max_depth': trial.suggest_int('max_depth', 3, 7),
                                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100, step=10),
                                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
                                }
                                max_leaves = min(127, (2 ** params['max_depth']) - 1)
                                params['num_leaves'] = trial.suggest_int('num_leaves', 7, max_leaves)
                                
                                try:
                                    model = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1, **params, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
                                    model.fit(X_train, y_train, sample_weight=sample_weights, categorical_feature=cat_features)
                                    preds = model.predict_proba(X_test)[:, 1]
                                    test_eval = test_data.copy()
                                    test_eval['pred_score'] = preds
                                    
                                    if test_eval['pred_score'].nunique() <= 1:
                                        return -1.0
                                    
                                    threshold = test_eval['pred_score'].quantile(0.85)
                                    target_df = test_eval[test_eval['pred_score'] >= threshold]
                                    if len(target_df) == 0: return -1.0
                                    
                                    precision = target_df['target'].mean()
                                    avg_diff = target_df['next_diff'].mean()
                                    coverage = len(target_df) / len(test_eval)
                                    score = (precision * 100) + (avg_diff / 10) + (coverage * 50)
                                    return score
                                except Exception:
                                    return -1.0
                                    
                            study = optuna.create_study(direction='maximize')
                            # 全店舗一括は時間がかかるため、1店舗あたりの探索回数を15回に抑える
                            study.optimize(objective_all, n_trials=15)
                            
                            best_params = study.best_params
                            max_leaves_best = min(127, (2 ** best_params['max_depth']) - 1)
                            if 'num_leaves' not in best_params:
                                best_params['num_leaves'] = study.best_trial.params.get('num_leaves', max_leaves_best)
                                
                            current_hp = st.session_state["shop_hyperparams"].get(shop_name, st.session_state["shop_hyperparams"].get("デフォルト", {}))
                            st.session_state["shop_hyperparams"][shop_name] = {
                                'train_months': current_hp.get('train_months', 3), 'n_estimators': best_params['n_estimators'], 'learning_rate': best_params['learning_rate'],
                                'num_leaves': best_params['num_leaves'], 'max_depth': best_params['max_depth'], 'min_child_samples': best_params['min_child_samples'],
                                'reg_alpha': best_params.get('reg_alpha', 0.0), 'reg_lambda': best_params.get('reg_lambda', 0.0)
                            }
                            progress_bar.progress((shop_idx + 1) / len(all_shops))
                            
                        backend.save_shop_ai_settings(st.session_state["shop_hyperparams"])
                        status_text.text("✅ 全店舗のOptunaチューニングが完了しました！")
                        st.toast("✅ 全店舗のAIパラメータを最適化しました！")
                        st.rerun()

            st.divider()
            with st.expander("🤖 総合原因分析 (AIの自己診断レポート)", expanded=True):
                st.markdown("精度検証の結果から、予測がうまくいっているか、あるいは**何が原因で精度が落ちているか**を総合的に診断します。")
                
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
                    final_action_msg = "AIの予測云々以前に、客層の見切りが早すぎて高設定が埋もれやすい状態です。採点システムは低稼働に合わせて優しくなっていますが、実際の勝率を上げるには**『自分自身でしっかり回して判別する』**覚悟が必要になります。または、もっと稼働が良い店を探すのも一つの手です。"
                    st.warning(f"**{final_action_title}**\n\n{final_action_msg}")

                elif diag_ai["status"] in ["🔴", "🟡"] or diag_feat["status"] in ["🔴", "🟡"]:
                    final_action_title = "⚙️ 【最終結論】アプリ内のAI設定をチューニングする"
                    tuning_hints = []
                    # 現在のパラメータを取得
                    fallback_hp = st.session_state["shop_hyperparams"].get("デフォルト", {})
                    current_hp = st.session_state["shop_hyperparams"].get(selected_shop, fallback_hp)
                    
                    if "過学習" in diag_ai["title"]: 
                        # 具体的な数値を提示
                        new_depth = max(3, current_hp.get('max_depth', 4) - 1)
                        new_samples = min(200, current_hp.get('min_child_samples', 50) + 20)
                        tuning_hints.append(f"「深さ制限(max_depth)」を **{new_depth}** に下げる、または「最小データ数」を **{new_samples}** に増やす")
                    if "評価甘め" in diag_ai["title"]: 
                        new_leaves = max(10, current_hp.get('num_leaves', 15) - 5)
                        new_lr = max(0.01, current_hp.get('learning_rate', 0.03) - 0.01)
                        tuning_hints.append(f"「葉の数(num_leaves)」を **{new_leaves}** に下げる、または「学習率」を **{new_lr:.2f}** に下げる")
                    if "評価慎重" in diag_ai["title"]: 
                        new_est = min(1000, current_hp.get('n_estimators', 300) + 100)
                        new_leaves = min(127, current_hp.get('num_leaves', 15) + 5)
                        tuning_hints.append(f"「学習回数」を **{new_est}** に増やす、または「葉の数」を **{new_leaves}** に上げる")
                    if "条件厳格化" in diag_feat["title"]:
                        new_depth = max(3, current_hp.get('max_depth', 4) - 1)
                        new_samples = min(200, current_hp.get('min_child_samples', 50) + 20)
                        tuning_hints.append(f"「深さ制限」を **{new_depth}** に下げる、または「最小データ数」を **{new_samples}** に増やす")
                    
                    final_action_msg = "データ量も店舗の素直さも悪くありませんが、AIの「クセの覚え方」が現在の店舗の状況とズレています。すぐ下の**【店舗専用 AIモデル設定】**から、以下の調整を試して再分析してください。\n\n"
                    for hint in set(tuning_hints):
                        final_action_msg += f"- {hint}\n"
                    st.info(f"**{final_action_title}**\n\n{final_action_msg}")

                else:
                    final_action_title = "🌟 【最終結論】現状維持でOK（素晴らしい状態です！）"
                    final_action_msg = "店舗の状況、データ量、AIの設定、すべてが完璧に噛み合っています。**今のAI設定のまま、自信を持って日々の立ち回りに活用してください！**"
                    st.success(f"**{final_action_title}**\n\n{final_action_msg}")
                    
        
    # --- 4. AIの弱点分析 (騙された台の共通点) ---
    st.divider()
    st.subheader("🧠 AIの弱点分析 (騙された台の共通点)")
    st.caption(f"【{selected_shop}】において、AIが予測を大きく外した台の特徴を店舗平均と比較して分析します。店舗ごとのAIのクセや弱点を把握するのに役立ちます。")

    display_df = merged_df.copy() # このセクションで使うDF
    bad_pred_df = display_df[(display_df['prediction_score'] >= 0.65) & (display_df['差枚_actual'] <= -1000)].copy()
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
            'is_corner': "配置: 角台の割合",
            'neighbor_reg_reliability_score': "配置: 隣台のREG信頼度",
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

    # --- 5. 予測エラー分析 (ワースト予測 ＆ 取りこぼし) ---
    st.divider()
    st.subheader("🔍 AIの予測エラー分析 (ワースト＆取りこぼし)")
    
    tab_worst, tab_missed = st.tabs(["📉 期待外れ台 (高評価→大負け)", "📈 逃したお宝台 (低評価→大勝ち)"])
    
    with tab_worst:
        st.caption("※この表は**「過去にAIが推奨台（店舗上位10%）として保存した予測ログ」**の中から抽出されています。\n\nAIが高く評価（期待度65%以上）したのに大きく負けて（-1000枚以下）しまった台のワーストランキングです。なぜ大負けしたのか（低稼働による見切りか、不発か、順当な低設定か）、なぜAIは評価を上げていたのかを振り返ることができます。")
        # 表示用に整理
        display_df = merged_df.copy()
        display_df['結果判定'] = display_df['差枚_actual'].apply(lambda x: 'Win 🔴' if x > 0 else 'Lose 🔵')
        display_df = display_df.sort_values('対象日付', ascending=False)
        
        if 'prediction_score' in display_df.columns:
            display_df['予想設定5以上確率'] = (display_df['prediction_score'] * 100).astype(int)
        else:
            display_df['予想設定5以上確率'] = 0

        def analyze_bad_reason(row):
            prev_g = row.get('prev_累計ゲーム', 0)
            prev_diff = row.get('prev_差枚', 0)
            cons_minus = row.get('連続マイナス日数', 0)
            prev_reg_prob = row.get('prev_REG確率', 0)
            neighbor_diff = row.get('neighbor_avg_diff', 0)
            is_corner = row.get('is_corner', 0)
            cons_minus_diff = row.get('cons_minus_total_diff', 0)
            
            if pd.isna(prev_g): prev_g = 0
            if pd.isna(prev_diff): prev_diff = 0
            if pd.isna(cons_minus): cons_minus = 0
            if pd.isna(prev_reg_prob): prev_reg_prob = 0
            if pd.isna(neighbor_diff): neighbor_diff = 0
            if pd.isna(is_corner): is_corner = 0
            if pd.isna(cons_minus_diff): cons_minus_diff = 0
            
            prev_reg_str = f"1/{int(1/prev_reg_prob)}" if prev_reg_prob > 0 else "-"
            
            extra_info = []
            if is_corner == 1: extra_info.append("角台")
            if neighbor_diff > 500: extra_info.append(f"両隣優秀(平均+{int(neighbor_diff)}枚)")
            elif neighbor_diff < -500: extra_info.append(f"両隣凹み(平均{int(neighbor_diff)}枚)")
            
            extra_str = f" [{', '.join(extra_info)}]" if extra_info else ""
            
            if cons_minus >= 3:
                diff_str = f" (計{int(cons_minus_diff)}枚吸込)" if cons_minus_diff < 0 else ""
                return f"{int(cons_minus)}日連続凹み{diff_str}の上げリセット狙い{extra_str}"
            elif prev_diff < -1000:
                return f"前日大凹み ({int(prev_diff)}枚) の上げリセット狙い{extra_str}"
            elif prev_diff > 1000:
                return f"前日大勝 (+{int(prev_diff)}枚) の据え置き狙い{extra_str}"
            elif prev_reg_prob >= (1/300):
                return f"前日REG優秀 ({prev_reg_str}) の据え置き狙い{extra_str}"
            else:
                return f"前日 {int(prev_g)}G / {int(prev_diff)}枚 / REG {prev_reg_str}{extra_str}"

        def analyze_bad_identity(row):
            g = row.get('結果_累計ゲーム', 0)
            score = row.get('設定5近似度', 0)
            
            if g < 2000:
                return "🏃 客の早見切り (不発の可能性残る)"
            elif score >= 40:
                return "💦 展開負け (REGは引けている等)"
            else:
                return "💀 順当に低設定 (AIの予測ミス)"

        display_df['AI高評価の要因(前日状況)'] = display_df.apply(analyze_bad_reason, axis=1)
        display_df['大負けの正体(結果分析)'] = display_df.apply(analyze_bad_identity, axis=1)

        cols_base = ['対象日付', shop_col, '台番号', '機種名', '予想設定5以上確率', '設定5近似度', '差枚_actual', '結果_累計ゲーム', '結果_BIG', '結果_BIG確率分母', '結果_REG', '結果_REG確率分母']
        bad_cols = ['対象日付', shop_col, '台番号', '機種名', '予想設定5以上確率', '結果_REG確率分母', 'AI高評価の要因(前日状況)', '設定5近似度', '大負けの正体(結果分析)', '差枚_actual', '結果_累計ゲーム', '結果_BIG', '結果_BIG確率分母', '結果_REG']

        config_dict = {
            "対象日付": st.column_config.DateColumn("予測対象日", format="MM/DD"),
            "予想設定5以上確率": st.column_config.NumberColumn("事前AI期待度", format="%d%%", help="AIが事前に予測した設定5以上の確率"),
            "設定5近似度": st.column_config.NumberColumn("近似度スコア", format="%d点", help="設定5近似度"),
            "差枚_actual": st.column_config.NumberColumn("差枚", format="%+d"),
            "結果_累計ゲーム": st.column_config.NumberColumn("総G数", format="%dG"),
            "結果_BIG": st.column_config.NumberColumn("BIG", format="%d"),
            "結果_BIG確率分母": st.column_config.NumberColumn("B確率", format="1/%d"),
            "結果_REG": st.column_config.NumberColumn("REG", format="%d"),
            "結果_REG確率分母": st.column_config.NumberColumn("実際REG確率", format="1/%d"),
            "AI高評価の要因(前日状況)": st.column_config.TextColumn("AI高評価の要因", help="AIが事前期待度を高く見積もった理由と思われる、前日の状況です。"),
            "大負けの正体(結果分析)": st.column_config.TextColumn("大負けの正体", help="低稼働による見切りか、回された上での低設定か等を判定します。"),
        }

        if 'prediction_score' in merged_df.columns:
            bad_pred_df = display_df[(display_df['prediction_score'] >= 0.65) & (display_df['差枚_actual'] <= -1000)].copy()
            
            if bad_pred_df.empty:
                st.success("現在、大きく期待を裏切った台はありません！")
            else:
                st.dataframe(bad_pred_df[bad_cols].sort_values('差枚_actual'), column_config=config_dict, width="stretch", hide_index=True)

    with tab_missed:
        st.caption("※この表は**「過去にAIが予測を保存した日」**の全稼働データの中から抽出されています。\n\nAIが低く評価（期待度20%未満）したのに大きく勝って（+2000枚以上）しまった台のランキングです。なぜ大勝ちしたのか（低設定の誤爆か、本物の高設定か）、なぜAIは事前に評価を下げていたのか（前日の稼働不足か、回収トレンドか）を振り返ることができます。")
        
        missed_df_raw = df_verify[df_verify[shop_col] == selected_shop].copy() if not df_verify.empty and shop_col in df_verify.columns else pd.DataFrame()
        
        if not missed_df_raw.empty and not df_pred_log.empty and '予測対象日_merge' in df_pred_log.columns:
            if shop_col in df_pred_log.columns:
                logged_dates = df_pred_log[df_pred_log[shop_col] == selected_shop]['予測対象日_merge'].dropna().dt.date.unique()
            else:
                logged_dates = df_pred_log['予測対象日_merge'].dropna().dt.date.unique()
                
            if 'next_date' in missed_df_raw.columns:
                missed_df_raw['tmp_target_date'] = pd.to_datetime(missed_df_raw['next_date'], errors='coerce').dt.date
            else:
                missed_df_raw['tmp_target_date'] = (pd.to_datetime(missed_df_raw['対象日付'], errors='coerce') + pd.Timedelta(days=1)).dt.date
                
            missed_df_raw = missed_df_raw[missed_df_raw['tmp_target_date'].isin(logged_dates)].copy()
        
        if missed_df_raw.empty or 'prediction_score' not in missed_df_raw.columns or 'next_diff' not in missed_df_raw.columns:
            st.info("分析に必要なデータがありません。")
        else:
            missed_df = missed_df_raw[(missed_df_raw['prediction_score'] < 0.20) & (missed_df_raw['next_diff'] >= 2000)].copy()
            missed_df = missed_df.sort_values('next_diff', ascending=False)
            
            if missed_df.empty:
                st.success("現在、大きく取りこぼしたお宝台はありません！")
            else:
                missed_df['予想設定5以上確率'] = (missed_df['prediction_score'] * 100).astype(int)
                
                def analyze_missed_reason(row):
                    g = row.get('累計ゲーム', 0)
                    diff = row.get('差枚', 0)
                    win_rate_7d = row.get('win_rate_7days', 0)
                    mean_7d = row.get('mean_7days_diff', 0)
                    cons_minus = row.get('連続マイナス日数', 0)
                    
                    if pd.isna(g): g = 0
                    if pd.isna(diff): diff = 0
                    if pd.isna(win_rate_7d): win_rate_7d = 0
                    
                    reasons = []
                    if g < 2000:
                        reasons.append(f"前日低稼働({int(g)}G)でデータ不足")
                    if win_rate_7d < 0.2 and mean_7d < -500:
                        reasons.append("週間トレンドが完全な回収モード")
                    if diff >= 1500:
                        reasons.append(f"前日大勝({int(diff)}枚)からの連勝ストップ警戒")
                    elif diff <= -2000 and cons_minus < 3:
                        reasons.append(f"前日大敗({int(diff)}枚)だが上げリセットの根拠弱")
                        
                    if not reasons:
                        return "特筆すべき強調データがなく全体的に評価が低かった"
                    return " / ".join(reasons)
                    
                def analyze_missed_identity(row):
                    g = row.get('next_累計ゲーム', 0)
                    b = row.get('next_BIG', 0)
                    r = row.get('next_REG', 0)
                    machine = row.get('機種名', '')
                    
                    if pd.isna(g): g = 0
                    if pd.isna(b): b = 0
                    if pd.isna(r): r = 0
                    
                    if g < 3000:
                        return "🎲 稼働不足での上振れ (まぐれ吹き)"
                        
                    reg_prob = g / r if r > 0 else 9999
                    
                    specs = backend.get_machine_specs()
                    matched_key = backend.get_matched_spec_key(machine, specs)
                    spec_r4 = specs[matched_key].get('設定4', {"REG": 300.0})["REG"] if matched_key in specs else 300.0
                    
                    if reg_prob > spec_r4 * 1.15: # 設定4より明らかに悪い
                        return "💣 低設定のBIG偏り (誤爆)"
                    else:
                        return "💎 本物の高設定 (AIの完全な取りこぼし)"

                missed_df['AI低評価の要因(前日状況)'] = missed_df.apply(analyze_missed_reason, axis=1)
                missed_df['大勝の正体(結果分析)'] = missed_df.apply(analyze_missed_identity, axis=1)
                
                missed_df['結果_BIG確率分母'] = np.where(missed_df['next_BIG'] > 0, missed_df['next_累計ゲーム'] / missed_df['next_BIG'], 0).astype(int)
                missed_df['結果_REG確率分母'] = np.where(missed_df['next_REG'] > 0, missed_df['next_累計ゲーム'] / missed_df['next_REG'], 0).astype(int)
                
                if 'next_date' in missed_df.columns:
                    missed_df['予測対象日'] = pd.to_datetime(missed_df['next_date'])
                else:
                    missed_df['予測対象日'] = pd.to_datetime(missed_df['対象日付']) + pd.Timedelta(days=1)
                    
                missed_cols = ['予測対象日', shop_col, '台番号', '機種名', '予想設定5以上確率', '結果_REG確率分母', 'AI低評価の要因(前日状況)', '大勝の正体(結果分析)', 'next_diff', 'next_累計ゲーム', 'next_BIG', '結果_BIG確率分母', 'next_REG']
                
                st.dataframe(
                    missed_df[missed_cols],
                    column_config={
                        "予測対象日": st.column_config.DateColumn("対象日", format="MM/DD"),
                        "予想設定5以上確率": st.column_config.NumberColumn("事前AI期待度", format="%d%%"),
                        "next_diff": st.column_config.NumberColumn("結果差枚", format="%+d"),
                        "next_累計ゲーム": st.column_config.NumberColumn("総G数", format="%dG"),
                        "next_BIG": st.column_config.NumberColumn("BIG", format="%d"),
                        "結果_BIG確率分母": st.column_config.NumberColumn("B確率", format="1/%d"),
                        "next_REG": st.column_config.NumberColumn("REG", format="%d"),
                        "結果_REG確率分母": st.column_config.NumberColumn("実際REG確率", format="1/%d"),
                        "AI低評価の要因(前日状況)": st.column_config.TextColumn("AI低評価の要因", help="AIが事前期待度を低く見積もった理由と思われる、前日の状況です。"),
                        "大勝の正体(結果分析)": st.column_config.TextColumn("大勝の正体", help="低設定のまぐれ吹きか、本物の高設定の取りこぼしかを判定します。"),
                    },
                    width="stretch",
                    hide_index=True
                )

    # --- 6. 全履歴データ (バックテスト結果) ---
    st.divider()
    st.subheader("📝 全履歴データ (バックテスト結果)")
    
    band_options = ['すべて', '50%以上', '40%〜49%', '30%〜39%', '20%〜29%', '15%〜19%', '10%〜14%', '10%未満']
    selected_band = st.selectbox("表示する期待度（確率帯）を選択", band_options, index=0)
    
    history_display_df = display_df.copy()
    if selected_band != 'すべて':
        if '確率帯' in history_display_df.columns:
            history_display_df = history_display_df[history_display_df['確率帯'] == selected_band]

    st.dataframe(
        history_display_df[cols_base],
        column_config=config_dict,
        width="stretch",
        hide_index=True
    )

    with tab_setting:
        st.subheader(f"⚙️ 【{selected_shop}】専用 AIモデル設定")
        st.caption("AIのパラメータを手動で調整するか、「自動チューニング」を試してください。各項目の意味が分からない場合は、まずは「自動チューニング」の実行をおすすめします。")
        
        if "shop_hyperparams" not in st.session_state:
            st.session_state["shop_hyperparams"] = {"デフォルト": {'train_months': 3, 'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50, 'reg_alpha': 0.0, 'reg_lambda': 0.0}}
            
        default_hp = st.session_state["shop_hyperparams"]["デフォルト"]
        current_hp = st.session_state["shop_hyperparams"].get(selected_shop, default_hp)
        
        with st.form(f"hp_form_{selected_shop}"):
            hp_train_months = st.slider("学習データ期間 (直近〇ヶ月)", 1, 12, int(current_hp.get('train_months', 3)), step=1, help="店の傾向が頻繁に変わるなら短く（1〜3ヶ月）、安定しているなら長く（6ヶ月以上）設定するのが有効です。")
            hp_n_estimators = st.slider("学習回数 (n_estimators)", 50, 1000, int(current_hp.get('n_estimators', 300)), step=50, help="AIが同じデータで何回繰り返し学習するか。多いほど複雑なパターンを学習できますが、過学習のリスクが増えます。過学習を抑えるには、学習率を下げつつこの値を調整します。")
            hp_learning_rate = st.slider("学習率 (learning_rate)", 0.01, 0.3, float(current_hp.get('learning_rate', 0.03)), step=0.01, help="1回の学習でどれだけ賢くなるかの度合い。小さいほど慎重に学習し、過学習しにくくなります。基本的には0.01〜0.05の範囲が推奨されます。")
            hp_num_leaves = st.slider("葉の数 (num_leaves)", 10, 127, int(current_hp.get('num_leaves', 15)), step=1, help="モデルが作成できる分岐（葉）の最大数。大きいほど複雑な条件分岐を作れますが、過学習しやすくなります。過学習を抑えるにはこの値を小さくします。")
            hp_max_depth = st.slider("深さ制限 (max_depth)", -1, 15, int(current_hp.get('max_depth', 4)), step=1, help="条件分岐の深さ。-1は無制限。深いほど複雑な条件の組み合わせを学習できますが、過学習しやすくなります。過学習を抑えるにはこの値を小さくします (例: 3〜7)。")
            hp_min_child_samples = st.slider("最小データ数 (min_child_samples)", 10, 200, int(current_hp.get('min_child_samples', 50)), step=10, help="1つの分岐（葉）を作るために必要な最小サンプル数。大きいほど、より一般的なルールを作るようになり、過学習を抑制します。")
            hp_reg_alpha = st.slider("L1正則化 (不要データの無視力)", 0.0, 5.0, float(current_hp.get('reg_alpha', 0.0)), step=0.1, help="値を大きくするほど、予測に不要な特徴量を無視しやすくなり、過学習を抑制します。")
            hp_reg_lambda = st.slider("L2正則化 (過学習の抑制力)", 0.0, 5.0, float(current_hp.get('reg_lambda', 0.0)), step=0.1, help="値を大きくするほど、AIが特定の特徴量に極端に依存するのを防ぎ、過学習を抑制します。")
            
            cols = st.columns(4)
            submitted = cols[0].form_submit_button("保存して再分析", type="primary")
            test_btn = cols[1].form_submit_button("🧪 カンニングなしテスト", help="現在スライダーで設定している値を使って、直近1ヶ月の結果を『カンニングなし』で予測するテストを実行します。")
            auto_tune_btn = cols[2].form_submit_button("✨ 自動チューニング", help=f"AIが【{selected_shop}】の過去データから最適なパラメータの組み合わせを自動で探索し、設定します。どの設定が良いか分からない場合にまずお試しください。")
            reset_btn = cols[3].form_submit_button("リセット")
            
        if submitted:
            st.session_state["shop_hyperparams"][selected_shop] = {
                'train_months': hp_train_months, 'n_estimators': hp_n_estimators, 'learning_rate': hp_learning_rate,
                'num_leaves': hp_num_leaves, 'max_depth': hp_max_depth, 'min_child_samples': hp_min_child_samples,
                'reg_alpha': hp_reg_alpha, 'reg_lambda': hp_reg_lambda
            }
            backend.save_shop_ai_settings(st.session_state["shop_hyperparams"])
            st.rerun()
            
        if reset_btn:
            if selected_shop in st.session_state["shop_hyperparams"]:
                del st.session_state["shop_hyperparams"][selected_shop]
                backend.save_shop_ai_settings(st.session_state["shop_hyperparams"])
                st.rerun()
                
        if test_btn:
            with st.spinner("直近1ヶ月のデータでカンニングなしのバックテストを実行中..."):
                import lightgbm as lgb
                actual_features = [f for f in backend.BASE_FEATURES if f in df_verify.columns]
                cat_features = [f for f in ['machine_code', 'shop_code', 'event_code', 'target_weekday', 'target_date_end_digit'] if f in actual_features]
                
                shop_df = df_verify[df_verify[shop_col] == selected_shop].copy()
                
                if len(shop_df) < 50:
                    st.error("データが少なすぎてテストを実行できません。")
                else:
                    shop_df['対象日付'] = pd.to_datetime(shop_df['対象日付'])
                    max_date = shop_df['対象日付'].max()
                    cutoff_date = max_date - pd.Timedelta(days=30)
                    
                    train_data = shop_df[shop_df['対象日付'] <= cutoff_date].copy()
                    test_data = shop_df[shop_df['対象日付'] > cutoff_date].copy()
                    
                    if len(train_data) < 30 or len(test_data) < 10:
                        st.error("直近1ヶ月またはそれ以前のデータが不足しているため、テストを実行できません。")
                    else:
                        X_train, y_train = train_data[actual_features], train_data['target']
                        X_test, y_test = test_data[actual_features], test_data['target']
                        
                        days_diff = (train_data['対象日付'].max() - train_data['対象日付']).dt.days
                        sample_weights = 0.995 ** days_diff
                        
                        params = {
                            'n_estimators': hp_n_estimators,
                            'learning_rate': hp_learning_rate,
                            'num_leaves': hp_num_leaves,
                            'max_depth': hp_max_depth,
                            'min_child_samples': hp_min_child_samples,
                            'reg_alpha': hp_reg_alpha,
                            'reg_lambda': hp_reg_lambda
                        }
                        
                        try:
                            reg_model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02)
                            reg_model.fit(X_train, train_data['next_diff'], sample_weight=sample_weights, categorical_feature=cat_features)
                            
                            X_train_st = X_train.copy()
                            X_train_st['predicted_diff'] = reg_model.predict(X_train)
                            
                            model = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1, **params, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02)
                            model.fit(X_train_st, y_train, sample_weight=sample_weights, categorical_feature=cat_features)
                            
                            X_test_st = X_test.copy()
                            X_test_st['predicted_diff'] = reg_model.predict(X_test)
                            preds = model.predict_proba(X_test_st)[:, 1]
                            test_data['pred_score'] = preds
                            
                            test_data['valid_play'] = (pd.to_numeric(test_data['next_累計ゲーム'], errors='coerce').fillna(0) >= 3000) | \
                                                   ((pd.to_numeric(test_data['next_累計ゲーム'], errors='coerce').fillna(0) < 3000) & \
                                                    ((pd.to_numeric(test_data['next_diff'], errors='coerce').fillna(0) <= -750) | (pd.to_numeric(test_data['next_diff'], errors='coerce').fillna(0) >= 750)))
                            test_data['valid_win'] = test_data['valid_play'] & (pd.to_numeric(test_data['next_diff'], errors='coerce').fillna(0) > 0)
                            test_data['valid_high_play'] = pd.to_numeric(test_data['next_累計ゲーム'], errors='coerce').fillna(0) >= 3000
                            test_data['valid_high'] = test_data['valid_high_play'] & (test_data['target'] == 1)
                            test_data['valid_next_diff'] = np.where(test_data['valid_play'], test_data['next_diff'], np.nan)
                            
                            # REG確率と合算確率の計算を追加
                            test_data['結果_REG確率_val'] = np.where(pd.to_numeric(test_data['next_累計ゲーム'], errors='coerce').fillna(0) > 0, pd.to_numeric(test_data['next_REG'], errors='coerce').fillna(0) / pd.to_numeric(test_data['next_累計ゲーム'], errors='coerce').fillna(0), 0)
                            test_data['valid_REG確率'] = np.where(test_data['valid_play'], test_data['結果_REG確率_val'], np.nan)
                            test_data['結果_合算確率_val'] = np.where(pd.to_numeric(test_data['next_累計ゲーム'], errors='coerce').fillna(0) > 0, (pd.to_numeric(test_data['next_BIG'], errors='coerce').fillna(0) + pd.to_numeric(test_data['next_REG'], errors='coerce').fillna(0)) / pd.to_numeric(test_data['next_累計ゲーム'], errors='coerce').fillna(0), 0)
                            test_data['valid_合算確率'] = np.where(test_data['valid_play'], test_data['結果_合算確率_val'], np.nan)
                            
                            def get_prob_band(score):
                                if score >= 0.50: return '50%以上'
                                elif score >= 0.40: return '40%〜49%'
                                elif score >= 0.30: return '30%〜39%'
                                elif score >= 0.20: return '20%〜29%'
                                elif score >= 0.15: return '15%〜19%'
                                elif score >= 0.10: return '10%〜14%'
                                else: return '10%未満'
                            
                            test_data['確率帯'] = test_data['pred_score'].apply(get_prob_band)
                            test_stats = test_data.groupby('確率帯').agg(
                                台数=('台番号', 'count'),
                                高設定数=('valid_high', 'sum'),
                                高設定有効数=('valid_high_play', 'sum'),
                                有効稼働数=('valid_play', 'sum'),
                                勝数=('valid_win', 'sum'),
                                平均差枚=('valid_next_diff', 'mean'),
                                合計差枚=('next_diff', 'sum'),
                                平均期待度=('pred_score', 'mean'),
                                平均REG確率=('valid_REG確率', 'mean'),
                                平均合算確率=('valid_合算確率', 'mean')
                            ).reset_index()
                            test_stats['高設定率'] = np.where(test_stats['高設定有効数'] > 0, test_stats['高設定数'] / test_stats['高設定有効数'] * 100, 0.0)
                            test_stats['勝率'] = np.where(test_stats['有効稼働数'] > 0, test_stats['勝数'] / test_stats['有効稼働数'] * 100, 0.0)
                            test_stats['平均期待度'] = test_stats['平均期待度'] * 100
                            test_stats['REG確率'] = test_stats['平均REG確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
                            test_stats['合算確率'] = test_stats['平均合算確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
                            
                            rank_order = {'50%以上': 1, '40%〜49%': 2, '30%〜39%': 3, '20%〜29%': 4, '15%〜19%': 5, '10%〜14%': 6, '10%未満': 7}
                            test_stats['sort'] = test_stats['確率帯'].map(rank_order).fillna(99)
                            test_stats = test_stats.sort_values('sort').drop('sort', axis=1)
                            test_stats['信頼度'] = test_stats['台数'].apply(get_confidence_indicator)
                            
                            st.session_state['backtest_result'] = test_stats
                        except Exception as e:
                            st.error(f"テスト実行中にエラーが発生しました: {e}")
                            
        if 'backtest_result' in st.session_state:
            st.success("✅ 直近1ヶ月のカンニングなしテスト結果")
            st.caption("現在設定されているパラメータで過去データのみを学習し、直近1ヶ月の結果を予測した「本物の実力」です。この表の上位の期待度帯（20%以上など）の勝率が高くなる設定を探してください。")
            st.dataframe(
                st.session_state['backtest_result'][['確率帯', '平均期待度', '台数', '有効稼働数', '高設定率', '勝率', '平均差枚', '合計差枚', 'REG確率', '合算確率', '信頼度']],
                column_config={
                    "確率帯": st.column_config.TextColumn("期待度"),
                    "平均期待度": st.column_config.ProgressColumn("平均期待度", format="%.1f%%", min_value=0, max_value=100),
                    "台数": st.column_config.NumberColumn("台数", format="%d台", help="検証数"),
                    "有効稼働数": st.column_config.NumberColumn("有効稼働", format="%d台"),
                    "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100),
                    "勝率": st.column_config.ProgressColumn("勝率(差枚)", format="%.1f%%", min_value=0, max_value=100),
                    "平均差枚": st.column_config.NumberColumn("平均", format="%+d枚", help="平均結果(差枚)"),
                    "合計差枚": st.column_config.NumberColumn("合計", format="%+d枚", help="合計収支(差枚)"),
                    "REG確率": st.column_config.TextColumn("平均REG確率"),
                    "合算確率": st.column_config.TextColumn("平均合算確率"),
                    "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                },
                hide_index=True,
                use_container_width=True
            )
                
        if auto_tune_btn:
            with st.spinner("AIが過去データを分割し、数多くの組み合わせから最適な設定を探索中... (約10〜20秒かかります)"):
                import lightgbm as lgb
                actual_features = [f for f in backend.BASE_FEATURES if f in df_verify.columns]
                cat_features = [f for f in ['machine_code', 'shop_code', 'event_code', 'target_weekday', 'target_date_end_digit'] if f in actual_features]
                
                shop_df = df_verify[df_verify[shop_col] == selected_shop].copy()
                
                if len(shop_df) < 150:
                    st.error("データが少なすぎて自動チューニングを実行できません。150件以上のデータが必要です。")
                else:
                    shop_df = shop_df.sort_values('対象日付')
                    split_idx = int(len(shop_df) * 0.8)
                    train_data = shop_df.iloc[:split_idx]
                    test_data = shop_df.iloc[split_idx:]
                    
                    X_train, y_train = train_data[actual_features], train_data['target']
                    X_test, y_test = test_data[actual_features], test_data['target']
                    
                    max_date = train_data['対象日付'].max()
                    days_diff = (max_date - train_data['対象日付']).dt.days
                    sample_weights = 0.995 ** days_diff
                    
                    param_candidates = [
                        {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 30, 'reg_alpha': 0.0, 'reg_lambda': 0.0},
                        {'n_estimators': 400, 'learning_rate': 0.02, 'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 40, 'reg_alpha': 0.5, 'reg_lambda': 0.5},
                        {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 20, 'max_depth': 5, 'min_child_samples': 30, 'reg_alpha': 1.0, 'reg_lambda': 0.0},
                        {'n_estimators': 500, 'learning_rate': 0.01, 'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 50, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
                        {'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 5, 'min_child_samples': 20, 'reg_alpha': 0.0, 'reg_lambda': 0.0},
                        {'n_estimators': 400, 'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6, 'min_child_samples': 30, 'reg_alpha': 0.5, 'reg_lambda': 0.5},
                        {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 60, 'reg_alpha': 2.0, 'reg_lambda': 0.0},
                        {'n_estimators': 600, 'learning_rate': 0.01, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 40, 'reg_alpha': 0.0, 'reg_lambda': 2.0},
                        {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 7, 'min_child_samples': 15, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
                        {'n_estimators': 100, 'learning_rate': 0.10, 'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 20, 'reg_alpha': 0.0, 'reg_lambda': 0.0},
                        {'n_estimators': 400, 'learning_rate': 0.02, 'num_leaves': 25, 'max_depth': 5, 'min_child_samples': 30, 'reg_alpha': 0.5, 'reg_lambda': 1.0},
                        {'n_estimators': 200, 'learning_rate': 0.04, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 20, 'reg_alpha': 1.0, 'reg_lambda': 1.0},
                    ]
                    
                    best_score = -9999
                    best_params = param_candidates[0]
                    progress_bar = st.progress(0)
                    
                    for i, params in enumerate(param_candidates):
                        try:
                            reg_model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02)
                            reg_model.fit(X_train, train_data['next_diff'], sample_weight=sample_weights, categorical_feature=cat_features)
                            
                            X_train_st = X_train.copy()
                            X_train_st['predicted_diff'] = reg_model.predict(X_train)
                            
                            model = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1, **params, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02)
                            model.fit(X_train_st, y_train, sample_weight=sample_weights, categorical_feature=cat_features)
                            
                            X_test_st = X_test.copy()
                            X_test_st['predicted_diff'] = reg_model.predict(X_test)
                            preds = model.predict_proba(X_test_st)[:, 1]
                            test_eval = test_data.copy()
                            test_eval['pred_score'] = preds
                            
                            if test_eval['pred_score'].nunique() <= 1:
                                score = -1
                            else:
                                # データ上の裏付けに基づく評価
                                threshold = test_eval['pred_score'].quantile(0.85)
                                target_df = test_eval[test_eval['pred_score'] >= threshold]
                                if len(target_df) == 0: score = -1
                                else:
                                    precision = target_df['target'].mean()
                                    avg_diff = target_df['next_diff'].mean()
                                    coverage = len(target_df) / len(test_eval)
                                    score = (precision * 100) + (avg_diff / 10) + (coverage * 50)
                            if score > best_score:
                                best_score = score
                                best_params = params
                        except Exception as e:
                            print(f"Auto tune error (Single): {e}")
                        progress_bar.progress((i + 1) / len(param_candidates))
                    
                    st.session_state["shop_hyperparams"][selected_shop] = {
                        'train_months': hp_train_months, 'n_estimators': best_params['n_estimators'], 'learning_rate': best_params['learning_rate'],
                        'num_leaves': best_params['num_leaves'], 'max_depth': best_params['max_depth'], 'min_child_samples': best_params['min_child_samples'],
                        'reg_alpha': best_params.get('reg_alpha', 0.0), 'reg_lambda': best_params.get('reg_lambda', 0.0)
                    }
                    backend.save_shop_ai_settings(st.session_state["shop_hyperparams"])
                    st.toast("✅ 自動チューニングが完了し、最も優秀だった設定を適用しました！")
                    st.rerun()

        with st.expander("🔧 AIパラメータ調整用のシミュレーション結果 (答え合わせ)", expanded=True):
            st.info("💡 **この表の役割について**\nこの表はAIが「答えを知っている状態」でのテスト結果であり、未来の勝率を表すものではありません。**「AIのパラメータ設定を変更した直後」**や**「AIが過学習（過去のまぐれを丸暗記）していないかの確認」**を行うための開発・メンテナンス用ツールです。")
            sim_df = df_verify[df_verify[shop_col] == selected_shop].copy() if not df_verify.empty and shop_col in df_verify.columns else pd.DataFrame()
            if not sim_df.empty and 'prediction_score' in sim_df.columns and 'target' in sim_df.columns and 'next_diff' in sim_df.columns:
                def get_prob_band(score):
                    if score >= 0.50: return '50%以上'
                    elif score >= 0.40: return '40%〜49%'
                    elif score >= 0.30: return '30%〜39%'
                    elif score >= 0.20: return '20%〜29%'
                    elif score >= 0.15: return '15%〜19%'
                    elif score >= 0.10: return '10%〜14%'
                    else: return '10%未満'
                sim_df['確率帯'] = sim_df['prediction_score'].apply(get_prob_band)
                sim_df['valid_play'] = (pd.to_numeric(sim_df['next_累計ゲーム'], errors='coerce').fillna(0) >= 3000) | \
                                       ((pd.to_numeric(sim_df['next_累計ゲーム'], errors='coerce').fillna(0) < 3000) & \
                                        ((pd.to_numeric(sim_df['next_diff'], errors='coerce').fillna(0) <= -750) | (pd.to_numeric(sim_df['next_diff'], errors='coerce').fillna(0) >= 750)))
                sim_df['valid_win'] = sim_df['valid_play'] & (pd.to_numeric(sim_df['next_diff'], errors='coerce').fillna(0) > 0)
                sim_df['valid_high_play'] = pd.to_numeric(sim_df['next_累計ゲーム'], errors='coerce').fillna(0) >= 3000
                sim_df['valid_high'] = sim_df['valid_high_play'] & (sim_df['target'] == 1)
                sim_df['valid_next_diff'] = np.where(sim_df['valid_play'], sim_df['next_diff'], np.nan)
                
                # REG確率の計算を追加
                sim_df['結果_REG確率_val'] = np.where(pd.to_numeric(sim_df['next_累計ゲーム'], errors='coerce').fillna(0) > 0, pd.to_numeric(sim_df['next_REG'], errors='coerce').fillna(0) / pd.to_numeric(sim_df['next_累計ゲーム'], errors='coerce').fillna(0), 0)
                sim_df['valid_REG確率'] = np.where(sim_df['valid_play'], sim_df['結果_REG確率_val'], np.nan)
                
                # 合算確率の計算を追加
                sim_df['結果_合算確率_val'] = np.where(pd.to_numeric(sim_df['next_累計ゲーム'], errors='coerce').fillna(0) > 0, (pd.to_numeric(sim_df['next_BIG'], errors='coerce').fillna(0) + pd.to_numeric(sim_df['next_REG'], errors='coerce').fillna(0)) / pd.to_numeric(sim_df['next_累計ゲーム'], errors='coerce').fillna(0), 0)
                sim_df['valid_合算確率'] = np.where(sim_df['valid_play'], sim_df['結果_合算確率_val'], np.nan)
                
                # 営業区分の付与
                if '対象日付' in sim_df.columns and not ai_recom_df.empty and '営業区分' in ai_recom_df.columns:
                    sim_df['対象日付_merge_key'] = pd.to_datetime(sim_df['対象日付']).dt.normalize()
                    day_eval_map = ai_recom_df[['対象日付', '営業区分']].drop_duplicates()
                    day_eval_map['対象日付_merge_key'] = pd.to_datetime(day_eval_map['対象日付']).dt.normalize()
                    sim_df = pd.merge(sim_df, day_eval_map[['対象日付_merge_key', '営業区分']], on='対象日付_merge_key', how='left')
                    sim_df = sim_df.drop(columns=['対象日付_merge_key'], errors='ignore')
                    sim_df['営業区分'] = sim_df['営業区分'].fillna("⚖️ 通常営業")
                else:
                    sim_df['営業区分'] = "⚖️ 通常営業"
                    
                tabs_sim = st.tabs(["全体", "🔥 還元日", "⚖️ 通常営業", "🥶 回収日"])
                
                def render_sim_stats(target_df):
                    if target_df.empty:
                        st.info("該当するデータがありません。")
                        return
                    s_stats = target_df.groupby('確率帯').agg(
                        台数=('台番号', 'count'),
                        高設定数=('valid_high', 'sum'),
                        高設定有効数=('valid_high_play', 'sum'),
                        有効稼働数=('valid_play', 'sum'),
                        勝数=('valid_win', 'sum'),
                        平均差枚=('valid_next_diff', 'mean'),
                        合計差枚=('next_diff', 'sum'),
                        平均期待度=('prediction_score', 'mean'),
                        平均REG確率=('valid_REG確率', 'mean'),
                        平均合算確率=('valid_合算確率', 'mean')
                    ).reset_index()
                    s_stats['勝率'] = np.where(s_stats['有効稼働数'] > 0, s_stats['勝数'] / s_stats['有効稼働数'] * 100, 0.0)
                    s_stats['高設定率'] = np.where(s_stats['高設定有効数'] > 0, s_stats['高設定数'] / s_stats['高設定有効数'] * 100, 0.0)
                    s_stats['平均期待度'] = s_stats['平均期待度'] * 100
                    s_stats['REG確率'] = s_stats['平均REG確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
                    s_stats['合算確率'] = s_stats['平均合算確率'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
                    
                    rank_order = {'50%以上': 1, '40%〜49%': 2, '30%〜39%': 3, '20%〜29%': 4, '15%〜19%': 5, '10%〜14%': 6, '10%未満': 7}
                    s_stats['sort'] = s_stats['確率帯'].map(rank_order).fillna(99)
                    s_stats = s_stats.sort_values('sort').drop('sort', axis=1)
                    s_stats['信頼度'] = s_stats['台数'].apply(get_confidence_indicator)
                    
                    st.dataframe(
                        s_stats[['確率帯', '平均期待度', '台数', '有効稼働数', '高設定率', '勝率', '平均差枚', '合計差枚', 'REG確率', '合算確率', '信頼度']],
                        column_config={
                            "確率帯": st.column_config.TextColumn("期待度"),
                            "平均期待度": st.column_config.ProgressColumn("平均期待度", format="%.1f%%", min_value=0, max_value=100),
                            "台数": st.column_config.NumberColumn("台数", format="%d台", help="検証数"),
                            "有効稼働数": st.column_config.NumberColumn("有効稼働", format="%d台"),
                            "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100),
                            "勝率": st.column_config.ProgressColumn("勝率(差枚)", format="%.1f%%", min_value=0, max_value=100),
                            "平均差枚": st.column_config.NumberColumn("平均", format="%+d枚", help="平均結果(差枚)"),
                            "合計差枚": st.column_config.NumberColumn("合計", format="%+d枚", help="合計収支(差枚)"),
                            "REG確率": st.column_config.TextColumn("平均REG確率"),
                            "合算確率": st.column_config.TextColumn("平均合算確率"),
                            "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                        },
                        width="stretch",
                        hide_index=True
                    )

                with tabs_sim[0]: render_sim_stats(sim_df)
                with tabs_sim[1]: render_sim_stats(sim_df[sim_df['営業区分'] == "🔥 還元日"])
                with tabs_sim[2]: render_sim_stats(sim_df[sim_df['営業区分'] == "⚖️ 通常営業"])
                with tabs_sim[3]: render_sim_stats(sim_df[sim_df['営業区分'] == "🥶 回収日"])
                
                with st.expander("🔍 シミュレーション詳細データを確認", expanded=False):
                    st.caption("シミュレーションで各確率帯に分類された台の具体的な日付と結果を確認できます。")
                    band_options_sim = ['すべて', '50%以上', '40%〜49%', '30%〜39%', '20%〜29%', '15%〜19%', '10%〜14%', '10%未満']
                    selected_band_sim = st.selectbox("表示する確率帯を選択", band_options_sim, index=0, key="sim_band_select")
                    
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
                        
                        display_cols_sim = ['予測対象日', '対象日付', '台番号', '機種名', '予想設定5以上確率', '高設定挙動', 'next_diff', 'next_累計ゲーム', 'next_BIG', 'next_REG']
                        if '根拠' in sim_display_df.columns:
                            display_cols_sim.append('根拠')
                        
                        st.dataframe(
                            sim_display_df[display_cols_sim],
                            column_config={
                                "予測対象日": st.column_config.DateColumn("予測日", format="MM/DD"),
                                "対象日付": st.column_config.DateColumn("稼働日(前日)", format="MM/DD", help="予測のベースとなった過去の稼働日"),
                                "予想設定5以上確率": st.column_config.NumberColumn("期待度", format="%d%%"),
                                "高設定挙動": st.column_config.TextColumn("挙動", help="設定5以上基準を満たしたか"),
                                "next_diff": st.column_config.NumberColumn("結果差枚", format="%+d"),
                                "next_累計ゲーム": st.column_config.NumberColumn("総G数", format="%dG"),
                                "next_BIG": st.column_config.NumberColumn("BIG", format="%d"),
                                "next_REG": st.column_config.NumberColumn("REG", format="%d"),
                                "根拠": st.column_config.TextColumn("AI推奨根拠", width="large"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
            else:
                st.info("シミュレーション用のデータがありません。")

        st.divider()
        with st.expander("🗑️ 古い予測ログの整理 (リセット)", expanded=False):
            st.warning("⚠️ プログラムの計算式変更などで基準が変わってしまい、過去の予測スコアが参考にならなくなった場合は、ここで古いログを削除してリセットできます。")
            
            del_months = st.selectbox("削除対象", ["すべてのログをリセット(全削除)", "1ヶ月より前のログを削除", "3ヶ月より前のログを削除"], index=0)
            
            if st.button("🗑️ 選択したログを削除する", type="primary"):
                months = 3 if "3ヶ月" in del_months else (1 if "1ヶ月" in del_months else 0)
                with st.spinner("古いログを削除しています..."):
                    deleted_count = backend.delete_old_prediction_logs(months=months)
                    if deleted_count > 0:
                        st.success(f"{deleted_count}件の古い予測ログを削除しました！\n\n（※サイドバーの「データ更新」を押すと画面に反映されます）")
                        backend.load_prediction_log.clear()
                        st.cache_data.clear()
                        st.rerun()