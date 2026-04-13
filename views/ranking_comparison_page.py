import pandas as pd
import numpy as np
import math
import streamlit as st # type: ignore
import backend

def render_ranking_comparison_page(df_pred_log, df_verify, df_predict, df_raw, selected_shop):
    st.subheader("🏆 日別の予測 vs 実際 ランキング比較")
    st.caption("指定した日付の「AI予測ランキング」と「実際の結果ランキング」を並べて表示し、AIの推奨台が実際のランキングにどれくらい食い込んでいるかを確認します。")

    if df_pred_log.empty:
        st.warning("保存された予測結果ログがありません。日々の予測結果は、ログイン時に自動で保存されます。")
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
        df_pred_log['実行日時'] = pd.to_datetime(df_pred_log['実行日時'], errors='coerce')
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
    if '機種名_saved' in base_df.columns:
        base_df['機種名'] = base_df['機種名_saved']
        
    # --- 実際の成績を df_raw から直接取得 (未稼働台のデータ落ちを防ぐ) ---
    if '台番号' in df_raw.columns:
        df_raw_temp = df_raw.copy()
        df_raw_temp['台番号'] = df_raw_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_raw_temp['対象日付'] = pd.to_datetime(df_raw_temp['対象日付'], errors='coerce')
        
        # 実績データ側の店舗カラム名を統一
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
        target_date = row.get('対象日付') 
        base_date = row.get('予測ベース日') 
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
            return pd.Series([0, 0, 0, 0, actual_date])
            
    results = base_df.apply(get_actual_result, axis=1)
    base_df[['差枚_actual', '結果_BIG', '結果_REG', '結果_累計ゲーム', '実際の稼働日']] = results
    
    cum_g = pd.to_numeric(base_df['結果_累計ゲーム'], errors='coerce').fillna(0)
    big_c = pd.to_numeric(base_df['結果_BIG'], errors='coerce').fillna(0)
    reg_c = pd.to_numeric(base_df['結果_REG'], errors='coerce').fillna(0)
    base_df['結果_合算確率分母'] = np.where((big_c + reg_c) > 0, cum_g / (big_c + reg_c), 0)
    base_df['結果_BIG確率分母'] = np.where(big_c > 0, cum_g / big_c, 0)
    base_df['結果_REG確率分母'] = np.where(reg_c > 0, cum_g / reg_c, 0)

    merged_df = base_df.dropna(subset=['差枚_actual', 'prediction_score']).copy()

    if merged_df.empty:
        st.info("まだ結果が判明している予測がありません。")
        return

    if shop_col in merged_df.columns:
        merged_df = merged_df[merged_df[shop_col] == selected_shop].copy()

    if '対象日付' in merged_df.columns and not merged_df.empty:
        available_dates = sorted(merged_df['対象日付'].dropna().dt.date.unique(), reverse=True)
        if available_dates:
            selected_date = st.selectbox("📅 比較する対象日付を選択", available_dates, key="compare_date")
            
            compare_shop = selected_shop
            
            if compare_shop:
                rank_metric = st.radio("📊 実際のランキング基準", ["差枚", "合算確率", "REG確率"], horizontal=True, help="合算確率とREG確率は、総ゲーム数3000G以上の台のみを対象とします。")

                # --- 全体実績: ランキングトップ3獲得回数や通算差枚 (店舗別) ---
                period_options = {"直近1週間": 7, "直近1ヶ月": 30, "直近3ヶ月": 90, "全期間": None}
                selected_period = st.radio("通算成績の集計期間", list(period_options.keys()), index=1, horizontal=True)

                base_eval_df = merged_df.copy()
                shop_machine_label = f"【{compare_shop}】"
                    
                if period_options[selected_period] is not None:
                    max_date = base_eval_df['実際の稼働日'].max()
                    cutoff_date = max_date - pd.Timedelta(days=period_options[selected_period])
                    base_eval_df = base_eval_df[base_eval_df['実際の稼働日'] > cutoff_date]

                eval_days = base_eval_df[['実際の稼働日', shop_col]].dropna().drop_duplicates()
                    
                top3_count = 0
                total_eval_days = len(eval_days)
                
                if total_eval_days > 0:
                    raw_subset = pd.merge(df_raw_temp, eval_days, left_on=['対象日付', shop_col], right_on=['実際の稼働日', shop_col], how='inner')
                        
                    # 確率計算用の分母を準備
                    raw_g = pd.to_numeric(raw_subset['累計ゲーム'], errors='coerce').fillna(0)
                    raw_b = pd.to_numeric(raw_subset['BIG'], errors='coerce').fillna(0)
                    raw_r = pd.to_numeric(raw_subset['REG'], errors='coerce').fillna(0)
                    raw_subset['合算確率分母'] = np.where((raw_b + raw_r) > 0, raw_g / (raw_b + raw_r), 0).astype(int)
                    raw_subset['REG確率分母'] = np.where(raw_r > 0, raw_g / raw_r, 0).astype(int)
                    
                    if rank_metric == "差枚":
                        raw_subset = raw_subset.dropna(subset=['差枚'])
                        raw_subset['actual_rank'] = raw_subset.groupby(['対象日付', shop_col])['差枚'].rank(method='first', ascending=False)
                    elif rank_metric == "合算確率":
                        raw_subset = raw_subset[(raw_g >= 3000) & (raw_subset['合算確率分母'] > 0)]
                        raw_subset['actual_rank'] = raw_subset.groupby(['対象日付', shop_col])['合算確率分母'].rank(method='first', ascending=True)
                    else:
                        raw_subset = raw_subset[(raw_g >= 3000) & (raw_subset['REG確率分母'] > 0)]
                        raw_subset['actual_rank'] = raw_subset.groupby(['対象日付', shop_col])['REG確率分母'].rank(method='first', ascending=True)
                        
                    if 'actual_rank' in raw_subset.columns:
                        top3_machines = raw_subset[raw_subset['actual_rank'] <= 3][['対象日付', shop_col, '台番号', 'actual_rank']]
                    else:
                        top3_machines = pd.DataFrame()

                    if not top3_machines.empty:
                        top3_machines = top3_machines.rename(columns={'対象日付': '実際の稼働日'})
                        
                    # 保存されている予測ログはすでに上位10%に絞られているため、そのまま使用する
                    valid_pred_df = base_eval_df.copy()
                    valid_pred_df['ai_daily_rank'] = valid_pred_df.groupby(['実際の稼働日', shop_col])['prediction_score'].rank(method='first', ascending=False)
                    
                    if not top3_machines.empty:
                        match_df = pd.merge(valid_pred_df, top3_machines, on=['実際の稼働日', shop_col, '台番号'], how='left')
                        match_df['is_top3'] = match_df['actual_rank'].notna().astype(int)
                    else:
                        match_df = valid_pred_df.copy()
                        match_df['is_top3'] = 0
                    
                    # 少なくとも1台がTop3に入った日数
                    top3_days = match_df[match_df['is_top3'] == 1].drop_duplicates(subset=['実際の稼働日', shop_col])
                    top3_count = len(top3_days)
                    
                    # AI推奨台(上位10%)の通算勝率の計算
                    pred_g = pd.to_numeric(valid_pred_df['結果_累計ゲーム'], errors='coerce').fillna(0)
                    pred_diff = pd.to_numeric(valid_pred_df['差枚_actual'], errors='coerce').fillna(0)
                    valid_for_win = valid_pred_df[
                        (pred_g >= 3000) | ((pred_g < 3000) & ((pred_diff <= -750) | (pred_diff >= 750)))
                    ]
                    win_rate_str = "- (0/0台)"
                    if not valid_for_win.empty:
                        total_win = (valid_for_win['差枚_actual'] > 0).sum()
                        total_valid = len(valid_for_win)
                        win_rate_str = f"{total_win/total_valid:.1%} ({total_win}/{total_valid}台)"
                        
                    # 上位10%の過去累計差枚
                    total_diff_sum = pred_diff.sum()
                            
                    st.info(f"👑 **{shop_machine_label}AI推奨台(上位10%)の通算実績 ({selected_period})**\n\n"
                            f"📅 **評価日数**: {total_eval_days} 日\n"
                            f"🏆 **トップ3獲得**: **{top3_count} 日** ランクイン (獲得率: {top3_count/total_eval_days:.1%})\n"
                            f"📈 **推奨台の勝率**: **{win_rate_str}**\n\n"
                            f"💰 **推奨台の合計差枚**: **{int(total_diff_sum):+d} 枚**\n\n"
                            f"※トップ3は実際のランキング({rank_metric}順)に基づく。勝率は有効稼働(3000G以上 or 差枚が+750枚以上・-750枚以下)の台のみを対象に集計しています。")
                    
                    # 各AI順位ごとの通算実績
                    with st.expander("📊 AI順位ごとの通算成績", expanded=False):
                        # 勝率計算用のフラグを追加
                        act_g = pd.to_numeric(match_df['結果_累計ゲーム'], errors='coerce').fillna(0)
                        act_diff = pd.to_numeric(match_df['差枚_actual'], errors='coerce').fillna(0)
                        match_df['valid_play'] = (act_g >= 3000) | ((act_g < 3000) & ((act_diff <= -750) | (act_diff >= 750)))
                        match_df['is_win'] = match_df['valid_play'] & (act_diff > 0)
                        match_df['valid_差枚_actual'] = np.where(match_df['valid_play'], match_df['差枚_actual'], np.nan)
                        
                        rank_stats = match_df.groupby('ai_daily_rank').agg(
                            検証台数=('台番号', 'count'),
                            有効稼働数=('valid_play', 'sum'),
                            勝数=('is_win', 'sum'),
                            合計差枚=('差枚_actual', 'sum'),
                            平均差枚=('valid_差枚_actual', 'mean'),
                            トップ3獲得数=('is_top3', 'sum'),
                            平均期待度=('prediction_score', 'mean')
                        ).reset_index()
                        
                        rank_stats['勝率'] = np.where(rank_stats['有効稼働数'] > 0, (rank_stats['勝数'] / rank_stats['有効稼働数']) * 100, 0.0)
                        rank_stats['平均期待度'] = rank_stats['平均期待度'] * 100
                        rank_stats['ai_daily_rank'] = rank_stats['ai_daily_rank'].astype(int)
                        rank_stats = rank_stats.sort_values('ai_daily_rank')
                        
                        # 集計行の計算と追加
                        total_count = rank_stats['検証台数'].sum()
                        total_valid = rank_stats['有効稼働数'].sum()
                        total_win = rank_stats['勝数'].sum()
                        total_diff = rank_stats['合計差枚'].sum()
                        total_top3 = rank_stats['トップ3獲得数'].sum()
                        avg_diff = total_diff / total_valid if total_valid > 0 else 0
                        total_win_rate = (total_win / total_valid) * 100 if total_valid > 0 else 0.0
                        total_avg_score = match_df['prediction_score'].mean() * 100 if not match_df.empty else 0.0
                        
                        rank_stats['ai_daily_rank'] = rank_stats['ai_daily_rank'].astype(str) + "位"
                        total_row = pd.DataFrame([{
                            'ai_daily_rank': '合計/平均',
                            '平均期待度': total_avg_score,
                            '検証台数': total_count,
                            '有効稼働数': total_valid,
                            '勝率': total_win_rate,
                            '合計差枚': total_diff,
                            '平均差枚': avg_diff,
                            'トップ3獲得数': total_top3
                        }])
                        rank_stats = pd.concat([rank_stats, total_row], ignore_index=True)
                        
                        st.dataframe(
                            rank_stats[['ai_daily_rank', '平均期待度', '検証台数', '有効稼働数', '勝率', '合計差枚', '平均差枚', 'トップ3獲得数']],
                            column_config={
                                "ai_daily_rank": st.column_config.TextColumn("AI予測順位"),
                                "平均期待度": st.column_config.ProgressColumn("平均期待度", format="%.1f%%", min_value=0, max_value=100),
                                "検証台数": st.column_config.NumberColumn("台数"),
                                "有効稼働数": st.column_config.NumberColumn("有効稼働"),
                                "勝率": st.column_config.ProgressColumn("勝率(有効稼働)", format="%.1f%%", min_value=0, max_value=100),
                                "合計差枚": st.column_config.NumberColumn("合計差枚", format="%+d 枚"),
                                "平均差枚": st.column_config.NumberColumn("平均差枚", format="%+d 枚"),
                                "トップ3獲得数": st.column_config.NumberColumn("Top3的中", format="%d 回"),
                            },
                            hide_index=True,
                            use_container_width=True
                        )

                # 実際のランキング のデータ準備 (平均ゲーム数計算のため先に実行)
                target_ts = pd.Timestamp(selected_date)
                actual_date = pd.NaT
                if compare_shop in shop_operating_dates:
                    for d in shop_operating_dates[compare_shop]:
                        if d >= target_ts:
                            actual_date = d
                            break
                            
                if pd.notna(actual_date) and not df_raw_temp.empty:
                    actual_df_day = df_raw_temp[
                        (df_raw_temp[shop_col] == compare_shop) & 
                        (df_raw_temp['対象日付'] == actual_date)
                    ].copy()
                else:
                    actual_df_day = pd.DataFrame()

                # AI予測ランキング のデータ準備
                pred_df_day = merged_df[merged_df['対象日付'].dt.date == selected_date].copy()
                pred_df_day = pred_df_day.sort_values('prediction_score', ascending=False)
                
                shop_avg_g_actual = actual_df_day['累計ゲーム'].mean() if not actual_df_day.empty else 4000

                # 設定5近似度の算出ロジックを定義
                def calculate_score(row, g_col='累計ゲーム', b_col='BIG', r_col='REG', m_col='機種名', d_col='差枚'):
                    g = pd.to_numeric(row.get(g_col, 0), errors='coerce')
                    act_b = pd.to_numeric(row.get(b_col, 0), errors='coerce')
                    act_r = pd.to_numeric(row.get(r_col, 0), errors='coerce')
                    diff = pd.to_numeric(row.get(d_col, 0), errors='coerce')
                    machine = row.get(m_col, '')
                    
                    penalty_reg = st.session_state.get('penalty_reg', 15)
                    penalty_big = st.session_state.get('penalty_big', 5)
                    low_g_penalty = st.session_state.get('low_g_penalty', 30)
                    
                    return backend.calculate_setting_score(
                        g=g, act_b=act_b, act_r=act_r, machine_name=machine, diff=diff,
                        shop_avg_g=shop_avg_g_actual, penalty_reg=penalty_reg, penalty_big=penalty_big,
                        low_g_penalty=low_g_penalty, use_strict_scoring=True, return_details=False
                    )

                # --- 実際のランキング上位10%を事前に計算 (照合用) ---
                actual_top_machines = []
                if not actual_df_day.empty:
                    act_g = pd.to_numeric(actual_df_day['累計ゲーム'], errors='coerce').fillna(0)
                    act_b = pd.to_numeric(actual_df_day['BIG'], errors='coerce').fillna(0)
                    act_r = pd.to_numeric(actual_df_day['REG'], errors='coerce').fillna(0)
                    actual_df_day['合算回数'] = act_b + act_r
                    actual_df_day['BIG確率分母'] = np.where(act_b > 0, act_g / act_b, 0).astype(int)
                    actual_df_day['REG確率分母'] = np.where(act_r > 0, act_g / act_r, 0).astype(int)
                    actual_df_day['合算確率分母'] = np.where(actual_df_day['合算回数'] > 0, act_g / actual_df_day['合算回数'], 0).astype(int)
                    
                    specs = backend.get_machine_specs()
                    spec_reg_val = actual_df_day['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
                    spec_tot_val = actual_df_day['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
                    actual_df_day['高設定'] = (((actual_df_day['REG確率分母'] > 0) & (actual_df_day['REG確率分母'] <= spec_reg_val)) | ((actual_df_day['合算確率分母'] > 0) & (actual_df_day['合算確率分母'] <= spec_tot_val))).apply(lambda x: '🌟' if x else '')
                    
                    act_total_machines = len(actual_df_day)
                    act_top_k = max(3, int(act_total_machines * 0.10)) if act_total_machines > 0 else 10

                    if rank_metric == "差枚":
                        actual_df_day = actual_df_day.sort_values('差枚', ascending=False).head(act_top_k)
                    else:
                        actual_df_day = actual_df_day[actual_df_day['累計ゲーム'] >= 3000]
                        if rank_metric == "合算確率":
                            actual_df_day = actual_df_day[actual_df_day['合算確率分母'] > 0].sort_values('合算確率分母', ascending=True).head(act_top_k)
                        else:
                            actual_df_day = actual_df_day[actual_df_day['REG確率分母'] > 0].sort_values('REG確率分母', ascending=True).head(act_top_k)
                            
                    actual_top_machines = actual_df_day['台番号'].tolist()
                    
                # 当日の店舗全体の答え合わせサマリー
                avg_pred_score = pred_df_day['prediction_score'].mean() if not pred_df_day.empty and 'prediction_score' in pred_df_day.columns else np.nan
                day_eval_str = "🔥 還元日予測" if pd.notna(avg_pred_score) and avg_pred_score >= 0.20 else "🥶 回収日予測" if pd.notna(avg_pred_score) and avg_pred_score < 0.10 else "⚖️ 通常営業予測" if pd.notna(avg_pred_score) else "不明"
                actual_avg_diff = actual_df_day['差枚'].mean() if not actual_df_day.empty and '差枚' in actual_df_day.columns else np.nan
                actual_diff_str = f"{int(actual_avg_diff):+d} 枚" if pd.notna(actual_avg_diff) else "不明"
                
                st.markdown(f"##### 🎯 【{compare_shop}】 {selected_date} のランキング比較")
                st.caption(f"**店舗全体のAI事前評価**: {avg_pred_score*100:.1f}% ({day_eval_str}) ｜ **実際の店舗全体平均差枚**: {actual_diff_str}")

                # --- 上段: AI予測ランキング ---
                with st.expander("🤖 AI推奨台 (上位10%)", expanded=True):
                    if not pred_df_day.empty:
                        pred_g = pd.to_numeric(pred_df_day['結果_累計ゲーム'], errors='coerce').fillna(0)
                        pred_diff = pd.to_numeric(pred_df_day['差枚_actual'], errors='coerce').fillna(0)
                        valid_pred = pred_df_day[(pred_g >= 3000) | ((pred_g < 3000) & ((pred_diff <= -750) | (pred_diff >= 750)))]
                        if not valid_pred.empty:
                            win_c = (valid_pred['差枚_actual'] > 0).sum()
                            daily_win_rate = f"{win_c / len(valid_pred):.1%} ({win_c}/{len(valid_pred)}台)"
                        else:
                            daily_win_rate = "- (0/0台)"
                        st.caption(f"※対象日: {selected_date} ｜ 🎯 勝率: **{daily_win_rate}** (有効稼働のみ)")
                    else:
                        st.caption(f"※対象日: {selected_date}")
                        
                    if pred_df_day.empty:
                        st.info("この日の予測ログがありません。")
                    else:
                        if 'prediction_score' in pred_df_day.columns:
                            pred_df_day['予想設定5以上確率'] = (pred_df_day['prediction_score'] * 100).astype(int)
                        else:
                            pred_df_day['予想設定5以上確率'] = 0

                        pred_df_day['結果点数'] = pred_df_day.apply(lambda row: calculate_score(row, '結果_累計ゲーム', '結果_BIG', '結果_REG', '機種名', '差枚_actual'), axis=1)

                        # 上位10%ランクインのマーク
                        pred_df_day['的中'] = pred_df_day['台番号'].apply(lambda x: '🎯' if x in actual_top_machines else '')

                        # 高設定挙動のマーク
                        specs = backend.get_machine_specs()
                        spec_reg_val_pred = pred_df_day['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
                        spec_tot_val_pred = pred_df_day['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
                        pred_df_day['高設定'] = (((pred_df_day['結果_REG確率分母'] > 0) & (pred_df_day['結果_REG確率分母'] <= spec_reg_val_pred)) | ((pred_df_day['結果_合算確率分母'] > 0) & (pred_df_day['結果_合算確率分母'] <= spec_tot_val_pred))).apply(lambda x: '🌟' if x else '')

                        display_cols_pred = ['的中', '高設定', '台番号', '機種名', '予想設定5以上確率', '結果点数', '差枚_actual', '結果_累計ゲーム', '結果_BIG確率分母', '結果_REG確率分母', '結果_合算確率分母']
                        
                        def highlight_positive(row):
                            if row.get('的中', '') == '🎯':
                                return ['background-color: rgba(255, 215, 0, 0.3)'] * len(row) # 的中なら少し強めの黄色
                            elif pd.notna(row['差枚_actual']) and row['差枚_actual'] > 0:
                                return ['background-color: rgba(255, 75, 75, 0.2)'] * len(row)
                            return [''] * len(row)
                            
                        styled_pred_df = pred_df_day[display_cols_pred].style.apply(highlight_positive, axis=1)
                        if '差枚_actual' in display_cols_pred:
                            styled_pred_df = styled_pred_df.bar(subset=['差枚_actual'], align='mid', color=['rgba(66, 165, 245, 0.5)', 'rgba(255, 112, 67, 0.5)'], vmin=-3000, vmax=3000)

                        st.dataframe(
                            styled_pred_df,
                            column_config={
                                "的中": st.column_config.TextColumn("的中", width="small"),
                                "高設定": st.column_config.TextColumn("挙動", width="small", help="設定5以上の確率(REGか合算)の台"),
                                "台番号": st.column_config.TextColumn("台番号"),
                                "機種名": st.column_config.TextColumn("機種", width="small"),
                                "予想設定5以上確率": st.column_config.ProgressColumn("期待度", format="%d%%", min_value=0, max_value=100),
                                "結果点数": st.column_config.NumberColumn("結果点数", format="%.1f点", help="実際の結果に基づく設定5近似度"),
                                "差枚_actual": st.column_config.NumberColumn("結果差枚", format="%+d"),
                                "結果_累計ゲーム": st.column_config.NumberColumn("総G", format="%d"),
                                "結果_BIG確率分母": st.column_config.NumberColumn("BB", format="1/%d"),
                                "結果_REG確率分母": st.column_config.NumberColumn("RB", format="1/%d"),
                                "結果_合算確率分母": st.column_config.NumberColumn("合算", format="1/%d"),
                            },
                            hide_index=True,
                            width="stretch"
                        )

                st.divider()

                # --- 下段: 実績ランキング ---
                date_str_actual = actual_date.strftime('%Y-%m-%d') if pd.notna(actual_date) else "不明"
                with st.expander(f"🎰 実績 上位10% (全体) - {date_str_actual}", expanded=True):
                    st.caption(f"※稼働日: {date_str_actual} ({rank_metric}順)")
                        
                    if actual_df_day.empty:
                        st.info("条件を満たす結果データがありません。")
                    else:
                        actual_df_day = actual_df_day.reset_index(drop=True)
                        actual_df_day.index = actual_df_day.index + 1
                        
                        def get_rank_medal(rank):
                            if rank == 1: return '🥇 1位'
                            elif rank == 2: return '🥈 2位'
                            elif rank == 3: return '🥉 3位'
                            else: return f'{rank}位'
                            
                        actual_df_day['順位'] = actual_df_day.index.map(get_rank_medal)
                        
                        actual_df_day['結果点数'] = actual_df_day.apply(lambda row: calculate_score(row, '累計ゲーム', 'BIG', 'REG', '機種名', '差枚'), axis=1)

                        # AIが推奨していた台にはマークをつける
                        pred_machines = pred_df_day['台番号'].tolist()
                        actual_df_day['AI推奨'] = actual_df_day['台番号'].apply(lambda x: '🎯' if x in pred_machines else '')
                        
                        display_cols = ['AI推奨', '高設定', '順位', '台番号', '機種名', '結果点数', '差枚', '累計ゲーム', 'BIG確率分母', 'REG確率分母', '合算確率分母']
                        
                        def highlight_top3(row):
                            if '1位' in row['順位']:
                                return ['background-color: rgba(255, 215, 0, 0.2)'] * len(row)
                            elif '2位' in row['順位']:
                                return ['background-color: rgba(192, 192, 192, 0.2)'] * len(row)
                            elif '3位' in row['順位']:
                                return ['background-color: rgba(205, 127, 50, 0.2)'] * len(row)
                            return [''] * len(row)
                            
                        styled_df = actual_df_day[display_cols].style.apply(highlight_top3, axis=1)
                        if '差枚' in display_cols:
                            styled_df = styled_df.bar(subset=['差枚'], align='mid', color=['rgba(66, 165, 245, 0.5)', 'rgba(255, 112, 67, 0.5)'], vmin=-3000, vmax=3000)
                        
                        st.dataframe(
                            styled_df,
                            column_config={
                                "AI推奨": st.column_config.TextColumn("予測", width="small"),
                                "高設定": st.column_config.TextColumn("挙動", width="small", help="設定5以上の確率(REGか合算)の台"),
                                "順位": st.column_config.TextColumn("順位"),
                                "台番号": st.column_config.TextColumn("台番号"),
                                "機種名": st.column_config.TextColumn("機種", width="small"),
                                "結果点数": st.column_config.NumberColumn("結果点数", format="%.1f点", help="実際の結果に基づく設定5近似度"),
                                "差枚": st.column_config.NumberColumn("差枚", format="%+d"),
                                "累計ゲーム": st.column_config.NumberColumn("総G", format="%d"),
                                "BIG確率分母": st.column_config.NumberColumn("BB", format="1/%d"),
                                "REG確率分母": st.column_config.NumberColumn("RB", format="1/%d"),
                                "合算確率分母": st.column_config.NumberColumn("合算", format="1/%d"),
                            },
                            hide_index=True,
                            use_container_width=True
                        )