import pandas as pd
import numpy as np
import math
import streamlit as st # type: ignore
import backend

def render_ranking_comparison_page(df_pred_log, df_verify, df_predict, df_raw):
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
    if '予測差枚数_saved' in base_df.columns:
        base_df['予測差枚数'] = base_df['予測差枚数_saved']
    if '機種名_saved' in base_df.columns:
        base_df['機種名'] = base_df['機種名_saved']
        
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

    if '対象日付' in merged_df.columns and not merged_df.empty:
        available_dates = sorted(merged_df['対象日付'].dropna().dt.date.unique(), reverse=True)
        if available_dates:
            selected_date = st.selectbox("📅 比較する対象日付を選択", available_dates, key="compare_date")
            
            shops_in_date = sorted(merged_df[merged_df['対象日付'].dt.date == selected_date][shop_col].unique())
            if shops_in_date:
                default_index = 0
                saved_shop = st.session_state.get("global_selected_shop", "")
                if saved_shop in shops_in_date:
                    default_index = shops_in_date.index(saved_shop)
                    
                compare_shop = st.selectbox("🏬 比較する店舗を選択", shops_in_date, index=default_index, key="compare_shop")
                st.session_state["global_selected_shop"] = compare_shop
            else:
                compare_shop = None
            
            if compare_shop:
                # --- 機種フィルター ---
                machines_in_date = ['すべての機種'] + sorted(merged_df[(merged_df['対象日付'].dt.date == selected_date) & (merged_df[shop_col] == compare_shop)]['機種名'].dropna().unique().tolist())
                selected_machine = st.selectbox("🎰 比較する機種を選択", machines_in_date, key="compare_machine")

                rank_metric = st.radio("📊 実際のランキング基準", ["差枚", "合算確率", "REG確率"], horizontal=True, help="合算確率とREG確率は、総ゲーム数3000G以上の台のみを対象とします。")

                # --- 全体実績: ランキングトップ3獲得回数 (店舗別) ---
                if selected_machine != 'すべての機種':
                    eval_days = merged_df[(merged_df[shop_col] == compare_shop) & (merged_df['機種名'] == selected_machine)][['実際の稼働日', shop_col]].dropna().drop_duplicates()
                    shop_machine_label = f"【{compare_shop} - {selected_machine}】"
                else:
                    eval_days = merged_df[merged_df[shop_col] == compare_shop][['実際の稼働日', shop_col]].dropna().drop_duplicates()
                    shop_machine_label = f"【{compare_shop}】"
                    
                top3_count = 0
                total_eval_days = len(eval_days)
                
                if total_eval_days > 0:
                    raw_subset = pd.merge(df_raw_temp, eval_days, left_on=['対象日付', shop_col], right_on=['実際の稼働日', shop_col], how='inner')
                    if selected_machine != 'すべての機種':
                        raw_subset = raw_subset[raw_subset['機種名'] == selected_machine]
                        
                    # 確率計算用の分母を準備
                    raw_g = pd.to_numeric(raw_subset['累計ゲーム'], errors='coerce').fillna(0)
                    raw_b = pd.to_numeric(raw_subset['BIG'], errors='coerce').fillna(0)
                    raw_r = pd.to_numeric(raw_subset['REG'], errors='coerce').fillna(0)
                    raw_subset['合算確率分母'] = np.where((raw_b + raw_r) > 0, raw_g / (raw_b + raw_r), 0).astype(int)
                    raw_subset['REG確率分母'] = np.where(raw_r > 0, raw_g / raw_r, 0).astype(int)
                    
                    if rank_metric == "差枚":
                        raw_subset = raw_subset.dropna(subset=['差枚'])
                        top3_machines = raw_subset.sort_values('差枚', ascending=False).groupby(['対象日付', shop_col]).head(3) if not raw_subset.empty else pd.DataFrame()
                    elif rank_metric == "合算確率":
                        raw_subset = raw_subset[(raw_g >= 3000) & (raw_subset['合算確率分母'] > 0)]
                        top3_machines = raw_subset.sort_values('合算確率分母', ascending=True).groupby(['対象日付', shop_col]).head(3) if not raw_subset.empty else pd.DataFrame()
                    else:
                        raw_subset = raw_subset[(raw_g >= 3000) & (raw_subset['REG確率分母'] > 0)]
                        top3_machines = raw_subset.sort_values('REG確率分母', ascending=True).groupby(['対象日付', shop_col]).head(3) if not raw_subset.empty else pd.DataFrame()

                    if not top3_machines.empty:
                        top3_machines = top3_machines[['対象日付', shop_col, '台番号']]
                        
                        top3_machines = top3_machines.rename(columns={'対象日付': '実際の稼働日'})
                        
                        # 画面表示と同じく「各日のAI予測Top10」に絞ってから合致判定を行う
                        valid_pred_df = merged_df[merged_df[shop_col] == compare_shop].copy()
                        if selected_machine != 'すべての機種':
                            valid_pred_df = valid_pred_df[valid_pred_df['機種名'] == selected_machine]
                        valid_pred_df = valid_pred_df.sort_values('prediction_score', ascending=False).groupby('実際の稼働日').head(10)
                        
                        match_df = pd.merge(valid_pred_df, top3_machines, on=['実際の稼働日', shop_col, '台番号'], how='inner')
                        top3_count = len(match_df.drop_duplicates(subset=['実際の稼働日', shop_col]))
                        
                        # AI推奨台(Top10)の通算勝率の計算
                        pred_g = pd.to_numeric(valid_pred_df['結果_累計ゲーム'], errors='coerce').fillna(0)
                        pred_diff = pd.to_numeric(valid_pred_df['差枚_actual'], errors='coerce').fillna(0)
                        valid_for_win = valid_pred_df[
                            (pred_g >= 3000) | ((pred_g < 3000) & (pred_diff.abs() >= 1000))
                        ]
                        win_rate_str = "- (0/0台)"
                        if not valid_for_win.empty:
                            total_win = (valid_for_win['差枚_actual'] > 0).sum()
                            total_valid = len(valid_for_win)
                            win_rate_str = f"{total_win/total_valid:.1%} ({total_win}/{total_valid}台)"
                            
                    st.info(f"👑 **{shop_machine_label}AI推奨台の通算実績 (過去 {total_eval_days} 日)**\n\n"
                            f"🏆 **トップ3獲得**: **{top3_count} 日** ランクイン (獲得率: {top3_count/total_eval_days:.1%})\n"
                            f"📈 **推奨台の勝率**: **{win_rate_str}**\n\n"
                            f"※トップ3は実際のランキング({rank_metric}順)に基づく。勝率は有効稼働(3000G以上 or 差枚±1000枚以上)の台のみを対象に集計しています。")

                # AI予測ランキング のデータ準備
                pred_df_day = merged_df[(merged_df['対象日付'].dt.date == selected_date) & (merged_df[shop_col] == compare_shop)].copy()
                if selected_machine != 'すべての機種':
                    pred_df_day = pred_df_day[pred_df_day['機種名'] == selected_machine]
                pred_df_day = pred_df_day.sort_values('prediction_score', ascending=False).head(10)
                
                # 設定5近似度の算出ロジックを定義
                specs = backend.get_machine_specs()
                def calculate_score(row, g_col='累計ゲーム', b_col='BIG', r_col='REG', m_col='機種名', diff_col='差枚'):
                    g = pd.to_numeric(row.get(g_col, 0), errors='coerce')
                    act_b = pd.to_numeric(row.get(b_col, 0), errors='coerce')
                    act_r = pd.to_numeric(row.get(r_col, 0), errors='coerce')
                    diff = pd.to_numeric(row.get(diff_col, 0), errors='coerce')
                    if pd.isna(g) or g <= 0: return np.nan
                    
                    machine = row.get(m_col, '')
                    matched_spec = backend.get_matched_spec_key(machine, specs)
                    p_b, p_r = 1/259.0, 1/255.0
                    if matched_spec and "設定5" in specs[matched_spec]:
                        s5 = specs[matched_spec]["設定5"]
                        if "BIG" in s5: p_b = 1.0 / s5["BIG"]
                        if "REG" in s5: p_r = 1.0 / s5["REG"]
                        
                    exp_b, exp_r = g * p_b, g * p_r
                    
                    sigma_r = math.sqrt(g * p_r * (1.0 - p_r)) if g > 0 else 0
                    sigma_b = math.sqrt(g * p_b * (1.0 - p_b)) if g > 0 else 0
                    
                    deficit_r = max(0, exp_r - act_r)
                    adjusted_deficit_r = max(0, deficit_r - (sigma_r * 0.5))
                    
                    deficit_b = max(0, exp_b - act_b)
                    adjusted_deficit_b = max(0, deficit_b - (sigma_b * 0.5))
                    
                    penalty_reg = st.session_state.get('penalty_reg', 15)
                    penalty_big = st.session_state.get('penalty_big', 5)
                    low_g_penalty = st.session_state.get('low_g_penalty', 30)
                    
                    score_r = max(0, 80 - (adjusted_deficit_r * penalty_reg))
                    score_b = max(0, 20 - (adjusted_deficit_b * penalty_big))
                    
                    total_score = score_r + score_b
                    
                    if g < 5000:
                        multiplier = 0.80 + (g / 5000.0) * 0.20
                        total_score *= multiplier
                        
                    if g < 1000:
                        total_score *= (1 - ((1000 - g) / 1000.0) * (low_g_penalty / 100.0))
                        
                    if g >= 7000 and adjusted_deficit_r <= 0:
                        bonus = min(5.0, (g - 7000) / 500.0)
                        total_score = min(100.0, total_score + bonus)
                        
                    is_abandoned = False
                    tot_b_r = act_b + act_r
                    if g >= 500 and tot_b_r == 0: is_abandoned = True
                    elif g >= 1000 and tot_b_r > 0 and (g / tot_b_r) >= 400: is_abandoned = True
                    elif g >= 1500 and tot_b_r > 0 and (g / tot_b_r) >= 300: is_abandoned = True
                    
                    valid_play = (g >= 3000) or (abs(diff) >= 1000)
                    
                    if not valid_play and not is_abandoned:
                        return np.nan
                        
                    if is_abandoned:
                        total_score *= 0.5
                        
                    # 確率ベースの強制減点 (G数が少ない場合の「0.5σ保護」が過剰に効いてしまうのを防ぐ)
                    reg_prob_den = g / act_r if act_r > 0 else 9999
                    tot_prob_den = g / tot_b_r if tot_b_r > 0 else 9999
                    
                    if reg_prob_den > 400: total_score -= 30
                    elif reg_prob_den > 300: total_score -= 15
                        
                    if tot_prob_den > 180: total_score -= 30
                    elif tot_prob_den > 150: total_score -= 15
                        
                    return max(0.0, total_score)

                # 実際のランキング のデータ準備
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
                    if selected_machine != 'すべての機種':
                        actual_df_day = actual_df_day[actual_df_day['機種名'] == selected_machine]
                else:
                    actual_df_day = pd.DataFrame()

                # --- 実際のランキングTop10を事前に計算 (照合用) ---
                actual_top10_machines = []
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
                    
                    if rank_metric == "差枚":
                        actual_df_day = actual_df_day.sort_values('差枚', ascending=False).head(10)
                    else:
                        actual_df_day = actual_df_day[actual_df_day['累計ゲーム'] >= 3000]
                        if rank_metric == "合算確率":
                            actual_df_day = actual_df_day[actual_df_day['合算確率分母'] > 0].sort_values('合算確率分母', ascending=True).head(10)
                        else:
                            actual_df_day = actual_df_day[actual_df_day['REG確率分母'] > 0].sort_values('REG確率分母', ascending=True).head(10)
                            
                    actual_top10_machines = actual_df_day['台番号'].tolist()

                # --- 上段: AI予測ランキング ---
                if selected_machine != 'すべての機種':
                    st.markdown(f"##### 🤖 AI推奨台 ({selected_machine})")
                else:
                    st.markdown("##### 🤖 AI推奨台 (全体)")
                
                if not pred_df_day.empty:
                    pred_g = pd.to_numeric(pred_df_day['結果_累計ゲーム'], errors='coerce').fillna(0)
                    pred_diff = pd.to_numeric(pred_df_day['差枚_actual'], errors='coerce').fillna(0)
                    valid_pred = pred_df_day[(pred_g >= 3000) | ((pred_g < 3000) & (pred_diff.abs() >= 1000))]
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

                    # トップ10ランクインのマーク
                    pred_df_day['的中'] = pred_df_day['台番号'].apply(lambda x: '🎯' if x in actual_top10_machines else '')

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
                if selected_machine != 'すべての機種':
                    st.markdown(f"##### 🎰 実績 Top10 ({selected_machine})")
                else:
                    st.markdown(f"##### 🎰 実績 Top10 (全体)")
                
                if not actual_df_day.empty:
                    act_g = pd.to_numeric(actual_df_day['累計ゲーム'], errors='coerce').fillna(0)
                    act_diff = pd.to_numeric(actual_df_day['差枚'], errors='coerce').fillna(0)
                    valid_act = actual_df_day[(act_g >= 3000) | ((act_g < 3000) & (act_diff.abs() >= 1000))]
                    if not valid_act.empty:
                        win_c_act = (valid_act['差枚'] > 0).sum()
                        daily_act_win_rate = f"{win_c_act / len(valid_act):.1%} ({win_c_act}/{len(valid_act)}台)"
                    else:
                        daily_act_win_rate = "- (0/0台)"
                    st.caption(f"※稼働日: {date_str_actual} ({rank_metric}順) ｜ 🎯 勝率: **{daily_act_win_rate}** (有効稼働のみ)")
                else:
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