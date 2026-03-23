import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import math
import backend

def render_daily_result_page(df_raw, df_events, df_island, shop_hyperparams):
    st.header("📅 日別 結果＆予測確認")
    st.caption("指定した日付の全台の結果と、「現在のAI設定」でその日を予測した場合のシミュレーション結果（期待度・信頼度）を照合できます。")

    if df_raw.empty:
        st.warning("データがありません。")
        return

    # 日付選択
    date_col = '対象日付'
    if date_col not in df_raw.columns:
        st.error("データに日付カラムがありません。")
        return

    # 日付型に変換して有効な日付リストを取得
    temp_df = df_raw.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
    available_dates = sorted(temp_df[date_col].dropna().dt.date.unique(), reverse=True)

    if not available_dates:
        st.warning("有効な日付データがありません。")
        return

    selected_date = st.selectbox("📅 確認する日付を選択", available_dates)

    # 店舗選択
    shop_col = '店名' if '店名' in temp_df.columns else ('店舗名' if '店舗名' in temp_df.columns else None)
    if not shop_col:
        st.error("データに店舗カラムがありません。")
        return

    df_day = temp_df[temp_df[date_col].dt.date == selected_date].copy()
    
    shops = ["店舗を選択してください"] + sorted(list(df_day[shop_col].unique()))
    
    default_index = 0
    saved_shop = st.session_state.get("global_selected_shop", "店舗を選択してください")
    if saved_shop in shops:
        default_index = shops.index(saved_shop)

    selected_shop = st.selectbox("🏬 店舗を選択", shops, index=default_index, key="daily_result_shop")
    
    if selected_shop != "店舗を選択してください":
        st.session_state["global_selected_shop"] = selected_shop

    if selected_shop == "店舗を選択してください":
        st.info("👆 店舗を選択すると、その日の全台結果とAIの事前の予測が表示されます。")
        return

    df_target = df_day[df_day[shop_col] == selected_shop].copy()
    df_target['台番号'] = df_target['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)

    # --- 現在のAIでバックテスト（シミュレーション）を実行 ---
    with st.spinner(f"🤖 現在のAI設定で {selected_date.strftime('%Y-%m-%d')} の予測をシミュレーション中..."):
        df_pred, _, _ = backend.run_analysis(df_raw, df_events, df_island, shop_hyperparams, target_date=selected_date)
        
    if not df_pred.empty:
        pred_shop_col = '店名' if '店名' in df_pred.columns else ('店舗名' if '店舗名' in df_pred.columns else None)
        if pred_shop_col:
            df_pred_target = df_pred[df_pred[pred_shop_col] == selected_shop].copy()
            
            if not df_pred_target.empty:
                df_pred_target['台番号'] = df_pred_target['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
                # 必要なカラムだけ残す
                cols_to_merge = ['台番号', 'prediction_score', '予測信頼度', '根拠']
                cols_to_merge = [c for c in cols_to_merge if c in df_pred_target.columns]
                
                df_target = pd.merge(df_target, df_pred_target[cols_to_merge], on='台番号', how='left')

    # 表示用データの整形
    display_df = df_target.copy()
    
    # 期待度をパーセント表記に
    if 'prediction_score' in display_df.columns:
        display_df['期待度'] = display_df['prediction_score'].apply(lambda x: f"{int(x * 100)}%" if pd.notna(x) and x != '' else "-")
        # 期待度に基づくAI順位の計算
        display_df['AI順位_num'] = display_df['prediction_score'].rank(method='min', ascending=False).fillna(999).astype(int)
        display_df['AI順位'] = display_df['AI順位_num'].apply(lambda x: f"{x}位" if x != 999 else "-")
    else:
        display_df['期待度'] = "-"
        display_df['AI順位'] = "-"

    if '予測信頼度' not in display_df.columns:
        display_df['予測信頼度'] = "-"

    display_df['総回転'] = display_df.get('累計ゲーム', 0)
    
    # 差枚や回数などを確実に整数型(int)にして小数点を消す
    for col in ['差枚', '総回転', 'BIG', 'REG']:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0).astype(int)
    
    
    # --- 結果点数（設定5近似度）の計算 ---
    specs = backend.get_machine_specs()
    def calculate_score(row, g_col='総回転', b_col='BIG', r_col='REG', m_col='機種名', diff_col='差枚'):
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
        
        # 学習基準(3000G)に合わせて、G数が増えるほど「確率のブレ(σ)による免除」を減らす
        sigma_multiplier = 0.5
        if g >= 5000:
            sigma_multiplier = 0.0
        elif g >= 3000:
            sigma_multiplier = 0.25
            
        deficit_r = max(0, exp_r - act_r)
        adjusted_deficit_r = max(0, deficit_r - (sigma_r * sigma_multiplier))
        
        deficit_b = max(0, exp_b - act_b)
        adjusted_deficit_b = max(0, deficit_b - (sigma_b * sigma_multiplier))
        
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
        
        if is_abandoned:
            total_score *= 0.5
            
        # 一定のゲーム数(2000G以上)回っていて確率が悪い場合は、言い訳無用で重く減点する
        if g >= 2000:
            reg_prob_den = g / act_r if act_r > 0 else 9999
            tot_prob_den = g / tot_b_r if tot_b_r > 0 else 9999
            
            if reg_prob_den > 400: total_score -= 30
            elif reg_prob_den > 300: total_score -= 15
                
            if tot_prob_den > 180: total_score -= 30
            elif tot_prob_den > 150: total_score -= 15
            
        return max(0.0, total_score)

    display_df['結果点数'] = display_df.apply(calculate_score, axis=1)
    
    if '推定ぶどう確率' in display_df.columns:
        display_df['ぶどう確率_str'] = display_df['推定ぶどう確率'].apply(lambda x: f"1/{x:.2f}" if pd.notna(x) else "-")
    else:
        display_df['ぶどう確率_str'] = "-"

    # --- 期待外れ台のフラグ計算 (ハイライト用) ---
    def check_bad_pred(row):
        score = row.get('prediction_score', 0)
        if pd.isna(score) or score < 0.70: return False
        
        g = row.get('総回転', 0)
        b = row.get('BIG', 0)
        r = row.get('REG', 0)
        diff = row.get('差枚', 0)
        tot_prob = (b + r) / g if g > 0 else 0
        
        # 低稼働かつ確率が死んでない台は免除 (0G、または1000G未満で合算1/200以上)
        if g < 1000 and (g == 0 or tot_prob >= (1/200.0)): return False
        if diff < 0 or row.get('結果点数', 0) < 40: return True
        return False
        
    display_df['is_bad_pred'] = display_df.apply(check_bad_pred, axis=1)

    # --- ランク変動の計算 ---
    if 'AI順位_num' in display_df.columns:
        # 結果点数で実際の結果順位を計算
        display_df['事後順位_num'] = display_df['結果点数'].rank(method='min', ascending=False).fillna(999).astype(int)
        
        def format_ai_rank(row):
            ai_r = row.get('AI順位_num', 999)
            if ai_r == 999: return "-"
            act_r = row.get('事後順位_num', 999)
            diff = ai_r - act_r
            if diff > 0:
                return f"{ai_r}位 (🔼+{diff})"
            elif diff < 0:
                return f"{ai_r}位 (🔻{diff})"
            else:
                return f"{ai_r}位 (➖)"
                
        display_df['AI順位'] = display_df.apply(format_ai_rank, axis=1)

    def format_prob(prob):
        try:
            p = float(prob)
            if p > 0: return f"1/{int(1/p)}"
        except: pass
        return "-"
        
    if 'BIG確率' in display_df.columns: display_df['BIG確率_str'] = display_df['BIG確率'].apply(format_prob)
    else: display_df['BIG確率_str'] = "-"
        
    if 'REG確率' in display_df.columns: display_df['REG確率_str'] = display_df['REG確率'].apply(format_prob)
    else: display_df['REG確率_str'] = "-"
    
    # 合算確率の計算とフォーマット
    display_df['tmp_total_prob'] = (display_df.get('BIG', 0).fillna(0) + display_df.get('REG', 0).fillna(0)) / display_df['総回転'].replace(0, np.nan)
    display_df['合算確率_str'] = display_df['tmp_total_prob'].apply(format_prob)
    
    # 並び替え
    sort_options = ["AI期待度順", "差枚が多い順", "合算確率が良い順", "REG確率が良い順", "台番号順"]
    sort_by = st.radio("並び替え", sort_options, horizontal=True)
    display_mode = st.radio("表示件数", ["厳選台 (上位10%)", "Top 10", "Top 20", "すべて"], horizontal=True, index=3)
    
    if sort_by == "差枚が多い順":
        display_df = display_df.sort_values('差枚', ascending=False)
    elif sort_by == "合算確率が良い順":
        display_df = display_df.sort_values('tmp_total_prob', ascending=False)
    elif sort_by == "REG確率が良い順":
        display_df['tmp_reg_prob'] = display_df.get('REG', 0).fillna(0) / display_df['総回転'].replace(0, np.nan)
        display_df = display_df.sort_values('tmp_reg_prob', ascending=False)
    elif sort_by == "AI期待度順":
        if 'prediction_score' in display_df.columns:
            display_df = display_df.sort_values('prediction_score', ascending=False)
    else:
        try:
            display_df['台番号_num'] = pd.to_numeric(display_df['台番号'])
            display_df = display_df.sort_values('台番号_num')
        except:
            display_df = display_df.sort_values('台番号')

    st.markdown(f"### 🎰 {selected_date.strftime('%Y-%m-%d')} の結果 ({selected_shop})")

    # --- 店舗全体の有効稼働勝率 ---
    all_g = pd.to_numeric(display_df['総回転'], errors='coerce').fillna(0)
    all_diff = pd.to_numeric(display_df['差枚'], errors='coerce').fillna(0)
    valid_all = display_df[(all_g >= 3000) | ((all_g < 3000) & (all_diff.abs() >= 1000))]
    if not valid_all.empty:
        all_win_c = (valid_all['差枚'] > 0).sum()
        all_win_rate_str = f"{all_win_c / len(valid_all):.1%} ({all_win_c}/{len(valid_all)}台)"
    else:
        all_win_rate_str = "- (0/0台)"
    st.caption(f"📊 店舗全体の勝率: **{all_win_rate_str}** (有効稼働のみ対象)")

    # --- 本日のAI精度パーセントの計算と表示 ---
    if 'prediction_score' in display_df.columns:
        ai_target_df = display_df[display_df['prediction_score'] >= 0.70]
        target_label = "AI推奨台(期待度70%以上)"
        if ai_target_df.empty and 'AI順位_num' in display_df.columns:
            ai_target_df = display_df[display_df['AI順位_num'] <= 10]
            target_label = "AI予測上位10台"
            
        if not ai_target_df.empty:
            hit_df = ai_target_df[(ai_target_df['差枚'] > 0) | (ai_target_df['結果点数'] >= 50)]
            hit_count = len(hit_df)
            total_target = len(ai_target_df)
            accuracy = hit_count / total_target * 100
            acc_color = "🟢" if accuracy >= 60 else "🟡" if accuracy >= 40 else "🔴"
            
            # --- AI推奨台の有効稼働勝率 ---
            act_g = pd.to_numeric(ai_target_df['総回転'], errors='coerce').fillna(0)
            act_diff = pd.to_numeric(ai_target_df['差枚'], errors='coerce').fillna(0)
            valid_df = ai_target_df[(act_g >= 3000) | ((act_g < 3000) & (act_diff.abs() >= 1000))]
            if not valid_df.empty:
                win_c = (valid_df['差枚'] > 0).sum()
                win_rate_str = f"{win_c / len(valid_df):.1%} ({win_c}/{len(valid_df)}台)"
            else:
                win_rate_str = "- (0/0台)"
                
            st.info(f"{acc_color} **本日のAI予測精度: {accuracy:.1f}%**\n\n{target_label} **{total_target}台** 中、**{hit_count}台** が見事プラス収支または高設定挙動でした！\n\n🎯 **推奨台の実質勝率: {win_rate_str}** (有効稼働のみ対象)\n\n※表の色について: 🟥赤背景=AI推奨の期待外れ台 / 🟨黄背景=結果点数が80点以上の優秀台")

    # --- 表示件数の絞り込み ---
    if display_mode == "厳選台 (上位10%)":
        limit = max(3, int(len(display_df) * 0.10))
        display_df = display_df.head(limit)
    elif display_mode == "Top 10":
        display_df = display_df.head(10)
    elif display_mode == "Top 20":
        display_df = display_df.head(20)

    cols = ['AI順位', '台番号', '機種名']
    if '期待度' in display_df.columns: cols.append('期待度')
    if '予測信頼度' in display_df.columns: cols.append('予測信頼度')
    cols.append('結果点数')
    cols.extend(['差枚', '総回転', 'BIG', 'REG', '合算確率_str', 'REG確率_str', 'BIG確率_str', 'ぶどう確率_str'])
    
    if '根拠' in display_df.columns:
        cols.append('根拠')
        
    available_cols = [c for c in cols if c in display_df.columns]

    # Pandas Stylerを使って期待外れ台を赤くハイライト
    def apply_row_style(row):
        row_data = display_df.loc[row.name]
        if row_data.get('is_bad_pred', False): 
            return ['background-color: rgba(255, 75, 75, 0.2)'] * len(available_cols)
        elif row_data.get('結果点数', 0) >= 80:
            return ['background-color: rgba(255, 215, 0, 0.2)'] * len(available_cols)
        return [''] * len(available_cols)
        
    styled_display_df = display_df[available_cols].style.apply(apply_row_style, axis=1)

    st.dataframe(
        styled_display_df,
        column_config={
            "AI順位": st.column_config.TextColumn("順位", width="small", help="AIの予測順位です。()内は実際の事後確率順位との変動（🔼:予測より好結果 / 🔻:予測より悪結果）を示します。"),
            "台番号": st.column_config.TextColumn("No.", width="small"),
            "機種名": st.column_config.TextColumn("機種", width="small"),
            "期待度": st.column_config.TextColumn("AI期待度", width="small", help="AIが前日時点で予測していた設定5以上の確率です。"),
            "予測信頼度": st.column_config.TextColumn("信頼度", width="small"),
            "結果点数": st.column_config.NumberColumn("結果点数", format="%.1f点", help="実際の結果に基づく設定5近似度"),
            "差枚": st.column_config.NumberColumn("差枚", format="%+d"),
            "総回転": st.column_config.NumberColumn("総回転", format="%d"),
            "BIG": st.column_config.NumberColumn("BIG", format="%d"),
            "REG": st.column_config.NumberColumn("REG", format="%d"),
            "合算確率_str": st.column_config.TextColumn("合算確率", width="small"),
            "BIG確率_str": st.column_config.TextColumn("BIG確率", width="small"),
            "REG確率_str": st.column_config.TextColumn("REG確率", width="small"),
            "ぶどう確率_str": st.column_config.TextColumn("🍇確率", width="small", help="差枚数から逆算した推定ぶどう確率 (ステータスOKの台のみ)"),
            "根拠": st.column_config.TextColumn("AI推奨根拠", width="large"),
        },
        width="stretch",
        hide_index=True
    )