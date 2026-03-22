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
    
    # --- 事後確率（設定5以上確率）の計算 ---
    specs = backend.get_machine_specs()
    def calc_post_prob(row):
        g = row.get('総回転', 0)
        act_b = row.get('BIG', 0)
        act_r = row.get('REG', 0)
        if pd.isna(g) or g <= 0: return 0
        
        machine = row.get('機種名', '')
        matched_spec = backend.get_matched_spec_key(machine, specs)
        
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
                
        return int(prob_5_over * 100)
        
    display_df['事後確率'] = display_df.apply(calc_post_prob, axis=1)
    
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
        post_prob = row.get('事後確率', 0)
        tot_prob = (b + r) / g if g > 0 else 0
        
        # 低稼働かつ確率が死んでない台は免除 (0G、または1000G未満で合算1/200以上)
        if g < 1000 and (g == 0 or tot_prob >= (1/200.0)): return False
        if diff < 0 or post_prob <= 30: return True
        return False
        
    display_df['is_bad_pred'] = display_df.apply(check_bad_pred, axis=1)

    # --- ランク変動の計算 ---
    if 'AI順位_num' in display_df.columns:
        # 事後確率（とタイブレーク用の差枚）で実際の結果順位を計算
        display_df['事後スコア'] = display_df['事後確率'] + (display_df.get('差枚', 0).fillna(0) / 100000.0)
        display_df['事後順位_num'] = display_df['事後スコア'].rank(method='min', ascending=False).fillna(999).astype(int)
        
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

    # --- 本日のAI精度パーセントの計算と表示 ---
    if 'prediction_score' in display_df.columns:
        ai_target_df = display_df[display_df['prediction_score'] >= 0.70]
        target_label = "AI推奨台(期待度70%以上)"
        if ai_target_df.empty and 'AI順位_num' in display_df.columns:
            ai_target_df = display_df[display_df['AI順位_num'] <= 10]
            target_label = "AI予測上位10台"
            
        if not ai_target_df.empty:
            hit_df = ai_target_df[(ai_target_df['差枚'] > 0) | (ai_target_df['事後確率'] >= 50)]
            hit_count = len(hit_df)
            total_target = len(ai_target_df)
            accuracy = hit_count / total_target * 100
            acc_color = "🟢" if accuracy >= 60 else "🟡" if accuracy >= 40 else "🔴"
            st.info(f"{acc_color} **本日のAI予測精度: {accuracy:.1f}%**\n\n{target_label} **{total_target}台** 中、**{hit_count}台** が見事プラス収支または高設定挙動でした！(赤背景は期待外れ台)")

    cols = ['AI順位', '台番号', '機種名']
    if '期待度' in display_df.columns: cols.append('期待度')
    if '予測信頼度' in display_df.columns: cols.append('予測信頼度')
    cols.append('事後確率')
    cols.extend(['差枚', '総回転', 'BIG', 'REG', '合算確率_str', 'REG確率_str', 'BIG確率_str', 'ぶどう確率_str'])
    
    if '根拠' in display_df.columns:
        cols.append('根拠')
        
    available_cols = [c for c in cols if c in display_df.columns]

    # Pandas Stylerを使って期待外れ台を赤くハイライト
    def apply_row_style(row):
        if display_df.loc[row.name, 'is_bad_pred']: return ['background-color: rgba(255, 75, 75, 0.2)'] * len(available_cols)
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
            "事後確率": st.column_config.ProgressColumn("結果(事後確率)", format="%d%%", min_value=0, max_value=100, help="実際の結果(BIG/REG回数)から統計的に逆算した、本当に設定5以上だった確率"),
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