import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import math
import backend

def render_daily_result_page(df_raw, df_events, df_island, shop_hyperparams):
    st.header("📅 日別 結果＆予測確認")
    st.caption("指定した日付の全台の結果と、「現在のAI設定」でその日を予測した場合のシミュレーション結果（期待度・信頼度）を照合できます。")

    specs = backend.get_machine_specs()

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
        df_pred, _, _ = backend.run_analysis(df_raw, _df_events=df_events, _df_island=df_island, shop_hyperparams=shop_hyperparams, target_date=selected_date)
        
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
    
    # REG確率がなければ計算する
    if 'REG確率' not in display_df.columns:
        display_df['REG確率'] = display_df['REG'] / display_df['総回転'].replace(0, np.nan)

    # --- 結果点数（設定5近似度）の計算 ---
    shop_avg_g = display_df['総回転'].mean() if not display_df.empty else 4000

    def calculate_score(row, g_col='総回転', b_col='BIG', r_col='REG', m_col='機種名', d_col='差枚'):
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
            shop_avg_g=shop_avg_g, penalty_reg=penalty_reg, penalty_big=penalty_big,
            low_g_penalty=low_g_penalty, use_strict_scoring=True, return_details=False
        )

    display_df['結果点数'] = display_df.apply(calculate_score, axis=1)
    
    if '推定ぶどう確率' in display_df.columns:
        display_df['ぶどう確率_str'] = display_df['推定ぶどう確率'].apply(lambda x: f"1/{x:.2f}" if pd.notna(x) else "-")
    else:
        display_df['ぶどう確率_str'] = "-"

    # --- REG確率が設定5基準を上回るか判定 (ハイライト用) ---
    def check_high_reg(row):
        if '機種名' in row and 'REG確率' in row and pd.notna(row['REG確率']) and row['REG確率'] > 0:
            machine_name = row['機種名']
            matched_spec_key = backend.get_matched_spec_key(machine_name, specs)
            if matched_spec_key:
                spec_reg5_prob = 1.0 / specs[matched_spec_key].get("設定5", {"REG": 260.0})["REG"]
                if row['REG確率'] >= spec_reg5_prob:
                    return True
        return False
    display_df['is_high_reg'] = display_df.apply(check_high_reg, axis=1)

    # --- 期待外れ台のフラグ計算 (ハイライト用) ---
    def check_bad_pred(row):
        score = row.get('prediction_score', 0)
        if pd.isna(score) or score < 0.30: return False
        
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
    display_df['tmp_total_prob'] = (display_df['BIG'] + display_df['REG']) / display_df['総回転'].replace(0, np.nan)
    display_df['合算確率_str'] = display_df['tmp_total_prob'].apply(format_prob)
    
    # 並び替え
    sort_options = ["REG確率が良い順", "AI期待度順", "差枚が多い順", "合算確率が良い順", "台番号順"]
    sort_by = st.radio("並び替え", sort_options, horizontal=True)
    display_mode = st.radio("表示件数", ["厳選台 (上位10%)", "Top 10", "Top 20", "すべて"], horizontal=True, index=3)
    
    if sort_by == "差枚が多い順":
        display_df = display_df.sort_values('差枚', ascending=False)
    elif sort_by == "合算確率が良い順":
        display_df = display_df.sort_values('tmp_total_prob', ascending=False)
    elif sort_by == "REG確率が良い順":
        display_df = display_df.sort_values('REG確率', ascending=False)
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
    valid_all = display_df[(all_g >= 3000) | ((all_g < 3000) & ((all_diff <= -750) | (all_diff >= 750)))]
    if not valid_all.empty:
        all_win_c = (valid_all['差枚'] > 0).sum()
        all_win_rate_str = f"{all_win_c / len(valid_all):.1%} ({all_win_c}/{len(valid_all)}台)"
    else:
        all_win_rate_str = "- (0/0台)"
    all_avg_diff = all_diff.mean() if not all_diff.empty else 0
    st.caption(f"📊 店舗全体の勝率: **{all_win_rate_str}** (有効稼働のみ対象) ｜ 店舗全体平均差枚: **{int(all_avg_diff):+d}枚**")

    # --- 本日のAI精度パーセントの計算と表示 ---
    if 'prediction_score' in display_df.columns:
        day_eval_str = backend.classify_shop_eval(
            display_df['予測差枚数'].mean() if '予測差枚数' in display_df.columns else np.nan, 
            display_df['台番号'].nunique() if not display_df.empty else 50, 
            is_prediction=True
        )

        limit = max(3, int(len(display_df) * 0.10))
        if 'AI順位_num' in display_df.columns:
            ai_target_df = display_df[display_df['AI順位_num'] <= limit]
        else:
            ai_target_df = display_df.sort_values('prediction_score', ascending=False).head(limit)
        target_label = f"AI推奨台(上位10%・計{len(ai_target_df)}台)"
            
        if not ai_target_df.empty:
            act_g = pd.to_numeric(ai_target_df['総回転'], errors='coerce').fillna(0)
            act_diff = pd.to_numeric(ai_target_df['差枚'], errors='coerce').fillna(0)
            
            # 有効稼働のみに絞る (勝率ベース：3000G以上 or 差枚±750枚以上)
            valid_mask = (act_g >= 3000) | ((act_g < 3000) & ((act_diff <= -750) | (act_diff >= 750)))
            valid_target_df = ai_target_df[valid_mask]
            
            if not valid_target_df.empty:
                hit_df = valid_target_df[(valid_target_df['差枚'] > 0) | (valid_target_df['結果点数'] >= 50)]
                hit_count = len(hit_df)
                total_target = len(valid_target_df)
                accuracy = hit_count / total_target * 100
                acc_color = "🟢" if accuracy >= 60 else "🟡" if accuracy >= 40 else "🔴"
                
                win_c = (valid_target_df['差枚'] > 0).sum()
                win_rate_str = f"{win_c / total_target:.1%} ({win_c}/{total_target}台)"
                
                st.info(f"{acc_color} **本日のAI予測精度: {accuracy:.1f}%** ｜ 店舗のAI評価: {day_eval_str}\n\n{target_label} のうち有効稼働した **{total_target}台** 中、**{hit_count}台** が見事プラス収支または高設定挙動でした！\n\n🎯 **推奨台の実質勝率: {win_rate_str}** (有効稼働のみ対象)\n\n※表の色について: 🟥赤背景=AI推奨の期待外れ台 / 🟨黄背景=結果点数が80点以上の優秀台 / 🟩緑背景=REG確率が設定5基準以上の優秀台")
            else:
                st.info(f"⚪ **本日のAI予測精度: データなし** ｜ 店舗のAI評価: {day_eval_str}\n\n{target_label} の中で有効稼働（3000G以上等）した台がありませんでした。")

    tab_list, tab_map = st.tabs(["📋 台データ一覧", "🗺️ 島マップ (神視点)"])

    with tab_list:
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
            elif row_data.get('is_high_reg', False):
                return ['background-color: rgba(102, 187, 106, 0.3)'] * len(available_cols)
            return [''] * len(available_cols)
            
        styled_display_df = display_df[available_cols].style.apply(apply_row_style, axis=1)
        if '差枚' in available_cols:
            styled_display_df = styled_display_df.bar(subset=['差枚'], align='mid', color=['rgba(66, 165, 245, 0.5)', 'rgba(255, 112, 67, 0.5)'], vmin=-3000, vmax=3000)

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

    with tab_map:
        st.subheader("🗺️ 島マップ (神視点ビュー)")
        st.caption("島マスターに登録された情報に基づき、各島（列）の出玉状況を上から見た図で直感的に確認できます。塊や並びの投入箇所が一目で分かります。")
        
        map_metric = st.radio("表示する指標", ["差枚", "REG確率", "合算確率", "AI期待度(予測時)"], horizontal=True)
        
        with st.expander("🔍 絞り込みフィルター (条件に合わない台をグレーアウト)", expanded=False):
            st.caption("条件に合致しない台を目立たなくし、目的の台（凹み台や高稼働台など）を浮き彫りにします。")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                filter_min_g = st.slider("最低回転数 (G以上)", min_value=0, max_value=10000, value=0, step=500)
            with col_f2:
                filter_diff_range = st.slider("差枚数の範囲", min_value=-5000, max_value=10000, value=(-5000, 10000), step=500)
            filter_min_diff, filter_max_diff = filter_diff_range
            
        if df_island is None or df_island.empty:
            st.warning("島マスターのデータがありません。サイドバーの「島マスター管理」から島を登録してください。")
        else:
            # 該当店舗の島情報をパース
            shop_islands = df_island[df_island['店名'] == selected_shop]
            if shop_islands.empty:
                st.info("この店舗に登録されている島情報がありません。サイドバーの「島マスター管理」から島を登録してください。")
            else:
                parsed_islands = []
                for _, i_row in shop_islands.iterrows():
                    i_name = i_row.get('島名')
                    machines = []
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
                    else:
                        try:
                            s = int(i_row.get('開始台番号', 0))
                            e = int(i_row.get('終了台番号', 0))
                            if s > 0 and e >= s: machines.extend(range(s, e + 1))
                        except: pass
                        
                    machines = sorted(list(set(machines)))
                    if machines:
                        parsed_islands.append({
                            'name': i_name,
                            'type': str(i_row.get('島属性', '普通')),
                            'corner': str(i_row.get('メイン角番', '')).strip(),
                            'machines': machines
                        })
                
                if not parsed_islands:
                    st.info("島に有効な台番号が登録されていません。")
                else:
                    mac_data_dict = {}
                    for _, row in df_target.iterrows():
                        mac_num = str(row['台番号']).replace('.0', '')
                        g = row.get('累計ゲーム', 0)
                        b = row.get('BIG', 0)
                        r = row.get('REG', 0)
                        diff = row.get('差枚', 0)
                        pred = row.get('prediction_score', np.nan)
                        mac_name = row.get('機種名', '')
                        
                        mac_data_dict[mac_num] = {
                            'g': g, 'b': b, 'r': r, 'diff': diff, 'pred': pred, 'mac_name': mac_name
                        }
                        
                    html_parts = ["<div style='display: flex; flex-direction: column; gap: 20px; font-family: sans-serif;'>"]
                    
                    for island in parsed_islands:
                        isl_name = island['name']
                        isl_type = island['type']
                        machines = island['machines']
                        
                        html_parts.append(f"<div style='border: 2px solid #ddd; border-radius: 8px; padding: 12px; background-color: #fcfcfc;'>")
                        html_parts.append(f"<h4 style='margin-top: 0; margin-bottom: 12px; color: #444; font-size: 16px;'>🏝️ {isl_name} <span style='font-size:12px; font-weight:normal; color:#888;'>({isl_type})</span></h4>")
                        html_parts.append("<div style='display: flex; flex-wrap: wrap; gap: 8px;'>")
                        
                        for m_num in machines:
                            m_str = str(m_num)
                            data = mac_data_dict.get(m_str)
                            
                            bg_color, text_color, main_text, sub_text, border_color = "#f5f5f5", "#aaa", "-", "データなし", "#e0e0e0"
                            opacity = "1.0"
                            
                            if data:
                                g = int(pd.to_numeric(data['g'], errors='coerce') or 0)
                                b = int(pd.to_numeric(data['b'], errors='coerce') or 0)
                                r = int(pd.to_numeric(data['r'], errors='coerce') or 0)
                                diff = int(pd.to_numeric(data['diff'], errors='coerce') or 0)
                                pred = float(pd.to_numeric(data['pred'], errors='coerce') or np.nan)
                                m_name = str(data['mac_name'])
                                
                                matched_key = backend.get_matched_spec_key(m_name, specs)
                                spec_r5 = specs[matched_key].get('設定5', {"REG": 260.0})["REG"] if matched_key in specs else 260.0
                                spec_t5 = specs[matched_key].get('設定5', {"合算": 128.0})["合算"] if matched_key in specs else 128.0
                                
                                reg_prob_val = g / r if r > 0 else 9999
                                tot_prob_val = g / (b + r) if (b + r) > 0 else 9999
                                
                                short_mac = m_name.replace('ジャグラー', 'J').replace('ガールズ', 'G').replace('ハッピー', 'ﾊｯﾋﾟｰ').replace('ファンキー', 'ﾌｧﾝｷｰ')
                                sub_text = f"{g}G / {short_mac}" if g > 0 else "0G"
                                
                                if map_metric == "差枚":
                                    main_text = f"{diff:+d}" if g > 0 else "-"
                                    if g == 0: bg_color = "#f5f5f5"; text_color = "#9e9e9e"
                                    elif diff >= 2000: bg_color = "#d32f2f"; text_color = "#fff"; border_color = "#b71c1c"
                                    elif diff >= 1000: bg_color = "#ef5350"; text_color = "#fff"; border_color = "#c62828"
                                    elif diff > 0: bg_color = "#ffcdd2"; text_color = "#d32f2f"; border_color = "#ef5350"
                                    elif diff > -1000: bg_color = "#e3f2fd"; text_color = "#1565c0"; border_color = "#42a5f5"
                                    else: bg_color = "#1976d2"; text_color = "#fff"; border_color = "#0d47a1"
                                        
                                elif map_metric == "REG確率":
                                    main_text = f"1/{int(reg_prob_val)}" if r > 0 else "-"
                                    if g < 1000: bg_color = "#f5f5f5"; text_color = "#9e9e9e"
                                    elif reg_prob_val <= spec_r5: bg_color = "#ef6c00"; text_color = "#fff"; border_color = "#e65100"
                                    elif reg_prob_val <= spec_r5 * 1.15: bg_color = "#ffe082"; text_color = "#f57f17"; border_color = "#ffb300"
                                    else: bg_color = "#eceff1"; text_color = "#546e7a"; border_color = "#cfd8dc"
                                        
                                elif map_metric == "合算確率":
                                    main_text = f"1/{int(tot_prob_val)}" if (b+r) > 0 else "-"
                                    if g < 1000: bg_color = "#f5f5f5"; text_color = "#9e9e9e"
                                    elif tot_prob_val <= spec_t5: bg_color = "#8e24aa"; text_color = "#fff"; border_color = "#6a1b9a"
                                    elif tot_prob_val <= spec_t5 * 1.1: bg_color = "#e1bee7"; text_color = "#6a1b9a"; border_color = "#ab47bc"
                                    else: bg_color = "#eceff1"; text_color = "#546e7a"; border_color = "#cfd8dc"
                                        
                                elif map_metric == "AI期待度(予測時)":
                                    if pd.notna(pred):
                                        main_text = f"{int(pred * 100)}%"
                                        if pred >= 0.50: bg_color = "#d32f2f"; text_color = "#fff"; border_color = "#b71c1c"
                                        elif pred >= 0.30: bg_color = "#ef5350"; text_color = "#fff"; border_color = "#c62828"
                                        elif pred >= 0.15: bg_color = "#ffcdd2"; text_color = "#d32f2f"; border_color = "#ef5350"
                                        else: bg_color = "#eceff1"; text_color = "#546e7a"; border_color = "#cfd8dc"
                                    else:
                                        main_text = "-"
                                        bg_color = "#f5f5f5"; text_color = "#9e9e9e"
                                        
                                # フィルターの適用
                                if g < filter_min_g or diff > filter_max_diff or diff < filter_min_diff:
                                    bg_color = "#fafafa"
                                    text_color = "#ccc"
                                    border_color = "#eee"
                                    opacity = "0.2"
                            else:
                                if filter_min_g > 0:
                                    opacity = "0.2"
                            
                            is_corner_mark = "✨" if str(m_num) == island['corner'] else ""
                                
                            html_parts.append(f"""
                                <div style='width: 82px; height: 75px; background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 6px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); position: relative; padding: 2px; box-sizing: border-box; opacity: {opacity};'>
                                    <div style='position: absolute; top: 2px; left: 4px; font-size: 11px; font-weight: bold; color: #555;'>{is_corner_mark}#{m_num}</div>
                                    <div style='font-size: 18px; font-weight: bold; color: {text_color}; margin-top: 12px; line-height: 1;'>{main_text}</div>
                                    <div style='font-size: 10px; color: #666; margin-top: 4px; width: 100%; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{sub_text}</div>
                                </div>
                            """)
                        html_parts.append("</div></div>")
                    html_parts.append("</div>")
                    
                    st.components.v1.html("".join(html_parts), height=800, scrolling=True)