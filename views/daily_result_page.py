import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import math
import backend
from utils import get_valid_play_mask

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
                cols_to_merge = ['台番号', 'prediction_score', 'sueoki_score', '予測信頼度', '根拠']
                cols_to_merge = [c for c in cols_to_merge if c in df_pred_target.columns]
                
                df_target = pd.merge(df_target, df_pred_target[cols_to_merge], on='台番号', how='left')

    # 表示用データの整形
    display_df = df_target.copy()
    
    # 期待度をパーセント表記に
    if 'prediction_score' in display_df.columns:
        if 'sueoki_score' in display_df.columns:
            display_df['max_score'] = display_df[['prediction_score', 'sueoki_score']].max(axis=1)
            display_df['狙い目'] = np.where(
                pd.to_numeric(display_df['prediction_score'], errors='coerce').fillna(0.0) >= 
                pd.to_numeric(display_df['sueoki_score'], errors='coerce').fillna(0.0), 
                "🚀変更", "🔁据置"
            )
        else:
            display_df['max_score'] = display_df['prediction_score']
            display_df['狙い目'] = "🚀変更"
            
        display_df['期待度'] = display_df['max_score'].apply(lambda x: f"{int(x * 100)}%" if pd.notna(x) and x != '' else "-")
        # 期待度に基づくAI順位の計算
        display_df['AI順位_num'] = display_df['max_score'].rank(method='min', ascending=False).fillna(999).astype(int)
        display_df['AI順位'] = display_df['AI順位_num'].apply(lambda x: f"{x}位" if x != 999 else "-")
    else:
        display_df['期待度'] = "-"
        display_df['AI順位'] = "-"
        display_df['狙い目'] = "-"

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
        return backend.get_setting_score_from_row(
            row, shop_avg_g=shop_avg_g, g_col=g_col, b_col=b_col, r_col=r_col, m_col=m_col, d_col=d_col
        )

    display_df['結果点数'] = display_df.apply(calculate_score, axis=1)
    
    # --- Zスコア（設定1否定度）の計算 ---
    def calculate_z_score(row):
        games = row.get('総回転', 0)
        reg = row.get('REG', 0)
        machine_name = row.get('機種名', '')
        if games <= 0: return 0.0
        
        matched_spec_key = backend.get_matched_spec_key(machine_name, specs)
        if matched_spec_key:
            spec_reg1_den = specs[matched_spec_key].get('設定1', {"REG": 400.0}).get("REG", 400.0)
            p1 = 1.0 / spec_reg1_den
            exp_reg1 = games * p1
            std_reg1 = math.sqrt(games * p1 * (1.0 - p1))
            if std_reg1 > 0:
                return (reg - exp_reg1) / std_reg1
        return 0.0

    display_df['Zスコア'] = display_df.apply(calculate_z_score, axis=1)
    
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
        c_score = row.get('prediction_score', 0)
        s_score = row.get('sueoki_score', 0)
        max_s = max(c_score, s_score)
        if pd.isna(max_s) or max_s < 0.30: return False
        
        g = row.get('総回転', 0)
        b = row.get('BIG', 0)
        r = row.get('REG', 0)
        diff = row.get('差枚', 0)
        tot_prob = (b + r) / g if g > 0 else 0.0
        
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
        if 'max_score' in display_df.columns:
            display_df = display_df.sort_values('max_score', ascending=False)
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
    valid_all = display_df[get_valid_play_mask(all_g, all_diff)]
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

        limit = max(1, int(len(display_df) * 0.10))
        if 'AI順位_num' in display_df.columns:
            ai_target_df = display_df[display_df['AI順位_num'] <= limit]
        else:
            ai_target_df = display_df.sort_values('max_score', ascending=False).head(limit)
        target_label = f"AI推奨台(上位10%・計{len(ai_target_df)}台)"
            
        if not ai_target_df.empty:
            act_g = pd.to_numeric(ai_target_df['総回転'], errors='coerce').fillna(0)
            act_diff = pd.to_numeric(ai_target_df['差枚'], errors='coerce').fillna(0)
            
            # 有効稼働のみに絞る (勝率ベース：3000G以上 or 差枚±750枚以上)
            valid_mask = get_valid_play_mask(act_g, act_diff)
            valid_target_df = ai_target_df[valid_mask]
            
            if not valid_target_df.empty:
                hit_df = valid_target_df[(valid_target_df['差枚'] > 0) | (valid_target_df['結果点数'] >= 50)]
                hit_count = len(hit_df)
                total_target = len(valid_target_df)
                accuracy = hit_count / total_target * 100
                acc_color = "🟢" if accuracy >= 60 else "🟡" if accuracy >= 40 else "🔴"
                
                win_c = (valid_target_df['差枚'] > 0).sum()
                win_rate_str = f"{win_c / total_target:.1%} ({win_c}/{total_target}台)"
                
                st.info(f"{acc_color} **本日のAI予測精度: {accuracy:.1f}%** ｜ 店舗のAI評価: {day_eval_str}\n\n{target_label} のうち有効稼働した **{total_target}台** 中、**{hit_count}台** が見事プラス収支または高設定挙動でした！\n\n🎯 **推奨台の実質勝率: {win_rate_str}** (有効稼働のみ対象)\n\n💡 **Zスコアについて:** 設定1のREG確率を基準とした「低設定の否定度」です。**1.64以上**で低設定では5%未満しか起こらない奇跡、**2.58以上**なら1%未満（ほぼ否定）となります。\n※表の色について: 🟥赤背景=AI推奨の期待外れ台 / 🟨黄背景=結果点数が80点以上の優秀台 / 🟩緑背景=REG確率が設定5基準以上の優秀台")
            else:
                st.info(f"⚪ **本日のAI予測精度: データなし** ｜ 店舗のAI評価: {day_eval_str}\n\n{target_label} の中で有効稼働（3000G以上等）した台がありませんでした。")

    # --- 表示件数の絞り込み ---
    if display_mode == "厳選台 (上位10%)":
        limit = max(1, int(len(display_df) * 0.10))
        display_df = display_df.head(limit)
    elif display_mode == "Top 10":
        display_df = display_df.head(10)
    elif display_mode == "Top 20":
        display_df = display_df.head(20)

    cols = ['AI順位', '狙い目', '台番号', '機種名']
    if '期待度' in display_df.columns: cols.append('期待度')
    if '予測信頼度' in display_df.columns: cols.append('予測信頼度')
    cols.append('結果点数')
    cols.extend(['Zスコア', '差枚', '総回転', 'BIG', 'REG', '合算確率_str', 'REG確率_str', 'BIG確率_str', 'ぶどう確率_str'])
    
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
            "狙い目": st.column_config.TextColumn("狙い目", width="small", help="AIが総合的により高く評価している方向（スコアが高い方）です。"),
            "台番号": st.column_config.TextColumn("No.", width="small"),
            "機種名": st.column_config.TextColumn("機種", width="small"),
            "期待度": st.column_config.TextColumn("AI期待度", width="small", help="AIが前日時点で予測していた設定5以上の確率です。"),
            "予測信頼度": st.column_config.TextColumn("信頼度", width="small"),
            "結果点数": st.column_config.NumberColumn("結果点数", format="%.1f点", help="実際の結果に基づく設定5近似度"),
            "Zスコア": st.column_config.NumberColumn("Zスコア", format="%.2f", help="設定1のREG確率を基準にした上振れ度（低設定の否定度）。1.64以上で低設定の可能性5%未満、2.58以上で1%未満（ほぼ否定）です。"),
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