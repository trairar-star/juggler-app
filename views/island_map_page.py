import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import backend

def render_island_map_page(df_raw, df_pred_log, df_island):
    st.header("🗺️ 島マップ (神視点ビュー)")
    st.caption("島マスターに登録された情報に基づき、各島（列）の出玉状況を上から見た図で直感的に確認できます。塊や並びの投入箇所が一目で分かります。")

    if df_raw.empty:
        st.warning("データがありません。")
        return

    specs = backend.get_machine_specs()
    date_col = '対象日付'
    temp_df = df_raw.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
    available_dates = sorted(temp_df[date_col].dropna().dt.date.unique(), reverse=True)

    if not available_dates:
        st.warning("有効な日付データがありません。")
        return

    col_d, col_s = st.columns(2)
    selected_date = col_d.selectbox("📅 確認する日付を選択", available_dates)

    shop_col = '店名' if '店名' in temp_df.columns else ('店舗名' if '店舗名' in temp_df.columns else None)
    df_day = temp_df[temp_df[date_col].dt.date == selected_date].copy()
    shops = ["店舗を選択してください"] + sorted(list(df_day[shop_col].unique()))
    
    default_index = 0
    saved_shop = st.session_state.get("global_selected_shop", "店舗を選択してください")
    if saved_shop in shops:
        default_index = shops.index(saved_shop)

    selected_shop = col_s.selectbox("🏬 店舗を選択", shops, index=default_index)
    if selected_shop != "店舗を選択してください":
        st.session_state["global_selected_shop"] = selected_shop

    if selected_shop == "店舗を選択してください":
        st.info("👆 店舗を選択すると、その日の島マップが表示されます。")
        return

    if df_island is None or df_island.empty:
        st.warning("島マスターのデータがありません。サイドバーの「島マスター管理」から島を登録してください。")
        return

    shop_islands = df_island[df_island['店名'] == selected_shop]
    if shop_islands.empty:
        st.info("この店舗に登録されている島情報がありません。サイドバーの「島マスター管理」から島を登録してください。")
        return

    # データ準備
    shop_all_df = temp_df[temp_df[shop_col] == selected_shop].copy()
    shop_all_df['台番号'] = shop_all_df['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
    shop_all_df = shop_all_df.sort_values(['台番号', date_col])
    shop_all_df['prev_差枚'] = shop_all_df.groupby('台番号')['差枚'].shift(1)
    df_target = shop_all_df[shop_all_df[date_col].dt.date == selected_date].copy()

    # AI期待度の結合
    if not df_pred_log.empty:
        log_temp = df_pred_log.copy()
        if '予測対象日' in log_temp.columns:
            log_temp['予測対象日_dt'] = pd.to_datetime(log_temp['予測対象日'], errors='coerce').dt.date
            log_day = log_temp[log_temp['予測対象日_dt'] == selected_date].copy()
            shop_col_log = '店名' if '店名' in log_day.columns else '店舗名'
            if shop_col_log in log_day.columns:
                log_shop = log_day[log_day[shop_col_log] == selected_shop].copy()
                if not log_shop.empty:
                    log_shop['台番号'] = log_shop['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
                    log_shop['prediction_score'] = pd.to_numeric(log_shop['prediction_score'], errors='coerce')
                    if '実行日時' in log_shop.columns:
                        log_shop = log_shop.sort_values('実行日時', ascending=False).drop_duplicates('台番号')
                    df_target = pd.merge(df_target, log_shop[['台番号', 'prediction_score']], on='台番号', how='left')

    st.divider()

    # 島のパース
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
        return

    # コントロールパネル
    col_c1, col_c2 = st.columns([1.5, 1])
    with col_c1:
        map_metric = st.radio("📊 表示する指標", ["差枚", "REG確率", "合算確率", "AI期待度(事前の予測)"], horizontal=True)
    with col_c2:
        layout_mode = st.radio("島のレイアウト", ["島ごとに改行 (縦積み)", "すべて横一列に繋げる"], horizontal=True)

    island_names = [isl['name'] for isl in parsed_islands]
    selected_island_names = st.multiselect(
        "🛠️ 表示する島と順番（ドラッグ操作や再選択で並び替えできます）", 
        options=island_names, 
        default=island_names,
        help="ここで選択した順番通りに島が配置されます。特定列だけを見たい場合や、背中合わせの島を隣同士に並べ替えたい場合に便利です。"
    )

    with st.expander("🔍 絞り込みフィルター (条件に合わない台をグレーアウト)", expanded=False):
        st.caption("条件に合致しない台を目立たなくし、目的の台（凹み台や高稼働台など）を浮き彫りにします。")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_min_g = st.slider("最低回転数 (G以上)", min_value=0, max_value=10000, value=0, step=500)
        with col_f2:
            filter_diff_range = st.slider("差枚数の範囲", min_value=-5000, max_value=10000, value=(-5000, 10000), step=500)
        filter_min_diff, filter_max_diff = filter_diff_range

    display_islands = []
    for name in selected_island_names:
        for isl in parsed_islands:
            if isl['name'] == name:
                display_islands.append(isl)
                break

    if not display_islands:
        st.info("表示する島が選択されていません。")
        return

    # 台データ辞書の作成
    mac_data_dict = {}
    for _, row in df_target.iterrows():
        mac_num = str(row['台番号']).replace('.0', '')
        mac_data_dict[mac_num] = {
            'g': row.get('累計ゲーム', 0), 'b': row.get('BIG', 0), 'r': row.get('REG', 0), 
            'diff': row.get('差枚', 0), 'pred': row.get('prediction_score', np.nan), 'mac_name': row.get('機種名', ''),
            'prev_diff': row.get('prev_差枚', np.nan)
        }

    if layout_mode == "すべて横一列に繋げる":
        container_style = "display: flex; flex-direction: row; gap: 20px; overflow-x: auto; padding-bottom: 20px; white-space: nowrap; width: 100%;"
        island_style = "display: flex; flex-direction: row; flex-wrap: nowrap; gap: 6px;"
    else:
        container_style = "display: flex; flex-direction: column; gap: 20px;"
        island_style = "display: flex; flex-wrap: wrap; gap: 6px;"

    html_parts = [f"<div style='{container_style} font-family: sans-serif;'>"]
    for island in display_islands:
        html_parts.append(f"<div style='border: 2px solid #ddd; border-radius: 8px; padding: 12px; background-color: #fcfcfc; min-width: max-content;'>")
        html_parts.append(f"<h4 style='margin-top: 0; margin-bottom: 12px; color: #444; font-size: 16px;'>🏝️ {island['name']} <span style='font-size:12px; font-weight:normal; color:#888;'>({island['type']})</span></h4>")
        html_parts.append(f"<div style='{island_style}'>")
        for m_num in island['machines']:
            data = mac_data_dict.get(str(m_num))
            bg_color, text_color, main_text, sub_text, border_color, opacity, prev_diff_str = "#f5f5f5", "#aaa", "-", "データなし", "#e0e0e0", "1.0", ""
            if data:
                g, b, r, diff, pred, m_name = int(pd.to_numeric(data['g'], errors='coerce') or 0), int(pd.to_numeric(data['b'], errors='coerce') or 0), int(pd.to_numeric(data['r'], errors='coerce') or 0), int(pd.to_numeric(data['diff'], errors='coerce') or 0), float(pd.to_numeric(data['pred'], errors='coerce') or np.nan), str(data['mac_name'])
                prev_diff_val = data.get('prev_diff')
                matched_key = backend.get_matched_spec_key(m_name, specs)
                spec_r5 = specs[matched_key].get('設定5', {"REG": 260.0})["REG"] if matched_key in specs else 260.0
                spec_t5 = specs[matched_key].get('設定5', {"合算": 128.0})["合算"] if matched_key in specs else 128.0
                reg_prob_val, tot_prob_val = g / r if r > 0 else 9999, g / (b + r) if (b + r) > 0 else 9999
                sub_text = f"{g}G / {m_name.replace('ジャグラー', 'J').replace('ガールズ', 'G').replace('ハッピー', 'ﾊｯﾋﾟｰ').replace('ファンキー', 'ﾌｧﾝｷｰ')}" if g > 0 else "0G"
                if map_metric == "差枚":
                    main_text = f"{diff:+d}" if g > 0 else "-"
                    if g == 0: bg_color, text_color = "#f5f5f5", "#9e9e9e"
                    elif diff >= 2000: bg_color, text_color, border_color = "#d32f2f", "#fff", "#b71c1c"
                    elif diff >= 1000: bg_color, text_color, border_color = "#ef5350", "#fff", "#c62828"
                    elif diff > 0: bg_color, text_color, border_color = "#ffcdd2", "#d32f2f", "#ef5350"
                    elif diff > -1000: bg_color, text_color, border_color = "#e3f2fd", "#1565c0", "#42a5f5"
                    else: bg_color, text_color, border_color = "#1976d2", "#fff", "#0d47a1"
                elif map_metric == "REG確率":
                    main_text = f"1/{int(reg_prob_val)}" if r > 0 else "-"
                    if g < 1000: bg_color, text_color = "#f5f5f5", "#9e9e9e"
                    elif reg_prob_val <= spec_r5: bg_color, text_color, border_color = "#ef6c00", "#fff", "#e65100"
                    elif reg_prob_val <= spec_r5 * 1.15: bg_color, text_color, border_color = "#ffe082", "#f57f17", "#ffb300"
                    else: bg_color, text_color, border_color = "#eceff1", "#546e7a", "#cfd8dc"
                elif map_metric == "合算確率":
                    main_text = f"1/{int(tot_prob_val)}" if (b+r) > 0 else "-"
                    if g < 1000: bg_color, text_color = "#f5f5f5", "#9e9e9e"
                    elif tot_prob_val <= spec_t5: bg_color, text_color, border_color = "#8e24aa", "#fff", "#6a1b9a"
                    elif tot_prob_val <= spec_t5 * 1.1: bg_color, text_color, border_color = "#e1bee7", "#6a1b9a", "#ab47bc"
                    else: bg_color, text_color, border_color = "#eceff1", "#546e7a", "#cfd8dc"
                elif map_metric == "AI期待度(事前の予測)":
                    main_text = f"{int(pred * 100)}%" if pd.notna(pred) else "-"
                    if pd.notna(pred):
                        if pred >= 0.50: bg_color, text_color, border_color = "#d32f2f", "#fff", "#b71c1c"
                        elif pred >= 0.30: bg_color, text_color, border_color = "#ef5350", "#fff", "#c62828"
                        elif pred >= 0.15: bg_color, text_color, border_color = "#ffcdd2", "#d32f2f", "#ef5350"
                        else: bg_color, text_color, border_color = "#eceff1", "#546e7a", "#cfd8dc"
                    else: bg_color, text_color = "#f5f5f5", "#9e9e9e"
                if pd.notna(prev_diff_val):
                    p_diff = int(prev_diff_val)
                    p_color = "#d32f2f" if p_diff > 0 else "#1565c0" if p_diff < 0 else "#9e9e9e"
                    prev_diff_str = f"<div style='position: absolute; top: 1px; right: 3px; font-size: 9px; font-weight: bold; color: {p_color}; letter-spacing: -0.5px;'>前:{p_diff:+d}</div>"
                if g < filter_min_g or diff > filter_max_diff or diff < filter_min_diff: bg_color, text_color, border_color, opacity = "#fafafa", "#ccc", "#eee", "0.2"
            else:
                if filter_min_g > 0: opacity = "0.2"
            html_parts.append(f"<div style='width: 72px; height: 66px; flex-shrink: 0; background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 6px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); position: relative; padding: 2px; box-sizing: border-box; opacity: {opacity};'><div style='position: absolute; top: 1px; left: 3px; font-size: 10px; font-weight: bold; color: #555;'>{'✨' if str(m_num) == island['corner'] else ''}#{m_num}</div>{prev_diff_str}<div style='font-size: 15px; font-weight: bold; color: {text_color}; margin-top: 10px; line-height: 1;'>{main_text}</div><div style='font-size: 9px; color: #666; margin-top: 3px; width: 100%; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{sub_text}</div></div>")
        html_parts.append("</div></div>")
    html_parts.append("</div>")
    st.components.v1.html("".join(html_parts), height=800, scrolling=True)