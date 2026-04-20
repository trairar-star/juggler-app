import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import backend

def render_island_map_page(df_raw, df_pred_log, df_island):
    # ==========================================
    # UIコントロール群を一つのコンテナにまとめる
    # ==========================================
    ui_container = st.container()
    
    with ui_container:
        st.header("🗺️ 島マップ (神視点ビュー)")
        st.caption("島マスターに登録された情報に基づき、各島（列）の出玉状況を上から見た図で直感的に確認できます。塊や並びの投入箇所が一目で分かります。")

        if df_raw.empty:
            st.warning("データがありません。")
            st.stop()

        specs = backend.get_machine_specs()
        date_col = '対象日付'
        temp_df = df_raw.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
        available_dates = sorted(temp_df[date_col].dropna().dt.date.unique(), reverse=True)

        if not available_dates:
            st.warning("有効な日付データがありません。")
            st.stop()

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
            st.stop()

        if df_island is None or df_island.empty:
            st.warning("島マスターのデータがありません。サイドバーの「島マスター管理」から島を登録してください。")
            st.stop()

        shop_islands = df_island[df_island['店名'] == selected_shop]
        if shop_islands.empty:
            st.info("この店舗に登録されている島情報がありません。サイドバーの「島マスター管理」から島を登録してください。")
            st.stop()

        st.divider()

        # --- 島のパース処理 (UIに表示するため先に実行) ---
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
            st.stop()

        # --- コントロールパネル (指標、レイアウト、並び替え、フィルター) ---
        col_fs, col_c1, col_c2 = st.columns([1, 2.5, 2])
        with col_fs:
            st.components.v1.html("""
                <button id="fs-btn" onclick="toggleFullscreen()" style="width: 100%; height: 40px; border-radius: 6px; background-color: #42A5F5; color: white; border: none; cursor: pointer; font-weight: bold; font-family: sans-serif; font-size: 14px; margin-top: 25px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                    🖥️ 領域を最大化
                </button>
                <script>
                function toggleFullscreen() {
                    const doc = window.parent.document;
                    const mainBlock = doc.querySelector('[data-testid="stMainBlockContainer"]');
                    const btn = document.getElementById('fs-btn');
                    
                    // UIコンテナ（一番上のブロック）を取得してSticky化する
                    const topBlock = mainBlock ? mainBlock.querySelector('[data-testid="stVerticalBlock"] > div:first-child') : null;
                    
                    if (!doc.fullscreenElement) {
                        if (mainBlock) {
                            mainBlock.requestFullscreen().then(() => {
                                mainBlock.style.backgroundColor = window.getComputedStyle(doc.body).backgroundColor || '#ffffff';
                                mainBlock.style.maxWidth = '100%';
                                mainBlock.style.padding = '1rem';
                                mainBlock.style.overflowY = 'auto';
                                
                                if (topBlock) {
                                    topBlock.style.position = 'sticky';
                                    topBlock.style.top = '-1rem';
                                    topBlock.style.zIndex = '9999';
                                    topBlock.style.backgroundColor = window.getComputedStyle(doc.body).backgroundColor || '#ffffff';
                                    topBlock.style.paddingTop = '1rem';
                                    topBlock.style.paddingBottom = '0.5rem';
                                    topBlock.style.borderBottom = '2px solid #ddd';
                                }
                                
                                btn.innerHTML = '↙️ 元に戻す';
                            }).catch(err => { console.log('Error:', err); });
                        }
                    } else { 
                        doc.exitFullscreen(); 
                    }
                }
                
                window.parent.document.addEventListener('fullscreenchange', () => {
                    const doc = window.parent.document;
                    const mainBlock = doc.querySelector('[data-testid="stMainBlockContainer"]');
                    const btn = document.getElementById('fs-btn');
                    const topBlock = mainBlock ? mainBlock.querySelector('[data-testid="stVerticalBlock"] > div:first-child') : null;
                    
                    if (!doc.fullscreenElement) {
                        if (mainBlock) { mainBlock.style.backgroundColor = ''; mainBlock.style.maxWidth = ''; mainBlock.style.padding = ''; }
                        if (topBlock) {
                            topBlock.style.position = '';
                            topBlock.style.top = '';
                            topBlock.style.zIndex = '';
                            topBlock.style.backgroundColor = '';
                            topBlock.style.paddingTop = '';
                            topBlock.style.paddingBottom = '';
                            topBlock.style.borderBottom = '';
                        }
                        if (btn) btn.innerHTML = '🖥️ 領域を最大化';
                    }
                });
                </script>
            """, height=75)
        with col_c1:
            map_metric = st.radio("📊 表示する指標", ["差枚", "REG確率", "合算確率", "AI期待度(事前の予測)", "結果点数(設定5近似度)"], horizontal=True)
        with col_c2:
            layout_mode = st.radio("島のレイアウト", ["島ごとに改行 (縦積み・島内横スクロール)", "すべて横一列に繋げる", "島内で折り返す (コンパクト)"], horizontal=True)

        island_names = [isl['name'] for isl in parsed_islands]
        
        shop_order_key = f"island_order_{selected_shop}"
        if shop_order_key not in st.session_state:
            st.session_state[shop_order_key] = island_names
        else:
            current_valid_order = [n for n in st.session_state[shop_order_key] if n in island_names]
            if current_valid_order != st.session_state[shop_order_key]:
                st.session_state[shop_order_key] = current_valid_order

        selected_island_names = st.multiselect(
            "🛠️ 表示する島と順番（×で消して選び直すことで順番を変更できます）", 
            options=island_names, 
            key=shop_order_key,
            help="ここで選択した順番通りに島が配置されます。×ボタンで島を消し、再度追加することで一番下に移動し、並べ替えができます。"
        )

        with st.expander("🔍 絞り込みフィルター (条件に合わない台をグレーアウト)", expanded=False):
            st.caption("条件に合致しない台を目立たなくし、目的の台（凹み台や高稼働台など）を浮き彫りにします。")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                filter_min_g = st.slider("最低回転数 (G以上)", min_value=0, max_value=10000, value=3000, step=500)
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
        st.stop()

    # ==========================================
    # データ処理 (UIには見えないバックグラウンド処理)
    # ==========================================
    
    # データ準備
    shop_all_df = temp_df[temp_df[shop_col] == selected_shop].copy()
    shop_all_df['台番号'] = shop_all_df['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
    shop_all_df = shop_all_df.sort_values(['台番号', date_col])

    # --- 1. AI期待度の結合 (全日程) ---
    if not df_pred_log.empty:
        log_temp = df_pred_log.copy()
        if '予測対象日' in log_temp.columns:
            log_temp['予測対象日_dt'] = pd.to_datetime(log_temp['予測対象日'], errors='coerce').dt.date
            shop_col_log = '店名' if '店名' in log_temp.columns else '店舗名'
            if shop_col_log in log_temp.columns:
                log_shop = log_temp[log_temp[shop_col_log] == selected_shop].copy()
                if not log_shop.empty:
                    log_shop['台番号'] = log_shop['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
                    log_shop['prediction_score'] = pd.to_numeric(log_shop['prediction_score'], errors='coerce')
                    if '実行日時' in log_shop.columns:
                        log_shop = log_shop.sort_values('実行日時', ascending=False).drop_duplicates(['台番号', '予測対象日_dt'])
                    shop_all_df['対象日付_dt'] = shop_all_df[date_col].dt.date
                    shop_all_df = pd.merge(shop_all_df, log_shop[['台番号', '予測対象日_dt', 'prediction_score']], left_on=['台番号', '対象日付_dt'], right_on=['台番号', '予測対象日_dt'], how='left')
                    shop_all_df = shop_all_df.drop(columns=['対象日付_dt', '予測対象日_dt'], errors='ignore')

    # --- 2. 結果点数の計算 (全日程) ---
    shop_avg_g = shop_all_df['累計ゲーム'].mean() if not shop_all_df.empty else 4000
    if pd.isna(shop_avg_g):
        shop_avg_g = 4000
        
    def calc_score_all(row):
        g = pd.to_numeric(row.get('累計ゲーム', 0), errors='coerce')
        act_b = pd.to_numeric(row.get('BIG', 0), errors='coerce')
        act_r = pd.to_numeric(row.get('REG', 0), errors='coerce')
        diff = pd.to_numeric(row.get('差枚', 0), errors='coerce')
        machine = row.get('機種名', '')
        
        penalty_reg = st.session_state.get('penalty_reg', 15)
        penalty_big = st.session_state.get('penalty_big', 5)
        low_g_penalty = st.session_state.get('low_g_penalty', 30)
        
        return backend.calculate_setting_score(
            g=g, act_b=act_b, act_r=act_r, machine_name=machine, diff=diff,
            shop_avg_g=shop_avg_g, penalty_reg=penalty_reg, penalty_big=penalty_big,
            low_g_penalty=low_g_penalty, use_strict_scoring=True, return_details=False
        )

    if not shop_all_df.empty:
        shop_all_df['結果点数'] = shop_all_df.apply(calc_score_all, axis=1)
    else:
        shop_all_df['結果点数'] = np.nan
        
    # --- 3. 前日データの計算と比較用シフト ---
    shop_all_df = shop_all_df.sort_values(['台番号', date_col])
    shop_all_df['prev_差枚'] = shop_all_df.groupby('台番号')['差枚'].shift(1)
    shop_all_df['prev_累計ゲーム'] = shop_all_df.groupby('台番号')['累計ゲーム'].shift(1)
    shop_all_df['prev_BIG'] = shop_all_df.groupby('台番号')['BIG'].shift(1)
    shop_all_df['prev_REG'] = shop_all_df.groupby('台番号')['REG'].shift(1)
    if 'prediction_score' in shop_all_df.columns:
        shop_all_df['prev_pred'] = shop_all_df.groupby('台番号')['prediction_score'].shift(1)
    if '結果点数' in shop_all_df.columns:
        shop_all_df['prev_score'] = shop_all_df.groupby('台番号')['結果点数'].shift(1)

    # 当日データに絞り込み
    df_target = shop_all_df[shop_all_df[date_col].dt.date == selected_date].copy()

    # 台データ辞書の作成
    mac_data_dict = {}
    for _, row in df_target.iterrows():
        mac_num = str(row['台番号']).replace('.0', '')
        mac_data_dict[mac_num] = {
            'g': row.get('累計ゲーム', 0), 'b': row.get('BIG', 0), 'r': row.get('REG', 0), 
            'diff': row.get('差枚', 0), 'pred': row.get('prediction_score', np.nan), 'mac_name': row.get('機種名', ''),
            'prev_diff': row.get('prev_差枚', np.nan),
            'prev_g': row.get('prev_累計ゲーム', 0),
            'prev_b': row.get('prev_BIG', 0),
            'prev_r': row.get('prev_REG', 0),
            'prev_pred': row.get('prev_pred', np.nan),
            'prev_score': row.get('prev_score', np.nan),
            'result_score': row.get('結果点数', np.nan)
        }

    if layout_mode == "すべて横一列に繋げる":
        container_style = "display: flex; flex-direction: row; gap: 20px; overflow-x: auto; padding-bottom: 20px; white-space: nowrap; width: 100%;"
        island_style = "display: flex; flex-direction: row; flex-wrap: nowrap; gap: 6px;"
        island_wrapper_style = "border: 2px solid #ddd; border-radius: 8px; padding: 12px; background-color: #fcfcfc; min-width: max-content;"
    elif layout_mode == "島ごとに改行 (縦積み・島内横スクロール)":
        container_style = "display: flex; flex-direction: column; gap: 20px; width: 100%;"
        island_style = "display: flex; flex-direction: row; flex-wrap: nowrap; gap: 6px; overflow-x: auto; padding-bottom: 10px;"
        island_wrapper_style = "border: 2px solid #ddd; border-radius: 8px; padding: 12px; background-color: #fcfcfc; width: 100%; box-sizing: border-box;"
    else:
        container_style = "display: flex; flex-direction: column; gap: 20px;"
        island_style = "display: flex; flex-wrap: wrap; gap: 6px;"
        island_wrapper_style = "border: 2px solid #ddd; border-radius: 8px; padding: 12px; background-color: #fcfcfc; width: 100%; box-sizing: border-box;"

    # --- ツールチップ(ホバー表示)用 CSS ---
    html_parts = [f"""
    <style>
    .island-container::-webkit-scrollbar {{
        height: 6px;
    }}
    .island-container::-webkit-scrollbar-track {{
        background: #f1f1f1;
        border-radius: 4px;
    }}
    .island-container::-webkit-scrollbar-thumb {{
        background: #ccc;
        border-radius: 4px;
    }}
    .island-container::-webkit-scrollbar-thumb:hover {{
        background: #aaa;
    }}
    .machine-box {{
        position: relative;
        cursor: crosshair;
        transition: transform 0.1s ease-in-out;
    }}
    .machine-box:hover {{
        z-index: 1000;
        transform: scale(1.05);
    }}
    .tooltip-text {{
        visibility: hidden;
        width: max-content;
        min-width: 140px;
        background-color: rgba(20, 20, 20, 0.95);
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        bottom: calc(100% + 8px);
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.2s;
        font-size: 11px;
        line-height: 1.5;
        pointer-events: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }}
    .machine-box:hover .tooltip-text {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    <div style='{container_style} font-family: sans-serif; padding-top: 10px;'>
    """]
    
    for island in display_islands:
        html_parts.append(f"<div style='border: 2px solid #ddd; border-radius: 8px; padding: 12px; background-color: #fcfcfc; min-width: max-content;'>")
        html_parts.append(f"<h4 style='margin-top: 0; margin-bottom: 12px; color: #444; font-size: 16px;'>🏝️ {island['name']} <span style='font-size:12px; font-weight:normal; color:#888;'>({island['type']})</span></h4>")
        html_parts.append(f"<div style='{island_style}'>")
        for m_num in island['machines']:
            data = mac_data_dict.get(str(m_num))
            bg_color, text_color, main_text, sub_text, border_color, opacity, prev_diff_str = "#f5f5f5", "#aaa", "-", "データなし", "#e0e0e0", "1.0", ""
            tooltip_html = ""
            
            if data:
                g_val = pd.to_numeric(data['g'], errors='coerce')
                b_val = pd.to_numeric(data['b'], errors='coerce')
                r_val = pd.to_numeric(data['r'], errors='coerce')
                diff_val = pd.to_numeric(data['diff'], errors='coerce')
                pred_val = pd.to_numeric(data['pred'], errors='coerce')
                
                g = int(g_val) if pd.notna(g_val) else 0
                b = int(b_val) if pd.notna(b_val) else 0
                r = int(r_val) if pd.notna(r_val) else 0
                diff = int(diff_val) if pd.notna(diff_val) else 0
                pred = float(pred_val) if pd.notna(pred_val) else np.nan
                m_name = str(data['mac_name']).replace('nan', '')

                prev_diff_val = data.get('prev_diff')
                rs_val = pd.to_numeric(data.get('result_score', np.nan), errors='coerce')
                result_score = float(rs_val) if pd.notna(rs_val) else np.nan
                
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
                elif map_metric == "結果点数(設定5近似度)":
                    if pd.notna(result_score):
                        main_text = f"{result_score:.1f}点"
                        if g < 1000: bg_color, text_color, border_color = "#f5f5f5", "#9e9e9e", "#e0e0e0"
                        elif result_score >= 80: bg_color, text_color, border_color = "#d32f2f", "#fff", "#b71c1c"
                        elif result_score >= 60: bg_color, text_color, border_color = "#ef5350", "#fff", "#c62828"
                        elif result_score >= 40: bg_color, text_color, border_color = "#ffcdd2", "#d32f2f", "#ef5350"
                        else: bg_color, text_color, border_color = "#eceff1", "#546e7a", "#cfd8dc"
                    else:
                        main_text = "-"
                        bg_color, text_color = "#f5f5f5", "#9e9e9e"
                if pd.notna(prev_diff_val):
                    p_diff = int(prev_diff_val)
                    p_color = "#d32f2f" if p_diff > 0 else "#1565c0" if p_diff < 0 else "#9e9e9e"
                    prev_diff_str = f"<div style='position: absolute; top: 1px; right: 3px; font-size: 9px; font-weight: bold; color: {p_color}; letter-spacing: -0.5px;'>前:{p_diff:+d}</div>"
                if g < filter_min_g or diff > filter_max_diff or diff < filter_min_diff: bg_color, text_color, border_color, opacity = "#fafafa", "#ccc", "#eee", "0.2"
            else:
                if filter_min_g > 0: opacity = "0.2"
                
            if data:
                b_prob_str = f"1/{int(g/b)}" if b > 0 else "-"
                r_prob_str = f"1/{int(g/r)}" if r > 0 else "-"
                t_prob_str = f"1/{int(g/(b+r))}" if (b+r) > 0 else "-"
                p_diff_str = f"{int(prev_diff_val):+d}枚" if pd.notna(prev_diff_val) else "-"
                pred_str = f"{pred*100:.1f}%" if pd.notna(pred) else "-"
                result_score_str = f"{result_score:.1f}点" if pd.notna(result_score) else "-"
                diff_color = "#EF5350" if diff > 0 else "#64B5F6" if diff < 0 else "#fff"
                
                tooltip_html = f"""
                <div class='tooltip-text'>
                    <div style='font-size:13px; color:#64B5F6; font-weight:bold; margin-bottom:4px;'>#{m_num} <span style='font-size:11px; color:#ccc; font-weight:normal;'>{m_name}</span></div>
                    <div style='display:flex; justify-content:space-between;'><span>総回転:</span><b>{g}G</b></div>
                    <div style='display:flex; justify-content:space-between;'><span>差枚数:</span><b style='color:{diff_color};'>{diff:+d}枚</b></div>
                    <div style='display:flex; justify-content:space-between; font-size:9px; color:#aaa; margin-top:-2px;'><span>(前日差枚:</span><span>{p_diff_str})</span></div>
                    <hr style='margin:6px 0; border:none; border-top:1px dashed #666;'>
                    <div style='display:flex; justify-content:space-between;'><span>BIG:</span><span>{b}回 ({b_prob_str})</span></div>
                    <div style='display:flex; justify-content:space-between;'><span>REG:</span><span>{r}回 ({r_prob_str})</span></div>
                    <div style='display:flex; justify-content:space-between;'><span>合算:</span><span>{t_prob_str}</span></div>
                    <hr style='margin:6px 0; border:none; border-top:1px dashed #666;'>
                    <div style='display:flex; justify-content:space-between;'><span>AI事前期待度:</span><b style='color:#FFCA28;'>{pred_str}</b></div>
                    <div style='display:flex; justify-content:space-between;'><span>当日結果点数:</span><b style='color:#FFCA28;'>{result_score_str}</b></div>
                </div>
                """
                
            html_parts.append(f"""
                <div class='machine-box' style='width: 72px; height: 66px; flex-shrink: 0; background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 6px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); padding: 2px; box-sizing: border-box; opacity: {opacity};'>
                    <div style='position: absolute; top: 1px; left: 3px; font-size: 10px; font-weight: bold; color: #555;'>{'✨' if str(m_num) == island['corner'] else ''}#{m_num}</div>
                    {prev_diff_str}
                    <div style='font-size: 15px; font-weight: bold; color: {text_color}; margin-top: 10px; line-height: 1;'>{main_text}</div>
                    <div style='font-size: 9px; color: #666; margin-top: 3px; width: 100%; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{sub_text}</div>
                    {tooltip_html}
                </div>
            """)
        html_parts.append("</div></div>")
    html_parts.append("</div>")
    st.components.v1.html("".join(html_parts), height=800, scrolling=True)