import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import backend

def render_island_map_page(df_raw, df_pred_log, df_island, df_predict=None):
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.header("📅 月間 台別データ表")
        st.caption("1ヶ月間の各台の成績表（REG確率や差枚）を一覧で確認できます。設定基準による色分けと、角台の強調表示により傾向を一目で掴めます。")
    with col_h2:
        st.components.v1.html("""
            <button id="fs-btn" onclick="toggleFullscreen()" style="width: 100%; height: 36px; border-radius: 6px; background-color: #42A5F5; color: white; border: none; cursor: pointer; font-weight: bold; font-family: sans-serif; font-size: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                🖥️ 領域を最大化
            </button>
            <script>
            function toggleFullscreen() {
                const doc = window.parent.document;
                const mainBlock = doc.querySelector('[data-testid="stMainBlockContainer"]');
                const btn = document.getElementById('fs-btn');
                
                if (!doc.fullscreenElement) {
                    if (mainBlock) {
                        mainBlock.requestFullscreen().then(() => {
                            mainBlock.style.backgroundColor = window.getComputedStyle(doc.body).backgroundColor || '#ffffff';
                            mainBlock.style.maxWidth = '100%';
                            mainBlock.style.padding = '1rem';
                            mainBlock.style.overflowY = 'auto';
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
                if (!doc.fullscreenElement) {
                    if (mainBlock) { mainBlock.style.backgroundColor = ''; mainBlock.style.maxWidth = ''; mainBlock.style.padding = ''; }
                    if (btn) btn.innerHTML = '🖥️ 領域を最大化';
                }
            });
            </script>
        """, height=50)

    if df_raw.empty:
        st.warning("データがありません。")
        st.stop()

    specs = backend.get_machine_specs()
    date_col = '対象日付'
    temp_df = df_raw.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
    available_dates = sorted(temp_df[date_col].dropna().dt.date.unique(), reverse=True)
    available_months = sorted(list(set([d.strftime('%Y-%m') for d in available_dates])), reverse=True)

    if not available_dates:
        st.warning("有効な日付データがありません。")
        st.stop()

    shop_col = '店名' if '店名' in temp_df.columns else ('店舗名' if '店舗名' in temp_df.columns else None)
    shops = ["店舗を選択してください"] + sorted(list(temp_df[shop_col].dropna().unique()))
    
    default_index = 0
    saved_shop = st.session_state.get("global_selected_shop", "店舗を選択してください")
    if saved_shop in shops:
        default_index = shops.index(saved_shop)

    with st.sidebar:
        st.markdown("---")
        st.subheader("🛠️ 表示設定")
        selected_shop = st.selectbox("🏬 店舗を選択", shops, index=default_index)
        
        if selected_shop != "店舗を選択してください":
            st.session_state["global_selected_shop"] = selected_shop

        st.markdown("---")
        st.subheader("📅 データ表 設定")
        table_metric = st.radio("📊 表示する指標", ["REG確率", "差枚"], horizontal=True)
        table_period = st.selectbox("表示期間", ["直近30日", "直近14日"] + available_months)
        day_filter_options = [
            "すべて", 
            "0のつく日", "1のつく日", "2のつく日", "3のつく日", "4のつく日", 
            "5のつく日", "6のつく日", "7のつく日", "8のつく日", "9のつく日",
            "月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"
        ]
        table_day_filter = st.selectbox("📅 日付フィルター (特定の日付/曜日で絞り込み)", day_filter_options)
        min_g_filter = st.slider("🌫️ グレーアウトする回転数 (G以下)", min_value=0, max_value=8000, value=1000, step=500, help="指定した回転数以下のデータは灰色で表示されます。")

    if selected_shop == "店舗を選択してください":
        st.info("👆 サイドバーから店舗を選択すると、台別データ表が表示されます。")
        st.stop()

    st.divider()

    # --- 最新の期待度データを取得 ---
    pred_dict = {}
    pred_date_disp = ""
    
    # 優先して df_predict (全台の最新予測データ) を使用する
    if df_predict is not None and not df_predict.empty:
        shop_col_pred = '店名' if '店名' in df_predict.columns else ('店舗名' if '店舗名' in df_predict.columns else None)
        if shop_col_pred:
            df_shop_pred = df_predict[df_predict[shop_col_pred] == selected_shop].copy()
            if not df_shop_pred.empty:
                if 'next_date' in df_shop_pred.columns:
                    latest_pred_date = pd.to_datetime(df_shop_pred['next_date'].max())
                    pred_date_disp = latest_pred_date.strftime('%m/%d') + " "
                elif '予測対象日' in df_shop_pred.columns:
                    latest_pred_date = pd.to_datetime(df_shop_pred['予測対象日'].max())
                    pred_date_disp = latest_pred_date.strftime('%m/%d') + " "
                
                if '台番号' in df_shop_pred.columns and 'prediction_score' in df_shop_pred.columns:
                    df_shop_pred['台番号_str'] = df_shop_pred['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
                    for _, r in df_shop_pred.iterrows():
                        pred_dict[r['台番号_str']] = r['prediction_score']

    # df_predict が渡されなかった場合は既存の df_pred_log (上位10%のみ) をフォールバックとして使用
    if not pred_dict and not df_pred_log.empty:
        temp_pred = df_pred_log.copy()
        if '予測対象日' in temp_pred.columns:
            temp_pred['予測対象日_merge'] = pd.to_datetime(temp_pred['予測対象日'], errors='coerce').fillna(pd.to_datetime(temp_pred['対象日付'], errors='coerce') + pd.Timedelta(days=1))
        else:
            temp_pred['予測対象日_merge'] = pd.to_datetime(temp_pred['対象日付'], errors='coerce') + pd.Timedelta(days=1)
            
        temp_pred['予測対象日_str'] = temp_pred['予測対象日_merge'].dt.strftime('%Y-%m-%d')
        
        shop_col_pred = '店名' if '店名' in temp_pred.columns else ('店舗名' if '店舗名' in temp_pred.columns else None)
        if shop_col_pred:
            df_shop_pred = temp_pred[temp_pred[shop_col_pred] == selected_shop].copy()
            if not df_shop_pred.empty:
                latest_pred_date_str = df_shop_pred['予測対象日_str'].max()
                try:
                    pred_date_disp = pd.to_datetime(latest_pred_date_str).strftime('%m/%d') + " "
                except:
                    pass
                df_latest_pred = df_shop_pred[df_shop_pred['予測対象日_str'] == latest_pred_date_str].copy()
                
                if '実行日時' in df_latest_pred.columns:
                    df_latest_pred['実行日時'] = pd.to_datetime(df_latest_pred['実行日時'], errors='coerce')
                    df_latest_pred = df_latest_pred.sort_values('実行日時', ascending=False).drop_duplicates(subset=['台番号'])
                
                if '台番号' in df_latest_pred.columns and 'prediction_score' in df_latest_pred.columns:
                    df_latest_pred['台番号_str'] = df_latest_pred['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
                    for _, r in df_latest_pred.iterrows():
                        pred_dict[r['台番号_str']] = r['prediction_score']

    df_shop = df_raw[df_raw[shop_col] == selected_shop].copy()
    df_shop['対象日付'] = pd.to_datetime(df_shop['対象日付'], errors='coerce')
    df_shop = df_shop.dropna(subset=['対象日付', '台番号'])
    df_shop['台番号'] = df_shop['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)

    if table_period == "直近30日":
        max_date = df_shop['対象日付'].max()
        cutoff_date = max_date - pd.Timedelta(days=30)
        df_month = df_shop[df_shop['対象日付'] > cutoff_date].copy()
    elif table_period == "直近14日":
        max_date = df_shop['対象日付'].max()
        cutoff_date = max_date - pd.Timedelta(days=14)
        df_month = df_shop[df_shop['対象日付'] > cutoff_date].copy()
    else:
        df_month = df_shop[df_shop['対象日付'].dt.strftime('%Y-%m') == table_period].copy()

    if table_day_filter != "すべて":
        if "のつく日" in table_day_filter:
            target_digit = int(table_day_filter.replace("のつく日", ""))
            df_month = df_month[df_month['対象日付'].dt.day % 10 == target_digit].copy()
        elif "曜日" in table_day_filter:
            weekdays = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]
            target_weekday = weekdays.index(table_day_filter)
            df_month = df_month[df_month['対象日付'].dt.dayofweek == target_weekday].copy()

    if df_month.empty:
        st.warning("表示するデータがありません。")
        st.stop()

    weekdays_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
    df_month['day_str'] = df_month['対象日付'].dt.strftime('%m/%d') + "(" + df_month['対象日付'].dt.dayofweek.map(weekdays_map) + ")"
    
    corner_macs = set()
    if 'is_corner' in df_month.columns:
        corners = df_month[df_month['is_corner'] == 1]['台番号'].unique()
        corner_macs.update(corners)
        
    if df_island is not None and not df_island.empty:
        shop_islands = df_island[df_island['店名'] == selected_shop]
        parsed_machines = []
        for _, r in shop_islands.iterrows():
            c = str(r.get('メイン角番', '')).strip()
            if c: corner_macs.add(c)
            
            rule = str(r.get('台番号ルール', ''))
            if rule and rule.strip() != '' and rule != 'nan':
                for part in rule.split(','):
                    part = part.strip()
                    if not part: continue
                    if '-' in part:
                        try:
                            s_str, e_str = part.split('-', 1)
                            parsed_machines.extend(range(int(s_str), int(e_str) + 1))
                        except: pass
                    else:
                        try: parsed_machines.append(int(part))
                        except: pass
            else:
                try:
                    s = int(r.get('開始台番号', 0))
                    e = int(r.get('終了台番号', 0))
                    if s > 0 and e >= s: parsed_machines.extend(range(s, e + 1))
                except: pass
        island_order = []
        for m in parsed_machines:
            if str(m) not in island_order:
                island_order.append(str(m))
    else:
        island_order = []

    df_month['g'] = pd.to_numeric(df_month['累計ゲーム'], errors='coerce').fillna(0)
    df_month['b'] = pd.to_numeric(df_month['BIG'], errors='coerce').fillna(0)
    df_month['r'] = pd.to_numeric(df_month['REG'], errors='coerce').fillna(0)
    df_month['diff'] = pd.to_numeric(df_month['差枚'], errors='coerce').fillna(0)

    df_month_dedup = df_month.drop_duplicates(subset=['台番号', '機種名', 'day_str'], keep='first')
    df_month_dedup = df_month_dedup.set_index(['台番号', '機種名', 'day_str'])
    
    pivot_g = df_month_dedup['g'].unstack()
    pivot_b = df_month_dedup['b'].unstack()
    pivot_r = df_month_dedup['r'].unstack()
    pivot_diff = df_month_dedup['diff'].unstack()
    
    date_cols = sorted([c for c in pivot_g.columns])
    
    records = []
    for (mac_num, mac_name) in pivot_g.index:
        mac_str = str(mac_num)
        
        # --- 機種スペックのツールチップ(ポップアップ)を作成 ---
        matched_key = backend.get_matched_spec_key(mac_name, specs)
        spec_tooltip = ""
        if matched_key and matched_key in specs:
            spec_info = specs[matched_key]
            tooltip_lines = [f"【{matched_key}】確率目安"]
            for s in ["設定1", "設定4", "設定5", "設定6"]:
                if s in spec_info:
                    sp = spec_info[s]
                    tooltip_lines.append(f"{s}: BB 1/{sp.get('BIG',0):.1f} | RB 1/{sp.get('REG',0):.1f} | 合算 1/{sp.get('合算',0):.1f}")
            spec_tooltip = "&#10;".join(tooltip_lines) # HTMLのtitle属性用の改行コード
            
        mac_disp = mac_name
        if mac_str in pred_dict:
            score = pred_dict[mac_str]
            if pd.notna(score):
                score_pct = int(score * 100)
                color = "#D32F2F" if score_pct >= 40 else "#EF6C00" if score_pct >= 30 else "#757575"
                mac_disp = f"{mac_name}<br><span class='pred-pct' style='color:{color};'>{pred_date_disp}期待度:{score_pct}%</span>"
                
        mac_disp = f"<div title='{spec_tooltip}' style='cursor: help;'>{mac_disp}</div>"
        
        row_data = {'台番号': mac_str, '機種名': mac_disp}
        for d in date_cols:
            g_val = pivot_g.loc[(mac_num, mac_name), d]
            b_val = pivot_b.loc[(mac_num, mac_name), d]
            r_val = pivot_r.loc[(mac_num, mac_name), d]
            diff_val = pivot_diff.loc[(mac_num, mac_name), d]
            if pd.isna(g_val):
                row_data[d] = np.nan
            else:
                row_data[d] = (g_val, b_val, r_val, diff_val)
        records.append(row_data)
        
    pivot_val = pd.DataFrame(records)

    def get_sort_key(m_str):
        if m_str in island_order:
            return island_order.index(m_str)
        else:
            return 999999 + int(m_str) if str(m_str).isdigit() else 999999
            
    pivot_val['sort_key'] = pivot_val['台番号'].apply(get_sort_key)
    pivot_val = pivot_val.sort_values('sort_key').drop(columns=['sort_key'])
    pivot_val = pivot_val[['台番号', '機種名'] + date_cols]

    pivot_val['角台'] = pivot_val['台番号'].apply(lambda x: 1 if str(x) in corner_macs else 0)

    def style_monthly_table(row):
        styles = [''] * len(row)
        mac_name = row['機種名']
        is_corner = row['角台']
        
        idx_num = row.index.get_loc('台番号')
        if is_corner:
            styles[idx_num] = 'background-color: #FFF9C4; color: #F57F17; font-weight: bold;'
            
        matched_key = backend.get_matched_spec_key(mac_name, specs)
        spec_r4_den = specs[matched_key].get('設定4', {"REG": 300.0})["REG"] if matched_key in specs else 300.0
        spec_r5_den = specs[matched_key].get('設定5', {"REG": 260.0})["REG"] if matched_key in specs else 260.0
        spec_r6_den = specs[matched_key].get('設定6', {"REG": 240.0})["REG"] if matched_key in specs else 240.0
        
        for i, col in enumerate(row.index):
            if col in ['台番号', '機種名', '角台']: continue
            val = row[col]
            if not isinstance(val, tuple): continue
            
            g, b, r, diff = val
            if g == 0: continue
            
            if g <= min_g_filter:
                styles[i] = 'background-color: #EEEEEE; color: #9E9E9E;'
                continue
                
            bg_color = ""
            text_color = ""
                
            if table_metric == "REG確率":
                if r > 0:
                    prob = g / r
                    if prob <= spec_r6_den: bg_color = "#FFCDD2"
                    elif prob <= spec_r5_den: bg_color = "#FFE082"
                    elif prob <= spec_r4_den: bg_color = "#FFF59D"
            elif table_metric == "差枚":
                if diff >= 2000: bg_color = "#FFCDD2"
                elif diff >= 1000: bg_color = "#FFE082"
                elif diff > 0: bg_color = "#FFF59D"
                elif diff <= -1000: bg_color = "#E3F2FD"
                
            if diff > 1000: text_color = "#D32F2F"
            elif diff > 0: text_color = "#EF6C00"
            elif diff < 0: text_color = "#1565C0"
            
            style_str = "vertical-align: middle; "
            if bg_color: style_str += f"background-color: {bg_color}; "
            if text_color:
                style_str += f"color: {text_color}; "
                if diff > 1000: style_str += "font-weight: bold; "
                
            styles[i] = style_str
                    
        return styles

    def fmt_cell(val):
        if isinstance(val, tuple):
            g, b, r, diff = val
            if g == 0: return "-"
            if table_metric == "REG確率":
                prob_str = f"1/{int(g/r)}" if r > 0 else "-"
                return f"<div class='cell-val' data-b='{int(b)}'>{int(g)}G<br>{int(r)}R ({prob_str})<br>{int(diff):+d}枚</div>"
            else:
                return f"<div class='cell-val' data-b='{int(b)}'>{int(g)}G<br>{int(r)}R<br>{int(diff):+d}枚</div>"
        return "-"

    format_dict = {c: fmt_cell for c in date_cols}
    styled_df = pivot_val.style.apply(style_monthly_table, axis=1).format(format_dict, na_rep="-")

    if table_metric == "REG確率":
        st.markdown("**(色分けの目安)** 🟥: 設定6基準以上 / 🟧: 設定5基準以上 / 🟨: 設定4基準以上 ｜ 台番号背景🟨: 角台")
    else:
        st.markdown("**(色分けの目安)** 🟥: +2000枚以上 / 🟧: +1000枚以上 / 🟨: プラス / 🟦: -1000枚以下 ｜ 台番号背景🟨: 角台")
        
    st.info("💡 **便利機能**: 表のセルをマウスでクリック＆ドラッグ（なぞって複数選択）すると、選択した台の **合計ゲーム数・REG回数・合算REG確率・合計差枚** が画面下部に自動計算されます！島や並びの判別に活用してください。")

    # Streamlitの仕様による行高さの制限を回避するため、HTML形式で描画
    html_table = styled_df.hide(axis="index").to_html(escape=False)
    
    custom_css = """
    <style>
        .scroll-container {
            position: relative;
            overflow: auto;
            max-height: 85vh;
            width: 100%;
            border: 1px solid #ccc;
        }
        .scroll-container table {
            border-collapse: separate;
            border-spacing: 0;
            min-width: 100%;
            font-size: 11px;
            font-family: sans-serif;
            text-align: center;
            user-select: none;
            -webkit-user-select: none;
        }
        .scroll-container th, .scroll-container td {
            border-right: 1px solid #ccc;
            border-bottom: 1px solid #ccc;
            padding: 4px;
            white-space: nowrap;
        }
        .scroll-container thead th {
            position: sticky;
            top: 0;
            background-color: #eeeeee;
            z-index: 10;
            border-top: 1px solid #ccc;
        }
        .scroll-container tr th:first-child, .scroll-container tr td:first-child {
            border-left: 1px solid #ccc;
        }
        .scroll-container th:nth-child(1), .scroll-container td:nth-child(1) {
            position: sticky;
            left: 0;
            background-color: #f9f9f9;
            z-index: 5;
        }
        .scroll-container th:nth-child(2), .scroll-container td:nth-child(2) {
            position: sticky;
            background-color: #f9f9f9;
            z-index: 5;
        }
        .scroll-container thead th:nth-child(1), .scroll-container thead th:nth-child(2) {
            z-index: 15;
            background-color: #eeeeee;
        }
        .selected-cell {
            box-shadow: inset 0 0 0 3px #E91E63 !important;
        }
        #calc-bar {
            position: sticky;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: rgba(30, 30, 30, 0.95);
            color: white;
            padding: 8px 15px;
            font-size: 14px;
            line-height: 1.4;
            font-weight: bold;
            z-index: 10;
            display: none;
            box-shadow: 0 -2px 8px rgba(0,0,0,0.4);
            border-radius: 4px 4px 0 0;
            justify-content: space-between;
            align-items: center;
            box-sizing: border-box;
        }
        #calc-content {
            text-align: left;
            flex-grow: 1;
        }
        #clear-btn {
            background-color: #F44336;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            cursor: pointer;
            font-weight: bold;
            font-size: 12px;
            margin-left: 10px;
            flex-shrink: 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        #clear-btn:active {
            background-color: #D32F2F;
        }
        .pred-pct {
            font-size: 10px;
            font-weight: bold;
        }
        .cell-val {
            line-height: 1.3;
            padding: 2px;
        }
        @media (max-width: 768px) {
            .scroll-container table {
                font-size: 9px;
            }
            .scroll-container th, .scroll-container td {
                padding: 2px;
            }
            .pred-pct {
                font-size: 8px;
            }
            .cell-val {
                line-height: 1.1;
                padding: 1px;
            }
            #calc-bar {
                font-size: 12px;
                padding: 6px 10px;
            }
            #clear-btn {
                padding: 6px 8px;
                font-size: 11px;
            }
        }
    </style>
    """
    
    html_content = f"""
    {custom_css}
    <div class='scroll-container' id='main-scroll'>
        {html_table}
        <div id="calc-bar">
            <div id="calc-content"></div>
            <button id="clear-btn" onclick="clearTableSelection()">✖ クリア</button>
        </div>
    </div>
    <script>
        (function() {{
            const container = document.getElementById('main-scroll');
            const table = container.querySelector('table');
            let isMouseDown = false;
            let selectionMode = true; // true: 選択(塗りつぶし), false: 解除(消しゴム)
            
            window.clearTableSelection = function() {{
                document.querySelectorAll('.selected-cell').forEach(c => c.classList.remove('selected-cell'));
                updateCalc();
            }};

            function updateStickyLeft() {{
                const th1 = table.querySelector('th:nth-child(1)');
                const th2 = table.querySelector('th:nth-child(2)');
                if(th1 && th2) {{
                    const w1 = th1.getBoundingClientRect().width;
                    const cells2 = table.querySelectorAll('th:nth-child(2), td:nth-child(2)');
                    cells2.forEach(c => {{
                        c.style.left = w1 + 'px';
                    }});
                }}
            }}
            
            setTimeout(updateStickyLeft, 50);
            setTimeout(updateStickyLeft, 500);
            window.addEventListener('resize', updateStickyLeft);

            function updateCalc() {{
                let totalG = 0;
                let totalB = 0;
                let totalR = 0;
                let totalDiff = 0;
                let count = 0;
                const selected = table.querySelectorAll('.selected-cell');
                selected.forEach(td => {{
                    const cellVal = td.querySelector('.cell-val');
                    if(cellVal) {{
                        const text = td.innerText || td.textContent;
                        const gMatch = text.match(/([0-9]+)G/);
                        const rMatch = text.match(/([0-9]+)R/);
                        const diffMatch = text.match(/([+-]?[0-9]+)枚/);
                        
                        let bVal = parseInt(cellVal.getAttribute('data-b') || '0', 10);
    
                        if(gMatch && rMatch && diffMatch) {{
                            totalG += parseInt(gMatch[1], 10);
                            totalR += parseInt(rMatch[1], 10);
                            totalDiff += parseInt(diffMatch[1].replace('+', ''), 10);
                            totalB += bVal;
                            count++;
                        }}
                    }}
                }});

                const calcBar = document.getElementById('calc-bar');
                const calcContent = document.getElementById('calc-content');
                if (count > 0) {{
                    let regProbStr = totalR > 0 ? "1/" + Math.floor(totalG / totalR) : "-";
                    let bigProbStr = totalB > 0 ? "1/" + Math.floor(totalG / totalB) : "-";
                    let totProbStr = (totalB + totalR) > 0 ? "1/" + Math.floor(totalG / (totalB + totalR)) : "-";
                    let diffSign = totalDiff > 0 ? "+" : "";
                    calcContent.innerHTML = `🎰 [選択: ${{count}}台] ｜ 総回転: ${{totalG}}G ｜ 差枚: <span style="color:${{totalDiff>0?'#FFCA28':'#81D4FA'}}">${{diffSign}}${{totalDiff}}枚</span><br>BIG: ${{totalB}}回 (<span style="color:#FFCA28">${{bigProbStr}}</span>) ｜ REG: ${{totalR}}回 (<span style="color:#FFCA28">${{regProbStr}}</span>) ｜ 合算: <span style="color:#FFCA28">${{totProbStr}}</span>`;
                    calcBar.style.display = 'flex';
                }} else {{
                    calcBar.style.display = 'none';
                }}
            }}

            table.addEventListener('mousedown', function(e) {{
                let td = e.target.closest('td');
                if (!td || td.cellIndex < 2) return;
                
                isMouseDown = true;
                selectionMode = !td.classList.contains('selected-cell');
                
                if (selectionMode) {{
                    td.classList.add('selected-cell');
                }} else {{
                    td.classList.remove('selected-cell');
                }}
                updateCalc();
            }});

            table.addEventListener('mouseover', function(e) {{
                if (!isMouseDown) return;
                let td = e.target.closest('td');
                if (!td || td.cellIndex < 2) return;
                
                if (selectionMode) {{
                    td.classList.add('selected-cell');
                }} else {{
                    td.classList.remove('selected-cell');
                }}
                updateCalc();
            }});

            document.addEventListener('mouseup', function(e) {{
                isMouseDown = false;
            }});
            
            // タッチデバイス(スマホ)でのスライド選択用
            table.addEventListener('touchstart', function(e) {{
                isMouseDown = true;
            }}, {{ passive: true }});
            
            table.addEventListener('touchmove', function(e) {{
                if (!isMouseDown) return;
                let touch = e.touches[0];
                let element = document.elementFromPoint(touch.clientX, touch.clientY);
                if (!element) return;
                
                let td = element.closest('td');
                if (!td || td.cellIndex < 2) return;
                
                if (selectionMode) {{
                    td.classList.add('selected-cell');
                }} else {{
                    td.classList.remove('selected-cell');
                }}
                updateCalc();
            }}, {{ passive: true }});
            
            document.addEventListener('touchend', function(e) {{
                isMouseDown = false;
            }});
        }})();
    </script>
    """
    
    st.components.v1.html(html_content, height=850, scrolling=False)