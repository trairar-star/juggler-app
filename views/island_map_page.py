import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import backend

def render_island_map_page(df_raw, df_pred_log, df_island):
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

    df_month['day_str'] = df_month['対象日付'].dt.strftime('%m/%d')
    
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
    pivot_r = df_month_dedup['r'].unstack()
    pivot_diff = df_month_dedup['diff'].unstack()
    
    date_cols = sorted([c for c in pivot_g.columns])
    
    records = []
    for (mac_num, mac_name) in pivot_g.index:
        row_data = {'台番号': str(mac_num), '機種名': mac_name}
        for d in date_cols:
            g_val = pivot_g.loc[(mac_num, mac_name), d]
            r_val = pivot_r.loc[(mac_num, mac_name), d]
            diff_val = pivot_diff.loc[(mac_num, mac_name), d]
            if pd.isna(g_val):
                row_data[d] = np.nan
            else:
                row_data[d] = (g_val, r_val, diff_val)
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
            
            g, r, diff = val
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
            g, r, diff = val
            if g == 0: return "-"
            if table_metric == "REG確率":
                prob_str = f"1/{int(g/r)}" if r > 0 else "-"
                return f"<div style='line-height:1.3; padding:2px;'>{int(g)}G<br>{int(r)}R ({prob_str})<br>{int(diff):+d}枚</div>"
            else:
                return f"<div style='line-height:1.3; padding:2px;'>{int(g)}G<br>{int(r)}R<br>{int(diff):+d}枚</div>"
        return "-"

    format_dict = {c: fmt_cell for c in date_cols}
    styled_df = pivot_val.style.apply(style_monthly_table, axis=1).format(format_dict, na_rep="-")

    if table_metric == "REG確率":
        st.markdown("**(色分けの目安)** 🟥: 設定6基準以上 / 🟧: 設定5基準以上 / 🟨: 設定4基準以上 ｜ 台番号背景🟨: 角台")
    else:
        st.markdown("**(色分けの目安)** 🟥: +2000枚以上 / 🟧: +1000枚以上 / 🟨: プラス / 🟦: -1000枚以下 ｜ 台番号背景🟨: 角台")

    # Streamlitの仕様による行高さの制限を回避するため、HTML形式で描画
    html_table = styled_df.hide(axis="index").to_html(escape=False)
    
    custom_css = """
    <style>
        .scroll-container {
            overflow: auto;
            max-height: 85vh;
            width: 100%;
            border: 1px solid #ccc;
        }
        .scroll-container table {
            border-collapse: collapse;
            min-width: 100%;
            font-size: 11px;
            font-family: sans-serif;
            text-align: center;
        }
        .scroll-container th, .scroll-container td {
            border: 1px solid #ccc;
            padding: 4px;
            white-space: nowrap;
        }
        .scroll-container thead th {
            position: sticky;
            top: 0;
            background-color: #eeeeee;
            z-index: 1;
        }
    </style>
    """
    st.components.v1.html(f"{custom_css}<div class='scroll-container'>{html_table}</div>", height=850, scrolling=False)