import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import backend

def render_island_map_page(df_raw, df_pred_log, df_island):
    st.header("📅 月間 台別データ表")
    st.caption("1ヶ月間の各台の成績表（REG確率や差枚）を一覧で確認できます。設定基準による色分けと、角台の強調表示により傾向を一目で掴めます。")

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
        for _, r in shop_islands.iterrows():
            c = str(r.get('メイン角番', '')).strip()
            if c: corner_macs.add(c)

    if table_metric == "REG確率":
        df_month['値'] = np.where(pd.to_numeric(df_month['REG'], errors='coerce').fillna(0) > 0, pd.to_numeric(df_month['累計ゲーム'], errors='coerce').fillna(0) / pd.to_numeric(df_month['REG'], errors='coerce').fillna(0), 0)
    else:
        df_month['値'] = pd.to_numeric(df_month['差枚'], errors='coerce').fillna(0)

    pivot_val = df_month.pivot_table(index=['台番号', '機種名'], columns='day_str', values='値', aggfunc='first').reset_index()

    pivot_val['台番号_num'] = pd.to_numeric(pivot_val['台番号'], errors='coerce')
    pivot_val = pivot_val.sort_values('台番号_num').drop(columns=['台番号_num'])

    date_cols = [c for c in pivot_val.columns if c not in ['台番号', '機種名']]
    date_cols = sorted(date_cols)

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
        spec_r4 = 1.0 / specs[matched_key].get('設定4', {"REG": 300.0})["REG"] if matched_key in specs else 1/300.0
        spec_r5 = 1.0 / specs[matched_key].get('設定5', {"REG": 260.0})["REG"] if matched_key in specs else 1/260.0
        spec_r6 = 1.0 / specs[matched_key].get('設定6', {"REG": 240.0})["REG"] if matched_key in specs else 1/240.0
        
        for i, col in enumerate(row.index):
            if col in ['台番号', '機種名', '角台']: continue
            val = row[col]
            if pd.isna(val) or val == 0:
                continue
                
            if table_metric == "REG確率":
                if val <= 0: continue
                prob = 1.0 / val
                if prob >= spec_r6:
                    styles[i] = 'background-color: #FFCDD2; color: #B71C1C; font-weight: bold;'
                elif prob >= spec_r5:
                    styles[i] = 'background-color: #FFE082; color: #E65100; font-weight: bold;'
                elif prob >= spec_r4:
                    styles[i] = 'background-color: #FFF59D; color: #F57F17;'
            elif table_metric == "差枚":
                if val >= 2000:
                    styles[i] = 'background-color: #FFCDD2; color: #B71C1C; font-weight: bold;'
                elif val >= 1000:
                    styles[i] = 'background-color: #FFE082; color: #E65100; font-weight: bold;'
                elif val > 0:
                    styles[i] = 'background-color: #FFF59D; color: #F57F17;'
                elif val <= -1000:
                    styles[i] = 'background-color: #E3F2FD; color: #1565C0;'
                    
        return styles

    format_dict = {}
    if table_metric == "REG確率":
        for c in date_cols:
            format_dict[c] = lambda x: f"1/{int(x)}" if pd.notna(x) and x > 0 else "-"
    else:
        for c in date_cols:
            format_dict[c] = lambda x: f"{int(x):+d}" if pd.notna(x) else "-"

    styled_df = pivot_val.style.apply(style_monthly_table, axis=1).format(format_dict, na_rep="-")

    config = {
        "台番号": st.column_config.TextColumn("台番号", width="small"),
        "機種名": st.column_config.TextColumn("機種名", width="small"),
        "角台": None
    }
    for c in date_cols:
        config[c] = st.column_config.TextColumn(c, width="small")

    if table_metric == "REG確率":
        st.markdown("**(色分けの目安)** 🟥: 設定6基準以上 / 🟧: 設定5基準以上 / 🟨: 設定4基準以上 ｜ 台番号背景🟨: 角台")
    else:
        st.markdown("**(色分けの目安)** 🟥: +2000枚以上 / 🟧: +1000枚以上 / 🟨: プラス / 🟦: -1000枚以下 ｜ 台番号背景🟨: 角台")

    st.dataframe(styled_df, column_config=config, use_container_width=True, hide_index=True)