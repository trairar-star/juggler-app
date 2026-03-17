import pandas as pd
import numpy as np
import streamlit as st # type: ignore
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
    else:
        display_df['期待度'] = "-"

    if '予測信頼度' not in display_df.columns:
        display_df['予測信頼度'] = "-"

    display_df['総回転'] = display_df.get('累計ゲーム', 0)
    
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
    sort_options = ["台番号順", "差枚が多い順", "合算確率が良い順", "REG確率が良い順", "期待度が高い順"]
    sort_by = st.radio("並び替え", sort_options, horizontal=True)
    
    if sort_by == "差枚が多い順":
        display_df = display_df.sort_values('差枚', ascending=False)
    elif sort_by == "合算確率が良い順":
        display_df = display_df.sort_values('tmp_total_prob', ascending=False)
    elif sort_by == "REG確率が良い順":
        display_df['tmp_reg_prob'] = display_df.get('REG', 0).fillna(0) / display_df['総回転'].replace(0, np.nan)
        display_df = display_df.sort_values('tmp_reg_prob', ascending=False)
    elif sort_by == "期待度が高い順":
        if 'prediction_score' in display_df.columns:
            display_df = display_df.sort_values('prediction_score', ascending=False)
    else:
        try:
            display_df['台番号_num'] = pd.to_numeric(display_df['台番号'])
            display_df = display_df.sort_values('台番号_num')
        except:
            display_df = display_df.sort_values('台番号')

    st.markdown(f"### 🎰 {selected_date.strftime('%Y-%m-%d')} の結果 ({selected_shop})")

    cols = ['台番号', '機種名']
    if '期待度' in display_df.columns: cols.append('期待度')
    if '予測信頼度' in display_df.columns: cols.append('予測信頼度')
    cols.extend(['差枚', '総回転', 'BIG', 'REG', '合算確率_str', 'BIG確率_str', 'REG確率_str'])
    
    if '根拠' in display_df.columns:
        cols.append('根拠')
        
    available_cols = [c for c in cols if c in display_df.columns]

    st.dataframe(
        display_df[available_cols],
        column_config={
            "台番号": st.column_config.TextColumn("No.", width="small"),
            "機種名": st.column_config.TextColumn("機種", width="small"),
            "期待度": st.column_config.TextColumn("AI期待度", width="small", help="AIが前日時点で予測していた設定5以上の確率です。"),
            "予測信頼度": st.column_config.TextColumn("信頼度", width="small"),
            "差枚": st.column_config.NumberColumn("差枚", format="%+d"),
            "総回転": st.column_config.NumberColumn("総回転"),
            "BIG": st.column_config.NumberColumn("BIG"),
            "REG": st.column_config.NumberColumn("REG"),
            "合算確率_str": st.column_config.TextColumn("合算確率", width="small"),
            "BIG確率_str": st.column_config.TextColumn("BIG確率", width="small"),
            "REG確率_str": st.column_config.TextColumn("REG確率", width="small"),
            "根拠": st.column_config.TextColumn("AI推奨根拠 (期待度70%以上だった台のみ)", width="large"),
        },
        use_container_width=True,
        hide_index=True
    )