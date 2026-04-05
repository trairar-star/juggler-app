import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import altair as alt # type: ignore
import backend

def render_my_balance_page(df_raw):
    st.header("💰 マイ収支管理")
    st.caption("あなたの稼働実績（投資・回収・収支）を記録・分析します。")

    # --- 1. 収支入力フォーム ---
    with st.expander("📝 収支データを登録", expanded=False):
        # 日付選択をフォーム外に配置し、変更時に即座に曜日を反映させる
        input_date = st.date_input("稼働日", pd.Timestamp.now(tz='Asia/Tokyo').date())
        weekdays = ['月', '火', '水', '木', '金', '土', '日']
        wd_str = weekdays[input_date.weekday()]

        with st.form("balance_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                # 既存データから店名リストを取得
                shops = []
                if not df_raw.empty:
                    shop_col = '店名' if '店名' in df_raw.columns else '店舗名'
                    if shop_col in df_raw.columns:
                        shops = list(df_raw[shop_col].unique())
                input_shop = st.selectbox("店舗名", shops + ["その他 (手入力)"])
                if input_shop == "その他 (手入力)":
                    input_shop = st.text_input("店舗名を入力")
                
                input_number = st.text_input("台番号", placeholder="例: 123")
                
            with col2:
                # 機種名リスト
                machines = []
                if not df_raw.empty and '機種名' in df_raw.columns:
                    machines = list(df_raw['機種名'].dropna().unique())
                    
                machine_options = machines + ["その他 (手入力)"]
                default_idx = machine_options.index("マイジャグラーV") if "マイジャグラーV" in machine_options else 0
                
                input_machine = st.selectbox("機種名", machine_options, index=default_idx)
                if input_machine == "その他 (手入力)":
                    input_machine = st.text_input("機種名を入力")
            
            c1, c2, c3 = st.columns(3)
            with c1: input_invest = st.number_input("投資金額 (円)", min_value=0, step=1000)
            with c2: input_recovery = st.number_input("回収金額 (円)", min_value=0, step=1000)
            with c3: input_hours = st.number_input("稼働時間 (h)", min_value=0.0, step=0.5, value=0.0, help="0.5時間=30分として記録します。")
            
            st.metric("収支", f"{(input_recovery - input_invest):+d} 円")
            
            input_memo = st.text_area("メモ", value=f"【{wd_str}曜】", placeholder="設定示唆、挙動など", height=80)
            
            submitted = st.form_submit_button("登録する", type="primary")
            if submitted:
                if not input_shop or not input_machine:
                    st.error("店舗名と機種名は必須です。")
                else:
                    if backend.save_my_balance(input_date, input_shop, input_machine, input_number, input_invest, input_recovery, input_hours, input_memo):
                        st.success("収支データを登録（または上書き更新）しました！")
                        st.cache_data.clear()
                        st.rerun()

    # --- 2. 収支データの表示 ---
    df_balance = backend.load_my_balance()
    
    if df_balance.empty:
        st.info("まだ収支データがありません。「収支データを登録」から記録をつけてみましょう。")
        return

    # 日付ソート
    df_balance = df_balance.sort_values('日付', ascending=False)

    # KPI 計算
    total_balance = df_balance['収支'].sum()
    total_invest = df_balance['投資'].sum()
    total_recovery = df_balance['回収'].sum()
    if '稼働時間' not in df_balance.columns:
        df_balance['稼働時間'] = 0.0
    df_balance['稼働時間'] = pd.to_numeric(df_balance['稼働時間'], errors='coerce').fillna(0)
    total_hours = df_balance['稼働時間'].sum()
    hourly_wage = total_balance / total_hours if total_hours > 0 else 0
    win_count = (df_balance['収支'] > 0).sum()
    total_count = len(df_balance)
    win_rate = win_count / total_count if total_count > 0 else 0
    
    st.subheader("📊 通算成績")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("総収支", f"{total_balance:+d} 円", delta_color="normal")
    k2.metric("回収率", f"{(total_recovery/total_invest*100):.1f} %" if total_invest > 0 else "-")
    k3.metric("勝率", f"{win_rate:.1%}")
    k4.metric("稼働数", f"{total_count} 回")
    k5.metric("時給", f"{int(hourly_wage):,} 円/h" if total_hours > 0 else "-")

    # --- 月別収支グラフ ---
    st.subheader("🗓️ 月別収支")
    df_balance['年月'] = df_balance['日付'].dt.strftime('%Y-%m')
    monthly_stats = df_balance.groupby('年月')['収支'].sum().reset_index()
    monthly_stats['累積収支'] = monthly_stats['収支'].cumsum()
    
    base_m = alt.Chart(monthly_stats).encode(x=alt.X('年月', title='年月'))
    bar_m = base_m.mark_bar(opacity=0.7).encode(
        y=alt.Y('収支', title='収支 (円)'),
        color=alt.condition(alt.datum.収支 > 0, alt.value("#ef5350"), alt.value("#42a5f5")),
        tooltip=['年月', alt.Tooltip('収支', format='+d')]
    )
    line_m = base_m.mark_line(color='#ffa726', point=True, strokeWidth=3).encode(
        y=alt.Y('累積収支', title='累積収支 (円)'),
        tooltip=['年月', alt.Tooltip('累積収支', format='+d')]
    )
    st.altair_chart(alt.layer(bar_m, line_m).resolve_scale(y='independent').interactive(), use_container_width=True)

    # --- 店舗別・機種別ランキング ---
    st.divider()
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.subheader("🏬 店舗別 成績")
        shop_rank = df_balance.groupby('店名').agg(
            総収支=('収支', 'sum'),
            勝率=('収支', lambda x: (x > 0).mean() * 100),
            稼働数=('収支', 'count'),
            稼働時間=('稼働時間', 'sum')
        ).sort_values('総収支', ascending=False).reset_index()
        shop_rank['時給'] = np.where(shop_rank['稼働時間'] > 0, shop_rank['総収支'] / shop_rank['稼働時間'], 0)
        
        st.dataframe(
            shop_rank[['店名', '総収支', '時給', '勝率', '稼働数']],
            column_config={
                "店名": st.column_config.TextColumn("店舗"),
                "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
                "時給": st.column_config.NumberColumn("時給", format="%+d 円/h"),
                "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=100),
                "稼働数": st.column_config.NumberColumn("回数"),
            },
            width="stretch",
            hide_index=True
        )

    with col_r2:
        st.subheader("🎰 機種別 成績")
        machine_rank = df_balance.groupby('機種名').agg(
            総収支=('収支', 'sum'),
            勝率=('収支', lambda x: (x > 0).mean() * 100),
            稼働数=('収支', 'count'),
            稼働時間=('稼働時間', 'sum')
        ).sort_values('総収支', ascending=False).reset_index()
        machine_rank['時給'] = np.where(machine_rank['稼働時間'] > 0, machine_rank['総収支'] / machine_rank['稼働時間'], 0)
        
        st.dataframe(
            machine_rank[['機種名', '総収支', '時給', '勝率', '稼働数']],
            column_config={
                "機種名": st.column_config.TextColumn("機種"),
                "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
                "時給": st.column_config.NumberColumn("時給", format="%+d 円/h"),
                "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=100),
                "稼働数": st.column_config.NumberColumn("回数"),
            },
            width="stretch",
            hide_index=True
        )

    # --- 曜日別 成績 ---
    st.divider()
    st.subheader("📅 曜日別 成績")
    
    # 曜日カラムの作成
    weekdays_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
    df_balance['曜日_num'] = df_balance['日付'].dt.dayofweek
    df_balance['曜日'] = df_balance['曜日_num'].map(weekdays_map)
    
    # 集計
    weekday_rank = df_balance.groupby(['曜日_num', '曜日']).agg(
        総収支=('収支', 'sum'),
        勝率=('収支', lambda x: (x > 0).mean() * 100),
        稼働数=('収支', 'count'),
        稼働時間=('稼働時間', 'sum')
    ).reset_index().sort_values('曜日_num')
    weekday_rank['時給'] = np.where(weekday_rank['稼働時間'] > 0, weekday_rank['総収支'] / weekday_rank['稼働時間'], 0)
    
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        base_w = alt.Chart(weekday_rank).encode(x=alt.X('曜日', sort=[weekdays_map[i] for i in range(7)], title='曜日'))
        bar_w = base_w.mark_bar(opacity=0.7).encode(
            y=alt.Y('総収支', title='総収支 (円)'),
            color=alt.condition(alt.datum.総収支 > 0, alt.value("#ef5350"), alt.value("#42a5f5")),
            tooltip=['曜日', alt.Tooltip('総収支', format='+d')]
        )
        line_w = base_w.mark_line(color='#ab47bc', point=True, strokeWidth=3).encode(
            y=alt.Y('勝率', title='勝率 (%)', scale=alt.Scale(domain=[0, 100])),
            tooltip=['曜日', alt.Tooltip('勝率', format='.1f', title='勝率(%)')]
        )
        st.altair_chart(alt.layer(bar_w, line_w).resolve_scale(y='independent').interactive(), width="stretch")
        
    with col_w2:
        st.dataframe(
            weekday_rank[['曜日', '総収支', '時給', '勝率', '稼働数']],
            column_config={
                "曜日": st.column_config.TextColumn("曜日"),
                "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
                "時給": st.column_config.NumberColumn("時給", format="%+d 円/h"),
                "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=100),
                "稼働数": st.column_config.NumberColumn("回数"),
            },
            width="stretch",
            hide_index=True
        )

    # グラフ (日別収支と累積収支)
    st.subheader("📈 資産推移")
    chart_data = df_balance.sort_values('日付').copy()
    chart_data['累積収支'] = chart_data['収支'].cumsum()
    
    # 複合グラフ: 棒(日別) + 線(累積)
    base = alt.Chart(chart_data).encode(x=alt.X('日付', title='日付'))
    
    bar = base.mark_bar(opacity=0.3).encode(
        y=alt.Y('収支', title='日別収支 (円)'),
        color=alt.condition(alt.datum.収支 > 0, alt.value("#ef5350"), alt.value("#42a5f5")),
        tooltip=['日付', '店名', '機種名', '台番号', '収支', 'メモ']
    )
    
    line = base.mark_line(point=True, color='#ffa726').encode(
        y=alt.Y('累積収支', title='累積収支 (円)'),
        tooltip=['日付', '累積収支']
    )
    
    st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent'), width="stretch")

    # テーブル表示
    st.subheader("📝 稼働履歴一覧")
    display_cols = ['日付', '店名', '台番号', '機種名', '投資', '回収', '収支', '稼働時間', 'メモ']
    available_cols = [c for c in display_cols if c in df_balance.columns]
    styled_balance = df_balance[available_cols].style.bar(subset=['収支'], align='mid', color=['rgba(66, 165, 245, 0.5)', 'rgba(255, 112, 67, 0.5)'], vmin=-30000, vmax=30000)
    st.dataframe(
        styled_balance,
        column_config={
            "日付": st.column_config.DateColumn("日付", format="YYYY-MM-DD"),
            "投資": st.column_config.NumberColumn("投資", format="%d 円"),
            "回収": st.column_config.NumberColumn("回収", format="%d 円"),
            "収支": st.column_config.NumberColumn("収支", format="%+d 円"),
            "稼働時間": st.column_config.NumberColumn("時間", format="%.1f h"),
        },
        width="stretch",
        hide_index=True
    )

    # --- 3. 収支データの編集・削除 ---
    if '登録日時' in df_balance.columns:
        st.divider()
        st.subheader("✏️ 収支データの編集・削除")
        
        # セレクトボックス用に表示名を作成
        def format_balance_label(uid):
            row = df_balance[df_balance['登録日時'] == uid].iloc[0]
            d_str = row['日付'].strftime('%Y-%m-%d') if pd.notna(row['日付']) else "不明"
            return f"{d_str} | {row['店名']} | {row['機種名']} ({row['収支']}円)"
        
        def on_edit_target_change():
            """編集対象が変更されたときにフォームの値を更新するコールバック"""
            target_uid = st.session_state.edit_balance_target
            if target_uid:
                try:
                    target_row = df_balance[df_balance['登録日時'] == target_uid].iloc[0]
                    st.session_state.eb_date = pd.to_datetime(target_row['日付']).date()
                    st.session_state.eb_shop = target_row['店名']
                    st.session_state.eb_num = str(target_row['台番号'])
                    st.session_state.eb_mac = target_row['機種名']
                    st.session_state.eb_inv = int(target_row['投資'])
                    st.session_state.eb_rec = int(target_row['回収'])
                    st.session_state.eb_hours = float(target_row.get('稼働時間', 0.0) if pd.notna(target_row.get('稼働時間')) and target_row.get('稼働時間') != '' else 0.0)
                    st.session_state.eb_memo = str(target_row.get('メモ', ''))
                except (IndexError, KeyError):
                    # データが見つからない場合は何もしない
                    pass

        target_uid = st.selectbox(
            "編集・削除するデータを選択", 
            df_balance['登録日時'].unique(), 
            format_func=format_balance_label, 
            key="edit_balance_target",
            on_change=on_edit_target_change
        )
        
        # 初回実行時にフォームの値を初期化
        if 'eb_date' not in st.session_state and not df_balance.empty:
            on_edit_target_change()
        
        if target_uid:
            with st.form("edit_balance_form"):
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    st.date_input("稼働日", key="eb_date")
                    st.text_input("店舗名", key="eb_shop")
                    st.text_input("台番号", key="eb_num")
                with e_col2:
                    st.text_input("機種名", key="eb_mac")
                    st.number_input("投資金額 (円)", min_value=0, step=1000, key="eb_inv")
                    st.number_input("回収金額 (円)", min_value=0, step=1000, key="eb_rec")
                st.number_input("稼働時間 (h)", min_value=0.0, step=0.5, key="eb_hours")
                st.text_area("メモ", height=80, key="eb_memo")
                if st.form_submit_button("更新を保存", type="primary"):
                    if backend.update_my_balance(target_uid, st.session_state.eb_date, st.session_state.eb_shop, st.session_state.eb_mac, st.session_state.eb_num, st.session_state.eb_inv, st.session_state.eb_rec, st.session_state.eb_hours, st.session_state.eb_memo):
                        st.success("収支データを更新しました！")
                        st.cache_data.clear(); st.rerun()
                    else: st.error("更新に失敗しました。")
            
            with st.form("delete_balance_form"):
                st.caption("※この操作は取り消せません")
                if st.form_submit_button("このデータを削除", type="primary"):
                    if backend.delete_my_balance(target_uid):
                        st.success("削除しました！"); st.cache_data.clear(); st.rerun()
                    else: st.error("削除に失敗しました。")