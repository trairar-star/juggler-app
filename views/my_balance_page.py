import pandas as pd
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
                    machines = list(df_raw['機種名'].unique())
                input_machine = st.selectbox("機種名", machines + ["その他 (手入力)"])
                if input_machine == "その他 (手入力)":
                    input_machine = st.text_input("機種名を入力")
            
            c1, c2, c3 = st.columns(3)
            with c1: input_invest = st.number_input("投資金額 (円)", min_value=0, step=1000)
            with c2: input_recovery = st.number_input("回収金額 (円)", min_value=0, step=1000)
            with c3: st.metric("収支", f"{(input_recovery - input_invest):+d} 円")
            
            input_memo = st.text_area("メモ", value=f"【{wd_str}曜】", placeholder="設定示唆、挙動など", height=80)
            
            submitted = st.form_submit_button("登録する", type="primary")
            if submitted:
                if not input_shop or not input_machine:
                    st.error("店舗名と機種名は必須です。")
                else:
                    if backend.save_my_balance(input_date, input_shop, input_machine, input_number, input_invest, input_recovery, input_memo):
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
    win_count = (df_balance['収支'] > 0).sum()
    total_count = len(df_balance)
    win_rate = win_count / total_count if total_count > 0 else 0
    
    st.subheader("📊 通算成績")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("総収支", f"{total_balance:+d} 円", delta_color="normal")
    k2.metric("回収率", f"{(total_recovery/total_invest*100):.1f} %" if total_invest > 0 else "-")
    k3.metric("勝率", f"{win_rate:.1%}")
    k4.metric("稼働数", f"{total_count} 回")

    # --- 月別収支グラフ ---
    st.subheader("🗓️ 月別収支")
    df_balance['年月'] = df_balance['日付'].dt.strftime('%Y-%m')
    monthly_stats = df_balance.groupby('年月')['収支'].sum().reset_index()
    
    monthly_chart = alt.Chart(monthly_stats).mark_bar().encode(
        x=alt.X('年月', title='年月'),
        y=alt.Y('収支', title='収支 (円)'),
        color=alt.condition(alt.datum.収支 > 0, alt.value("#ef5350"), alt.value("#42a5f5")),
        tooltip=['年月', alt.Tooltip('収支', format='+d')]
    ).interactive()
    st.altair_chart(monthly_chart, use_container_width=True)

    # --- 店舗別・機種別ランキング ---
    st.divider()
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.subheader("🏬 店舗別 成績")
        shop_rank = df_balance.groupby('店名').agg(
            総収支=('収支', 'sum'),
            勝率=('収支', lambda x: (x > 0).mean()),
            稼働数=('収支', 'count')
        ).sort_values('総収支', ascending=False).reset_index()
        
        st.dataframe(
            shop_rank,
            column_config={
                "店名": st.column_config.TextColumn("店舗"),
                "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
                "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=1),
                "稼働数": st.column_config.NumberColumn("回数"),
            },
            use_container_width=True,
            hide_index=True
        )

    with col_r2:
        st.subheader("🎰 機種別 成績")
        machine_rank = df_balance.groupby('機種名').agg(
            総収支=('収支', 'sum'),
            勝率=('収支', lambda x: (x > 0).mean()),
            稼働数=('収支', 'count')
        ).sort_values('総収支', ascending=False).reset_index()
        
        st.dataframe(
            machine_rank,
            column_config={
                "機種名": st.column_config.TextColumn("機種"),
                "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
                "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=1),
                "稼働数": st.column_config.NumberColumn("回数"),
            },
            use_container_width=True,
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
        勝率=('収支', lambda x: (x > 0).mean()),
        稼働数=('収支', 'count')
    ).reset_index().sort_values('曜日_num')
    
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        weekday_chart = alt.Chart(weekday_rank).mark_bar().encode(
            x=alt.X('曜日', sort=[weekdays_map[i] for i in range(7)], title='曜日'),
            y=alt.Y('総収支', title='総収支 (円)'),
            color=alt.condition(alt.datum.総収支 > 0, alt.value("#ef5350"), alt.value("#42a5f5")),
            tooltip=['曜日', alt.Tooltip('総収支', format='+d'), alt.Tooltip('勝率', format='.1%')]
        ).interactive()
        st.altair_chart(weekday_chart, use_container_width=True)
        
    with col_w2:
        st.dataframe(
            weekday_rank[['曜日', '総収支', '勝率', '稼働数']],
            column_config={
                "曜日": st.column_config.TextColumn("曜日"),
                "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
                "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=1),
                "稼働数": st.column_config.NumberColumn("回数"),
            },
            use_container_width=True,
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
    
    st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent'), use_container_width=True)

    # テーブル表示
    st.subheader("📝 稼働履歴一覧")
    st.dataframe(
        df_balance[['日付', '店名', '台番号', '機種名', '投資', '回収', '収支', 'メモ']],
        column_config={
            "日付": st.column_config.DateColumn("日付", format="YYYY-MM-DD"),
            "投資": st.column_config.NumberColumn("投資", format="%d 円"),
            "回収": st.column_config.NumberColumn("回収", format="%d 円"),
            "収支": st.column_config.NumberColumn("収支", format="%+d 円"),
        },
        use_container_width=True,
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

        target_uid = st.selectbox("編集・削除するデータを選択", df_balance['登録日時'].unique(), format_func=format_balance_label, key="edit_balance_target")
        
        if target_uid:
            target_row = df_balance[df_balance['登録日時'] == target_uid].iloc[0]
            
            with st.form("edit_balance_form"):
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    try: default_date = pd.to_datetime(target_row['日付']).date()
                    except: default_date = pd.Timestamp.now(tz='Asia/Tokyo').date()
                    edit_date = st.date_input("稼働日", value=default_date, key="eb_date")
                    edit_shop = st.text_input("店舗名", value=target_row['店名'], key="eb_shop")
                    edit_number = st.text_input("台番号", value=str(target_row['台番号']), key="eb_num")
                with e_col2:
                    edit_machine = st.text_input("機種名", value=target_row['機種名'], key="eb_mac")
                    edit_invest = st.number_input("投資金額 (円)", value=int(target_row['投資']), min_value=0, step=1000, key="eb_inv")
                    edit_recovery = st.number_input("回収金額 (円)", value=int(target_row['回収']), min_value=0, step=1000, key="eb_rec")
                edit_memo = st.text_area("メモ", value=str(target_row.get('メモ', '')), height=80, key="eb_memo")
                if st.form_submit_button("更新を保存", type="primary"):
                    if backend.update_my_balance(target_uid, edit_date, edit_shop, edit_machine, edit_number, edit_invest, edit_recovery, edit_memo):
                        st.success("収支データを更新しました！")
                        st.cache_data.clear(); st.rerun()
                    else: st.error("更新に失敗しました。")
            
            with st.form("delete_balance_form"):
                st.caption("※この操作は取り消せません")
                if st.form_submit_button("このデータを削除", type="primary"):
                    if backend.delete_my_balance(target_uid):
                        st.success("削除しました！"); st.cache_data.clear(); st.rerun()
                    else: st.error("削除に失敗しました。")