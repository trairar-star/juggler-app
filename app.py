import os
import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import matplotlib.pyplot as plt # type: ignore
import altair as alt # type: ignore

# バックエンド処理をインポート
import backend

# ---------------------------------------------------------
# ページ設定 (スマホ閲覧を意識して layout="centered" 推奨)
# ---------------------------------------------------------
st.set_page_config(
    page_title="スロット予測ビューアー",
    page_icon="🎰",
    layout="centered"
)

# --- ページ描画関数: 全店分析サマリー ---
def render_summary_page(df, df_raw, shop_col):
    st.header("📊 全店分析サマリー")
    
    if shop_col in df.columns:
        st.subheader("🏬 店舗別 期待度ランキング")
        
        # 店舗ごとの集計
        shop_stats = df.groupby(shop_col).agg(
            平均予測差枚=('予測差枚数', 'mean'),
            平均スコア=('prediction_score', 'mean'),
            推奨台数=('おすすめ度', lambda x: x.isin(['A', 'B']).sum()),
            全台数=('台番号', 'count')
        ).reset_index()
        
        # 平均スコアが高い順にソート
        shop_stats = shop_stats.sort_values('平均スコア', ascending=False)
        
        st.dataframe(
            shop_stats,
            column_config={
                shop_col: st.column_config.TextColumn("店舗名"),
                "平均予測差枚": st.column_config.NumberColumn("平均予想差枚", format="%d 枚"),
                "平均スコア": st.column_config.ProgressColumn("平均期待度", min_value=0, max_value=1.0, format="%.2f"),
                "推奨台数": st.column_config.NumberColumn("推奨台数 (A/B)", format="%d 台"),
                "全台数": st.column_config.NumberColumn("全台数", format="%d 台"),
            },
            use_container_width=True,
            hide_index=True
        )
        st.divider()
        
        # --- 店舗の質 vs 量 分析 (散布図) ---
        st.subheader("📊 店舗の質 vs 量 分析")
        st.caption("横軸が「推奨台数（量）」、縦軸が「平均予想差枚（質）」です。右上に位置するほど優良店と判断できます。")

        if not shop_stats.empty:
            # 散布図用に店舗名をインデックスに設定
            chart_data = shop_stats.set_index(shop_col)
            st.scatter_chart(
                chart_data,
                x='推奨台数',
                y='平均予測差枚',
                size='平均スコア',
                color="#FF4B4B" # Red color for bubbles
            )
        st.divider()

# --- ページ描画関数: 店舗別詳細データ ---
def render_shop_detail_page(df, df_raw, shop_col):
    st.header("🏪 店舗別 詳細データ")
    
    # 店舗選択 (サイドバー)
    selected_shop = '全て'
    if shop_col in df.columns:
        shops = ['全て'] + list(df[shop_col].unique())
        selected_shop = st.sidebar.selectbox("店舗名を選択", shops)
        
        if selected_shop != '全て':
            df = df[df[shop_col] == selected_shop]
            if not df_raw.empty:
                df_raw_shop = df_raw[df_raw[shop_col] == selected_shop]
            else:
                df_raw_shop = pd.DataFrame()
        else:
            df_raw_shop = df_raw.copy()
    
    # データがない場合のガード
    if df.empty:
        st.info("表示するデータがありません。")
        return

    # 機種名フィルター (サイドバー)
    if '機種名' in df.columns:
        machines = ['全て'] + list(df['機種名'].unique())
        selected_machine = st.sidebar.selectbox("機種名を選択", machines)
        if selected_machine != '全て':
            df = df[df['機種名'] == selected_machine]

    # --- メインコンテンツ: ランキング表示 (上部に配置) ---
    st.subheader("🏆 予測期待度ランキング (Top 10)")

    sort_cols = []
    if 'おすすめ度' in df.columns: sort_cols.append('おすすめ度')
    if 'prediction_score' in df.columns: sort_cols.append('prediction_score')
    if '予測差枚数' in df.columns: sort_cols.append('予測差枚数')
    
    ascending_list = [True] + [False] * (len(sort_cols) - 1)
    
    if sort_cols:
        df_sorted = df.sort_values(by=sort_cols, ascending=ascending_list).reset_index(drop=True)
    else:
        df_sorted = df

    # トップ10に絞る
    df_top10 = df_sorted.head(10)

    # スマホで見やすいようにカラムを厳選（「全て」の店が選ばれている時だけ「店名」を表示）
    base_cols = ['台番号', '機種名', 'おすすめ度', '予測差枚数']
    if selected_shop == '全て' and shop_col in df.columns:
        base_cols.insert(0, shop_col)
        
    display_cols = [c for c in base_cols if c in df_top10.columns]

    # データフレームの表示設定
    st.dataframe(
        df_top10[display_cols],
        column_config={
            shop_col: st.column_config.TextColumn("店舗", width="small"),
            "台番号": st.column_config.TextColumn("No.", width="small"),
            "機種名": st.column_config.TextColumn("機種", width="small"),
            "おすすめ度": st.column_config.TextColumn("評価", width="small"),
            "予測差枚数": st.column_config.NumberColumn("予想差枚", format="%d", width="small"),
        },
        use_container_width=True,
        hide_index=True
    )

    # --- 詳細分析: 上位台の根拠とスペック ---
    st.divider()
    st.subheader("🧐 上位台の詳細データ・根拠")
    st.caption("ランキング上位10台について、AIの判断根拠と詳細数値を表示します。")

    for i, row in df_top10.iterrows():
        shop_name = row.get(shop_col, '')
        machine_no = row.get('台番号', 'Unknown')
        machine_name = row.get('機種名', '')
        rating = row.get('おすすめ度', '-')
        diff_pred = row.get('予測差枚数', 0)
        
        label_prefix = f"【{shop_name}】 " if selected_shop == '全て' else ""
        label = f"{label_prefix}#{machine_no} {machine_name} (評価:{rating} / +{diff_pred}枚)"
        
        with st.expander(label, expanded=(i == 0)):
            if '根拠' in row and pd.notna(row['根拠']) and str(row['根拠']).strip() != "":
                st.markdown(f"**💡 AIの推奨根拠:**")
                st.info(row['根拠'])
            
            ref_date = row.get('対象日付', pd.NaT)
            date_str = ref_date.strftime('%Y-%m-%d') if pd.notna(ref_date) else "日付不明"
            st.markdown(f"**📊 参考データ ({date_str} の実績):**")
            
            def format_val(v):
                try: return f"{int(float(v))}"
                except: return str(v)
            
            def format_prob(v):
                try:
                    p = float(v)
                    if p > 0: return f"1/{int(1/p)}"
                except: pass
                return "-"

            # スマホ対応 2カラム x 3行構成
            c1, c2 = st.columns(2)
            with c1: st.metric("累計ゲーム", format_val(row.get('累計ゲーム', '-')))
            with c2: st.metric("週間平均差枚", f"{int(row.get('mean_7days_diff', 0)):+d}枚")
            
            c3, c4 = st.columns(2)
            with c3: st.metric("BIG回数", format_val(row.get('BIG', '-')))
            with c4: st.metric("REG回数", format_val(row.get('REG', '-')))
            
            c5, c6 = st.columns(2)
            with c5: st.metric("BIG確率", format_prob(row.get('BIG確率', 0)))
            with c6: st.metric("REG確率", format_prob(row.get('REG確率', 0)))
            
            # --- 過去の同曜日成績 ---
            target_wd = row.get('weekday', -1)
            try: target_wd = int(target_wd)
            except: target_wd = -1

            if target_wd != -1 and not df_raw.empty:
                weekdays_jp = ['月', '火', '水', '木', '金', '土', '日']
                wd_name = weekdays_jp[target_wd] if 0 <= target_wd <= 6 else '不明'
                
                st.markdown(f"**🔄 過去の同曜日 ({wd_name}曜) 平均成績:**")
                
                if shop_col in df_raw.columns and '台番号' in df_raw.columns and '対象日付' in df_raw.columns:
                    history_subset = df_raw[(df_raw[shop_col] == shop_name) & (df_raw['台番号'] == machine_no)].copy()
                    
                    if not history_subset.empty:
                        history_subset['wd'] = history_subset['対象日付'].dt.dayofweek
                        same_wd_df = history_subset[history_subset['wd'] == target_wd]
                        
                        if not same_wd_df.empty:
                            count = len(same_wd_df)
                            avg_diff = same_wd_df['差枚'].mean()
                            win_rate = (same_wd_df['差枚'] > 0).mean() * 100
                            avg_reg = same_wd_df['REG'].mean()
                            
                            # スマホ対応 2カラム x 2行
                            sw1, sw2 = st.columns(2)
                            with sw1: st.metric("集計回数", f"{count} 回")
                            with sw2: st.metric("勝率", f"{win_rate:.1f} %")
                            
                            sw3, sw4 = st.columns(2)
                            with sw3: st.metric("平均差枚", f"{int(avg_diff):+d} 枚")
                            with sw4: st.metric("平均REG", f"{avg_reg:.1f} 回")
                        else:
                            st.caption(f"※ 過去に{wd_name}曜日のデータが存在しません。")

            if 'prediction_score' in row:
                st.progress(float(row['prediction_score']), text=f"AI信頼度スコア: {float(row['prediction_score']):.2%}")
            
            # --- 設定期待度（円グラフ） ---
            if 'prediction_score' in row:
                st.markdown("**🎯 AI推定の設定期待度 (擬似分布):**")
                score = float(row['prediction_score'])
                
                p_low = max(0, (1.0 - score) / 3)
                p_high = max(0, score / 3)
                
                sizes = [p_low, p_low, p_low, p_high, p_high, p_high]
                labels = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5', 'Set 6']
                colors = ['#cfd8dc', '#b0bec5', '#90a4ae', '#fff59d', '#ffcc80', '#ffab91']
                explode = (0, 0, 0, 0, 0.05, 0.1)

                fig, ax = plt.subplots(figsize=(6, 3))
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                
                wedges, texts, autotexts = ax.pie(
                    sizes, 
                    labels=labels,
                    autopct='%1.0f%%',
                    startangle=90,
                    counterclock=False,
                    colors=colors,
                    explode=explode,
                    textprops={'fontsize': 8}
                )
                
                plt.setp(autotexts, weight="bold", color="black")
                
                st.pyplot(fig)
                plt.close(fig)

            # --- 過去の差枚推移グラフ ---
            st.markdown("**📉 過去7日間の差枚推移:**")
            
            if not df_raw.empty and shop_col in df_raw.columns:
                history_df = df_raw[
                    (df_raw[shop_col] == shop_name) & 
                    (df_raw['台番号'] == machine_no)
                ].sort_values('対象日付').tail(7)
                
                if not history_df.empty:
                    history_df['DisplayDate'] = history_df['対象日付'].dt.strftime('%m-%d')
                    
                    chart = alt.Chart(history_df).mark_line(point=True).encode(
                        x=alt.X('DisplayDate', title='日付', sort=None),
                        y=alt.Y('差枚', title='差枚数'),
                        tooltip=[
                            alt.Tooltip('DisplayDate', title='日付'),
                            alt.Tooltip('差枚', title='差枚'),
                            alt.Tooltip('BIG', title='BIG回数'),
                            alt.Tooltip('REG', title='REG回数'),
                            alt.Tooltip('累計ゲーム', title='総回転数')
                        ]
                    ).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.caption("過去データが見つかりませんでした。")

    # --- 店舗別 傾向分析 (下部に移動) ---
    if selected_shop != '全て' and not df_raw_shop.empty:
        st.divider()
        st.subheader(f"📅 {selected_shop} の傾向分析")
        st.caption("過去データに基づく、この店舗のイベント日や曜日ごとの平均差枚数です。")
        
        # スマホ対応: 縦に並べる
        if '日付要素' in df_raw_shop.columns and not df_raw_shop['日付要素'].isnull().all():
            st.markdown("**🔥 イベント別 平均差枚**")
            event_summary = df_raw_shop.groupby('日付要素')['差枚'].mean().sort_values(ascending=False)
            st.bar_chart(event_summary, color="#FF4B4B")
        
        st.markdown("**📅 曜日別 平均差枚**")
        viz_df = df_raw_shop.copy()
        if '曜日' not in viz_df.columns and '対象日付' in viz_df.columns:
            day_map = {'Monday': '月', 'Tuesday': '火', 'Wednesday': '水', 'Thursday': '木', 'Friday': '金', 'Saturday': '土', 'Sunday': '日'}
            viz_df['曜日'] = viz_df['対象日付'].dt.day_name().map(day_map)
        
        if '曜日' in viz_df.columns:
            weekday_shop_stats = viz_df.groupby('曜日')['差枚'].mean().sort_values(ascending=False)
            st.bar_chart(weekday_shop_stats, color="#4B4BFF")
        
        # --- 月間トレンド分析 (月初・月末) ---
        st.divider()
        st.subheader("🗓️ 月間トレンド (月初・月末の傾向)")
        st.caption("過去データにおける、日付（1日〜31日）ごとの平均差枚数です。")
        
        trend_df = df_raw_shop.copy()
        if '対象日付' in trend_df.columns:
            trend_df['day'] = trend_df['対象日付'].dt.day
            
            def classify_period(d):
                if d <= 7: return '月初 (1-7日)'
                elif d >= 25: return '月末 (25日-)'
                else: return '中旬 (8-24日)'
            
            trend_df['period'] = trend_df['day'].apply(classify_period)
            period_stats = trend_df.groupby('period')['差枚'].mean()
            
            # スマホ対応: 少し狭いがmetricは自動調整されるのでそのまま
            m1, m2, m3 = st.columns(3)
            val_start = period_stats.get('月初 (1-7日)', 0)
            val_mid = period_stats.get('中旬 (8-24日)', 0)
            val_end = period_stats.get('月末 (25日-)', 0)
            
            m1.metric("🌙 月初 (1-7)", f"{int(val_start):+d} 枚")
            m2.metric("☀️ 中旬 (8-24)", f"{int(val_mid):+d} 枚")
            m3.metric("🌑 月末 (25-)", f"{int(val_end):+d} 枚")
            
            st.markdown("👇 **期間を選択すると、その期間に強い機種が表示されます**")
            selected_period = st.radio("期間選択", ['月初 (1-7日)', '中旬 (8-24日)', '月末 (25日-)'], horizontal=True, label_visibility="collapsed")

            if selected_period:
                period_df = trend_df[trend_df['period'] == selected_period]
                if not period_df.empty:
                    st.markdown(f"🎰 **{selected_period} の機種別ランキング**")
                    machine_rank = period_df.groupby('機種名').agg(
                        平均差枚=('差枚', 'mean'),
                        勝率=('差枚', lambda x: (x > 0).mean()),
                        設置台数=('台番号', 'nunique')
                    ).sort_values('平均差枚', ascending=False).reset_index()
                    
                    st.dataframe(
                        machine_rank,
                        column_config={
                            "平均差枚": st.column_config.NumberColumn(format="%+d 枚"),
                            "勝率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                            "設置台数": st.column_config.NumberColumn(format="%d 台")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    st.markdown(f"🔢 **{selected_period} の末尾番号傾向 (0-9)**")
                    if '末尾番号' in period_df.columns:
                        digit_rank = period_df.groupby('末尾番号').agg(
                            平均差枚=('差枚', 'mean'),
                            勝率=('差枚', lambda x: (x > 0).mean()),
                            サンプル数=('差枚', 'count')
                        ).sort_index()
                        
                        st.bar_chart(digit_rank['平均差枚'], color="#29b6f6")
                        st.dataframe(
                            digit_rank.style.background_gradient(subset=['平均差枚'], cmap='RdYlGn', vmin=-300, vmax=300),
                            column_config={
                                "平均差枚": st.column_config.NumberColumn(format="%+d 枚"),
                                "勝率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                                "サンプル数": st.column_config.NumberColumn(format="%d 件")
                            },
                            use_container_width=True
                        )

            st.markdown("**📅 日付別 平均差枚推移**")
            day_stats = trend_df.groupby('day')['差枚'].mean()
            st.bar_chart(day_stats, color="#00E676")

    # --- 視覚化: 機種ごとの分析 ---
    st.divider()
    st.subheader("📊 機種別 平均予想差枚")
    
    if '機種名' in df.columns and '予測差枚数' in df.columns:
        # 機種ごとの平均差枚を算出
        machine_stats = df.groupby("機種名")["予測差枚数"].mean().sort_values(ascending=False)
        
        # 棒グラフで表示
        st.bar_chart(machine_stats, color="#FF4B4B")
        
        st.caption("※ 各機種の予測差枚数の平均値です（0枚以上の台のみ対象とするなど調整可）")

# --- ページ描画関数: 精度検証 (答え合わせ) ---
def render_verification_page(df_log, df_raw):
    st.header("✅ 精度検証 (答え合わせ)")
    st.caption("保存した「予測ログ」と実際の「翌日の結果」を照合し、AIの予測精度を可視化します。")

    if df_log.empty:
        st.warning("保存された予測ログがありません。「店舗別詳細データ」で予測を行い、「ログ保存」ボタンを押してください。")
        return

    if df_raw.empty:
        st.warning("稼働データ（正解データ）がありません。")
        return

    # --- データ結合処理 (予測ログ + 正解データ) ---
    # 1. ログデータの日付型変換
    if '対象日付' in df_log.columns:
        df_log['対象日付'] = pd.to_datetime(df_log['対象日付'])
    
    # 2. 正解データの作成
    # 予測の「対象日付」の翌日のデータが正解となる
    # マージするために、正解データ（df_raw）の日付を「1日戻す」ことで、予測ログの日付と一致させる
    ans_df = df_raw.copy()
    if '対象日付' in ans_df.columns:
        ans_df['対象日付'] = pd.to_datetime(ans_df['対象日付'])
        # 「明日のデータ」を「今日」にマッピングするイメージ
        ans_df['join_date'] = ans_df['対象日付'] - pd.Timedelta(days=1)
    
    # 店舗カラム名の統一
    shop_col_log = '店名' if '店名' in df_log.columns else '店舗名'
    shop_col_raw = '店名' if '店名' in ans_df.columns else '店舗名'
    
    # マージ実行 (Left Joinだと結果待ちのものも残るが、答え合わせなのでInner Joinで「結果が出たもの」だけに絞る)
    merged_df = pd.merge(
        df_log,
        ans_df[[shop_col_raw, '台番号', 'join_date', '差枚']], # 必要なカラムのみ
        left_on=[shop_col_log, '台番号', '対象日付'],
        right_on=[shop_col_raw, '台番号', 'join_date'],
        how='inner',
        suffixes=('', '_actual')
    )
    
    if merged_df.empty:
        st.info("まだ結果が判明している予測ログがありません。（予測日の翌日のデータが取り込まれるまでお待ちください）")
        # ログだけ表示しておく
        st.markdown("📝 **保存済み予測ログ一覧 (未検証)**")
        st.dataframe(df_log.sort_values('実行日時', ascending=False), hide_index=True)
        return

    # --- 1. 全体成績 (KPI) & 円グラフ ---
    st.subheader("📊 保存した予測の通算成績")
    
    total_count = len(merged_df)
    win_count = (merged_df['差枚_actual'] > 0).sum()
    lose_count = total_count - win_count
    win_rate = win_count / total_count if total_count > 0 else 0
    avg_diff = merged_df['差枚_actual'].mean()
    total_diff = merged_df['差枚_actual'].sum()
    
    col_kpi, col_pie = st.columns([2, 1])
    
    with col_kpi:
        k1, k2 = st.columns(2)
        k1.metric("検証台数", f"{total_count} 台")
        k2.metric("勝率", f"{win_rate:.1%}")
        
        k3, k4 = st.columns(2)
        k3.metric("平均差枚", f"{int(avg_diff):+d} 枚")
        k4.metric("合計収支", f"{int(total_diff):+d} 枚")
        
    with col_pie:
        # 勝敗円グラフ (Altair)
        pie_data = pd.DataFrame({
            'Category': ['Win', 'Lose'],
            'Count': [win_count, lose_count]
        })
        
        pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=35).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Category", type="nominal", 
                            scale=alt.Scale(domain=['Win', 'Lose'], range=['#FF4B4B', '#4B4BFF']),
                            legend=alt.Legend(title="勝敗", orient="bottom")),
            tooltip=['Category', 'Count']
        ).properties(height=200)
        
        st.altair_chart(pie_chart, use_container_width=True)

    # --- 2. 時系列推移 (勝率 & 収支) ---
    st.subheader("📈 日別の勝率・収支推移")
    
    daily_stats = merged_df.groupby('対象日付').agg(
        win_rate=('差枚_actual', lambda x: (x > 0).mean()),
        total_profit=('差枚_actual', 'sum'),
        count=('台番号', 'count')
    ).reset_index()
    
    if not daily_stats.empty:
        daily_stats['date_str'] = daily_stats['対象日付'].dt.strftime('%m/%d')
        
        base_chart = alt.Chart(daily_stats).encode(x=alt.X('date_str', title='日付', sort=None))
        
        # 棒グラフ: 収支 (左軸)
        bar_chart = base_chart.mark_bar(opacity=0.6).encode(
            y=alt.Y('total_profit', title='日別収支 (枚)'),
            color=alt.condition(alt.datum.total_profit > 0, alt.value("#FF4B4B"), alt.value("#4B4BFF")),
            tooltip=['date_str', alt.Tooltip('total_profit', format='+d'), 'count']
        )
        
        # 折れ線グラフ: 勝率 (右軸)
        line_chart = base_chart.mark_line(point=True, color='#FFA726', strokeWidth=3).encode(
            y=alt.Y('win_rate', title='勝率 (%)', axis=alt.Axis(format='%')),
            tooltip=['date_str', alt.Tooltip('win_rate', format='.1%')]
        )
        
        # 2軸グラフの合成
        st.altair_chart(alt.layer(bar_chart, line_chart).resolve_scale(y='independent'), use_container_width=True)

    # --- 2. ランク別分析 ---
    st.subheader("📈 ランク別 精度分析")
    if 'おすすめ度' in merged_df.columns:
        rank_stats = merged_df.groupby('おすすめ度').agg(
            台数=('台番号', 'count'),
            勝率=('差枚_actual', lambda x: (x > 0).mean()),
            平均差枚=('差枚_actual', 'mean'),
            合計差枚=('差枚_actual', 'sum')
        ).reset_index()
        
        # ソート順序固定
        rank_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
        rank_stats['sort'] = rank_stats['おすすめ度'].map(rank_order).fillna(99)
        rank_stats = rank_stats.sort_values('sort').drop('sort', axis=1)
        
        st.dataframe(
            rank_stats,
            column_config={
                "おすすめ度": st.column_config.TextColumn("AI評価"),
                "台数": st.column_config.NumberColumn("検証数", format="%d 台"),
                "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=1),
                "平均差枚": st.column_config.NumberColumn("平均結果", format="%+d 枚"),
                "合計差枚": st.column_config.NumberColumn("合計収支", format="%+d 枚"),
            },
            use_container_width=True,
            hide_index=True
        )

    # --- 3. 詳細ログリスト ---
    st.divider()
    st.subheader("📝 詳細ログ (予実比較)")
    
    # 表示用に整理
    display_df = merged_df.copy()
    display_df['結果判定'] = display_df['差枚_actual'].apply(lambda x: 'Win 🔴' if x > 0 else 'Lose 🔵')
    display_df = display_df.sort_values('対象日付', ascending=False)
    
    # カラム選択
    cols = ['対象日付', shop_col_log, '台番号', '機種名', 'おすすめ度', '予測差枚数', '差枚_actual', '結果判定']
    
    st.dataframe(
        display_df[cols],
        column_config={
            "対象日付": st.column_config.DateColumn("予測日", format="MM/DD"),
            "予測差枚数": st.column_config.NumberColumn("予想", format="%d"),
            "差枚_actual": st.column_config.NumberColumn("結果", format="%+d 枚"),
        },
        use_container_width=True,
        hide_index=True
    )

# --- ページ描画関数: AI学習データ分析 (勝利の法則) ---
def render_feature_analysis_page(df_train):
    st.header("🔬 AI学習データ分析 (勝利の法則)")
    st.caption("過去の全データから、「勝った台（翌日プラス差枚）」と「負けた台」の傾向を分析し、勝ちやすい台の特徴を可視化します。")

    if df_train.empty:
        st.warning("分析可能な過去データがありません。")
        return
    
    # データ準備
    analysis_df = df_train.copy()
    # REG確率分母を計算 (0除算回避)
    analysis_df['REG分母'] = analysis_df['REG確率'].apply(lambda x: int(1/x) if x > 0 else 9999)
    
    # --- 1. REG確率別の勝率 (最重要) ---
    st.subheader("📊 REG確率と勝率の関係")
    st.caption("「REG確率が良い台は本当に翌日勝てるのか？」を検証します。回転数が少ないと確率がブレてノイズになるため、最低回転数で絞り込めます。")
    
    # ノイズ除去用のゲーム数フィルター
    min_g = st.slider("集計対象の最低回転数", min_value=0, max_value=8000, value=3000, step=500, help="指定した回転数以上回っている台のみを集計します。")
    
    reg_df = analysis_df[analysis_df['累計ゲーム'] >= min_g].copy()
    
    if reg_df.empty:
        st.warning("条件に一致するデータがありません。最低回転数を下げてください。")
    else:
        # REG分母をビン分割
        bins = [0, 200, 240, 280, 320, 360, 400, 500, 10000]
        labels = ['~1/200 (極良)', '1/200~240 (高)', '1/240~280 (良)', '1/280~320 (中)', '1/320~360 (低)', '1/360~400 (悪)', '1/400~500 (極悪)', '1/500~ (論外)']
        
        reg_df['REG区間'] = pd.cut(reg_df['REG分母'], bins=bins, labels=labels)
        
        reg_stats = reg_df.groupby('REG区間', observed=True).agg(
            勝率=('target', 'mean'),
            平均翌日差枚=('next_diff', 'mean'),
            サンプル数=('target', 'count')
        ).reset_index()
        
        # 複合グラフ: 棒グラフ(勝率) + 折れ線(差枚)
        base = alt.Chart(reg_stats).encode(x=alt.X('REG区間', title='前日のREG確率区分'))
        
        bar = base.mark_bar(color='#66BB6A', opacity=0.7).encode(
            y=alt.Y('勝率', axis=alt.Axis(format='%', title='勝率')),
            tooltip=['REG区間', alt.Tooltip('勝率', format='.1%'), 'サンプル数']
        )
        
        line = base.mark_line(color='#FF7043', point=True).encode(
            y=alt.Y('平均翌日差枚', axis=alt.Axis(title='平均翌日差枚 (枚)')),
            tooltip=['REG区間', alt.Tooltip('平均翌日差枚', format='+.0f')]
        )
        
        st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent'), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # --- 2. 回転数(稼働量)と勝率 ---
    with col1:
        st.markdown("**🎰 回転数と期待値の関係**")
        # 回転数をビン分割
        g_bins = [0, 2000, 4000, 6000, 8000, 15000]
        g_labels = ['~2000G', '2000~4000G', '4000~6000G', '6000~8000G', '8000G~']
        
        analysis_df['G数区間'] = pd.cut(analysis_df['累計ゲーム'], bins=g_bins, labels=g_labels)
        
        g_stats = analysis_df.groupby('G数区間', observed=True)['next_diff'].mean().reset_index()

        chart_g = alt.Chart(g_stats).mark_bar().encode(
            x=alt.X('G数区間', title='前日の回転数'),
            y=alt.Y('next_diff', title='翌日の平均差枚'),
            color=alt.condition(alt.datum.next_diff > 0, alt.value("#FF7043"), alt.value("#42A5F5"))
        )
        st.altair_chart(chart_g, use_container_width=True)

    # --- 3. 前日差枚の影響 (上げ狙い vs 据え置き) ---
    with col2:
        st.markdown("**📉 前日差枚と翌日の勝率**")
        # 差枚をビン分割
        d_bins = [-10000, -2000, -500, 500, 2000, 10000]
        d_labels = ['大負け', '負け', 'トントン', '勝ち', '大勝ち']
        analysis_df['差枚区間'] = pd.cut(analysis_df['差枚'], bins=d_bins, labels=d_labels)
        
        d_stats = analysis_df.groupby('差枚区間', observed=True)['target'].mean().reset_index()
        
        chart_d = alt.Chart(d_stats).mark_bar(color='#FFA726').encode(
            x=alt.X('差枚区間', title='前日の結果', sort=None),
            y=alt.Y('target', title='翌日勝率', axis=alt.Axis(format='%'))
        )
        st.altair_chart(chart_d, use_container_width=True)

    # --- 4. イベントランク別の設定投入傾向 ---
    st.divider()
    st.subheader("🎉 イベントランク別の設定投入傾向")
    st.caption(f"指定した回転数（{min_g}G）以上回っている台のうち、「REG確率が1/300より良い台（高設定挙動）」の割合をイベントの強さごとに比較します。")
    
    if 'イベントランク' in reg_df.columns:
        event_df = reg_df.copy()
        # NaNや空文字を「通常日」として扱う
        event_df['イベントランク'] = event_df['イベントランク'].fillna('通常日').replace('', '通常日')
        
        # REG分母が300以下の台を高設定挙動とみなす
        event_df['高設定挙動'] = (event_df['REG分母'] <= 300).astype(int)
        
        event_stats = event_df.groupby('イベントランク').agg(
            高設定投入率=('高設定挙動', 'mean'),
            平均差枚=('差枚', 'mean'),
            サンプル数=('台番号', 'count')
        ).reset_index()
        
        # 並び替え順の指定 (S -> A -> B -> C -> 通常日)
        rank_order = {'S': 1, 'A': 2, 'B': 3, 'C': 4, '通常日': 5}
        event_stats['sort'] = event_stats['イベントランク'].map(rank_order).fillna(99)
        event_stats = event_stats.sort_values('sort').drop(columns=['sort'])
        
        if len(event_stats['イベントランク'].unique()) > 1 or '通常日' not in event_stats['イベントランク'].values:
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                chart_e = alt.Chart(event_stats).mark_bar(color='#AB47BC', opacity=0.8).encode(
                    x=alt.X('イベントランク', sort=[k for k in rank_order.keys()], title='イベントの強さ'),
                    y=alt.Y('高設定投入率', axis=alt.Axis(format='%', title='REG 1/300以上の割合')),
                    tooltip=['イベントランク', alt.Tooltip('高設定投入率', format='.1%'), 'サンプル数']
                ).interactive()
                st.altair_chart(chart_e, use_container_width=True)
            
            with col_e2:
                st.dataframe(
                    event_stats,
                    column_config={
                        "高設定投入率": st.column_config.ProgressColumn("高設定割合", format="%.1f%%", min_value=0, max_value=1),
                        "平均差枚": st.column_config.NumberColumn("台平均差枚", format="%+d 枚"),
                        "サンプル数": st.column_config.NumberColumn("集計台数", format="%d 台")
                    },
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.info("イベントランクが登録されたデータがまだありません。サイドバーからイベントを登録すると傾向が表示されます。")
    else:
        st.info("イベントデータが結合されていません。")

# --- ページ描画関数: イベント管理 ---
def render_event_management_page():
    st.header("📝 イベント管理")
    st.caption("登録済みの店舗イベント一覧です。不要なイベントはここから削除できます。")
    
    df_events = backend.load_shop_events()
    
    if df_events.empty:
        st.info("現在、登録されているイベントはありません。")
        return

    # 削除用の一意キーと表示用日付を作成
    df_events['date_str'] = df_events['イベント日付'].dt.strftime('%Y-%m-%d')
    df_events['uid'] = df_events['店名'] + " | " + df_events['date_str'] + " | " + df_events['イベント名']
    
    # 表示用データ (日付降順)
    df_display = df_events.sort_values(['イベント日付', '店名'], ascending=[False, True])
    
    st.dataframe(
        df_display[['イベント日付', '店名', 'イベント名', 'イベントランク']],
        column_config={
            "イベント日付": st.column_config.DateColumn("日付", format="YYYY-MM-DD"),
            "店名": st.column_config.TextColumn("店舗"),
            "イベント名": st.column_config.TextColumn("イベント"),
            "イベントランク": st.column_config.TextColumn("ランク"),
        },
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    st.subheader("🗑 イベント削除")
    
    # 削除フォーム
    with st.form("delete_event_form"):
        target_uid = st.selectbox("削除するイベントを選択", df_display['uid'].unique())
        if st.form_submit_button("削除実行", type="primary"):
            # 選択されたIDから元データを復元
            target_row = df_display[df_display['uid'] == target_uid].iloc[0]
            
            if backend.delete_shop_event(target_row['店名'], target_row['イベント日付'], target_row['イベント名']):
                st.success(f"削除しました: {target_uid}")
                st.cache_data.clear() # キャッシュクリア
                st.rerun()
            else:
                st.error("削除に失敗しました。")

# --- ページ描画関数: マイ収支管理 ---
def render_my_balance_page(df_raw):
    st.header("💰 マイ収支管理")
    st.caption("あなたの稼働実績（投資・回収・収支）を記録・分析します。")

    # --- 1. 収支入力フォーム ---
    with st.expander("📝 収支データを登録", expanded=False):
        # 日付選択をフォーム外に配置し、変更時に即座に曜日を反映させる
        input_date = st.date_input("稼働日", pd.Timestamp.now())
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
                        st.success("収支を登録しました！")
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

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
def main():
    # --- パスワード認証（ログイン機能） ---
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        st.title("🔒 ログイン")
        password = st.text_input("パスワードを入力してください", type="password")
        if st.button("ログイン"):
            # Secretsからパスワードを取得（設定されていない場合は '1234' とする）
            correct_password = st.secrets.get("app_password", "1234")
            if password == correct_password:
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("パスワードが違います。")
        return

    # タイトルや設定は共通
    st.title("🎰 明日のスロット予測")
    
    # --- ページ切り替えメニュー (サイドバーの一番上) ---
    page = st.sidebar.radio(
        "メニュー", 
        ["店舗別詳細データ", "全店分析サマリー", "AI傾向分析 (勝利の法則)", "精度検証 (答え合わせ)", "イベント管理", "💰 マイ収支管理"]
    )
    st.sidebar.divider()

    # データ更新ボタン (サイドバー)
    if st.sidebar.button("🔄 データ更新 (再読み込み)"):
        st.cache_data.clear()
        st.rerun()
        
    # 予測保存ボタン (サイドバー)
    if st.sidebar.button("💾 予測結果をログ保存"):
        # このボタンを押すと、後述の処理で計算された df を保存します
        st.session_state['save_requested'] = True

    # データのロード
    with st.spinner("スプレッドシートからデータを読み込み中..."):
        df_raw = backend.load_data()
    
    if df_raw.empty:
        st.warning("データが取得できませんでした。")
        return

    # イベントデータのロード
    df_events = backend.load_shop_events()

    # --- 学習データの統計情報を表示 (追加) ---
    if '対象日付' in df_raw.columns and not df_raw.empty:
        min_date = df_raw['対象日付'].min()
        max_date = df_raw['対象日付'].max()
        total_records = len(df_raw)
        
        min_str = min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else "不明"
        max_str = max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else "不明"
        
        st.sidebar.info(f"📚 **学習データ統計**\n\n期間: {min_str} 〜 {max_str}\n総数: {total_records:,} 件")

    # --- 店舗イベント登録 (サイドバー) ---
    with st.sidebar.expander("📅 店舗イベント登録", expanded=False):
        st.caption("店舗独自のイベント（取材、特定日など）を登録すると、AIがその傾向を学習します。")
        
        # 店舗リスト取得
        shop_col = '店名' if '店名' in df_raw.columns else '店舗名'
        if shop_col in df_raw.columns:
            unique_shops = df_raw[shop_col].unique()
            
            with st.form("event_reg_form", clear_on_submit=True):
                reg_shop = st.selectbox("店舗", unique_shops)
                reg_date = st.date_input("日付", pd.Timestamp.now())
                reg_name = st.text_input("イベント名 (例: ○○取材, リニューアル)")
                reg_rank = st.selectbox("イベントの強さ (期待度)", ["S", "A", "B", "C"], index=1, help="S:激アツ, A:強い, B:普通, C:弱め")
                submitted = st.form_submit_button("イベントを登録")
                
                if submitted:
                    if backend.save_shop_event(reg_shop, reg_date, reg_name, reg_rank):
                        st.success(f"{reg_shop} のイベントを登録しました！")
                        st.cache_data.clear() # データ更新のためキャッシュクリア
                        st.rerun() # 画面リロード

    # --- ハイパーパラメータ調整 (サイドバー) ---
    with st.sidebar.expander("⚙️ AIモデル設定 (調整)", expanded=False):
        hp_n_estimators = st.slider("学習回数 (n_estimators)", 50, 1000, 200, step=50, help="値を大きくすると学習量が増えますが、時間がかかり過学習のリスクもあります。")
        hp_learning_rate = st.slider("学習率 (learning_rate)", 0.01, 0.3, 0.05, step=0.01, help="値を小さくすると丁寧に学習しますが、回数を増やす必要があります。")
        hp_num_leaves = st.slider("葉の数 (num_leaves)", 10, 127, 20, step=1, help="モデルの複雑さ。スロットのようなノイズが多いデータは小さめ(15〜20)がおすすめです。")
        hp_max_depth = st.slider("深さ制限 (max_depth)", -1, 15, 5, step=1, help="木の深さの上限。ノイズ対策として3〜7程度に制限するのがおすすめです。-1は無制限。")
        
        hyperparams = {
            'n_estimators': hp_n_estimators,
            'learning_rate': hp_learning_rate,
            'num_leaves': hp_num_leaves,
            'max_depth': hp_max_depth
        }

    # 分析実行
    with st.spinner("AIがデータを分析し、予測を生成しています..."):
        # キャッシュキーとしてデータの長さを利用（簡易的）
        df, df_verify = backend.run_analysis(df_raw, df_events, hyperparams)
    
    if df.empty:
        st.warning("分析対象のデータがありません。")
        return

    # 日付表示
    if '対象日付' in df.columns:
        target_date = df['対象日付'].max()
        st.caption(f"データ基準日: {target_date.strftime('%Y-%m-%d')} (この日のデータを基に明日を予測)")
    
    # 必要なカラムの補完
    for col in ['おすすめ度', 'prediction_score', '予測差枚数', '根拠']:
        if col not in df.columns:
            df[col] = 0 if col == '予測差枚数' else '-'

    # カラム名判定
    shop_col = '店名' if '店名' in df.columns else '店舗名'

    # --- 保存処理の実行 (キャッシュ外へ移動) ---
    if st.session_state.get('save_requested'):
        backend.save_prediction_log(df)
        st.session_state['save_requested'] = False

    if page == "全店分析サマリー":
        render_summary_page(df, df_raw, shop_col)
    elif page == "精度検証 (答え合わせ)":
        # 予測ログをロードして渡す
        df_log = backend.load_prediction_log()
        render_verification_page(df_log, df_raw)
    elif page == "AI傾向分析 (勝利の法則)":
        render_feature_analysis_page(df_verify)
    elif page == "イベント管理":
        render_event_management_page()
    elif page == "💰 マイ収支管理":
        render_my_balance_page(df_raw)
    else:
        render_shop_detail_page(df, df_raw, shop_col)

if __name__ == "__main__":
    main()