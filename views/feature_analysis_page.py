import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import altair as alt # type: ignore
from utils import get_confidence_indicator
import backend

def _render_monthly_trend_analysis(viz_df, analysis_df=None):
    with st.expander("🗓️ 月間トレンド (月初・月末の傾向)", expanded=False):
        st.caption("過去データにおける、日付（1日〜31日）ごとの平均差枚数や高設定率です。")
        
        chart_metric_shop = st.radio("📊 グラフの表示指標", ["平均差枚", "高設定率", "REG確率"], horizontal=True, key="monthly_trend_metric")
        
        trend_df = viz_df.copy()
        trend_df['REG確率_val'] = np.where(trend_df['累計ゲーム'] > 0, trend_df['REG'] / trend_df['累計ゲーム'], 0)
        
        y_col = "差枚" if chart_metric_shop == "平均差枚" else "高設定_rate" if chart_metric_shop == "高設定率" else "REG確率_val"

        if '対象日付' in trend_df.columns:
            trend_df['day'] = trend_df['対象日付'].dt.day
            
            def classify_period(d):
                if d <= 7: return '月初 (1-7日)'
                elif d >= 25: return '月末 (25日-)'
                else: return '中旬 (8-24日)'
            
            trend_df['period'] = trend_df['day'].apply(classify_period)
            period_stats = trend_df.groupby('period')[y_col].mean()
            
            m1, m2, m3 = st.columns(3)
            val_start = period_stats.get('月初 (1-7日)', 0)
            val_mid = period_stats.get('中旬 (8-24日)', 0)
            val_end = period_stats.get('月末 (25日-)', 0)
            
            if chart_metric_shop == "平均差枚":
                m1.metric("🌙 月初 (1-7)", f"{int(val_start):+d} 枚")
                m2.metric("☀️ 中旬 (8-24)", f"{int(val_mid):+d} 枚")
                m3.metric("🌑 月末 (25-)", f"{int(val_end):+d} 枚")
            elif chart_metric_shop == "高設定率":
                m1.metric("🌙 月初 (1-7)", f"{val_start:.1f}%")
                m2.metric("☀️ 中旬 (8-24)", f"{val_mid:.1f}%")
                m3.metric("🌑 月末 (25-)", f"{val_end:.1f}%")
            else:
                m1.metric("🌙 月初 (1-7)", f"1/{int(1/val_start)}" if val_start > 0 else "-")
                m2.metric("☀️ 中旬 (8-24)", f"1/{int(1/val_mid)}" if val_mid > 0 else "-")
                m3.metric("🌑 月末 (25-)", f"1/{int(1/val_end)}" if val_end > 0 else "-")
            
            st.markdown("👇 **期間を選択すると、その期間に強い機種が表示されます**")
            selected_period = st.radio("期間選択", ['月初 (1-7日)', '中旬 (8-24日)', '月末 (25日-)'], horizontal=True, label_visibility="collapsed")
    
            if selected_period:
                period_df = trend_df[trend_df['period'] == selected_period]
                if not period_df.empty:
                    st.markdown(f"🎰 **{selected_period} の機種別ランキング**")
                    machine_rank = period_df.groupby('機種名').agg(平均差枚=('差枚', 'mean'), 高設定率=('高設定_rate', 'mean'), REG確率_val=('REG確率_val', 'mean'), 設置台数=('台番号', 'nunique')).sort_values('高設定率', ascending=False).reset_index()
                    machine_rank['REG確率'] = machine_rank['REG確率_val'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
                    machine_rank['信頼度'] = machine_rank['設置台数'].apply(get_confidence_indicator)
                    st.dataframe(machine_rank[['機種名', '平均差枚', '高設定率', 'REG確率', '設置台数', '信頼度']], column_config={"平均差枚": st.column_config.NumberColumn(format="%+d 枚"), "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100), "REG確率": st.column_config.TextColumn("REG確率"), "設置台数": st.column_config.NumberColumn(format="%d 台"), "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, hide_index=True, width="stretch")
                    
                    agg_y_col = "平均差枚" if chart_metric_shop == "平均差枚" else "高設定率" if chart_metric_shop == "高設定率" else "REG確率_val"
                    
                    st.markdown(f" **{selected_period} の末尾番号傾向 (0-9)**")
                    if '末尾番号' in period_df.columns:
                        digit_rank = period_df.groupby('末尾番号').agg(平均差枚=('差枚', 'mean'), 高設定率=('高設定_rate', 'mean'), REG確率_val=('REG確率_val', 'mean'), サンプル数=('差枚', 'count')).sort_index().reset_index()
                        digit_rank['REG確率'] = digit_rank['REG確率_val'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
                        digit_rank['信頼度'] = digit_rank['サンプル数'].apply(get_confidence_indicator)
                        st.bar_chart(digit_rank.set_index('末尾番号')[agg_y_col], color="#29b6f6" if chart_metric_shop == "平均差枚" else "#FF9800" if chart_metric_shop == "REG確率" else "#AB47BC")
                        st.dataframe(digit_rank[['末尾番号', '平均差枚', '高設定率', 'REG確率', 'サンプル数', '信頼度']].style.background_gradient(subset=['平均差枚'], cmap='RdYlGn', vmin=-300, vmax=300), column_config={"平均差枚": st.column_config.NumberColumn(format="%+d 枚"), "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100), "REG確率": st.column_config.TextColumn("REG確率"), "サンプル数": st.column_config.NumberColumn(format="%d 件"), "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, width="stretch")
                        
                    st.markdown(f"📅 **{selected_period} の曜日別傾向**")
                    if '曜日' in period_df.columns:
                        wd_rank = period_df.groupby('曜日').agg(平均差枚=('差枚', 'mean'), 高設定率=('高設定_rate', 'mean'), REG確率_val=('REG確率_val', 'mean'), サンプル数=('差枚', 'count')).reset_index()
                        wd_rank['REG確率'] = wd_rank['REG確率_val'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
                        day_order = {'月': 1, '火': 2, '水': 3, '木': 4, '金': 5, '土': 6, '日': 7}
                        wd_rank['sort'] = wd_rank['曜日'].map(day_order).fillna(99)
                        wd_rank = wd_rank.sort_values('sort').drop(columns=['sort'])
                        wd_rank['信頼度'] = wd_rank['サンプル数'].apply(get_confidence_indicator)
                        st.bar_chart(wd_rank.set_index('曜日')[agg_y_col], color="#4B4BFF" if chart_metric_shop == "平均差枚" else "#FF9800" if chart_metric_shop == "REG確率" else "#AB47BC")
                        st.dataframe(wd_rank[['曜日', '平均差枚', '高設定率', 'REG確率', 'サンプル数', '信頼度']].style.background_gradient(subset=['平均差枚'], cmap='RdYlGn', vmin=-300, vmax=300), column_config={"平均差枚": st.column_config.NumberColumn(format="%+d 枚"), "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100), "REG確率": st.column_config.TextColumn("REG確率"), "サンプル数": st.column_config.NumberColumn(format="%d 件"), "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, hide_index=True, width="stretch")
    
                    if '日付要素' in period_df.columns and not period_df['日付要素'].isnull().all():
                        st.markdown(f"🔥 **{selected_period} のイベント別傾向**")
                        ev_rank = period_df.groupby('日付要素').agg(平均差枚=('差枚', 'mean'), 高設定率=('高設定_rate', 'mean'), REG確率_val=('REG確率_val', 'mean'), サンプル数=('差枚', 'count')).reset_index().sort_values(chart_metric_shop if chart_metric_shop != "REG確率" else "REG確率_val", ascending=False)
                        ev_rank['REG確率'] = ev_rank['REG確率_val'].apply(lambda x: f"1/{int(1/x)}" if x > 0 else "-")
                        ev_rank['信頼度'] = ev_rank['サンプル数'].apply(get_confidence_indicator)
                        st.bar_chart(ev_rank.set_index('日付要素')[agg_y_col], color="#FF4B4B" if chart_metric_shop == "平均差枚" else "#FF9800" if chart_metric_shop == "REG確率" else "#AB47BC")
                        st.dataframe(ev_rank[['日付要素', '平均差枚', '高設定率', 'REG確率', 'サンプル数', '信頼度']].style.background_gradient(subset=['平均差枚'], cmap='RdYlGn', vmin=-300, vmax=300), column_config={"平均差枚": st.column_config.NumberColumn(format="%+d 枚"), "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100), "REG確率": st.column_config.TextColumn("REG確率"), "サンプル数": st.column_config.NumberColumn(format="%d 件"), "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, hide_index=True, width="stretch")
    
            st.markdown(f"**📅 日付別 {chart_metric_shop}推移**")
            day_stats = trend_df.groupby('day')[y_col].mean()
            st.bar_chart(day_stats, color="#00E676" if chart_metric_shop == "平均差枚" else "#FF9800" if chart_metric_shop == "REG確率" else "#AB47BC")
            
        if analysis_df is not None and not analysis_df.empty and 'shop_monthly_cumulative_diff' in analysis_df.columns:
            st.divider()
            st.markdown("### 💼 店長の予算（ノルマ進捗）別 営業傾向")
            st.caption("「前日までの月間累計差枚（店の赤字/黒字）」と「月の中の時期」を掛け合わせ、店がいつ回収し、いつ還元するかのクセを分析します。")
            
            daily_shop_df = analysis_df.groupby('対象日付').agg(
                店舗平均差枚=('差枚', 'mean'),
                月間累計差枚=('shop_monthly_cumulative_diff', 'first')
            ).reset_index()
            
            daily_shop_df['時期'] = daily_shop_df['対象日付'].dt.day.apply(lambda d: '1.月初 (1-7日)' if d<=7 else '3.月末 (25日-)' if d>=25 else '2.中旬 (8-24日)')
            
            def get_budget_status(diff):
                # 差枚は客側。客がマイナス＝店が黒字
                if diff <= -15000: return '1.🟦 余裕あり (店の大黒字: 客-1.5万枚〜)'
                elif diff <= 0: return '2.🟩 トントン (店の黒字: 客-1.5万〜0枚)'
                else: return '3.🟥 厳しい (店の赤字: 客+0枚〜)'
                
            daily_shop_df['予算状況(前日まで)'] = daily_shop_df['月間累計差枚'].apply(get_budget_status)
            
            budget_pivot = daily_shop_df.pivot_table(
                index='予算状況(前日まで)',
                columns='時期',
                values='店舗平均差枚',
                aggfunc='mean'
            ).round(0)
            
            for col in ['1.月初 (1-7日)', '2.中旬 (8-24日)', '3.月末 (25日-)']:
                if col not in budget_pivot.columns: budget_pivot[col] = np.nan
            budget_pivot = budget_pivot[['1.月初 (1-7日)', '2.中旬 (8-24日)', '3.月末 (25日-)']]
            
            st.info("💡 **表の見方**: 値は「その日の店舗の平均差枚」です。プラス（赤色）なら還元傾向、マイナス（青色）なら回収傾向を示します。たとえば『店が厳しい(赤字)時の月末』が真っ青なら、ノルマ達成のためのド回収が行われている証拠です。")
            st.dataframe(
                budget_pivot.style.background_gradient(cmap='RdYlBu_r', axis=None, vmin=-150, vmax=150).format("{:+.0f} 枚", na_rep="-"),
                width="stretch"
            )

def _render_shop_trend_analysis(selected_shop, df_raw_shop, top_trends_df, worst_trends_df, base_win_rate, specs, df_events=None, analysis_df=None):
    with st.expander(f"📅 {selected_shop} の傾向分析", expanded=True):
        st.caption("過去データに基づく、この店舗の店癖やイベント日・曜日ごとの傾向です。")
        
        # --- 店舗全体の還元日 / 回収日の傾向 ---
        if not df_raw_shop.empty and '対象日付' in df_raw_shop.columns:
            st.markdown(f"**💰 {selected_shop} の店舗全体 還元日 / 回収日 の傾向**")
            st.caption("店舗全体の平均差枚から、どの日が甘く（還元）、どの日が辛い（回収）かを示します。")
            
            shop_daily_df = df_raw_shop.groupby('対象日付').agg(
                店舗平均差枚=('差枚', 'mean')
            ).reset_index()
            
            shop_daily_df['曜日'] = shop_daily_df['対象日付'].dt.dayofweek
            shop_daily_df['末尾'] = shop_daily_df['対象日付'].dt.day % 10
            
            if df_events is not None and not df_events.empty:
                events_shop = df_events[df_events['店名'] == selected_shop].drop_duplicates(subset=['イベント日付'], keep='last')
                events_shop['イベント日付'] = pd.to_datetime(events_shop['イベント日付'])
                shop_daily_df = pd.merge(shop_daily_df, events_shop[['イベント日付', 'イベントランク']], left_on='対象日付', right_on='イベント日付', how='left')
                shop_daily_df['イベント有無'] = shop_daily_df['イベント日付'].notna().map({True: 'イベント日', False: '通常日'})
                shop_daily_df['イベントランク'] = shop_daily_df['イベントランク'].fillna('通常営業')
            else:
                shop_daily_df['イベント有無'] = '通常日'
                shop_daily_df['イベントランク'] = '通常営業'
            
            wd_shop_stats = shop_daily_df.groupby('曜日').agg(平均差枚=('店舗平均差枚', 'mean')).reset_index()
            digit_shop_stats = shop_daily_df.groupby('末尾').agg(平均差枚=('店舗平均差枚', 'mean')).reset_index()
            ev_shop_stats = shop_daily_df.groupby('イベント有無').agg(平均差枚=('店舗平均差枚', 'mean')).reset_index()
            rank_shop_stats = shop_daily_df.groupby('イベントランク').agg(平均差枚=('店舗平均差枚', 'mean')).reset_index()
            
            if not wd_shop_stats.empty and not digit_shop_stats.empty:
                best_wd = wd_shop_stats.loc[wd_shop_stats['平均差枚'].idxmax()]
                worst_wd = wd_shop_stats.loc[wd_shop_stats['平均差枚'].idxmin()]
                
                best_digit = digit_shop_stats.loc[digit_shop_stats['平均差枚'].idxmax()]
                worst_digit = digit_shop_stats.loc[digit_shop_stats['平均差枚'].idxmin()]
                
                weekdays_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
                
                ev_hot_str = ""
                ev_cold_str = ""
                if not ev_shop_stats.empty and 'イベント日' in ev_shop_stats['イベント有無'].values and '通常日' in ev_shop_stats['イベント有無'].values:
                    ev_diff = ev_shop_stats[ev_shop_stats['イベント有無']=='イベント日']['平均差枚'].iloc[0]
                    norm_diff = ev_shop_stats[ev_shop_stats['イベント有無']=='通常日']['平均差枚'].iloc[0]
                    
                    rank_str_list = []
                    for _, r in rank_shop_stats[rank_shop_stats['イベントランク'] != '通常営業'].sort_values('平均差枚', ascending=False).iterrows():
                        rank_str_list.append(f"{r['イベントランク']}: {int(r['平均差枚']):+d}枚")
                    rank_details = f" (ランク別: {', '.join(rank_str_list)})" if rank_str_list else ""
                    
                    if ev_diff > norm_diff:
                        ev_hot_str = f"\n- **イベント日** (店舗全体平均 {int(ev_diff):+d}枚 / 通常営業 {int(norm_diff):+d}枚){rank_details}"
                    else:
                        ev_cold_str = f"\n- **イベント日** (店舗全体平均 {int(ev_diff):+d}枚 / 通常営業 {int(norm_diff):+d}枚){rank_details} ※イベント回収傾向"
                
                st.info(f"🔥 **還元傾向が強い日 (甘い日)**\n- **{int(best_digit['末尾'])}のつく日** (店舗全体平均 {int(best_digit['平均差枚']):+d}枚)\n- **{weekdays_map[int(best_wd['曜日'])]}曜日** (店舗全体平均 {int(best_wd['平均差枚']):+d}枚){ev_hot_str}")
                st.warning(f"🥶 **回収傾向が強い日 (辛い日)**\n- **{int(worst_digit['末尾'])}のつく日** (店舗全体平均 {int(worst_digit['平均差枚']):+d}枚)\n- **{weekdays_map[int(worst_wd['曜日'])]}曜日** (店舗全体平均 {int(worst_wd['平均差枚']):+d}枚){ev_cold_str}")

            st.divider()
            st.markdown(f"**🔄 前日の営業結果（反動）による当日の傾向**")
            st.caption("前日お店全体が「回収（お店の黒字）」だったか「還元（お店の赤字）」だったかによって、翌日の営業がどう変わるか（お詫び還元があるか、連続回収か）を分析します。")
            
            shop_daily_df = shop_daily_df.sort_values('対象日付')
            shop_daily_df['前日_対象日付'] = shop_daily_df['対象日付'].shift(1)
            shop_daily_df['前日_店舗平均差枚'] = shop_daily_df['店舗平均差枚'].shift(1)
            
            def classify_prev_diff(d):
                if pd.isna(d): return np.nan
                if d <= -150: return '1. 大回収 (-150枚以下)'
                elif d < 0: return '2. チョイ回収 (-149〜-1枚)'
                elif d < 150: return '3. チョイ還元 (+0〜+149枚)'
                else: return '4. 大還元 (+150枚以上)'
                
            shop_daily_df['前日の営業'] = shop_daily_df['前日_店舗平均差枚'].apply(classify_prev_diff)
            
            rebound_stats = shop_daily_df.dropna(subset=['前日の営業']).groupby('前日の営業').agg(
                当日平均差枚=('店舗平均差枚', 'mean'),
                サンプル日数=('店舗平均差枚', 'count')
            ).reset_index().sort_values('前日の営業')
            
            if not rebound_stats.empty:
                col_r1, col_r2 = st.columns([1.2, 1])
                with col_r1:
                    chart_rebound = alt.Chart(rebound_stats).mark_bar().encode(
                        x=alt.X('前日の営業', title='前日の営業結果'),
                        y=alt.Y('当日平均差枚', title='当日の店舗平均差枚'),
                        color=alt.condition(alt.datum.当日平均差枚 > 0, alt.value("#FF7043"), alt.value("#42A5F5")),
                        tooltip=['前日の営業', alt.Tooltip('当日平均差枚', format='+.0f'), 'サンプル日数']
                    ).interactive()
                    st.altair_chart(chart_rebound, width="stretch")
                with col_r2:
                    st.dataframe(
                        rebound_stats,
                        column_config={
                            "前日の営業": st.column_config.TextColumn("前日の結果"),
                            "当日平均差枚": st.column_config.NumberColumn("当日の平均差枚", format="%+d 枚"),
                            "サンプル日数": st.column_config.NumberColumn("該当する日数", format="%d 日")
                        },
                        hide_index=True,
                        width="stretch"
                    )

            if analysis_df is not None and not analysis_df.empty:
                st.divider()
                st.markdown(f"**🔄 「据え置き」と「上げリセット（凹み反発）」の傾向**")
                st.caption("連続で勝っている台（据え置き）と、連続で凹んでいる台（上げリセット）の、翌日の高設定投入率（設定5基準）を比較します。何日目から狙い目になるかが分かります。")
                
                # 計算用フラグ
                sum_df = analysis_df.copy()
                sum_df['valid_high_play'] = sum_df['next_累計ゲーム'] >= 3000
                sum_df['target_rate'] = np.where(sum_df['valid_high_play'], sum_df['target'], np.nan) * 100
                
                def classify_minus(d):
                    if d == 1: return '1日凹み'
                    elif d == 2: return '2日連続凹み'
                    elif d >= 3: return '3日以上連続凹み'
                    return None
                    
                def classify_plus(d):
                    if d == 1: return '1日勝ち'
                    elif d == 2: return '2日連続勝ち'
                    elif d >= 3: return '3日以上連続勝ち'
                    return None
                    
                if '連続マイナス日数' in sum_df.columns:
                    sum_df['凹み状況'] = sum_df['連続マイナス日数'].apply(classify_minus)
                    minus_stats = sum_df.dropna(subset=['凹み状況']).groupby('凹み状況').agg(
                        上げ確率=('target_rate', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index()
                else:
                    minus_stats = pd.DataFrame()
                    
                if '連続プラス日数' in sum_df.columns:
                    sum_df['勝ち状況'] = sum_df['連続プラス日数'].apply(classify_plus)
                    plus_stats = sum_df.dropna(subset=['勝ち状況']).groupby('勝ち状況').agg(
                        据え置き確率=('target_rate', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index()
                else:
                    plus_stats = pd.DataFrame()
                
                col_up, col_stay = st.columns(2)
                with col_up:
                    st.markdown("**📈 凹み台の上げリセット確率**")
                    if not minus_stats.empty:
                        minus_stats['信頼度'] = minus_stats['サンプル数'].apply(get_confidence_indicator)
                        minus_order = {'1日凹み': 1, '2日連続凹み': 2, '3日以上連続凹み': 3}
                        minus_stats['sort'] = minus_stats['凹み状況'].map(minus_order)
                        minus_stats = minus_stats.sort_values('sort').drop(columns=['sort'])
                        st.dataframe(
                            minus_stats[['凹み状況', '上げ確率', 'サンプル数', '信頼度']],
                            column_config={
                                "凹み状況": st.column_config.TextColumn("状況"),
                                "上げ確率": st.column_config.ProgressColumn("上げ確率", format="%.1f%%", min_value=0, max_value=100),
                                "サンプル数": st.column_config.NumberColumn("対象台数", format="%d 台")
                            },
                            hide_index=True,
                            width="stretch"
                        )
                    else:
                        st.info("データがありません。")
                        
                with col_stay:
                    st.markdown("**🔁 勝ち台の据え置き確率**")
                    if not plus_stats.empty:
                        plus_stats['信頼度'] = plus_stats['サンプル数'].apply(get_confidence_indicator)
                        plus_order = {'1日勝ち': 1, '2日連続勝ち': 2, '3日以上連続勝ち': 3}
                        plus_stats['sort'] = plus_stats['勝ち状況'].map(plus_order)
                        plus_stats = plus_stats.sort_values('sort').drop(columns=['sort'])
                        st.dataframe(
                            plus_stats[['勝ち状況', '据え置き確率', 'サンプル数', '信頼度']],
                            column_config={
                                "勝ち状況": st.column_config.TextColumn("状況"),
                                "据え置き確率": st.column_config.ProgressColumn("据え置き確率", format="%.1f%%", min_value=0, max_value=100),
                                "サンプル数": st.column_config.NumberColumn("対象台数", format="%d 台")
                            },
                            hide_index=True,
                            width="stretch"
                        )
                    else:
                        st.info("データがありません。")

        if top_trends_df is not None or worst_trends_df is not None:
            st.markdown(f"**🤖 AIが発見した {selected_shop} の店癖/警戒条件**")
            if top_trends_df is not None and not top_trends_df.empty:
                st.caption("AIが過去データから見つけた、この店舗で特に翌日に高設定が入りやすい『激アツ条件 (🔥)』です。")
                top_trends_df['信頼度'] = top_trends_df['サンプル'].apply(get_confidence_indicator)
                st.dataframe(top_trends_df, column_config={"条件": st.column_config.TextColumn("激アツ条件"), "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100, help="条件合致時の高設定率"), "通常時との差": st.column_config.NumberColumn("差分", format="%+.1fpt", help="通常時との高設定率の差"), "サンプル": st.column_config.NumberColumn("台数", format="%d台", help="サンプル数"), "信頼度": st.column_config.TextColumn("信頼", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, hide_index=True, width="stretch")
            if worst_trends_df is not None and not worst_trends_df.empty:
                st.caption("AIが過去データから見つけた、この店舗で特に翌日に高設定が入りにくい『警戒条件 (⚠️)』です。")
                worst_trends_df['信頼度'] = worst_trends_df['サンプル'].apply(get_confidence_indicator)
                st.dataframe(worst_trends_df, column_config={"条件": st.column_config.TextColumn("警戒条件"), "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100, help="条件合致時の高設定率"), "通常時との差": st.column_config.NumberColumn("差分", format="%+.1fpt", help="通常時との高設定率の差"), "サンプル": st.column_config.NumberColumn("台数", format="%d台", help="サンプル数"), "信頼度": st.column_config.TextColumn("信頼", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, hide_index=True, width="stretch")
            st.caption(f"※この店舗の通常時の平均高設定率は **{base_win_rate:.1f}%** です。")
        

def render_feature_analysis_page(df_train, df_importance=None, df_events=None, df_raw=None, passed_shop_col=None, pre_selected_shop=None):
    if not pre_selected_shop:
        st.header("🔬 AI学習データ分析 (勝利の法則)")
    
    base_analysis_df = df_train.copy()

    if base_analysis_df.empty:
        st.warning("分析可能な過去データがありません。")
        return
        
    # --- データ期間フィルター ---
    if '対象日付' in base_analysis_df.columns:
        max_date = base_analysis_df['対象日付'].max()
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            selected_months = st.slider("📅 分析対象の期間 (直近〇ヶ月)", min_value=1, max_value=12, value=12, help="サイドバーの「AIモデル設定」で指定された学習データ期間の範囲内で、さらに期間を絞り込んで傾向を分析できます。")
            
        cutoff_date = max_date - pd.DateOffset(months=selected_months)
        base_analysis_df = base_analysis_df[base_analysis_df['対象日付'] >= cutoff_date]
        
        if base_analysis_df.empty:
            st.warning("指定された期間のデータがありません。")
            return
            
        actual_min_date = base_analysis_df['対象日付'].min()
        st.info(f"📅 **現在の集計期間**: {actual_min_date.strftime('%Y-%m-%d')} 〜 {max_date.strftime('%Y-%m-%d')} (対象: {len(base_analysis_df):,}件)\n\n※大元のデータ上限は、サイドバーの「⚙️ AIモデル設定」の学習データ期間に依存します。")
    
    # --- 店舗フィルター ---
    shop_col = '店名' if '店名' in base_analysis_df.columns else ('店舗名' if '店舗名' in base_analysis_df.columns else None)
    if not shop_col:
        shop_col = passed_shop_col
    
    if not shop_col:
        st.warning("店舗データがありません。")
        return

    if pre_selected_shop:
        selected_shop = pre_selected_shop
    else:
        shops = ["店舗を選択してください"] + sorted(list(base_analysis_df[shop_col].unique()))
        
        default_index = 0
        saved_shop = st.session_state.get("global_selected_shop", "店舗を選択してください")
        if saved_shop in shops:
            default_index = shops.index(saved_shop)
    
        selected_shop = st.selectbox("分析対象の店舗を選択", shops, index=default_index, key="feature_analysis_shop")
        
        if selected_shop != "店舗を選択してください":
            st.session_state["global_selected_shop"] = selected_shop
    
        if selected_shop == "店舗を選択してください":
            st.info("👆 分析対象の店舗を選択してください。店舗ごとの傾向や勝利の法則を分析します。")
            return

    analysis_df = base_analysis_df[base_analysis_df[shop_col] == selected_shop].copy()
    st.caption(f"【{selected_shop}】の過去データから、この店で高設定が入りやすい台の特徴を可視化します。")

    # --- 移管: 店舗別傾向・月間トレンド・AI店癖 ---
    has_raw = df_raw is not None and not df_raw.empty and shop_col in df_raw.columns
    if has_raw:
        df_raw_shop = df_raw[df_raw[shop_col] == selected_shop].copy()
        specs = backend.get_machine_specs()
        all_trends_dict = backend._calculate_shop_trends(base_analysis_df, shop_col, specs)
        base_win_rate = 0
        top_trends_df = None
        worst_trends_df = None
        if selected_shop in all_trends_dict:
            base_win_rate = all_trends_dict[selected_shop]['base_win_rate']
            top_trends_df = all_trends_dict[selected_shop]['top_df']
            worst_trends_df = all_trends_dict[selected_shop]['worst_df']
            
        viz_df = df_raw_shop.copy()
        if not viz_df.empty:
            viz_df['合算確率'] = (viz_df['BIG'] + viz_df['REG']) / viz_df['累計ゲーム'].replace(0, np.nan)
            spec_reg = viz_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
            spec_tot = viz_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
            spec_reg3 = viz_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定3', {"REG": 300.0})["REG"])
            viz_df['高設定'] = (
                (viz_df['累計ゲーム'] >= 3000) & 
                ((viz_df['REG確率'] >= spec_reg) | ((viz_df['合算確率'] >= spec_tot) & (viz_df['REG確率'] >= spec_reg3)))
            ).astype(int)
            viz_df['高設定_rate'] = np.where(viz_df['累計ゲーム'] >= 3000, viz_df['高設定'], np.nan) * 100
        else:
            viz_df['高設定_rate'] = np.nan

    tab_summary, tab_detail = st.tabs(["🏆 最重要サマリー (店癖・狙い目)", "📊 詳細分析データ (深掘り)"])
    
    with tab_summary:
        st.info("💡 **ここだけ見ればOK！** この店舗の基本的な設定配分のクセや、狙うべきポイントのまとめです。")
        if has_raw:
            _render_shop_trend_analysis(selected_shop, df_raw_shop, top_trends_df, worst_trends_df, base_win_rate, specs, df_events, analysis_df)

            # --- 当たり機種(全台系)の投入頻度 ---
            with st.expander(f"🎯 {selected_shop} の「全台系(当たり機種)」投入傾向", expanded=True):
                st.caption("この店舗における各機種の平均的な扱い（設定5近似度、高設定率、平均差枚など）です。店舗の「推し機種」や「冷遇機種」を見抜くのに役立ちます。")
                
                if not df_raw_shop.empty:
                    mac_df = df_raw_shop.copy()
                    mac_df['valid_play'] = (mac_df['累計ゲーム'] >= 3000) | ((mac_df['累計ゲーム'] < 3000) & ((mac_df['差枚'] <= -750) | (mac_df['差枚'] >= 750)))
                    
                    shop_avg_g = mac_df['累計ゲーム'].mean() if not mac_df.empty else 4000
                    
                    def calc_score_for_mac(row):
                        return backend.calculate_setting_score(
                            g=row.get('累計ゲーム', 0), act_b=row.get('BIG', 0), act_r=row.get('REG', 0), machine_name=row.get('機種名', ''), diff=row.get('差枚', 0),
                            shop_avg_g=shop_avg_g, penalty_reg=15, penalty_big=5, low_g_penalty=30, use_strict_scoring=True, return_details=False
                        )
                    
                    mac_df['設定5近似度'] = mac_df.apply(calc_score_for_mac, axis=1)
                    
                    mac_df['合算確率'] = (mac_df['BIG'] + mac_df['REG']) / mac_df['累計ゲーム'].replace(0, np.nan)
                    spec_reg = mac_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
                    spec_tot = mac_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
                    spec_reg3 = mac_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定3', {"REG": 300.0})["REG"])
                    
                    mac_df['高設定挙動'] = (
                        (mac_df['累計ゲーム'] >= 3000) & 
                        ((mac_df['REG確率'] >= spec_reg) | ((mac_df['合算確率'] >= spec_tot) & (mac_df['REG確率'] >= spec_reg3)))
                    ).astype(int)
                    mac_df['高設定率'] = np.where(mac_df['valid_play'], mac_df['高設定挙動'], np.nan) * 100
                    
                    mac_df['valid_差枚'] = np.where(mac_df['valid_play'], mac_df['差枚'], np.nan)
                    mac_df['valid_設定5近似度'] = np.where(mac_df['valid_play'], mac_df['設定5近似度'], np.nan)
                    mac_df['valid_累計ゲーム'] = np.where(mac_df['valid_play'], mac_df['累計ゲーム'], np.nan)

                    mac_stats = mac_df.groupby('機種名').agg(
                        平均差枚=('valid_差枚', 'mean'),
                        設定5近似度=('valid_設定5近似度', 'mean'),
                        高設定率=('高設定率', 'mean'),
                        平均回転数=('valid_累計ゲーム', 'mean'),
                        サンプル数=('台番号', 'count')
                    ).reset_index().sort_values('設定5近似度', ascending=False)
                    
                    mac_stats['信頼度'] = mac_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_m1, col_m2 = st.columns([1, 1.2])
                    with col_m1:
                        chart_mac = alt.Chart(mac_stats).mark_bar().encode(
                            x=alt.X('設定5近似度', title='設定5近似度 (平均点)'),
                            y=alt.Y('機種名', sort='-x', title='機種'),
                            color=alt.condition(alt.datum.設定5近似度 >= 40, alt.value("#FF7043"), alt.value("#42A5F5")),
                            tooltip=['機種名', alt.Tooltip('設定5近似度', format='.1f'), alt.Tooltip('高設定率', format='.1f', title='高設定率 (%)'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_mac, width="stretch")
                    with col_m2:
                        st.dataframe(
                            mac_stats,
                            column_config={
                                "機種名": st.column_config.TextColumn("機種"),
                                "設定5近似度": st.column_config.NumberColumn("設定5近似度", format="%.1f点", help="100点満点での平均的な設定5近似度"),
                                "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "平均差枚": st.column_config.NumberColumn("平均差枚", format="%+d 枚"),
                                "平均回転数": st.column_config.NumberColumn("平均回転", format="%d G"),
                                "サンプル数": st.column_config.NumberColumn("サンプル", format="%d 件"),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度"),
                            },
                            hide_index=True,
                            width="stretch"
                        )

            with st.expander(f"📅 {selected_shop} の機種別 日々の全台合算成績", expanded=False):
                st.caption("選択した機種の、その日の全台の合計データ（総回転数・BIG/REG・合算確率・REG確率）を日別に確認できます。特定機種のイベント日や、ベース設定の上げ下げを見抜くのに役立ちます。")
                
                if not df_raw_shop.empty and '機種名' in df_raw_shop.columns and '対象日付' in df_raw_shop.columns:
                    mac_list_for_daily = sorted(df_raw_shop['機種名'].dropna().unique().tolist())
                    selected_mac_daily = st.selectbox("確認する機種を選択", mac_list_for_daily, key="daily_mac_select_summary")
                    
                    if selected_mac_daily:
                        daily_mac_df = df_raw_shop[df_raw_shop['機種名'] == selected_mac_daily].copy()
                        
                        if not daily_mac_df.empty:
                            # 日別に集計
                            daily_mac_stats = daily_mac_df.groupby('対象日付').agg(
                                設置台数=('台番号', 'nunique'),
                                総回転=('累計ゲーム', 'sum'),
                                BIG=('BIG', 'sum'),
                                REG=('REG', 'sum'),
                                平均差枚=('差枚', 'mean'),
                                合計差枚=('差枚', 'sum')
                            ).reset_index().sort_values('対象日付', ascending=False)
                            
                            # 勝率の計算 (有効稼働ベース)
                            daily_mac_df['valid_play'] = (daily_mac_df['累計ゲーム'] >= 3000) | ((daily_mac_df['累計ゲーム'] < 3000) & ((daily_mac_df['差枚'] <= -750) | (daily_mac_df['差枚'] >= 750)))
                            daily_mac_df['is_win'] = daily_mac_df['valid_play'] & (daily_mac_df['差枚'] > 0)
                            
                            win_stats = daily_mac_df.groupby('対象日付').agg(
                                有効稼働=('valid_play', 'sum'),
                                勝台数=('is_win', 'sum')
                            ).reset_index()
                            
                            daily_mac_stats = pd.merge(daily_mac_stats, win_stats, on='対象日付', how='left')
                            daily_mac_stats['勝率'] = np.where(daily_mac_stats['有効稼働'] > 0, (daily_mac_stats['勝台数'] / daily_mac_stats['有効稼働']) * 100, 0.0)
                            
                            # 確率の計算
                            daily_mac_stats['合算確率分母'] = np.where((daily_mac_stats['BIG'] + daily_mac_stats['REG']) > 0, 
                                                                  daily_mac_stats['総回転'] / (daily_mac_stats['BIG'] + daily_mac_stats['REG']), 0)
                            daily_mac_stats['REG確率分母'] = np.where(daily_mac_stats['REG'] > 0, 
                                                                  daily_mac_stats['総回転'] / daily_mac_stats['REG'], 0)
                            
                            daily_mac_stats['対象日付_str'] = daily_mac_stats['対象日付'].dt.strftime('%Y-%m-%d')
                            daily_mac_stats['合算確率'] = daily_mac_stats['合算確率分母'].apply(lambda x: f"1/{x:.1f}" if x > 0 else "-")
                            daily_mac_stats['REG確率'] = daily_mac_stats['REG確率分母'].apply(lambda x: f"1/{x:.1f}" if x > 0 else "-")
                            
                            matched_key = backend.get_matched_spec_key(selected_mac_daily, specs)
                            spec_r5 = specs[matched_key].get('設定5', {}).get('REG', 260.0) if matched_key in specs else 260.0
                            
                            display_cols = ['対象日付_str', '設置台数', '総回転', 'BIG', 'REG', '合算確率', 'REG確率', '平均差枚', '合計差枚', '勝率']
                            
                            def highlight_reg(val):
                                if val == "-": return ""
                                try:
                                    den = float(val.replace("1/", ""))
                                    if den <= spec_r5: return 'background-color: rgba(255, 75, 75, 0.2)'
                                    elif den <= spec_r5 * 1.15: return 'background-color: rgba(255, 215, 0, 0.2)'
                                except: pass
                                return ""
                                
                            styled_df = daily_mac_stats[display_cols].style.map(highlight_reg, subset=['REG確率'])
                            styled_df = styled_df.bar(subset=['平均差枚'], align='mid', color=['rgba(66, 165, 245, 0.5)', 'rgba(255, 112, 67, 0.5)'], vmin=-500, vmax=500)
                            
                            st.dataframe(
                                styled_df,
                                column_config={
                                    "対象日付_str": st.column_config.TextColumn("日付"),
                                    "設置台数": st.column_config.NumberColumn("稼働台数", format="%d台"),
                                    "総回転": st.column_config.NumberColumn("総回転数", format="%d G"),
                                    "BIG": st.column_config.NumberColumn("BIG", format="%d 回"),
                                    "REG": st.column_config.NumberColumn("REG", format="%d 回"),
                                    "合算確率": st.column_config.TextColumn("全台合算確率"),
                                    "REG確率": st.column_config.TextColumn("全台REG確率", help=f"設定5の目安(1/{spec_r5})より良ければ赤色、少し悪ければ黄色になります。"),
                                    "平均差枚": st.column_config.NumberColumn("1台平均差枚", format="%+d 枚"),
                                    "合計差枚": st.column_config.NumberColumn("機種合計差枚", format="%+d 枚"),
                                    "勝率": st.column_config.ProgressColumn("勝率(有効稼働)", format="%.1f%%", min_value=0, max_value=100)
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                else:
                    st.info("データがありません。")

            # --- 並び(塊)の投入頻度 ---
            with st.expander(f"🤝 {selected_shop} の「並び(塊)」投入傾向", expanded=True):
                st.caption("日によって3台以上の並び（塊）が用意される店舗を見抜くための分析です。\n差枚が+1000枚以上の台が、台番号順で3台以上連続して存在した日を「並びが存在した日」として集計します。")
                
                if '対象日付' in df_raw_shop.columns and '台番号' in df_raw_shop.columns and '差枚' in df_raw_shop.columns:
                    # 台番号を数値化してソート可能にする
                    temp_df = df_raw_shop[['対象日付', '台番号', '差枚']].copy()
                    temp_df['台番号'] = pd.to_numeric(temp_df['台番号'], errors='coerce')
                    temp_df = temp_df.dropna(subset=['台番号']).sort_values(['対象日付', '台番号'])
                    
                    daily_narabi_records = []
                    for date, group in temp_df.groupby('対象日付'):
                        group = group.sort_values('台番号')
                        group['is_hot'] = group['差枚'] >= 1000
                        group['block'] = (group['is_hot'] != group['is_hot'].shift()).cumsum()
                        hot_blocks = group[group['is_hot']].groupby('block').size()
                        max_consecutive = hot_blocks.max() if not hot_blocks.empty else 0
                        
                        daily_narabi_records.append({
                            '対象日付': date,
                            '最大並び台数': max_consecutive,
                            '並びあり': max_consecutive >= 3
                        })
                        
                    if daily_narabi_records:
                        narabi_df = pd.DataFrame(daily_narabi_records)
                        hit_days = narabi_df['並びあり'].sum()
                        total_days = len(narabi_df)
                        hit_rate = (hit_days / total_days * 100) if total_days > 0 else 0
                        
                        st.metric("3台並び(塊) 投入率 (発生頻度)", f"{hit_rate:.1f}%", f"{hit_days}日 / 全{total_days}日")
                        
                        if hit_rate >= 30:
                            st.success("✨ **並び 多用店**: 3割以上の営業日で明確な「3台以上の並び」が存在します。両隣の挙動を常にチェックし、好調台の隣を積極的に狙う立ち回りが極めて有効です！")
                        elif hit_rate >= 15:
                            st.info("💡 **並び 散見**: 時折、3台以上の並び（塊）が作られる日があります。イベント日などに絞って並びを意識すると良いかもしれません。")
                        else:
                            st.warning("⚠️ **並び 傾向薄**: 3台以上連続して出ている箇所は少ないようです。並びよりも、単品や末尾、全台系などを意識した方が良さそうです。")
                            
                        with st.expander("📅 直近の並び投入履歴", expanded=False):
                            recent_hits = narabi_df[narabi_df['並びあり']].sort_values('対象日付', ascending=False).head(10)
                            if not recent_hits.empty:
                                recent_hits['対象日付_str'] = recent_hits['対象日付'].dt.strftime('%Y-%m-%d')
                                st.dataframe(
                                    recent_hits[['対象日付_str', '最大並び台数']],
                                    column_config={
                                        "対象日付_str": st.column_config.TextColumn("対象日付"),
                                        "最大並び台数": st.column_config.NumberColumn("最大連続台数", format="%d台")
                                    },
                                    hide_index=True,
                                    width="stretch"
                                )
                            else:
                                st.info("直近で明確な3台並びが存在した日がありません。")
                    else:
                        st.info("集計に必要なデータが不足しています。")
                else:
                    st.info("集計に必要なデータが不足しています。")

            # --- 角台の投入頻度 ---
            with st.expander(f"🪑 {selected_shop} の「角台」優遇傾向", expanded=True):
                st.caption("日によって角台が特別に強くなる店舗を見抜くための分析です。\n角台の平均差枚が店舗平均より+500枚以上高く、かつ角台平均自体が+500枚以上の対象日を「角台が優遇された日」として集計します。")
                
                if '対象日付' in df_raw_shop.columns and 'is_corner' in df_raw_shop.columns and '差枚' in df_raw_shop.columns:
                    # 角台の成績を集計
                    daily_corner_stats = df_raw_shop[df_raw_shop['is_corner'] == 1].groupby('対象日付').agg(
                        角台平均差枚=('差枚', 'mean'),
                        角台サンプル数=('台番号', 'count')
                    ).reset_index()
                    
                    daily_shop_stats = df_raw_shop.groupby('対象日付').agg(店舗平均差枚=('差枚', 'mean')).reset_index()
                    
                    if not daily_corner_stats.empty:
                        daily_merged = pd.merge(daily_corner_stats, daily_shop_stats, on='対象日付')
                        
                        # 角台優遇判定： 角台平均が+500枚以上 かつ 店舗平均を500枚以上上回る
                        daily_merged['角台優遇あり'] = (daily_merged['角台平均差枚'] >= 500) & ((daily_merged['角台平均差枚'] - daily_merged['店舗平均差枚']) >= 500)
                        
                        hit_days = daily_merged['角台優遇あり'].sum()
                        total_days = len(daily_merged)
                        hit_rate = (hit_days / total_days * 100) if total_days > 0 else 0
                        
                        st.metric("角台 優遇率 (発生頻度)", f"{hit_rate:.1f}%", f"{hit_days}日 / 全{total_days}日")
                        
                        if hit_rate >= 30:
                            st.success("✨ **角台 多用店**: 3割以上の営業日で角台が明確に優遇されています。台選びに迷ったら角台に座るのが最も期待値が高い立ち回りになります！")
                        elif hit_rate >= 15:
                            st.info("💡 **角台 散見**: 時折、角台が強くなる日があります。角台の挙動をチェックし、強そうなら粘る価値があります。")
                        else:
                            st.warning("⚠️ **角台 傾向薄**: 角台が特別に優遇される日は少ないようです。角というだけで過信せず、他の要素を重視したほうが良さそうです。")
                            
                        with st.expander("📅 直近の角台優遇履歴", expanded=False):
                            recent_hits = daily_merged[daily_merged['角台優遇あり']].sort_values('対象日付', ascending=False).head(10)
                            if not recent_hits.empty:
                                recent_hits['対象日付_str'] = recent_hits['対象日付'].dt.strftime('%Y-%m-%d')
                                st.dataframe(
                                    recent_hits[['対象日付_str', '角台サンプル数', '角台平均差枚', '店舗平均差枚']],
                                    column_config={
                                        "対象日付_str": st.column_config.TextColumn("対象日付"),
                                        "角台サンプル数": st.column_config.NumberColumn("角台数", format="%d台"),
                                        "角台平均差枚": st.column_config.NumberColumn("角台の平均差枚", format="%+d 枚"),
                                        "店舗平均差枚": st.column_config.NumberColumn("店舗全体の平均", format="%+d 枚")
                                    },
                                    hide_index=True,
                                    width="stretch"
                                )
                            else:
                                st.info("直近で明確な角台優遇が存在した日がありません。")
                    else:
                        st.info("集計に必要なデータ（角台のサンプル）が不足しています。")
                else:
                    st.info("集計に必要なデータが不足しています。事前にサイドバーの「島マスター管理」から角台を登録し、データ更新を行ってください。")

        if top_trends_df is not None or worst_trends_df is not None:
            st.markdown(f"**🤖 AIが発見した {selected_shop} の店癖/警戒条件**")

            # --- 並び(塊)の投入頻度 ---
            with st.expander(f"🤝 {selected_shop} の「並び(塊)」投入傾向", expanded=True):
                st.caption("日によって3台以上の並び（塊）が用意される店舗を見抜くための分析です。\n差枚が+1000枚以上の台が、台番号順で3台以上連続して存在した日を「並びが存在した日」として集計します。")
                
                if '対象日付' in df_raw_shop.columns and '台番号' in df_raw_shop.columns and '差枚' in df_raw_shop.columns:
                    # 台番号を数値化してソート可能にする
                    temp_df = df_raw_shop[['対象日付', '台番号', '差枚']].copy()
                    temp_df['台番号'] = pd.to_numeric(temp_df['台番号'], errors='coerce')
                    temp_df = temp_df.dropna(subset=['台番号']).sort_values(['対象日付', '台番号'])
                    
                    daily_narabi_records = []
                    for date, group in temp_df.groupby('対象日付'):
                        group = group.sort_values('台番号')
                        group['is_hot'] = group['差枚'] >= 1000
                        group['block'] = (group['is_hot'] != group['is_hot'].shift()).cumsum()
                        hot_blocks = group[group['is_hot']].groupby('block').size()
                        max_consecutive = hot_blocks.max() if not hot_blocks.empty else 0
                        
                        daily_narabi_records.append({
                            '対象日付': date,
                            '最大並び台数': max_consecutive,
                            '並びあり': max_consecutive >= 3
                        })
                        
                    if daily_narabi_records:
                        narabi_df = pd.DataFrame(daily_narabi_records)
                        hit_days = narabi_df['並びあり'].sum()
                        total_days = len(narabi_df)
                        hit_rate = (hit_days / total_days * 100) if total_days > 0 else 0
                        
                        st.metric("3台並び(塊) 投入率 (発生頻度)", f"{hit_rate:.1f}%", f"{hit_days}日 / 全{total_days}日")
                        
                        if hit_rate >= 30:
                            st.success("✨ **並び 多用店**: 3割以上の営業日で明確な「3台以上の並び」が存在します。両隣の挙動を常にチェックし、好調台の隣を積極的に狙う立ち回りが極めて有効です！")
                        elif hit_rate >= 15:
                            st.info("💡 **並び 散見**: 時折、3台以上の並び（塊）が作られる日があります。イベント日などに絞って並びを意識すると良いかもしれません。")
                        else:
                            st.warning("⚠️ **並び 傾向薄**: 3台以上連続して出ている箇所は少ないようです。並びよりも、単品や末尾、全台系などを意識した方が良さそうです。")
                            
                        with st.expander("📅 直近の並び投入履歴", expanded=False):
                            recent_hits = narabi_df[narabi_df['並びあり']].sort_values('対象日付', ascending=False).head(10)
                            if not recent_hits.empty:
                                recent_hits['対象日付_str'] = recent_hits['対象日付'].dt.strftime('%Y-%m-%d')
                                st.dataframe(
                                    recent_hits[['対象日付_str', '最大並び台数']],
                                    column_config={
                                        "対象日付_str": st.column_config.TextColumn("対象日付"),
                                        "最大並び台数": st.column_config.NumberColumn("最大連続台数", format="%d台")
                                    },
                                    hide_index=True,
                                    width="stretch"
                                )
                            else:
                                st.info("直近で明確な3台並びが存在した日がありません。")
                    else:
                        st.info("集計に必要なデータが不足しています。")
                else:
                    st.info("集計に必要なデータが不足しています。")

            # --- 角台の投入頻度 ---
            with st.expander(f"🪑 {selected_shop} の「角台」優遇傾向", expanded=True):
                st.caption("日によって角台が特別に強くなる店舗を見抜くための分析です。\n角台の平均差枚が店舗平均より+500枚以上高く、かつ角台平均自体が+500枚以上の対象日を「角台が優遇された日」として集計します。")
                
                if '対象日付' in df_raw_shop.columns and 'is_corner' in df_raw_shop.columns and '差枚' in df_raw_shop.columns:
                    # 角台の成績を集計
                    daily_corner_stats = df_raw_shop[df_raw_shop['is_corner'] == 1].groupby('対象日付').agg(
                        角台平均差枚=('差枚', 'mean'),
                        角台サンプル数=('台番号', 'count')
                    ).reset_index()
                    
                    daily_shop_stats = df_raw_shop.groupby('対象日付').agg(店舗平均差枚=('差枚', 'mean')).reset_index()
                    
                    if not daily_corner_stats.empty:
                        daily_merged = pd.merge(daily_corner_stats, daily_shop_stats, on='対象日付')
                        
                        # 角台優遇判定： 角台平均が+500枚以上 かつ 店舗平均を500枚以上上回る
                        daily_merged['角台優遇あり'] = (daily_merged['角台平均差枚'] >= 500) & ((daily_merged['角台平均差枚'] - daily_merged['店舗平均差枚']) >= 500)
                        
                        hit_days = daily_merged['角台優遇あり'].sum()
                        total_days = len(daily_merged)
                        hit_rate = (hit_days / total_days * 100) if total_days > 0 else 0
                        
                        st.metric("角台 優遇率 (発生頻度)", f"{hit_rate:.1f}%", f"{hit_days}日 / 全{total_days}日")
                        
                        if hit_rate >= 30:
                            st.success("✨ **角台 多用店**: 3割以上の営業日で角台が明確に優遇されています。台選びに迷ったら角台に座るのが最も期待値が高い立ち回りになります！")
                        elif hit_rate >= 15:
                            st.info("💡 **角台 散見**: 時折、角台が強くなる日があります。角台の挙動をチェックし、強そうなら粘る価値があります。")
                        else:
                            st.warning("⚠️ **角台 傾向薄**: 角台が特別に優遇される日は少ないようです。角というだけで過信せず、他の要素を重視したほうが良さそうです。")
                            
                        with st.expander("📅 直近の角台優遇履歴", expanded=False):
                            recent_hits = daily_merged[daily_merged['角台優遇あり']].sort_values('対象日付', ascending=False).head(10)
                            if not recent_hits.empty:
                                recent_hits['対象日付_str'] = recent_hits['対象日付'].dt.strftime('%Y-%m-%d')
                                st.dataframe(
                                    recent_hits[['対象日付_str', '角台サンプル数', '角台平均差枚', '店舗平均差枚']],
                                    column_config={
                                        "対象日付_str": st.column_config.TextColumn("対象日付"),
                                        "角台サンプル数": st.column_config.NumberColumn("角台数", format="%d台"),
                                        "角台平均差枚": st.column_config.NumberColumn("角台の平均差枚", format="%+d 枚"),
                                        "店舗平均差枚": st.column_config.NumberColumn("店舗全体の平均", format="%+d 枚")
                                    },
                                    hide_index=True,
                                    width="stretch"
                                )
                            else:
                                st.info("直近で明確な角台優遇が存在した日がありません。")
                    else:
                        st.info("集計に必要なデータ（角台のサンプル）が不足しています。")
                else:
                    st.info("集計に必要なデータが不足しています。事前にサイドバーの「島マスター管理」から角台を登録し、データ更新を行ってください。")

        else:
            st.warning("生データがないためサマリーを表示できません。詳細タブを確認してください。")

    with tab_detail:
        st.info("💡 **より詳しく分析したい方向け**。機種別の成績や詳細な条件ごとの傾向、AIの特徴量重要度などを確認できます。")
        
        if has_raw:
            _render_monthly_trend_analysis(viz_df, analysis_df)

            # --- 機種別 成績 (設定5近似度など) ---
            with st.expander(f"🎰 {selected_shop} の機種別 基本成績 (設定投入傾向)", expanded=False):
                st.caption("この店舗における各機種の平均的な扱い（設定5近似度、高設定率、平均差枚など）です。店舗の「推し機種」や「冷遇機種」を見抜くのに役立ちます。")
                
                if not df_raw_shop.empty:
                    mac_df = df_raw_shop.copy()
                    mac_df['valid_play'] = (mac_df['累計ゲーム'] >= 3000) | ((mac_df['累計ゲーム'] < 3000) & ((mac_df['差枚'] <= -750) | (mac_df['差枚'] >= 750)))
                    
                    shop_avg_g = mac_df['累計ゲーム'].mean() if not mac_df.empty else 4000
                    
                    def calc_score_for_mac(row):
                        return backend.calculate_setting_score(
                            g=row.get('累計ゲーム', 0), act_b=row.get('BIG', 0), act_r=row.get('REG', 0), machine_name=row.get('機種名', ''), diff=row.get('差枚', 0),
                            shop_avg_g=shop_avg_g, penalty_reg=15, penalty_big=5, low_g_penalty=30, use_strict_scoring=True, return_details=False
                        )
                    
                    mac_df['設定5近似度'] = mac_df.apply(calc_score_for_mac, axis=1)
                    
                    mac_df['合算確率'] = (mac_df['BIG'] + mac_df['REG']) / mac_df['累計ゲーム'].replace(0, np.nan)
                    spec_reg = mac_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
                    spec_tot = mac_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
                    spec_reg3 = mac_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定3', {"REG": 300.0})["REG"])
                    
                    mac_df['高設定挙動'] = (
                        (mac_df['累計ゲーム'] >= 3000) & 
                        ((mac_df['REG確率'] >= spec_reg) | ((mac_df['合算確率'] >= spec_tot) & (mac_df['REG確率'] >= spec_reg3)))
                    ).astype(int)
                    mac_df['高設定率'] = np.where(mac_df['valid_play'], mac_df['高設定挙動'], np.nan) * 100
                    
                    mac_df['valid_差枚'] = np.where(mac_df['valid_play'], mac_df['差枚'], np.nan)
                    mac_df['valid_設定5近似度'] = np.where(mac_df['valid_play'], mac_df['設定5近似度'], np.nan)
                    mac_df['valid_累計ゲーム'] = np.where(mac_df['valid_play'], mac_df['累計ゲーム'], np.nan)

                    mac_stats = mac_df.groupby('機種名').agg(
                        平均差枚=('valid_差枚', 'mean'),
                        設定5近似度=('valid_設定5近似度', 'mean'),
                        高設定率=('高設定率', 'mean'),
                        平均回転数=('valid_累計ゲーム', 'mean'),
                        サンプル数=('台番号', 'count')
                    ).reset_index().sort_values('設定5近似度', ascending=False)
                    
                    mac_stats['信頼度'] = mac_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_m1, col_m2 = st.columns([1, 1.2])
                    with col_m1:
                        chart_mac = alt.Chart(mac_stats).mark_bar().encode(
                            x=alt.X('設定5近似度', title='設定5近似度 (平均点)'),
                            y=alt.Y('機種名', sort='-x', title='機種'),
                            color=alt.condition(alt.datum.設定5近似度 >= 40, alt.value("#FF7043"), alt.value("#42A5F5")),
                            tooltip=['機種名', alt.Tooltip('設定5近似度', format='.1f'), alt.Tooltip('高設定率', format='.1f', title='高設定率 (%)'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_mac, width="stretch")
                    with col_m2:
                        st.dataframe(
                            mac_stats,
                            column_config={
                                "機種名": st.column_config.TextColumn("機種"),
                                "設定5近似度": st.column_config.NumberColumn("設定5近似度", format="%.1f点", help="100点満点での平均的な設定5近似度"),
                                "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "平均差枚": st.column_config.NumberColumn("平均差枚", format="%+d 枚"),
                                "平均回転数": st.column_config.NumberColumn("平均回転", format="%d G"),
                                "サンプル数": st.column_config.NumberColumn("サンプル", format="%d 件"),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度"),
                            },
                            hide_index=True,
                            width="stretch"
                        )

            # --- 機種別 日々の全台合算成績 ---
            with st.expander(f"📅 {selected_shop} の機種別 日々の全台合算成績", expanded=False):
                st.caption("選択した機種の、その日の全台の合計データ（総回転数・BIG/REG・合算確率・REG確率）を日別に確認できます。特定機種のイベント日や、ベース設定の上げ下げを見抜くのに役立ちます。")
                
                if not df_raw_shop.empty and '機種名' in df_raw_shop.columns and '対象日付' in df_raw_shop.columns:
                    mac_list_for_daily = sorted(df_raw_shop['機種名'].dropna().unique().tolist())
                    selected_mac_daily = st.selectbox("確認する機種を選択", mac_list_for_daily, key="daily_mac_select")
                    
                    if selected_mac_daily:
                        daily_mac_df = df_raw_shop[df_raw_shop['機種名'] == selected_mac_daily].copy()
                        
                        if not daily_mac_df.empty:
                            # 日別に集計
                            daily_mac_stats = daily_mac_df.groupby('対象日付').agg(
                                設置台数=('台番号', 'nunique'),
                                総回転=('累計ゲーム', 'sum'),
                                BIG=('BIG', 'sum'),
                                REG=('REG', 'sum'),
                                平均差枚=('差枚', 'mean'),
                                合計差枚=('差枚', 'sum')
                            ).reset_index().sort_values('対象日付', ascending=False)
                            
                            # 勝率の計算 (有効稼働ベース)
                            daily_mac_df['valid_play'] = (daily_mac_df['累計ゲーム'] >= 3000) | ((daily_mac_df['累計ゲーム'] < 3000) & ((daily_mac_df['差枚'] <= -750) | (daily_mac_df['差枚'] >= 750)))
                            daily_mac_df['is_win'] = daily_mac_df['valid_play'] & (daily_mac_df['差枚'] > 0)
                            
                            win_stats = daily_mac_df.groupby('対象日付').agg(
                                有効稼働=('valid_play', 'sum'),
                                勝台数=('is_win', 'sum')
                            ).reset_index()
                            
                            daily_mac_stats = pd.merge(daily_mac_stats, win_stats, on='対象日付', how='left')
                            daily_mac_stats['勝率'] = np.where(daily_mac_stats['有効稼働'] > 0, (daily_mac_stats['勝台数'] / daily_mac_stats['有効稼働']) * 100, 0.0)
                            
                            # 確率の計算
                            daily_mac_stats['合算確率分母'] = np.where((daily_mac_stats['BIG'] + daily_mac_stats['REG']) > 0, 
                                                                  daily_mac_stats['総回転'] / (daily_mac_stats['BIG'] + daily_mac_stats['REG']), 0)
                            daily_mac_stats['REG確率分母'] = np.where(daily_mac_stats['REG'] > 0, 
                                                                  daily_mac_stats['総回転'] / daily_mac_stats['REG'], 0)
                            
                            daily_mac_stats['対象日付_str'] = daily_mac_stats['対象日付'].dt.strftime('%Y-%m-%d')
                            daily_mac_stats['合算確率'] = daily_mac_stats['合算確率分母'].apply(lambda x: f"1/{x:.1f}" if x > 0 else "-")
                            daily_mac_stats['REG確率'] = daily_mac_stats['REG確率分母'].apply(lambda x: f"1/{x:.1f}" if x > 0 else "-")
                            
                            matched_key = backend.get_matched_spec_key(selected_mac_daily, specs)
                            spec_r5 = specs[matched_key].get('設定5', {}).get('REG', 260.0) if matched_key in specs else 260.0
                            
                            display_cols = ['対象日付_str', '設置台数', '総回転', 'BIG', 'REG', '合算確率', 'REG確率', '平均差枚', '合計差枚', '勝率']
                            
                            def highlight_reg(val):
                                if val == "-": return ""
                                try:
                                    den = float(val.replace("1/", ""))
                                    if den <= spec_r5: return 'background-color: rgba(255, 75, 75, 0.2)'
                                    elif den <= spec_r5 * 1.15: return 'background-color: rgba(255, 215, 0, 0.2)'
                                except: pass
                                return ""
                                
                            styled_df = daily_mac_stats[display_cols].style.map(highlight_reg, subset=['REG確率'])
                            styled_df = styled_df.bar(subset=['平均差枚'], align='mid', color=['rgba(66, 165, 245, 0.5)', 'rgba(255, 112, 67, 0.5)'], vmin=-500, vmax=500)
                            
                            st.dataframe(
                                styled_df,
                                column_config={
                                    "対象日付_str": st.column_config.TextColumn("日付"),
                                    "設置台数": st.column_config.NumberColumn("稼働台数", format="%d台"),
                                    "総回転": st.column_config.NumberColumn("総回転数", format="%d G"),
                                    "BIG": st.column_config.NumberColumn("BIG", format="%d 回"),
                                    "REG": st.column_config.NumberColumn("REG", format="%d 回"),
                                    "合算確率": st.column_config.TextColumn("全台合算確率"),
                                    "REG確率": st.column_config.TextColumn("全台REG確率", help=f"設定5の目安(1/{spec_r5})より良ければ赤色、少し悪ければ黄色になります。"),
                                    "平均差枚": st.column_config.NumberColumn("1台平均差枚", format="%+d 枚"),
                                    "合計差枚": st.column_config.NumberColumn("機種合計差枚", format="%+d 枚"),
                                    "勝率": st.column_config.ProgressColumn("勝率(有効稼働)", format="%.1f%%", min_value=0, max_value=100)
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                else:
                    st.info("データがありません。")
        tab_detail.divider()
        tab_detail.markdown("### 🔍 詳細条件別の傾向分析")
        tab_detail.caption("ここから下の各分析項目（基本指標・イベント・曜日・特殊パターン）に共通して適用される絞り込み条件です。")
        
        col_f1, col_f2 = tab_detail.columns(2)

    # --- 曜日フィルター ---
    with col_f1:
        if 'target_weekday' in analysis_df.columns:
            day_options = ["すべての曜日", "平日 (月〜金)", "週末 (土・日)", "月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"]
            selected_day = st.selectbox("分析対象の曜日を絞り込み", day_options)
            
            if selected_day != "すべての曜日":
                if selected_day == "平日 (月〜金)":
                    analysis_df = analysis_df[analysis_df['target_weekday'] < 5]
                elif selected_day == "週末 (土・日)":
                    analysis_df = analysis_df[analysis_df['target_weekday'] >= 5]
                else:
                    day_idx = ["月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"].index(selected_day)
                    analysis_df = analysis_df[analysis_df['target_weekday'] == day_idx]

    if analysis_df.empty:
        st.warning("選択された条件の分析データがありません。")
        return

    # REG確率分母を計算 (0除算回避)
    analysis_df['REG分母'] = analysis_df['REG確率'].apply(lambda x: int(1/x) if x > 0 else 9999)
    
    analysis_df['valid_high_play'] = analysis_df['next_累計ゲーム'] >= 3000
    analysis_df['target_rate'] = np.where(analysis_df['valid_high_play'], analysis_df['target'], np.nan) * 100
    analysis_df['valid_play_next'] = (analysis_df['next_累計ゲーム'] >= 3000) | ((analysis_df['next_累計ゲーム'] < 3000) & ((analysis_df['next_diff'] <= -750) | (analysis_df['next_diff'] >= 750)))
    analysis_df['valid_next_diff'] = np.where(analysis_df['valid_play_next'], analysis_df['next_diff'], np.nan)
    
    # ノイズ除去用のゲーム数フィルター
    with col_f2:
        min_g = st.slider("集計対象の最低回転数", min_value=0, max_value=8000, value=3000, step=500, help="指定した回転数以上回っている台のみを集計します。")
    
    reg_df = analysis_df[analysis_df['累計ゲーム'] >= min_g].copy()
    
    with tab_detail.expander("📊 基本指標 (REG・稼働)", expanded=False):
        # --- 1. REG確率別の翌日高設定率 (最重要) ---
        st.markdown("### 📊 REG確率と高設定据え置きの関係")
        st.caption("「前日のREG確率が良い台は、翌日も高設定のまま（据え置き）になるのか？」を検証します。")
        
        if reg_df.empty:
            st.warning("条件に一致するデータがありません。最低回転数を下げてください。")
        else:
            # REG分母をビン分割
            bins = [0, 200, 240, 280, 320, 360, 400, 500, 10000]
            labels = ['~1/200 (極良)', '1/200~240 (高)', '1/240~280 (良)', '1/280~320 (中)', '1/320~360 (低)', '1/360~400 (悪)', '1/400~500 (極悪)', '1/500~ (論外)']
            
            reg_df['REG区間'] = pd.cut(reg_df['REG分母'], bins=bins, labels=labels)
            
            reg_stats = reg_df.groupby('REG区間', observed=True).agg(
                高設定率=('target_rate', 'mean'),
                平均翌日差枚=('valid_next_diff', 'mean'),
                サンプル数=('target', 'count')
            ).reset_index()
            reg_stats['信頼度'] = reg_stats['サンプル数'].apply(get_confidence_indicator)
            
            # 複合グラフ: 棒グラフ(勝率) + 折れ線(差枚)
            base = alt.Chart(reg_stats).encode(x=alt.X('REG区間', title='前日のREG確率区分'))
            
            bar = base.mark_bar(color='#66BB6A', opacity=0.7).encode(
                y=alt.Y('高設定率', axis=alt.Axis(title='高設定率 (%)')),
                tooltip=['REG区間', alt.Tooltip('高設定率', format='.1f', title='高設定率 (%)'), 'サンプル数', '信頼度']
            )
            
            line = base.mark_line(color='#FF7043', point=True).encode(
                y=alt.Y('平均翌日差枚', axis=alt.Axis(title='平均翌日差枚 (枚)')),
                tooltip=['REG区間', alt.Tooltip('平均翌日差枚', format='+.0f')]
            )
            
            st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent'), width="stretch")
        
        col1, col2 = st.columns(2)
        
        # --- 2. 回転数(稼働量)と勝率 ---
        with col1:
            st.markdown("**🎰 回転数と期待値の関係**")
            # 回転数をビン分割
            g_bins = [0, 2000, 4000, 6000, 8000, 15000]
            g_labels = ['~2000G', '2000~4000G', '4000~6000G', '6000~8000G', '8000G~']
            
            analysis_df['G数区間'] = pd.cut(analysis_df['累計ゲーム'], bins=g_bins, labels=g_labels)
            
            g_stats = analysis_df.groupby('G数区間', observed=True)['valid_next_diff'].mean().reset_index(name='next_diff')
    
            chart_g = alt.Chart(g_stats).mark_bar().encode(
                x=alt.X('G数区間', title='前日の回転数'),
                y=alt.Y('next_diff', title='翌日の平均差枚'),
                color=alt.condition(alt.datum.next_diff > 0, alt.value("#FF7043"), alt.value("#42A5F5")),
                tooltip=['G数区間', alt.Tooltip('next_diff', format='+.0f', title='翌日の平均差枚')]
            )
            st.altair_chart(chart_g, width="stretch")
    
        # --- 3. 前日差枚と高設定率 ---
        with col2:
            st.markdown("**📉 前日差枚と翌日の高設定率**")
            # 差枚をビン分割
            d_bins = [-10000, -2000, -500, 500, 2000, 10000]
            d_labels = ['大負け', '負け', 'トントン', '勝ち', '大勝ち']
            analysis_df['差枚区間'] = pd.cut(analysis_df['差枚'], bins=d_bins, labels=d_labels)
            
            d_stats = analysis_df.groupby('差枚区間', observed=True)['target_rate'].mean().reset_index()
            
            chart_d = alt.Chart(d_stats).mark_bar(color='#FFA726').encode(
                x=alt.X('差枚区間', title='前日の結果', sort=None),
                y=alt.Y('target_rate', title='翌日高設定率 (%)'),
                tooltip=['差枚区間', alt.Tooltip('target_rate', format='.1f', title='翌日高設定率 (%)')]
            )
            st.altair_chart(chart_d, width="stretch")

    with tab_detail.expander("🏝️ 島（列）別の成績・傾向", expanded=False):
        st.markdown("### 🏝️ 島（列）別の成績・傾向")
        st.caption("島マスターに登録されている「島・列」ごとの過去の平均成績を比較します。どの島が普段から甘く使われているかがわかります。")
        if 'island_id' in reg_df.columns:
            island_df = reg_df[reg_df['island_id'] != "Unknown"].copy()
            if not island_df.empty:
                island_df['島名'] = island_df['island_id'].apply(lambda x: str(x).split('_', 1)[1] if '_' in str(x) else str(x))
                
                island_stats = island_df.groupby('島名').agg(
                    高設定率=('target_rate', 'mean'),
                    平均翌日差枚=('valid_next_diff', 'mean'),
                    サンプル数=('target', 'count')
                ).reset_index().sort_values('高設定率', ascending=False)
                
                island_stats['信頼度'] = island_stats['サンプル数'].apply(get_confidence_indicator)
                
                chart_metric_isl = st.radio("📊 グラフの表示指標", ["平均翌日差枚", "高設定率"], horizontal=True, key="isl_metric")
                y_field_isl = "平均翌日差枚" if chart_metric_isl == "平均翌日差枚" else "高設定率"
                y_title_isl = "平均翌日差枚 (枚)" if chart_metric_isl == "平均翌日差枚" else "高設定率 (%)"
                y_format_isl = "" if chart_metric_isl == "平均翌日差枚" else ".1f"
                y_axis_isl = alt.Axis(format=y_format_isl, title=y_title_isl) if y_format_isl else alt.Axis(title=y_title_isl)
                color_cond_isl = alt.condition(alt.datum.平均翌日差枚 > 0, alt.value("#FF7043"), alt.value("#42A5F5")) if chart_metric_isl == "平均翌日差枚" else alt.value("#AB47BC")

                col_isl1, col_isl2 = st.columns([1, 1.2])
                with col_isl1:
                    st.dataframe(
                        island_stats,
                        column_config={
                            "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100),
                            "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            "サンプル数": st.column_config.NumberColumn("集計台数", format="%d 台"),
                            "信頼度": st.column_config.TextColumn("信頼度")
                        },
                        hide_index=True,
                        width="stretch"
                    )
                with col_isl2:
                    chart_isl = alt.Chart(island_stats).mark_bar().encode(
                        x=alt.X('島名', title='島名', sort='-y'),
                        y=alt.Y(y_field_isl, axis=y_axis_isl),
                        color=color_cond_isl,
                        tooltip=['島名', alt.Tooltip('高設定率', format='.1f', title='高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                    ).interactive()
                    st.altair_chart(chart_isl, width="stretch")
            else:
                st.info("島マスターに登録された台の稼働データがありません。サイドバーの「島マスター管理」から島を登録してください。")
        else:
            st.info("島データがありません。")

    with tab_detail.expander("🎉 イベント傾向", expanded=False):
        # --- 4. イベントランク別の設定投入傾向 ---
        st.markdown("### 🎉 イベントランク別の設定投入傾向")
        st.caption(f"指定した回転数（{min_g}G）以上回っている台のうち、「設定5以上の基準を満たした台（高設定挙動）」の割合をイベントの強さごとに比較します。")
        
        if df_events is not None and not df_events.empty and shop_col in reg_df.columns:
            event_df = reg_df.copy()
            
            # 既存の 'イベントランク' カラムがある場合は翌日用のものなので削除
            if 'イベントランク' in event_df.columns:
                event_df = event_df.drop(columns=['イベントランク'])
                
            events_unique = df_events.drop_duplicates(subset=['店名', 'イベント日付'], keep='last').copy()
            merge_cols = ['店名', 'イベント日付']
            for c in ['イベントランク', '対象機種', 'イベント種別']:
                if c in events_unique.columns: merge_cols.append(c)
                
            if len(merge_cols) > 2:
                event_df = pd.merge(event_df, events_unique[merge_cols], left_on=[shop_col, '対象日付'], right_on=['店名', 'イベント日付'], how='left')
            else:
                event_df['イベントランク'] = np.nan
                
            # NaNや空文字を「通常日」として扱う
            event_df['イベントランク'] = event_df['イベントランク'].fillna('通常日').replace('', '通常日')
            
            # 機種別基準を適用した 'is_win' を高設定フラグとして利用
            event_df['valid_play'] = (event_df['next_累計ゲーム'] >= 3000) | ((event_df['next_累計ゲーム'] < 3000) & ((event_df['next_diff'] <= -750) | (event_df['next_diff'] >= 750)))
            event_df['高設定挙動'] = np.where(event_df['valid_play'], event_df.get('is_win', (event_df['REG分母'] <= 260).astype(int)), np.nan) * 100
            
            event_stats = event_df.groupby('イベントランク').agg(
                高設定投入率=('高設定挙動', 'mean'),
                平均差枚=('差枚', 'mean'),
                サンプル数=('台番号', 'count')
            ).reset_index()
            event_stats['信頼度'] = event_stats['サンプル数'].apply(get_confidence_indicator)
            
            # 並び替え順の指定
            rank_order = {'SS (周年)': 0, 'S': 1, 'A': 2, 'B': 3, 'C': 4, '通常日': 5}
            event_stats['sort'] = event_stats['イベントランク'].map(rank_order).fillna(99)
            event_stats = event_stats.sort_values('sort').drop(columns=['sort'])
            
            if len(event_stats['イベントランク'].unique()) > 1 or '通常日' not in event_stats['イベントランク'].values:
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    chart_e = alt.Chart(event_stats).mark_bar(color='#AB47BC', opacity=0.8).encode(
                        x=alt.X('イベントランク', sort=[k for k in rank_order.keys()], title='イベントの強さ'),
                        y=alt.Y('高設定投入率', axis=alt.Axis(title='高設定(設定5基準)の割合 (%)')),
                        tooltip=['イベントランク', alt.Tooltip('高設定投入率', format='.1f', title='高設定投入率 (%)'), 'サンプル数', '信頼度']
                    ).interactive()
                    st.altair_chart(chart_e, width="stretch")
                
                with col_e2:
                    st.dataframe(
                        event_stats,
                        column_config={
                            "高設定投入率": st.column_config.ProgressColumn("高設定割合", format="%.1f%%", min_value=0, max_value=100),
                            "平均差枚": st.column_config.NumberColumn("台平均差枚", format="%+d 枚"),
                            "サンプル数": st.column_config.NumberColumn("集計台数", format="%d 台"),
                            "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                        },
                        hide_index=True,
                        width="stretch"
                    )
                    
                # --- 対象機種・対象外のシワ寄せ分析 ---
                if '対象機種' in event_df.columns:
                    st.markdown("#### 🎯 特定機種イベント時の「対象外機種」の成績")
                    st.caption("イベントで特定の機種が対象になった際、自分が打っている機種（対象外）がどれくらい回収に回されているかを確認します。")
                    
                    def classify_target(row):
                        rank = str(row.get('イベントランク', '通常日'))
                        if rank == '通常日': return np.nan
                        
                        t_mac = str(row.get('対象機種', '指定なし'))
                        my_mac = str(row.get('機種名', ''))
                        e_type = str(row.get('イベント種別', '全体')).replace('スロット/全体', '全体')
                        
                        if e_type == '対象外(無効)': return np.nan
                        if e_type == 'パチンコ専用': return '🎰 パチンコ特日 (スロット回収警戒)'
                        
                        if t_mac in ['指定なし', 'スロット全体', 'ジャグラー全体', '全体', 'nan', 'None']:
                            return '🎈 全体対象イベント'
                        if t_mac == 'ジャグラー以外 (パチスロ他機種)':
                            return '⚠️ ジャグラー以外対象 (回収警戒)'
                        if my_mac in t_mac or t_mac in my_mac:
                            return '🎯 対象機種 (大チャンス)'
                        return '⚠️ 対象外機種 (シワ寄せ回収警戒)'
                        
                    event_df['対象ステータス'] = event_df.apply(classify_target, axis=1)
                    target_stats = event_df.dropna(subset=['対象ステータス']).groupby('対象ステータス').agg(
                        高設定投入率=('高設定挙動', 'mean'),
                        平均差枚=('差枚', 'mean'),
                        サンプル数=('台番号', 'count')
                    ).reset_index()
                    
                    if not target_stats.empty:
                        target_stats['信頼度'] = target_stats['サンプル数'].apply(get_confidence_indicator)
                        
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            chart_t = alt.Chart(target_stats).mark_bar(color='#FF7043', opacity=0.8).encode(
                                x=alt.X('高設定投入率', title='高設定(設定5基準)の割合 (%)'),
                                y=alt.Y('対象ステータス', sort='-x', title='イベント時の自分の台の立場'),
                                tooltip=['対象ステータス', alt.Tooltip('高設定投入率', format='.1f', title='高設定投入率 (%)'), alt.Tooltip('平均差枚', format='+.0f'), 'サンプル数', '信頼度']
                            ).interactive()
                            st.altair_chart(chart_t, width="stretch")
                        with col_t2:
                            st.dataframe(
                                target_stats.sort_values('高設定投入率', ascending=False),
                                column_config={
                                    "対象ステータス": st.column_config.TextColumn("イベント時の立場"),
                                    "高設定投入率": st.column_config.ProgressColumn("高設定割合", format="%.1f%%", min_value=0, max_value=100),
                                    "平均差枚": st.column_config.NumberColumn("台平均差枚", format="%+d 枚"),
                                    "サンプル数": st.column_config.NumberColumn("集計台数", format="%d 台"),
                                    "信頼度": st.column_config.TextColumn("信頼度")
                                },
                                hide_index=True,
                                width="stretch"
                            )
            else:
                st.info("イベントランクが登録されたデータがまだありません。サイドバーからイベントを登録すると傾向が表示されます。")
        else:
            st.info("イベントデータが登録されていないか、結合に失敗しました。")

    with tab_detail.expander("📅 曜日・末尾傾向", expanded=False):
        # --- 5. 予測日ベースの曜日・末尾別の傾向 ---
        st.markdown("### 📅 予測日ベースの傾向 (曜日・末尾)")
        st.caption(f"指定した回転数（{min_g}G）以上回っている台を対象に、「予測日の曜日」や「台の末尾番号」ごとの成績を比較します。店舗のクセを見抜くのに役立ちます。")
    
        if not reg_df.empty:
            tab_wd, tab_end = st.tabs(["📅 曜日別の傾向", "🔢 末尾番号の傾向"])
    
            with tab_wd:
                if 'target_weekday' in reg_df.columns:
                    weekdays_map = {0: '月曜', 1: '火曜', 2: '水曜', 3: '木曜', 4: '金曜', 5: '土曜', 6: '日曜'}
                    wd_df = reg_df.dropna(subset=['target_weekday']).copy()
                    wd_df['曜日'] = wd_df['target_weekday'].map(weekdays_map)
    
                    wd_stats = wd_df.groupby(['target_weekday', '曜日']).agg(
                        高設定率=('target_rate', 'mean'),
                        平均翌日差枚=('valid_next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('target_weekday')
    
                    wd_stats['信頼度'] = wd_stats['サンプル数'].apply(get_confidence_indicator)
    
                    col_w1, col_w2 = st.columns([1.2, 1])
                    with col_w1:
                        chart_wd = alt.Chart(wd_stats).mark_bar().encode(
                            x=alt.X('曜日', sort=[weekdays_map[i] for i in range(7)], title='予測日の曜日'),
                            y=alt.Y('平均翌日差枚', title='平均翌日差枚 (枚)'),
                            color=alt.condition(alt.datum.平均翌日差枚 > 0, alt.value("#FF7043"), alt.value("#42A5F5")),
                            tooltip=['曜日', alt.Tooltip('高設定率', format='.1f', title='高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_wd, width="stretch")
                    with col_w2:
                        st.dataframe(
                            wd_stats[['曜日', '高設定率', '平均翌日差枚', 'サンプル数', '信頼度']],
                            column_config={
                                "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
                else:
                    st.info("曜日データがありません。")
    
            with tab_end:
                if '末尾番号' in reg_df.columns:
                    end_df = reg_df.dropna(subset=['末尾番号']).copy()
                    end_stats = end_df.groupby('末尾番号').agg(
                        高設定率=('target_rate', 'mean'),
                        平均翌日差枚=('valid_next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('末尾番号')
                    
                    end_stats['末尾番号'] = end_stats['末尾番号'].astype(int).astype(str)
                    end_stats['信頼度'] = end_stats['サンプル数'].apply(get_confidence_indicator)
    
                    st.markdown("**📊 全期間の末尾別 平均成績**")
                    st.caption("過去の全データにおいて、特定の末尾が常に強いか（固定の当たり末尾があるか）を確認します。")
                    col_e1, col_e2 = st.columns([1.2, 1])
                    with col_e1:
                        chart_end = alt.Chart(end_stats).mark_bar().encode(
                            x=alt.X('末尾番号', title='末尾番号', sort=None),
                            y=alt.Y('平均翌日差枚', title='平均翌日差枚 (枚)'),
                            color=alt.condition(alt.datum.平均翌日差枚 > 0, alt.value("#FF7043"), alt.value("#42A5F5")),
                            tooltip=['末尾番号', alt.Tooltip('高設定率', format='.1f', title='高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_end, width="stretch")
                    with col_e2:
                        st.dataframe(
                            end_stats[['末尾番号', '高設定率', '平均翌日差枚', 'サンプル数', '信頼度']],
                            column_config={
                                "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
                        
                    st.divider()
                    st.markdown("**🎯 日替わり「当たり末尾」の投入頻度 (ランダム末尾を探せ)**")
                    st.caption("日によって当たり末尾が変わる店舗（全期間平均では目立たないが、1日単位で見ると特定の末尾だけ突出して出ている店）を見抜くための分析です。\n「その日一番出た末尾」が平均+500枚以上、かつ店舗全体の平均差枚を+500枚以上上回っている日を「当たり末尾が存在した日」として集計します。")
                    
                    if '対象日付' in end_df.columns and 'next_diff' in end_df.columns:
                        daily_end_stats = end_df.groupby(['対象日付', '末尾番号']).agg(
                            末尾平均差枚=('valid_next_diff', 'mean'),
                            サンプル数=('target', 'count')
                        ).reset_index()
                        
                        # サンプル数が少なすぎる末尾（3台未満）は極端な値になりやすいので除外
                        daily_end_stats = daily_end_stats[daily_end_stats['サンプル数'] >= 3]
                        daily_end_stats = daily_end_stats.dropna(subset=['末尾平均差枚'])
                        
                        if not daily_end_stats.empty:
                            idx_max = daily_end_stats.groupby('対象日付')['末尾平均差枚'].idxmax()
                            daily_top = daily_end_stats.loc[idx_max].copy()
                            daily_top = daily_top.rename(columns={'末尾番号': 'トップ末尾', '末尾平均差枚': 'トップ末尾差枚'})
                            
                            daily_shop = end_df.groupby('対象日付').agg(店舗平均差枚=('next_diff', 'mean')).reset_index()
                            daily_merged = pd.merge(daily_top[['対象日付', 'トップ末尾', 'トップ末尾差枚']], daily_shop, on='対象日付')
                            
                            daily_merged['当たり末尾あり'] = (daily_merged['トップ末尾差枚'] >= 500) & ((daily_merged['トップ末尾差枚'] - daily_merged['店舗平均差枚']) >= 500)
                            
                            hit_days = daily_merged['当たり末尾あり'].sum()
                            total_days = len(daily_merged)
                            hit_rate = (hit_days / total_days * 100) if total_days > 0 else 0
                            
                            st.metric("当たり末尾 投入率 (発生頻度)", f"{hit_rate:.1f}%", f"{hit_days}日 / 全{total_days}日")
                            
                            if hit_rate >= 30:
                                st.success("✨ **当たり末尾 多用店**: 3割以上の営業日で明確な「当たり末尾」が存在します。事前に予測できなくても、当日の周りの挙動（特に末尾）に注目して立ち回るのが非常に有効な店舗です！")
                            elif hit_rate >= 15:
                                st.info("💡 **当たり末尾 散見**: 時折、特定の末尾が強くなる日があります。イベント日などに絞って当たり末尾を探す立ち回りが有効かもしれません。")
                            else:
                                st.warning("⚠️ **当たり末尾 傾向薄**: 日替わりで特定の末尾が極端に強くなることは少ないようです。末尾狙いよりも、並びや他の傾向を重視したほうが良さそうです。")
                                
                            with st.expander("📅 直近の当たり末尾履歴", expanded=False):
                                recent_hits = daily_merged[daily_merged['当たり末尾あり']].sort_values('対象日付', ascending=False).head(10)
                                if not recent_hits.empty:
                                    recent_hits['対象日付_str'] = recent_hits['対象日付'].dt.strftime('%Y-%m-%d')
                                    recent_hits['トップ末尾'] = recent_hits['トップ末尾'].astype(int).astype(str)
                                    st.dataframe(
                                        recent_hits[['対象日付_str', 'トップ末尾', 'トップ末尾差枚', '店舗平均差枚']],
                                        column_config={
                                            "対象日付_str": st.column_config.TextColumn("対象日付"),
                                            "トップ末尾": st.column_config.TextColumn("一番出た末尾"),
                                            "トップ末尾差枚": st.column_config.NumberColumn("末尾の平均差枚", format="%+d 枚"),
                                            "店舗平均差枚": st.column_config.NumberColumn("店舗全体の平均", format="%+d 枚")
                                        },
                                        hide_index=True,
                                        width="stretch"
                                    )
                                else:
                                    st.info("直近で明確な当たり末尾が存在した日がありません。")
                        else:
                            st.info("集計に必要なデータ（各末尾3台以上のサンプル）が不足しています。")
                else:
                    st.info("末尾番号データがありません。")

    # --- 6. 特殊パターンの検証 ---
    with tab_detail.expander("🕵️‍♂️ 特殊パターンの検証", expanded=False):
        st.markdown("### 🕵️‍♂️ 特殊パターンの検証 (REG先行・凹み反発・大勝のその後)")
        st.caption(f"指定した回転数（{min_g}G）以上回っている台を対象に、スロット特有のパターンの翌日の成績を調査します。")
    
        if not reg_df.empty:
            chart_metric = st.radio("📊 グラフの表示指標", ["平均翌日差枚", "翌日高設定率"], horizontal=True)
            y_field = "平均翌日差枚" if chart_metric == "平均翌日差枚" else "翌日高設定率"
            y_title = "平均翌日差枚 (枚)" if chart_metric == "平均翌日差枚" else "翌日高設定率 (%)"
            y_format = "" if chart_metric == "平均翌日差枚" else ".1f"
            y_axis = alt.Axis(format=y_format, title=y_title) if y_format else alt.Axis(title=y_title)
            color_cond = alt.condition(alt.datum.平均翌日差枚 > 0, alt.value("#FF7043"), alt.value("#42A5F5")) if chart_metric == "平均翌日差枚" else alt.value("#AB47BC")
    
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["REG先行", "大凹み・大勝", "2日間トレンド", "連続凹み/連続勝ち", "安定度vs一撃", "BB極端欠損", "相対稼働率", "曜日別リセット傾向"])
            
            with tab1:
                if 'BIG' in reg_df.columns and 'REG' in reg_df.columns and 'REG確率' in reg_df.columns and 'BIG確率' in reg_df.columns:
                    reg_lead_df = reg_df.copy()
                    specs = backend.get_machine_specs()
                    
                    reg_lead_df['spec_reg_5'] = reg_lead_df['機種名'].apply(
                        lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"]
                    )
                    reg_lead_df['spec_big_5'] = reg_lead_df['機種名'].apply(
                        lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"BIG": 260.0})["BIG"]
                    )
                    
                    def classify_reg_lead(row):
                        if row['REG'] > row['BIG']:
                            if row.get('REG確率', 0) >= row.get('spec_reg_5', 1/260.0):
                                if row.get('差枚', 0) <= 0:
                                    return "① REG先行 & 差枚マイナス (BB欠損・完全不発)"
                                else:
                                    return "② REG先行 & 差枚プラス (チョイ浮き)"
                            else:
                                return "③ REG先行 & REG確率不足 (低設定の偏り)"
                        else:
                            if row.get('BIG確率', 0) >= row.get('spec_big_5', 1/260.0):
                                return "④ BIG先行/同数 & BIG設定5以上 (BIGヒキ強)"
                            else:
                                return "⑤ BIG先行/同数 & BIG確率不足 (マグレ吹き/低設定)"
                    
                    reg_lead_df['REG先行分類'] = reg_lead_df.apply(classify_reg_lead, axis=1)
                    
                    rl_stats = reg_lead_df.groupby('REG先行分類').agg(
                        翌日高設定率=('target_rate', 'mean'),
                        平均翌日差枚=('valid_next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index()
    
                    rl_stats['信頼度'] = rl_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_rl1, col_rl2 = st.columns([1, 1.2])
                    with col_rl1:
                        st.dataframe(
                            rl_stats,
                            column_config={
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
                    with col_rl2:
                        chart_rl = alt.Chart(rl_stats).mark_bar().encode(
                            x=alt.X('REG先行分類', title='REG先行の分類'),
                            y=alt.Y(y_field, axis=y_axis),
                            color=color_cond,
                            tooltip=['REG先行分類', alt.Tooltip('翌日高設定率', format='.1f', title='翌日高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_rl, width="stretch")
    
                else:
                    st.info("BIG/REG回数のデータがありません。")
    
            with tab2:
                st.markdown("**🔍 前日大凹み台の反発(底上げ) / 大勝台のその後**")
                st.caption("大きく負けた台は翌日反発するのか？ 逆に出すぎた台は回収されるのか？ を検証します。")
                
                diff_pat_df = reg_df.copy()
                def classify_diff_pat(d):
                    if d <= -2000: return "① 大凹み (-2000枚以下)"
                    elif d <= -1000: return "② 凹み (-1000〜-1999枚)"
                    elif d <= 0: return "③ チョイ負け (-1〜-999枚)"
                    elif d <= 1000: return "④ チョイ勝ち (+0〜+999枚)"
                    elif d <= 2000: return "⑤ 勝ち (+1000〜+1999枚)"
                    else: return "⑥ 大勝 (+2000枚以上)"
                    
                diff_pat_df['前日結果'] = diff_pat_df['差枚'].apply(classify_diff_pat)
                
                dp_stats = diff_pat_df.groupby('前日結果').agg(
                    翌日高設定率=('target_rate', 'mean'),
                    平均翌日差枚=('valid_next_diff', 'mean'),
                    サンプル数=('target', 'count')
                ).reset_index().sort_values('前日結果')
                dp_stats['信頼度'] = dp_stats['サンプル数'].apply(get_confidence_indicator)
    
                col_dp1, col_dp2 = st.columns([1, 1.2])
                with col_dp1:
                    st.dataframe(
                        dp_stats,
                        column_config={
                            "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                            "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=100),
                            "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                        },
                        hide_index=True,
                        width="stretch"
                    )
                with col_dp2:
                    chart_dp = alt.Chart(dp_stats).mark_bar().encode(
                        x=alt.X('前日結果', title='前日差枚'),
                        y=alt.Y(y_field, axis=y_axis),
                        color=color_cond,
                        tooltip=['前日結果', alt.Tooltip('翌日高設定率', format='.1f', title='翌日高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                    ).interactive()
                    st.altair_chart(chart_dp, width="stretch")
    
                st.divider()
                st.markdown("**📉 大凹み台の「総ゲーム数」別 反発期待度**")
                st.caption("前日大きく凹んだ(-1000枚以下)台について、あまり回されずに放置された台と、タコ粘りされて凹んだ台で翌日の反発(上げ)期待度がどう違うか検証します。")
                
                big_lose_df = diff_pat_df[diff_pat_df['差枚'] <= -1000].copy()
                if not big_lose_df.empty:
                    g_bins2 = [0, 3000, 5000, 7000, 15000]
                    g_labels2 = ['① ~3000G (放置)', '② 3000~5000G', '③ 5000~7000G', '④ 7000G~ (タコ粘り)']
                    
                    big_lose_df['G数区間'] = pd.cut(big_lose_df['累計ゲーム'], bins=g_bins2, labels=g_labels2)
                    
                    bl_stats = big_lose_df.groupby('G数区間', observed=True).agg(
                        翌日高設定率=('target_rate', 'mean'),
                        平均翌日差枚=('valid_next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index()
                    bl_stats['信頼度'] = bl_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_bl1, col_bl2 = st.columns([1, 1.2])
                    with col_bl1:
                        st.dataframe(
                            bl_stats,
                            column_config={
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
                    with col_bl2:
                        chart_bl = alt.Chart(bl_stats).mark_bar().encode(
                            x=alt.X('G数区間', title='前日の総ゲーム数'),
                            y=alt.Y(y_field, axis=y_axis),
                            color=color_cond,
                            tooltip=['G数区間', alt.Tooltip('翌日高設定率', format='.1f', title='翌日高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_bl, width="stretch")
                else:
                    st.info("大凹み（-1000枚以下）のデータがありません。")
    
            with tab3:
                st.markdown("**🔍 2日間の差枚トレンド (連勝・連敗・V字回復)**")
                st.caption("前々日と前日の差枚パターンから、翌日の高設定投入率（反発しやすいか、据え置かれやすいか）を検証します。")
                
                if 'prev_差枚' in reg_df.columns and '差枚' in reg_df.columns:
                    trend2d_df = reg_df.dropna(subset=['prev_差枚']).copy()
                    
                    def classify_2days_trend(row):
                        prev2 = row['prev_差枚']
                        prev1 = row['差枚']
                        
                        if prev2 <= -1000 and prev1 <= -1000:
                            return "① 連続大負け (-1000枚以下が2日連続)"
                        elif prev2 < 0 and prev1 < 0:
                            return "② 連敗 (2日連続マイナス)"
                        elif prev2 < 0 and prev1 >= 0:
                            return "③ V字反発 (前々日負け → 前日勝ち)"
                        elif prev2 >= 1000 and prev1 >= 1000:
                            return "④ 連続大勝 (+1000枚以上が2日連続)"
                        elif prev2 >= 0 and prev1 >= 0:
                            return "⑤ 連勝 (2日連続プラス)"
                        elif prev2 >= 0 and prev1 < 0:
                            return "⑥ 下落傾向 (前々日勝ち → 前日負け)"
                        else:
                            return "その他"
                            
                    trend2d_df['2日間トレンド'] = trend2d_df.apply(classify_2days_trend, axis=1)
                    
                    t2_stats = trend2d_df.groupby('2日間トレンド').agg(
                        翌日高設定率=('target_rate', 'mean'),
                        平均翌日差枚=('valid_next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('2日間トレンド')
                    t2_stats['信頼度'] = t2_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_t1, col_t2 = st.columns([1, 1.2])
                    with col_t1:
                        st.dataframe(
                            t2_stats,
                            column_config={
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
                    with col_t2:
                        chart_t2 = alt.Chart(t2_stats).mark_bar().encode(
                            x=alt.X('2日間トレンド', title='2日間の成績パターン'),
                            y=alt.Y(y_field, axis=y_axis),
                            color=color_cond,
                            tooltip=['2日間トレンド', alt.Tooltip('翌日高設定率', format='.1f', title='翌日高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_t2, width="stretch")
    
                else:
                    st.info("前々日の差枚データが不足しています。")
    
            with tab4:
                st.markdown("**🔍 連続マイナス台の「上げリセット」検証**")
                st.caption("何日間「実質マイナス（差枚+500枚未満）」が続くと「上げリセット（設定変更）」されやすくなるのか、店舗ごとの見切りラインを検証します。")
                
                if '連続マイナス日数' in reg_df.columns:
                    reset_df = reg_df.copy()
                    
                    def classify_cons_minus(d):
                        d = int(d)
                        if d == 0: return "① 0日 (前日放出: +500枚以上)"
                        elif d == 1: return "② 1日 実質マイナス"
                        elif d == 2: return "③ 2日連続 実質マイナス"
                        elif d == 3: return "④ 3日連続 実質マイナス"
                        elif d >= 4: return "⑤ 4日以上連続 実質マイナス"
                        return "不明"
                        
                    reset_df['マイナス継続状況'] = reset_df['連続マイナス日数'].apply(classify_cons_minus)
                    
                    r_stats = reset_df.groupby('マイナス継続状況').agg(
                        翌日高設定率=('target_rate', 'mean'),
                        平均翌日差枚=('valid_next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('マイナス継続状況')
                    r_stats['信頼度'] = r_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_r1, col_r2 = st.columns([1, 1.2])
                    with col_r1:
                        st.dataframe(
                            r_stats,
                            column_config={
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
                    with col_r2:
                        chart_r = alt.Chart(r_stats).mark_bar().encode(
                            x=alt.X('マイナス継続状況', title='マイナス継続状況'),
                            y=alt.Y(y_field, axis=y_axis),
                            color=color_cond,
                            tooltip=['マイナス継続状況', alt.Tooltip('翌日高設定率', format='.1f', title='翌日高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_r, width="stretch")
    
                else:
                    st.info("連続マイナス日数のデータがありません。")

                st.divider()
                st.markdown("**🔍 連続プラス台の「据え置き」検証**")
                st.caption("何日間「実質プラス（差枚>0）」が続くと据え置きされにくくなる（回収される）のか検証します。")
                
                if '連続プラス日数' in reg_df.columns:
                    plus_df = reg_df.copy()
                    
                    def classify_cons_plus(d):
                        d = int(d)
                        if d == 0: return "① 0日 (前日マイナス)"
                        elif d == 1: return "② 1日 プラス"
                        elif d == 2: return "③ 2日連続 プラス"
                        elif d == 3: return "④ 3日連続 プラス"
                        elif d >= 4: return "⑤ 4日以上連続 プラス"
                        return "不明"
                        
                    plus_df['プラス継続状況'] = plus_df['連続プラス日数'].apply(classify_cons_plus)
                    
                    p_stats = plus_df.groupby('プラス継続状況').agg(
                        翌日高設定率=('target_rate', 'mean'),
                        平均翌日差枚=('valid_next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('プラス継続状況')
                    p_stats['信頼度'] = p_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_p1, col_p2 = st.columns([1, 1.2])
                    with col_p1:
                        st.dataframe(
                            p_stats,
                            column_config={
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
                    with col_p2:
                        chart_p = alt.Chart(p_stats).mark_bar().encode(
                            x=alt.X('プラス継続状況', title='プラス継続状況'),
                            y=alt.Y(y_field, axis=y_axis),
                            color=color_cond,
                            tooltip=['プラス継続状況', alt.Tooltip('翌日高設定率', format='.1f', title='翌日高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_p, width="stretch")
                else:
                    st.info("連続プラス日数のデータがありません。")
    
            with tab5:
                st.markdown("**🔍 「安定台」vs「一撃荒波台」の翌日成績**")
                st.caption("週間平均差枚がプラスの好調台について、「毎日コツコツ高設定挙動の台（安定台）」と「まぐれで一撃出ただけの台（一撃台）」で翌日の高設定率に差があるか検証します。")
                
                if 'mean_7days_diff' in reg_df.columns and 'win_rate_7days' in reg_df.columns:
                    stab_df = reg_df.copy()
                    
                    def classify_stability(row):
                        mean_7d = row['mean_7days_diff']
                        wr = row['win_rate_7days']
                        
                        if mean_7d >= 500:
                            if wr >= 0.5:
                                return "① 安定・優秀台 (週間+500枚以上 & 高設定率50%以上)"
                            else:
                                return "② 一撃・荒波台 (週間+500枚以上 & 高設定率50%未満)"
                        elif mean_7d >= 0:
                            return "③ チョイ浮き台 (週間0〜+499枚)"
                        elif mean_7d >= -500:
                            return "④ チョイ沈み台 (週間-1〜-500枚)"
                        else:
                            return "⑤ 不調台 (週間-500枚以下)"
                            
                    stab_df['安定度分類'] = stab_df.apply(classify_stability, axis=1)
                    
                    stab_stats = stab_df.groupby('安定度分類').agg(
                        翌日高設定率=('target_rate', 'mean'),
                        平均翌日差枚=('valid_next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('安定度分類')
                    stab_stats['信頼度'] = stab_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_s1, col_s2 = st.columns([1, 1.2])
                    with col_s1:
                        st.dataframe(
                            stab_stats,
                            column_config={
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
                    with col_s2:
                        chart_stab = alt.Chart(stab_stats).mark_bar().encode(
                            x=alt.X('安定度分類', title='台の性質'),
                            y=alt.Y(y_field, axis=y_axis),
                            color=color_cond,
                            tooltip=['安定度分類', alt.Tooltip('翌日高設定率', format='.1f', title='翌日高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_stab, width="stretch")
    
                else:
                    st.info("週間勝率や平均差枚のデータがありません。")
    
            with tab6:
                st.markdown("**🔍 BIG極端欠損台の反発期待度**")
                st.caption("BIG確率が極端に悪い（1/400以下など）台が、翌日どうなるか（反発するか、そのまま放置か）を検証します。")
                
                if 'BIG確率' in reg_df.columns and 'REG確率' in reg_df.columns:
                    bb_def_df = reg_df.copy()
                    specs = backend.get_machine_specs()
                    
                    bb_def_df['spec_reg_5'] = bb_def_df['機種名'].apply(
                        lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"]
                    )
                    bb_def_df['BIG分母'] = bb_def_df['BIG確率'].apply(lambda x: 1/x if x > 0 else 9999)
                    
                    def classify_bb_deficit(row):
                        if row['BIG分母'] >= 400:
                            if row.get('REG確率', 0) >= row.get('spec_reg_5', 1/260.0):
                                return "① BIG 1/400以下 & REGは高設定基準 (超不発台)"
                            else:
                                return "② BIG 1/400以下 & REGも基準未達 (低設定の極み)"
                        elif row['BIG分母'] >= 300:
                            return "③ BIG 1/300〜1/400 (やや不発)"
                        else:
                            return "④ BIG 1/300より良い (欠損なし)"
                    
                    bb_def_df['BB欠損分類'] = bb_def_df.apply(classify_bb_deficit, axis=1)
                    
                    bb_stats = bb_def_df.groupby('BB欠損分類').agg(
                        翌日高設定率=('target_rate', 'mean'),
                        平均翌日差枚=('valid_next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('BB欠損分類')
                    
                    bb_stats['信頼度'] = bb_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_bb1, col_bb2 = st.columns([1, 1.2])
                    with col_bb1:
                        st.dataframe(
                            bb_stats,
                            column_config={
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
                    with col_bb2:
                        chart_bb = alt.Chart(bb_stats).mark_bar().encode(
                            x=alt.X('BB欠損分類', title='BB欠損の度合い'),
                            y=alt.Y(y_field, axis=y_axis),
                            color=color_cond,
                            tooltip=['BB欠損分類', alt.Tooltip('翌日高設定率', format='.1f', title='翌日高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_bb, width="stretch")
    
                else:
                    st.info("BIG確率のデータがありません。")
    
            with tab7:
                st.markdown("**🔍 相対的な粘られ度 (店舗平均との比較)**")
                st.caption("その日の店舗平均稼働に対して、どれくらい回されていたか（放置されたか、タコ粘りされたか）による翌日の反発・据え置き期待度を検証します。")
                
                if 'relative_games_ratio' in reg_df.columns:
                    rel_df = reg_df.copy()
                    
                    # ビン分割
                    rel_bins = [0, 0.5, 0.8, 1.2, 1.5, 10.0]
                    rel_labels = ['① 0.5倍未満 (完全放置)', '② 0.5〜0.8倍 (早め見切り)', '③ 0.8〜1.2倍 (平均的)', '④ 1.2〜1.5倍 (よく粘られた)', '⑤ 1.5倍以上 (タコ粘り)']
                    
                    rel_df['相対稼働率'] = pd.cut(rel_df['relative_games_ratio'], bins=rel_bins, labels=rel_labels)
                    
                    rel_stats = rel_df.groupby('相対稼働率', observed=True).agg(
                        翌日高設定率=('target_rate', 'mean'),
                        平均翌日差枚=('valid_next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('相対稼働率')
                    
                    rel_stats['信頼度'] = rel_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_rel1, col_rel2 = st.columns([1, 1.2])
                    with col_rel1:
                        st.dataframe(
                            rel_stats,
                            column_config={
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=100),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
                    with col_rel2:
                        chart_rel = alt.Chart(rel_stats).mark_bar().encode(
                            x=alt.X('相対稼働率', title='店舗平均に対する稼働割合'),
                            y=alt.Y(y_field, axis=y_axis),
                            color=color_cond,
                            tooltip=['相対稼働率', alt.Tooltip('翌日高設定率', format='.1f', title='翌日高設定率 (%)'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_rel, width="stretch")
    
                else:
                    st.info("相対稼働率のデータがありません。")

            with tab8:
                st.markdown("**🔍 曜日別 上げリセット・据え置き傾向**")
                st.caption("前日大きく凹んだ台（-1000枚以下）が翌日高設定になる「上げリセット率」と、前日大きく勝った台（+1000枚以上）が翌日高設定になる「据え置き率」を曜日ごとに比較します。")
                
                if 'target_weekday' in reg_df.columns and '差枚' in reg_df.columns:
                    weekdays_map = {0: '月曜', 1: '火曜', 2: '水曜', 3: '木曜', 4: '金曜', 5: '土曜', 6: '日曜'}
                    wd_trend_df = reg_df.dropna(subset=['target_weekday']).copy()
                    wd_trend_df['曜日'] = wd_trend_df['target_weekday'].map(weekdays_map)
                    
                    # 上げリセット（前日-1000枚以下）
                    reset_df = wd_trend_df[wd_trend_df['差枚'] <= -1000].groupby(['target_weekday', '曜日']).agg(
                        上げリセット率=('target_rate', 'mean'),
                        凹み台サンプル=('target', 'count')
                    ).reset_index()
                    
                    # 据え置き（前日+1000枚以上）
                    sue_df = wd_trend_df[wd_trend_df['差枚'] >= 1000].groupby(['target_weekday', '曜日']).agg(
                        据え置き率=('target_rate', 'mean'),
                        大勝台サンプル=('target', 'count')
                    ).reset_index()
                    
                    # 結合
                    wd_cross_stats = pd.merge(reset_df, sue_df, on=['target_weekday', '曜日'], how='outer').sort_values('target_weekday')
                    
                    if not wd_cross_stats.empty:
                        col_wxt1, col_wxt2 = st.columns([1.2, 1])
                        
                        # グラフ用にデータを縦持ちに変換
                        melted_stats = pd.melt(wd_cross_stats, id_vars=['曜日', 'target_weekday'], value_vars=['上げリセット率', '据え置き率'], var_name='パターン', value_name='高設定率')
                        
                        with col_wxt1:
                            chart_wxt = alt.Chart(melted_stats).mark_bar().encode(
                                x=alt.X('曜日', sort=[weekdays_map[i] for i in range(7)], title='予測日の曜日 (前日→当日)'),
                                y=alt.Y('高設定率', title='高設定率 (%)'),
                                color=alt.Color('パターン', scale=alt.Scale(domain=['上げリセット率', '据え置き率'], range=['#FF7043', '#42A5F5'])),
                                xOffset=alt.XOffset('パターン', sort=['上げリセット率', '据え置き率']),
                                tooltip=['曜日', 'パターン', alt.Tooltip('高設定率', format='.1f', title='高設定率 (%)')]
                            ).interactive()
                            st.altair_chart(chart_wxt, width="stretch")
                            
                        with col_wxt2:
                            st.dataframe(
                                wd_cross_stats[['曜日', '上げリセット率', '据え置き率', '凹み台サンプル', '大勝台サンプル']],
                                column_config={
                                    "上げリセット率": st.column_config.ProgressColumn("上げリセット", format="%.1f%%", min_value=0, max_value=100, help="前日-1000枚以下の台が翌日高設定になった割合"),
                                    "据え置き率": st.column_config.ProgressColumn("据え置き", format="%.1f%%", min_value=0, max_value=100, help="前日+1000枚以上の台が翌日高設定のままだった割合"),
                                },
                                hide_index=True,
                                width="stretch"
                            )
                    else:
                        st.info("集計に必要なデータが不足しています。")
                else:
                    st.info("曜日または差枚のデータがありません。")

    # --- 7. 特徴量重要度 ---
    with tab_detail.expander("🧠 AI特徴量重要度", expanded=False):
        if df_importance is not None and not df_importance.empty:
            st.markdown("### 🧠 AIが重視したポイント (特徴量重要度)")
    
            feature_name_map = {
                '累計ゲーム': '前日: 累計ゲーム数', 'REG確率': '前日: REG確率', 'BIG確率': '前日: BIG確率',
                '差枚': '前日: 差枚数', '末尾番号': '台番号: 末尾', 'target_weekday': '予測日: 曜日',
                'target_date_end_digit': '予測日: 日付末尾 (7のつく日等)', 'weekday_avg_diff': '店舗: 曜日平均差枚',
                'mean_7days_diff': '台: 直近7日平均差枚', 'median_7days_diff': '台: 直近7日中央値差枚(平常ベース)', 'win_rate_7days': '台: 直近7日間高設定率 (一撃排除用)', 'plus_rate_7days': '台: 直近7日間勝率 (プラス差枚割合)',
                '連続マイナス日数': '台: 連続実質マイナス日数(+500枚未満)', '連続低稼働日数': '台: 連続低稼働日数(1500G未満)', 'machine_code': '機種', 'shop_code': '店舗',
                'reg_ratio': '前日: REG比率', 'is_corner': '配置: 角台', 'is_main_corner': '配置: メイン通路側 角台', 'is_main_island': '島: メイン通路沿い(目立つ)', 'is_wall_island': '島: 壁側(目立たない)',
                'neighbor_avg_diff': '配置: 両隣の平均差枚 (※片側の大爆発によるフェイク注意)',
                'left_diff': '配置: 左隣の差枚', 'right_diff': '配置: 右隣の差枚', 'neighbor_positive_count': '配置: 両隣のプラス台数 (塊検知)',
                'event_avg_diff': 'イベント: 平均差枚',
                'event_code': 'イベント: 種類', 'event_rank_score': 'イベント: ランク', 'prev_差枚': '前々日: 差枚数',
                'prev_event_rank_score': 'イベント: 前日(特日)のランク(据え置き/回収反動)',
                'prev_REG確率': '前々日: REG確率', 'prev_累計ゲーム': '前々日: 累計ゲーム数',
                'shop_avg_diff': '店舗: 当日平均差枚', 'island_avg_diff': '島: 当日平均差枚',
                'shop_high_rate': '店舗: 当日高設定率', 'island_high_rate': '島: 当日高設定率',
                'prev_island_reg_prob': '前日: 島全体のREG確率', 'shop_heavy_lose_rate': '店舗: 当日大負け率(-1000枚以下)',
                'shop_play_rate': '店舗: 当日遊べる割合(±500枚以内)',
                'relative_games_ratio': '台: 相対稼働率(店舗平均比)',
                'is_new_machine': '台: 新台導入(導入後7日以内)',
                'is_moved_machine': '台: 配置変更(移動後7日以内)',
                'shop_7days_avg_diff': '店舗: 週間還元/回収モード(直近7日差枚)',
                'prev_shop_daily_avg_diff': '店舗: 前日の平均差枚(日次ノルマ反動)',
                'machine_30days_avg_diff': '機種: 機種ごとの扱い(直近30日差枚)',
                'machine_avg_diff': '機種: 当日平均差枚', 'machine_high_rate': '機種: 当日高設定率',
                'machine_heavy_lose_rate': '機種: 当日大負け率(-1000枚以下)',
                'machine_play_rate': '機種: 当日遊べる割合(±500枚以内)',
                'prev_推定ぶどう確率': '前日: 推定ぶどう確率(小役)',
                'shop_avg_games': '店舗: 平均稼働ゲーム数(客層レベル)',
                'shop_abandon_rate': '店舗: 見切り台の割合(見切りスピード)',
                'event_x_machine_avg_diff': '複合: イベント×機種の平均差枚',
                'event_x_end_digit_avg_diff': '複合: イベント×末尾の平均差枚',
                'cons_minus_total_diff': '台: 連続マイナス期間の合計吸い込み(枚)',
                'machine_no_30days_avg_diff': '台番号: その場所の強さ(直近30日差枚)',
                'is_beginning_of_month': '予測日: 月初(1-7日)', 'is_end_of_month': '予測日: 月末(25日-)',
                'is_pension_day': '予測日: 年金支給日(14-16日)',
                'shop_monthly_cumulative_diff': '店舗: 月間累計差枚(ノルマ進捗)',
                'prev_bonus_balance': '前日: BIG・REGの偏り(REG-BIG)',
                'prev_unlucky_gap': '前日: 不発度合い(REG回数と差枚のギャップ)',
                'is_low_play_high_reg': '複合: 前日低稼働(1000-3000G)＆高設定挙動',
                'is_hot_wd_and_heavy_lose': '複合: 還元曜日＆週間大凹み',
                'predicted_diff': 'AI予測: 予測差枚数(ST)'
            }
    
            # 全店舗の重要度データを準備
            imp_all = df_importance[df_importance['shop_name'] == '全店舗'].copy()
            if not imp_all.empty:
                imp_all['特徴量名'] = imp_all['feature'].map(lambda x: feature_name_map.get(x, x))
                imp_all = imp_all.sort_values('importance', ascending=False)
    
            # 店舗別の重要度データを準備
            imp_shop = df_importance[df_importance['shop_name'] == selected_shop].copy()
            if not imp_shop.empty:
                imp_shop['特徴量名'] = imp_shop['feature'].map(lambda x: feature_name_map.get(x, x))
                imp_shop = imp_shop.sort_values('importance', ascending=False)
    
            st.info("💡 **グラフの見方**: バーの長さが「重要度（予測への影響力）」を、**色**が「影響の方向（プラスかマイナスか）」を表します。\n\n"
                    "- 🟥 **赤系 (プラス)**: その値が大きいほど、高設定になりやすい（例: 前日差枚が多いほど据え置きされやすい）\n"
                    "- 🟦 **青系 (マイナス)**: その値が小さいほど、高設定になりやすい（例: 前日差枚が少ないほど上げられやすい）")
            
            # 相関データが存在しない過去のキャッシュへのフォールバック
            if 'correlation' not in imp_all.columns:
                imp_all['correlation'] = 0.0
            if not imp_shop.empty and 'correlation' not in imp_shop.columns:
                imp_shop['correlation'] = 0.0

            color_condition = alt.condition(
                alt.datum.correlation >= 0,
                alt.value("#FF7043"),  # プラスならオレンジ/赤系
                alt.value("#42A5F5")   # マイナスなら青系
            )
            
            tooltip_def = [
                alt.Tooltip('特徴量名:N', title='特徴量'),
                alt.Tooltip('importance:Q', title='重要度'),
                alt.Tooltip('correlation:Q', title='相関係数(方向)', format='+.2f')
            ]

            if not imp_shop.empty:
                # 店舗別モデルの重要度順でリストを作成し、両方のグラフのY軸ソート順を統一する
                sort_order = imp_shop['特徴量名'].tolist()
                
                # 店舗別モデルと全体モデルを比較表示
                st.caption(f"AIが【{selected_shop}】の台を予測する際に重視したデータ（左）と、全店舗共通の傾向（右）を比較します。")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**【{selected_shop}】専用モデル**")
                    chart_shop = alt.Chart(imp_shop).mark_bar().encode(
                        x=alt.X('importance:Q', title='重要度スコア'),
                        y=alt.Y('特徴量名:N', title='特徴量', sort=sort_order, axis=alt.Axis(labelLimit=0)),
                        color=color_condition,
                        tooltip=tooltip_def
                    ).properties(height=500).interactive()
                    st.altair_chart(chart_shop, width="stretch")
                    
                with col2:
                    st.markdown("**【全店舗】共通モデル**")
                    if not imp_all.empty:
                        chart_all = alt.Chart(imp_all).mark_bar().encode(
                            x=alt.X('importance:Q', title='重要度スコア'),
                            y=alt.Y('特徴量名:N', title=None, sort=sort_order, axis=alt.Axis(labelLimit=0)),
                            color=color_condition,
                            tooltip=tooltip_def
                        ).properties(height=500).interactive()
                        st.altair_chart(chart_all, width="stretch")
                    else:
                        st.info("全店舗共通モデルのデータがありません。")
    
            elif not imp_all.empty:
                # 店舗別モデルがなく、全体モデルのみの場合
                st.caption(f"【{selected_shop}】専用の学習モデルはまだありません（データ不足）。代わりに全店舗共通の傾向を表示します。")
                chart_imp = alt.Chart(imp_all).mark_bar().encode(
                    x=alt.X('importance:Q', title='重要度スコア'),
                    y=alt.Y('特徴量名:N', title='特徴量', sort='-x', axis=alt.Axis(labelLimit=0)),
                    color=color_condition,
                    tooltip=tooltip_def
                ).properties(height=500).interactive()
                
                st.altair_chart(chart_imp, width="stretch")
        else:
            st.info("特徴量重要度のデータがありません。")