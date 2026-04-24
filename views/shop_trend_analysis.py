import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import altair as alt # type: ignore
from utils import get_confidence_indicator
import backend

def render_shop_trend_analysis(selected_shop, df_raw_shop, top_trends_df, worst_trends_df, base_win_rate, specs, df_events=None, analysis_df=None):
    with st.expander(f"📅 {selected_shop} の傾向分析", expanded=True):
        st.caption("過去データに基づく、この店舗の店癖やイベント日・曜日ごとの傾向です。")
        
        # --- 店舗全体の還元日 / 回収日の傾向 ---
        if not df_raw_shop.empty and '対象日付' in df_raw_shop.columns:
            st.markdown(f"**💰 {selected_shop} の店舗全体 還元日 / 回収日 の傾向**")
            st.caption("店舗全体の平均REG確率から、どの日が甘く（還元）、どの日が辛い（回収）かを示します。")
            
            shop_daily_df = df_raw_shop.groupby('対象日付').agg(
                店舗平均差枚=('差枚', 'mean'),
                合計REG=('REG', 'sum'),
                合計ゲーム数=('累計ゲーム', 'sum')
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
            
            wd_shop_stats = shop_daily_df.groupby('曜日').agg(合計REG=('合計REG', 'sum'), 合計ゲーム数=('合計ゲーム数', 'sum')).reset_index()
            wd_shop_stats['REG確率分母'] = np.where(wd_shop_stats['合計REG'] > 0, wd_shop_stats['合計ゲーム数'] / wd_shop_stats['合計REG'], np.nan)
            digit_shop_stats = shop_daily_df.groupby('末尾').agg(合計REG=('合計REG', 'sum'), 合計ゲーム数=('合計ゲーム数', 'sum')).reset_index()
            digit_shop_stats['REG確率分母'] = np.where(digit_shop_stats['合計REG'] > 0, digit_shop_stats['合計ゲーム数'] / digit_shop_stats['合計REG'], np.nan)
            ev_shop_stats = shop_daily_df.groupby('イベント有無').agg(合計REG=('合計REG', 'sum'), 合計ゲーム数=('合計ゲーム数', 'sum')).reset_index()
            ev_shop_stats['REG確率分母'] = np.where(ev_shop_stats['合計REG'] > 0, ev_shop_stats['合計ゲーム数'] / ev_shop_stats['合計REG'], np.nan)
            rank_shop_stats = shop_daily_df.groupby('イベントランク').agg(合計REG=('合計REG', 'sum'), 合計ゲーム数=('合計ゲーム数', 'sum')).reset_index()
            rank_shop_stats['REG確率分母'] = np.where(rank_shop_stats['合計REG'] > 0, rank_shop_stats['合計ゲーム数'] / rank_shop_stats['合計REG'], np.nan)
            
            if not wd_shop_stats.empty and not digit_shop_stats.empty:
                best_wd = wd_shop_stats.loc[wd_shop_stats['REG確率分母'].idxmin()]
                worst_wd = wd_shop_stats.loc[wd_shop_stats['REG確率分母'].idxmax()]
                
                best_digit = digit_shop_stats.loc[digit_shop_stats['REG確率分母'].idxmin()]
                worst_digit = digit_shop_stats.loc[digit_shop_stats['REG確率分母'].idxmax()]
                
                weekdays_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
                
                ev_hot_str = ""
                ev_cold_str = ""
                if not ev_shop_stats.empty and 'イベント日' in ev_shop_stats['イベント有無'].values and '通常日' in ev_shop_stats['イベント有無'].values:
                    ev_reg_prob = ev_shop_stats[ev_shop_stats['イベント有無']=='イベント日']['REG確率分母'].iloc[0]
                    norm_reg_prob = ev_shop_stats[ev_shop_stats['イベント有無']=='通常日']['REG確率分母'].iloc[0]
                    
                    rank_str_list = []
                    for _, r in rank_shop_stats[rank_shop_stats['イベントランク'] != '通常営業'].sort_values('REG確率分母', ascending=True).iterrows():
                        if pd.notna(r['REG確率分母']):
                            rank_str_list.append(f"{r['イベントランク']}: 1/{int(r['REG確率分母'])}")
                    rank_details = f" (ランク別: {', '.join(rank_str_list)})" if rank_str_list else ""
                    
                    if pd.notna(ev_reg_prob) and pd.notna(norm_reg_prob) and ev_reg_prob < norm_reg_prob:
                        ev_hot_str = f"\n- **イベント日** (店舗全体REG 1/{int(ev_reg_prob)} / 通常営業 1/{int(norm_reg_prob)}){rank_details}"
                    else:
                        ev_cold_str = f"\n- **イベント日** (店舗全体REG 1/{int(ev_reg_prob) if pd.notna(ev_reg_prob) else 999} / 通常営業 1/{int(norm_reg_prob) if pd.notna(norm_reg_prob) else 999}){rank_details} ※イベント回収傾向"
                
                st.info(f"🔥 **還元傾向が強い日 (甘い日)**\n- **{int(best_digit['末尾'])}のつく日** (店舗全体REG 1/{int(best_digit['REG確率分母'])})\n- **{weekdays_map[int(best_wd['曜日'])]}曜日** (店舗全体REG 1/{int(best_wd['REG確率分母'])}){ev_hot_str}")
                st.warning(f"🥶 **回収傾向が強い日 (辛い日)**\n- **{int(worst_digit['末尾'])}のつく日** (店舗全体REG 1/{int(worst_digit['REG確率分母'])})\n- **{weekdays_map[int(worst_wd['曜日'])]}曜日** (店舗全体REG 1/{int(worst_wd['REG確率分母'])}){ev_cold_str}")

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