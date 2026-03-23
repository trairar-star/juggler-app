import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import altair as alt # type: ignore
from utils import get_confidence_indicator
import backend
from views.shop_detail_page import _calculate_shop_trends

def _render_monthly_trend_analysis(viz_df):
    with st.expander("🗓️ 月間トレンド (月初・月末の傾向)", expanded=False):
        st.caption("過去データにおける、日付（1日〜31日）ごとの平均差枚数や高設定率です。")
        
        chart_metric_shop = st.radio("📊 グラフの表示指標", ["平均差枚", "高設定率"], horizontal=True, key="monthly_trend_metric")
        y_col = "差枚" if chart_metric_shop == "平均差枚" else "高設定_rate"
        
        trend_df = viz_df.copy()
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
            else:
                m1.metric("🌙 月初 (1-7)", f"{val_start:.1%}")
                m2.metric("☀️ 中旬 (8-24)", f"{val_mid:.1%}")
                m3.metric("🌑 月末 (25-)", f"{val_end:.1%}")
            
            st.markdown("👇 **期間を選択すると、その期間に強い機種が表示されます**")
            selected_period = st.radio("期間選択", ['月初 (1-7日)', '中旬 (8-24日)', '月末 (25日-)'], horizontal=True, label_visibility="collapsed")
    
            if selected_period:
                period_df = trend_df[trend_df['period'] == selected_period]
                if not period_df.empty:
                    st.markdown(f"🎰 **{selected_period} の機種別ランキング**")
                    machine_rank = period_df.groupby('機種名').agg(平均差枚=('差枚', 'mean'), 高設定率=('高設定_rate', 'mean'), 設置台数=('台番号', 'nunique')).sort_values('高設定率', ascending=False).reset_index()
                    machine_rank['信頼度'] = machine_rank['設置台数'].apply(get_confidence_indicator)
                    st.dataframe(machine_rank, column_config={"平均差枚": st.column_config.NumberColumn(format="%+d 枚"), "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1), "設置台数": st.column_config.NumberColumn(format="%d 台"), "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, hide_index=True, width="stretch")
                    
                    st.markdown(f" **{selected_period} の末尾番号傾向 (0-9)**")
                    if '末尾番号' in period_df.columns:
                        digit_rank = period_df.groupby('末尾番号').agg(平均差枚=('差枚', 'mean'), 高設定率=('高設定_rate', 'mean'), サンプル数=('差枚', 'count')).sort_index().reset_index()
                        digit_rank['信頼度'] = digit_rank['サンプル数'].apply(get_confidence_indicator)
                        st.bar_chart(digit_rank.set_index('末尾番号')[chart_metric_shop], color="#29b6f6" if chart_metric_shop == "平均差枚" else "#AB47BC")
                        st.dataframe(digit_rank.style.background_gradient(subset=['平均差枚'], cmap='RdYlGn', vmin=-300, vmax=300), column_config={"平均差枚": st.column_config.NumberColumn(format="%+d 枚"), "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1), "サンプル数": st.column_config.NumberColumn(format="%d 件"), "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, width="stretch")
                        
                    st.markdown(f"📅 **{selected_period} の曜日別傾向**")
                    if '曜日' in period_df.columns:
                        wd_rank = period_df.groupby('曜日').agg(平均差枚=('差枚', 'mean'), 高設定率=('高設定_rate', 'mean'), サンプル数=('差枚', 'count')).reset_index()
                        day_order = {'月': 1, '火': 2, '水': 3, '木': 4, '金': 5, '土': 6, '日': 7}
                        wd_rank['sort'] = wd_rank['曜日'].map(day_order).fillna(99)
                        wd_rank = wd_rank.sort_values('sort').drop(columns=['sort'])
                        wd_rank['信頼度'] = wd_rank['サンプル数'].apply(get_confidence_indicator)
                        st.bar_chart(wd_rank.set_index('曜日')[chart_metric_shop], color="#4B4BFF" if chart_metric_shop == "平均差枚" else "#AB47BC")
                        st.dataframe(wd_rank.style.background_gradient(subset=['平均差枚'], cmap='RdYlGn', vmin=-300, vmax=300), column_config={"平均差枚": st.column_config.NumberColumn(format="%+d 枚"), "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1), "サンプル数": st.column_config.NumberColumn(format="%d 件"), "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, hide_index=True, width="stretch")
    
                    if '日付要素' in period_df.columns and not period_df['日付要素'].isnull().all():
                        st.markdown(f"🔥 **{selected_period} のイベント別傾向**")
                        ev_rank = period_df.groupby('日付要素').agg(平均差枚=('差枚', 'mean'), 高設定率=('高設定_rate', 'mean'), サンプル数=('差枚', 'count')).reset_index().sort_values(chart_metric_shop, ascending=False)
                        ev_rank['信頼度'] = ev_rank['サンプル数'].apply(get_confidence_indicator)
                        st.bar_chart(ev_rank.set_index('日付要素')[chart_metric_shop], color="#FF4B4B" if chart_metric_shop == "平均差枚" else "#AB47BC")
                        st.dataframe(ev_rank.style.background_gradient(subset=['平均差枚'], cmap='RdYlGn', vmin=-300, vmax=300), column_config={"平均差枚": st.column_config.NumberColumn(format="%+d 枚"), "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1), "サンプル数": st.column_config.NumberColumn(format="%d 件"), "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, hide_index=True, width="stretch")
    
            st.markdown(f"**📅 日付別 {chart_metric_shop}推移**")
            day_stats = trend_df.groupby('day')[y_col].mean()
            st.bar_chart(day_stats, color="#00E676" if chart_metric_shop == "平均差枚" else "#AB47BC")

def _render_shop_trend_analysis(selected_shop, df_raw_shop, top_trends_df, worst_trends_df, base_win_rate, specs):
    with st.expander(f"📅 {selected_shop} の傾向分析", expanded=True):
        st.caption("過去データに基づく、この店舗の店癖やイベント日・曜日ごとの傾向です。")
        
        if top_trends_df is not None or worst_trends_df is not None:
            st.markdown(f"**🤖 AIが発見した {selected_shop} の店癖/警戒条件**")
            if top_trends_df is not None and not top_trends_df.empty:
                st.caption("AIが過去データから見つけた、この店舗で特に翌日に高設定が入りやすい『激アツ条件 (🔥)』です。")
                top_trends_df['信頼度'] = top_trends_df['サンプル'].apply(get_confidence_indicator)
                st.dataframe(top_trends_df, column_config={"条件": st.column_config.TextColumn("激アツ条件"), "高設定率": st.column_config.ProgressColumn("高設定率", format="%.2f", min_value=0, max_value=1, help="条件合致時の高設定率"), "通常時との差": st.column_config.NumberColumn("差分", format="%+.1fpt", help="通常時との高設定率の差"), "サンプル": st.column_config.NumberColumn("台数", format="%d台", help="サンプル数"), "信頼度": st.column_config.TextColumn("信頼", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, hide_index=True, width="stretch")
            if worst_trends_df is not None and not worst_trends_df.empty:
                st.caption("AIが過去データから見つけた、この店舗で特に翌日に高設定が入りにくい『警戒条件 (⚠️)』です。")
                worst_trends_df['信頼度'] = worst_trends_df['サンプル'].apply(get_confidence_indicator)
                st.dataframe(worst_trends_df, column_config={"条件": st.column_config.TextColumn("警戒条件"), "高設定率": st.column_config.ProgressColumn("高設定率", format="%.2f", min_value=0, max_value=1, help="条件合致時の高設定率"), "通常時との差": st.column_config.NumberColumn("差分", format="%+.1fpt", help="通常時との高設定率の差"), "サンプル": st.column_config.NumberColumn("台数", format="%d台", help="サンプル数"), "信頼度": st.column_config.TextColumn("信頼", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")}, hide_index=True, width="stretch")
            st.caption(f"※この店舗の通常時の平均高設定率は **{base_win_rate:.1%}** です。")
        
        viz_df = df_raw_shop.copy()
        if not viz_df.empty:
            viz_df['合算確率'] = (viz_df['BIG'] + viz_df['REG']) / viz_df['累計ゲーム'].replace(0, np.nan)
            spec_reg = viz_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
            spec_tot = viz_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
            viz_df['valid_play'] = (viz_df['累計ゲーム'] >= 3000) | ((viz_df['累計ゲーム'] < 3000) & (viz_df['差枚'].abs() >= 1000))
            viz_df['高設定'] = ((viz_df['REG確率'] >= spec_reg) | (viz_df['合算確率'] >= spec_tot)).astype(int)
            viz_df['高設定_rate'] = np.where(viz_df['valid_play'], viz_df['高設定'], np.nan)
        else:
            viz_df['高設定_rate'] = np.nan
        
    _render_monthly_trend_analysis(viz_df)

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
    if df_raw is not None and not df_raw.empty and shop_col in df_raw.columns:
        df_raw_shop = df_raw[df_raw[shop_col] == selected_shop].copy()
        specs = backend.get_machine_specs()
        all_trends_dict = _calculate_shop_trends(base_analysis_df, shop_col, specs)
        base_win_rate = 0
        top_trends_df = None
        worst_trends_df = None
        if selected_shop in all_trends_dict:
            base_win_rate = all_trends_dict[selected_shop]['base_win_rate']
            top_trends_df = all_trends_dict[selected_shop]['top_df']
            worst_trends_df = all_trends_dict[selected_shop]['worst_df']
            
        _render_shop_trend_analysis(selected_shop, df_raw_shop, top_trends_df, worst_trends_df, base_win_rate, specs)

    st.divider()
    st.markdown("### 🔍 詳細条件別の傾向分析")
    st.caption("ここから下の各分析項目（基本指標・イベント・曜日・特殊パターン）に共通して適用される絞り込み条件です。")
    
    col_f1, col_f2 = st.columns(2)

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
    
    analysis_df['valid_play'] = (analysis_df['next_累計ゲーム'] >= 3000) | ((analysis_df['next_累計ゲーム'] < 3000) & (analysis_df['next_diff'].abs() >= 1000))
    analysis_df['target_rate'] = np.where(analysis_df['valid_play'], analysis_df['target'], np.nan)
    
    # ノイズ除去用のゲーム数フィルター
    with col_f2:
        min_g = st.slider("集計対象の最低回転数", min_value=0, max_value=8000, value=3000, step=500, help="指定した回転数以上回っている台のみを集計します。")
    
    reg_df = analysis_df[analysis_df['累計ゲーム'] >= min_g].copy()
    
    with st.expander("📊 基本指標 (REG・稼働)", expanded=False):
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
                平均翌日差枚=('next_diff', 'mean'),
                サンプル数=('target', 'count')
            ).reset_index()
            reg_stats['信頼度'] = reg_stats['サンプル数'].apply(get_confidence_indicator)
            
            # 複合グラフ: 棒グラフ(勝率) + 折れ線(差枚)
            base = alt.Chart(reg_stats).encode(x=alt.X('REG区間', title='前日のREG確率区分'))
            
            bar = base.mark_bar(color='#66BB6A', opacity=0.7).encode(
                y=alt.Y('高設定率', axis=alt.Axis(format='%', title='高設定率')),
                tooltip=['REG区間', alt.Tooltip('高設定率', format='.1%'), 'サンプル数', '信頼度']
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
            
            g_stats = analysis_df.groupby('G数区間', observed=True)['next_diff'].mean().reset_index()
    
            chart_g = alt.Chart(g_stats).mark_bar().encode(
                x=alt.X('G数区間', title='前日の回転数'),
                y=alt.Y('next_diff', title='翌日の平均差枚'),
                color=alt.condition(alt.datum.next_diff > 0, alt.value("#FF7043"), alt.value("#42A5F5"))
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
                y=alt.Y('target_rate', title='翌日高設定率', axis=alt.Axis(format='%'))
            )
            st.altair_chart(chart_d, width="stretch")

    with st.expander("🎉 イベント傾向", expanded=False):
        # --- 4. イベントランク別の設定投入傾向 ---
        st.markdown("### 🎉 イベントランク別の設定投入傾向")
        st.caption(f"指定した回転数（{min_g}G）以上回っている台のうち、「REG確率が1/300より良い台（高設定挙動）」の割合をイベントの強さごとに比較します。")
        
        if df_events is not None and not df_events.empty and shop_col in reg_df.columns:
            event_df = reg_df.copy()
            
            # 既存の 'イベントランク' カラムがある場合は翌日用のものなので削除
            if 'イベントランク' in event_df.columns:
                event_df = event_df.drop(columns=['イベントランク'])
                
            events_unique = df_events.drop_duplicates(subset=['店名', 'イベント日付'], keep='last').copy()
            if 'イベントランク' in events_unique.columns:
                event_df = pd.merge(event_df, events_unique[['店名', 'イベント日付', 'イベントランク']], left_on=[shop_col, '対象日付'], right_on=['店名', 'イベント日付'], how='left')
            else:
                event_df['イベントランク'] = np.nan
                
            # NaNや空文字を「通常日」として扱う
            event_df['イベントランク'] = event_df['イベントランク'].fillna('通常日').replace('', '通常日')
            
            # 機種別基準を適用した 'is_win' を高設定フラグとして利用
            event_df['valid_play'] = (event_df['next_累計ゲーム'] >= 3000) | ((event_df['next_累計ゲーム'] < 3000) & (event_df['next_diff'].abs() >= 1000))
            event_df['高設定挙動'] = np.where(event_df['valid_play'], event_df.get('is_win', (event_df['REG分母'] <= 260).astype(int)), np.nan)
            
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
                        y=alt.Y('高設定投入率', axis=alt.Axis(format='%', title='高設定(設定5基準)の割合')),
                        tooltip=['イベントランク', alt.Tooltip('高設定投入率', format='.1%'), 'サンプル数', '信頼度']
                    ).interactive()
                    st.altair_chart(chart_e, width="stretch")
                
                with col_e2:
                    st.dataframe(
                        event_stats,
                        column_config={
                            "高設定投入率": st.column_config.ProgressColumn("高設定割合", format="%.1f%%", min_value=0, max_value=1),
                            "平均差枚": st.column_config.NumberColumn("台平均差枚", format="%+d 枚"),
                            "サンプル数": st.column_config.NumberColumn("集計台数", format="%d 台"),
                            "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                        },
                        hide_index=True,
                        width="stretch"
                    )
            else:
                st.info("イベントランクが登録されたデータがまだありません。サイドバーからイベントを登録すると傾向が表示されます。")
        else:
            st.info("イベントデータが登録されていないか、結合に失敗しました。")

    with st.expander("📅 曜日・末尾傾向", expanded=False):
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
                        平均翌日差枚=('next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('target_weekday')
    
                    wd_stats['信頼度'] = wd_stats['サンプル数'].apply(get_confidence_indicator)
    
                    col_w1, col_w2 = st.columns([1.2, 1])
                    with col_w1:
                        chart_wd = alt.Chart(wd_stats).mark_bar().encode(
                            x=alt.X('曜日', sort=[weekdays_map[i] for i in range(7)], title='予測日の曜日'),
                            y=alt.Y('平均翌日差枚', title='平均翌日差枚 (枚)'),
                            color=alt.condition(alt.datum.平均翌日差枚 > 0, alt.value("#FF7043"), alt.value("#42A5F5")),
                            tooltip=['曜日', alt.Tooltip('高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_wd, width="stretch")
                    with col_w2:
                        st.dataframe(
                            wd_stats[['曜日', '高設定率', '平均翌日差枚', 'サンプル数', '信頼度']],
                            column_config={
                                "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=1),
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
                        平均翌日差枚=('next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('末尾番号')
                    
                    end_stats['末尾番号'] = end_stats['末尾番号'].astype(int).astype(str)
                    end_stats['信頼度'] = end_stats['サンプル数'].apply(get_confidence_indicator)
    
                    col_e1, col_e2 = st.columns([1.2, 1])
                    with col_e1:
                        chart_end = alt.Chart(end_stats).mark_bar().encode(
                            x=alt.X('末尾番号', title='末尾番号', sort=None),
                            y=alt.Y('平均翌日差枚', title='平均翌日差枚 (枚)'),
                            color=alt.condition(alt.datum.平均翌日差枚 > 0, alt.value("#FF7043"), alt.value("#42A5F5")),
                            tooltip=['末尾番号', alt.Tooltip('高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_end, width="stretch")
                    with col_e2:
                        st.dataframe(
                            end_stats[['末尾番号', '高設定率', '平均翌日差枚', 'サンプル数', '信頼度']],
                            column_config={
                                "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=1),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            width="stretch"
                        )
                else:
                    st.info("末尾番号データがありません。")

    # --- 6. 特殊パターンの検証 (REG先行・大凹み・大勝) ---
    with st.expander("🕵️‍♂️ 特殊パターンの検証", expanded=False):
        st.markdown("### 🕵️‍♂️ 特殊パターンの検証 (REG先行・凹み反発・大勝のその後)")
        st.caption(f"指定した回転数（{min_g}G）以上回っている台を対象に、スロット特有のパターンの翌日の成績を調査します。")
    
        if not reg_df.empty:
            chart_metric = st.radio("📊 グラフの表示指標", ["平均翌日差枚", "翌日高設定率"], horizontal=True)
            y_field = "平均翌日差枚" if chart_metric == "平均翌日差枚" else "翌日高設定率"
            y_title = "平均翌日差枚 (枚)" if chart_metric == "平均翌日差枚" else "翌日高設定率"
            y_format = "" if chart_metric == "平均翌日差枚" else "%"
            y_axis = alt.Axis(format=y_format, title=y_title) if y_format else alt.Axis(title=y_title)
            color_cond = alt.condition(alt.datum.平均翌日差枚 > 0, alt.value("#FF7043"), alt.value("#42A5F5")) if chart_metric == "平均翌日差枚" else alt.value("#AB47BC")
    
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["REG先行", "大凹み・大勝", "2日間トレンド", "上げリセット", "安定度vs一撃", "BB極端欠損", "相対稼働率"])
            
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
                        平均翌日差枚=('next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index()
    
                    rl_stats['信頼度'] = rl_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_rl1, col_rl2 = st.columns([1, 1.2])
                    with col_rl1:
                        st.dataframe(
                            rl_stats,
                            column_config={
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
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
                            tooltip=['REG先行分類', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
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
                    平均翌日差枚=('next_diff', 'mean'),
                    サンプル数=('target', 'count')
                ).reset_index().sort_values('前日結果')
                dp_stats['信頼度'] = dp_stats['サンプル数'].apply(get_confidence_indicator)
    
                col_dp1, col_dp2 = st.columns([1, 1.2])
                with col_dp1:
                    st.dataframe(
                        dp_stats,
                        column_config={
                            "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                            "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
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
                        tooltip=['前日結果', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
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
                        平均翌日差枚=('next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index()
                    bl_stats['信頼度'] = bl_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_bl1, col_bl2 = st.columns([1, 1.2])
                    with col_bl1:
                        st.dataframe(
                            bl_stats,
                            column_config={
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
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
                            tooltip=['G数区間', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
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
                        平均翌日差枚=('next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('2日間トレンド')
                    t2_stats['信頼度'] = t2_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_t1, col_t2 = st.columns([1, 1.2])
                    with col_t1:
                        st.dataframe(
                            t2_stats,
                            column_config={
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
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
                            tooltip=['2日間トレンド', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
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
                        平均翌日差枚=('next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('マイナス継続状況')
                    r_stats['信頼度'] = r_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_r1, col_r2 = st.columns([1, 1.2])
                    with col_r1:
                        st.dataframe(
                            r_stats,
                            column_config={
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
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
                            tooltip=['マイナス継続状況', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_r, width="stretch")
    
                else:
                    st.info("連続マイナス日数のデータがありません。")
    
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
                        平均翌日差枚=('next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('安定度分類')
                    stab_stats['信頼度'] = stab_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_s1, col_s2 = st.columns([1, 1.2])
                    with col_s1:
                        st.dataframe(
                            stab_stats,
                            column_config={
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
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
                            tooltip=['安定度分類', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
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
                        平均翌日差枚=('next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('BB欠損分類')
                    
                    bb_stats['信頼度'] = bb_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_bb1, col_bb2 = st.columns([1, 1.2])
                    with col_bb1:
                        st.dataframe(
                            bb_stats,
                            column_config={
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
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
                            tooltip=['BB欠損分類', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
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
                        平均翌日差枚=('next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('相対稼働率')
                    
                    rel_stats['信頼度'] = rel_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    col_rel1, col_rel2 = st.columns([1, 1.2])
                    with col_rel1:
                        st.dataframe(
                            rel_stats,
                            column_config={
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
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
                            tooltip=['相対稼働率', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                        ).interactive()
                        st.altair_chart(chart_rel, width="stretch")
    
                else:
                    st.info("相対稼働率のデータがありません。")

    # --- 7. 特徴量重要度 (Feature Importance) ---
    with st.expander("🧠 AI特徴量重要度", expanded=False):
        if df_importance is not None and not df_importance.empty:
            st.markdown("### 🧠 AIが重視したポイント (特徴量重要度)")
    
            feature_name_map = {
                '累計ゲーム': '前日: 累計ゲーム数', 'REG確率': '前日: REG確率', 'BIG確率': '前日: BIG確率',
                '差枚': '前日: 差枚数', '末尾番号': '台番号: 末尾', 'target_weekday': '予測日: 曜日',
                'target_date_end_digit': '予測日: 日付末尾 (7のつく日等)', 'weekday_avg_diff': '店舗: 曜日平均差枚',
                'mean_7days_diff': '台: 直近7日平均差枚', 'win_rate_7days': '台: 直近7日間高設定率 (一撃排除用)',
                '連続マイナス日数': '台: 連続実質マイナス日数(+500枚未満)', '連続低稼働日数': '台: 連続低稼働日数(1500G未満)', 'machine_code': '機種', 'shop_code': '店舗',
                'reg_ratio': '前日: REG比率', 'is_corner': '配置: 角台', 'neighbor_avg_diff': '配置: 両隣の平均差枚',
                'event_avg_diff': 'イベント: 平均差枚',
                'event_code': 'イベント: 種類', 'event_rank_score': 'イベント: ランク', 'prev_差枚': '前々日: 差枚数',
                'prev_REG確率': '前々日: REG確率', 'prev_累計ゲーム': '前々日: 累計ゲーム数',
                'shop_avg_diff': '店舗: 当日平均差枚', 'island_avg_diff': '島: 当日平均差枚',
                'relative_games_ratio': '台: 相対稼働率(店舗平均比)',
                'is_new_machine': '台: 新台/配置変更(導入7日以内)',
                'shop_7days_avg_diff': '店舗: 週間還元/回収モード(直近7日差枚)',
                'machine_30days_avg_diff': '機種: 機種ごとの扱い(直近30日差枚)',
                'prev_推定ぶどう確率': '前日: 推定ぶどう確率(小役)',
                'shop_avg_games': '店舗: 平均稼働ゲーム数(客層レベル)',
                'shop_abandon_rate': '店舗: 見切り台の割合(見切りスピード)'
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
    
            if not imp_shop.empty:
                # 店舗別モデルと全体モデルを比較表示
                st.caption(f"AIが【{selected_shop}】の台を予測する際に重視したデータ（左）と、全店舗共通の傾向（右）を比較します。")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**【{selected_shop}】専用モデル**")
                    chart_shop = alt.Chart(imp_shop).mark_bar(color='#AB47BC').encode(
                        x=alt.X('importance:Q', title='重要度スコア'),
                        y=alt.Y('特徴量名:N', title='特徴量', sort='-x', axis=alt.Axis(labelLimit=0)),
                        tooltip=['特徴量名', 'importance']
                    ).properties(height=500).interactive()
                    st.altair_chart(chart_shop, width="stretch")
                    
                with col2:
                    st.markdown("**【全店舗】共通モデル**")
                    if not imp_all.empty:
                        chart_all = alt.Chart(imp_all).mark_bar(color='#5C6BC0').encode(
                            x=alt.X('importance:Q', title='重要度スコア'),
                            y=alt.Y('特徴量名:N', title=None, sort='-x', axis=alt.Axis(labelLimit=0)),
                            tooltip=['特徴量名', 'importance']
                        ).properties(height=500).interactive()
                        st.altair_chart(chart_all, width="stretch")
                    else:
                        st.info("全店舗共通モデルのデータがありません。")
    
            elif not imp_all.empty:
                # 店舗別モデルがなく、全体モデルのみの場合
                st.caption(f"【{selected_shop}】専用の学習モデルはまだありません（データ不足）。代わりに全店舗共通の傾向を表示します。")
                chart_imp = alt.Chart(imp_all).mark_bar(color='#5C6BC0').encode(
                    x=alt.X('importance:Q', title='重要度スコア'),
                    y=alt.Y('特徴量名:N', title='特徴量', sort='-x', axis=alt.Axis(labelLimit=0)),
                    tooltip=['特徴量名', 'importance']
                ).properties(height=500).interactive()
                
                st.altair_chart(chart_imp, width="stretch")
        else:
            st.info("特徴量重要度のデータがありません。")