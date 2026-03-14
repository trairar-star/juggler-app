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
def render_summary_page(df, df_raw, shop_col, df_events=None):
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
        
        # --- 月間トレンド分析 (全店比較) ---
        if not df_raw.empty and '対象日付' in df_raw.columns:
            st.subheader("📈 店舗別の強さ推移 (月間トレンド)")
            st.caption("各店舗の「高設定挙動台（REG確率 1/300以上）」の投入割合が、月ごとにどう変化しているかを分析します。最近勢いのある店舗を見つけるのに役立ちます。")

            trend_df = df_raw.copy()
            
            # --- 絞り込みフィルターUI ---
            f_col1, f_col2 = st.columns(2)
            filter_type = f_col1.selectbox("集計対象の絞り込み", ["すべて", "曜日で絞る", "イベント日で絞る"])
            
            if filter_type == "曜日で絞る":
                day_options = ["平日 (月〜金)", "週末 (土・日)", "月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"]
                selected_day = f_col2.selectbox("曜日を選択", day_options)
                trend_df['weekday'] = trend_df['対象日付'].dt.dayofweek
                if selected_day == "平日 (月〜金)": trend_df = trend_df[trend_df['weekday'] < 5]
                elif selected_day == "週末 (土・日)": trend_df = trend_df[trend_df['weekday'] >= 5]
                else:
                    day_idx = ["月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"].index(selected_day)
                    trend_df = trend_df[trend_df['weekday'] == day_idx]
                    
            elif filter_type == "イベント日で絞る":
                if df_events is not None and not df_events.empty:
                    rank_options = ["すべてのイベント"] + [r for r in ["S", "A", "B", "C"] if r in df_events['イベントランク'].values]
                    selected_rank = f_col2.selectbox("イベントランク", rank_options)
                    events_subset = df_events.drop_duplicates(subset=['店名', 'イベント日付']).copy()
                    if selected_rank != "すべてのイベント":
                        events_subset = events_subset[events_subset['イベントランク'] == selected_rank]
                    trend_df = pd.merge(trend_df, events_subset[['店名', 'イベント日付']], left_on=[shop_col, '対象日付'], right_on=['店名', 'イベント日付'], how='inner')
                else:
                    f_col2.info("登録されたイベントがありません")
                    trend_df = pd.DataFrame()

            if trend_df.empty:
                st.warning("指定した条件に一致するデータがありません。")
            else:
                trend_df['年月'] = trend_df['対象日付'].dt.strftime('%Y-%m')
                # REG確率が1/300 (約0.00333) 以上を高設定挙動と定義
                trend_df['高設定'] = (trend_df['REG確率'] >= (1/300)).astype(int)

                trend_stats = trend_df.groupby(['年月', shop_col]).agg(
                    高設定投入率=('高設定', 'mean'),
                    平均差枚=('差枚', 'mean'),
                    集計台数=('台番号', 'count')
                ).reset_index()

                trend_chart = alt.Chart(trend_stats).mark_line(point=True, strokeWidth=3).encode(
                    x=alt.X('年月', title='年月'),
                    y=alt.Y('高設定投入率', title='高設定投入率 (REG 1/300以上)', axis=alt.Axis(format='%')),
                    color=alt.Color(f'{shop_col}:N', title='店舗名'),
                    tooltip=['年月', shop_col, alt.Tooltip('高設定投入率', format='.1%'), alt.Tooltip('平均差枚', format='+.0f'), '集計台数']
                ).interactive()

                st.altair_chart(trend_chart, use_container_width=True)
                
                # --- 最新月の店舗別成績一覧 ---
                st.divider()
                latest_month = trend_df['年月'].max()
                st.markdown(f"**🏅 {latest_month} の店舗別成績 (絞り込み適用)**")
                
                latest_month_df = trend_df[trend_df['年月'] == latest_month]
                latest_stats = latest_month_df.groupby(shop_col).agg(
                    平均差枚=('差枚', 'mean'),
                    勝率=('差枚', lambda x: (x > 0).mean()),
                    集計台数=('台番号', 'count')
                ).reset_index().sort_values('平均差枚', ascending=False)
                
                st.dataframe(
                    latest_stats,
                    column_config={
                        shop_col: st.column_config.TextColumn("店舗名"),
                        "平均差枚": st.column_config.NumberColumn("平均差枚", format="%+d 枚"),
                        "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=1),
                        "集計台数": st.column_config.NumberColumn("集計台数", format="%d 台")
                    },
                    use_container_width=True,
                    hide_index=True
                )

# --- ページ描画関数: 店舗別詳細データ ---
def render_shop_detail_page(df, df_raw, shop_col, df_events=None, df_train=None):
    st.header("🏪 店舗別 詳細データ")
    
    # 店舗・機種選択 (メイン画面上部)
    col_filter1, col_filter2 = st.columns(2)
    
    selected_shop = '全て'
    if shop_col in df.columns:
        shops = ['全て'] + list(df[shop_col].unique())
        selected_shop = col_filter1.selectbox("店舗名を選択", shops)
        
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

    # 機種名フィルター
    if '機種名' in df.columns:
        machines = ['全て'] + list(df['機種名'].unique())
        selected_machine = col_filter2.selectbox("機種名を選択", machines)
        if selected_machine != '全て':
            df = df[df['機種名'] == selected_machine]

    # --- 店癖トップ5の抽出と明日の候補台へのマッピング ---
    top_trends_df = None
    base_win_rate = 0
    if df_train is not None and not df_train.empty and selected_shop != '全て':
        train_shop = df_train[df_train[shop_col] == selected_shop]
        if len(train_shop) > 0:
            base_win_rate = train_shop['target'].mean()
            trends = []
            
            if 'is_corner' in train_shop.columns:
                subset = train_shop[train_shop['is_corner'] == 1]
                if len(subset) >= 5: trends.append({"id": "corner", "条件": "角台", "勝率": subset['target'].mean(), "サンプル": len(subset)})
            if 'REG' in train_shop.columns and 'BIG' in train_shop.columns and 'REG確率' in train_shop.columns:
                subset = train_shop[(train_shop['REG'] > train_shop['BIG']) & (train_shop['REG確率'] >= (1/300))]
                if len(subset) >= 5: trends.append({"id": "reg_lead", "条件": "REG先行 (高設定不発狙い)", "勝率": subset['target'].mean(), "サンプル": len(subset)})
            if '連続マイナス日数' in train_shop.columns:
                subset = train_shop[train_shop['連続マイナス日数'] >= 3]
                if len(subset) >= 5: trends.append({"id": "cons_minus", "条件": "3日以上連続凹み (上げリセット狙い)", "勝率": subset['target'].mean(), "サンプル": len(subset)})
            if '差枚' in train_shop.columns:
                subset = train_shop[train_shop['差枚'] <= -1000]
                if len(subset) >= 5: trends.append({"id": "prev_lose", "条件": "前日大負け (-1000枚以下) からの反発", "勝率": subset['target'].mean(), "サンプル": len(subset)})
                subset = train_shop[train_shop['差枚'] >= 1000]
                if len(subset) >= 5: trends.append({"id": "prev_win", "条件": "前日大勝ち (+1000枚以上) の据え置き", "勝率": subset['target'].mean(), "サンプル": len(subset)})
            if '対象日付' in train_shop.columns:
                target_dates = train_shop['対象日付'] + pd.Timedelta(days=1)
                for d in [0, 5, 7]:
                    subset = train_shop[target_dates.dt.day % 10 == d]
                    if len(subset) >= 5: trends.append({"id": f"day_{d}", "条件": f"{d}のつく日 (予測日)", "勝率": subset['target'].mean(), "サンプル": len(subset)})
            if '末尾番号' in train_shop.columns:
                best_m, best_wr, best_count = -1, 0, 0
                for m in range(10):
                    subset = train_shop[train_shop['末尾番号'] == m]
                    if len(subset) >= 10:
                        wr = subset['target'].mean()
                        if wr > best_wr: best_m, best_wr, best_count = m, wr, len(subset)
                if best_m != -1: trends.append({"id": f"end_{int(best_m)}", "条件": f"末尾【{int(best_m)}】", "勝率": best_wr, "サンプル": best_count})

            df['店癖マッチ'] = ""
            if trends:
                top_trends_df = pd.DataFrame(trends).sort_values('勝率', ascending=False).head(5)
                top_trends_df['通常時との差'] = (top_trends_df['勝率'] - base_win_rate) * 100
                top_ids = top_trends_df['id'].tolist()
                
                def get_matched_trends(row):
                    matched = []
                    if "corner" in top_ids and row.get('is_corner') == 1: matched.append("角")
                    if "reg_lead" in top_ids and row.get('REG', 0) > row.get('BIG', 0) and row.get('REG確率', 0) >= (1/300): matched.append("REG先行")
                    if "cons_minus" in top_ids and row.get('連続マイナス日数', 0) >= 3: matched.append("連凹")
                    if "prev_lose" in top_ids and row.get('差枚', 0) <= -1000: matched.append("負反発")
                    if "prev_win" in top_ids and row.get('差枚', 0) >= 1000: matched.append("勝据え")
                    for tid in top_ids:
                        if tid.startswith("day_") and pd.notna(row.get('対象日付')):
                            if (row['対象日付'] + pd.Timedelta(days=1)).day % 10 == int(tid.split("_")[1]): matched.append(f"{int(tid.split('_')[1])}のつく日")
                        elif tid.startswith("end_") and row.get('末尾番号') == int(tid.split("_")[1]): matched.append(f"末尾{int(tid.split('_')[1])}")
                    return "🔥" + " ".join(matched) if matched else ""
                df['店癖マッチ'] = df.apply(get_matched_trends, axis=1)

                # --- 激アツ条件によるスコア加算 (店癖ボーナス) ---
                def apply_bonus(row):
                    score = row.get('prediction_score', 0)
                    match_str = row.get('店癖マッチ', '')
                    if match_str.startswith('🔥'):
                        # マッチした条件1つにつき +0.05 (5%) のボーナスを加算
                        bonus = 0.05 * len(match_str.replace('🔥', '').strip().split())
                        score = min(1.0, score + bonus) # 上限は1.0 (100%)
                    return score
                
                if 'prediction_score' in df.columns:
                    df['prediction_score'] = df.apply(apply_bonus, axis=1)
                    # ボーナス加算後のスコアで「おすすめ度」を再評価
                    def update_rating(score):
                        if score >= 0.8: return 'A'
                        elif score >= 0.6: return 'B'
                        elif score >= 0.4: return 'C'
                        elif score >= 0.2: return 'D'
                        else: return 'E'
                    df['おすすめ度'] = df['prediction_score'].apply(update_rating)

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
    base_cols = ['台番号', '機種名', '店癖マッチ', 'おすすめ度', '予測差枚数']
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
            "店癖マッチ": st.column_config.TextColumn("激アツ条件", width="medium"),
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
                ].sort_values('対象日付').tail(7).copy()
                
                if not history_df.empty:
                    history_df['DisplayDate'] = history_df['対象日付'].dt.strftime('%m-%d')
                    
                    # イベント情報を結合
                    if df_events is not None and not df_events.empty:
                        shop_events = df_events[df_events['店名'] == shop_name].drop_duplicates(subset=['イベント日付'])
                        history_df = pd.merge(history_df, shop_events[['イベント日付', 'イベント名', 'イベントランク']], left_on='対象日付', right_on='イベント日付', how='left')
                        history_df['イベント情報'] = history_df.apply(lambda r: f"{r['イベント名']} ({r.get('イベントランク', '-')})" if pd.notna(r['イベント名']) and str(r['イベント名']).strip() != '' else "なし", axis=1)
                    else:
                        history_df['イベント情報'] = "なし"
                    
                    base = alt.Chart(history_df).encode(
                        x=alt.X('DisplayDate', title='日付', sort=None),
                        y=alt.Y('差枚', title='差枚数'),
                        tooltip=[
                            alt.Tooltip('DisplayDate', title='日付'),
                            alt.Tooltip('差枚', title='差枚'),
                            alt.Tooltip('イベント情報', title='イベント'),
                            alt.Tooltip('BIG', title='BIG回数'),
                            alt.Tooltip('REG', title='REG回数'),
                            alt.Tooltip('累計ゲーム', title='総回転数')
                        ]
                    )
                    
                    line_chart = base.mark_line(point=True)
                    event_points = base.transform_filter(alt.datum.イベント情報 != 'なし').mark_point(color='#FF4B4B', size=150, filled=True)
                    
                    st.altair_chart((line_chart + event_points).interactive(), use_container_width=True)
                else:
                    st.caption("過去データが見つかりませんでした。")

    # --- 店舗別 傾向分析 (下部に移動) ---
    if selected_shop != '全て' and not df_raw_shop.empty:
        st.divider()
        st.subheader(f"📅 {selected_shop} の傾向分析")
        st.caption("過去データに基づく、この店舗のイベント日や曜日ごとの平均差枚数です。")
        
        # --- 🤖 AIが発見した店癖トップ5 ---
        if top_trends_df is not None and not top_trends_df.empty:
            st.markdown(f"**🤖 AIが発見した {selected_shop} の店癖 トップ5**")
            st.caption("AIが過去データから見つけた、この店舗で特に勝率が高い特定の条件（店癖）です。ランキングの「🔥」マークはこの条件に合致しています。")
            
            st.dataframe(
                top_trends_df,
                        column_config={
                            "条件": st.column_config.TextColumn("激アツ条件 (店癖)"),
                            "勝率": st.column_config.ProgressColumn("条件合致時の勝率", format="%.2f", min_value=0, max_value=1),
                            "通常時との差": st.column_config.NumberColumn("通常時との差", format="%+.1f pt"),
                            "サンプル": st.column_config.NumberColumn("サンプル", format="%d 台")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
            st.caption(f"※この店舗の通常時の平均勝率は **{base_win_rate:.1%}** です。")
        
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
def render_feature_analysis_page(df_train, df_importance=None):
    st.header("🔬 AI学習データ分析 (勝利の法則)")
    st.caption("過去の全データから、「勝った台（翌日プラス差枚）」と「負けた台」の傾向を分析し、勝ちやすい台の特徴を可視化します。")

    if df_train.empty:
        st.warning("分析可能な過去データがありません。")
        return
    
    # データ準備
    analysis_df = df_train.copy()
    shop_col = '店名' if '店名' in analysis_df.columns else ('店舗名' if '店舗名' in analysis_df.columns else None)
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

    # --- 5. 特殊パターンの検証 (REG先行・大凹み・大勝) ---
    st.divider()
    st.subheader("🕵️‍♂️ 特殊パターンの検証 (REG先行・凹み反発・大勝のその後)")
    st.caption(f"指定した回転数（{min_g}G）以上回っている台を対象に、スロット特有のパターンの翌日の成績を調査します。")

    if not reg_df.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["① REG先行 (BIG欠損)", "② 大凹み vs 大勝", "③ 2日間のトレンド", "④ 上げリセット傾向(連続凹み)"])

        with tab1:
            st.markdown("**🔍 REG先行 (BIG欠損) 台の翌日成績**")
            st.caption("BIGが引けなかったがREGは付いてきている台が、翌日「高設定の不発」として出る(底上げされる)のか検証します。")
            
            if 'BIG' in reg_df.columns and 'REG' in reg_df.columns:
                reg_lead_df = reg_df.copy()
                def classify_reg_lead(row):
                    if row['REG'] > row['BIG']:
                        if row.get('REG分母', 9999) <= 300:
                            return "REG先行 & REG確率1/300以上 (高設定不発候補)"
                        else:
                            return "REG先行 & REG確率1/300未満"
                    else:
                        return "BIG先行 または 同数"
                
                reg_lead_df['REG先行分類'] = reg_lead_df.apply(classify_reg_lead, axis=1)
                
                rl_stats = reg_lead_df.groupby('REG先行分類').agg(
                    翌日勝率=('target', 'mean'),
                    平均翌日差枚=('next_diff', 'mean'),
                    サンプル数=('target', 'count')
                ).reset_index()

                st.dataframe(
                    rl_stats,
                    column_config={
                        "翌日勝率": st.column_config.ProgressColumn("翌日勝率", format="%.1f%%", min_value=0, max_value=1),
                        "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                    },
                    hide_index=True,
                    use_container_width=True
                )

                if shop_col:
                    st.divider()
                    st.markdown("**🏬 店舗別: 高設定不発候補 (REG先行&1/300以上) の翌日成績**")
                    st.caption("このパターンが発生したとき、どの店舗が一番底上げ（上げ狙い成功）しやすいかを比較します。")
                    target_reg_df = reg_lead_df[reg_lead_df['REG先行分類'] == "REG先行 & REG確率1/300以上 (高設定不発候補)"]
                    if not target_reg_df.empty:
                        shop_reg_stats = target_reg_df.groupby(shop_col).agg(
                            翌日勝率=('target', 'mean'),
                            平均翌日差枚=('next_diff', 'mean'),
                            サンプル数=('target', 'count')
                        ).reset_index().sort_values('平均翌日差枚', ascending=False)
                        
                        st.dataframe(
                            shop_reg_stats,
                            column_config={
                                shop_col: st.column_config.TextColumn("店舗名"),
                                "翌日勝率": st.column_config.ProgressColumn("翌日勝率", format="%.1f%%", min_value=0, max_value=1),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
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
                翌日勝率=('target', 'mean'),
                平均翌日差枚=('next_diff', 'mean'),
                サンプル数=('target', 'count')
            ).reset_index().sort_values('前日結果')

            col_dp1, col_dp2 = st.columns([1, 1.2])
            with col_dp1:
                st.dataframe(
                    dp_stats,
                    column_config={
                        "翌日勝率": st.column_config.ProgressColumn("翌日勝率", format="%.1f%%", min_value=0, max_value=1),
                        "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
            with col_dp2:
                chart_dp = alt.Chart(dp_stats).mark_bar().encode(
                    x=alt.X('前日結果', title='前日差枚'),
                    y=alt.Y('平均翌日差枚', title='平均翌日差枚 (枚)'),
                    color=alt.condition(alt.datum.平均翌日差枚 > 0, alt.value("#FF7043"), alt.value("#42A5F5")),
                    tooltip=['前日結果', alt.Tooltip('翌日勝率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数']
                ).interactive()
                st.altair_chart(chart_dp, use_container_width=True)

            if shop_col:
                st.divider()
                st.markdown("**🏬 店舗別: パターン別翌日成績の比較**")
                
                patterns = sorted(diff_pat_df['前日結果'].unique())
                selected_pattern = st.selectbox(
                    "比較する前日パターンを選択して、店舗ごとの扱いを確認",
                    patterns,
                    index=0
                )
                
                target_dp_df = diff_pat_df[diff_pat_df['前日結果'] == selected_pattern]
                if not target_dp_df.empty:
                    shop_dp_stats = target_dp_df.groupby(shop_col).agg(
                        翌日勝率=('target', 'mean'),
                        平均翌日差枚=('next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('平均翌日差枚', ascending=False)
                    
                    st.dataframe(
                        shop_dp_stats,
                        column_config={
                            shop_col: st.column_config.TextColumn("店舗名"),
                            "翌日勝率": st.column_config.ProgressColumn("翌日勝率", format="%.1f%%", min_value=0, max_value=1),
                            "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            "サンプル数": st.column_config.NumberColumn("サンプル数", format="%d 台")
                        },
                        hide_index=True,
                        use_container_width=True
                    )

        with tab3:
            st.markdown("**🔍 2日間の差枚トレンド (連勝・連敗・V字回復)**")
            st.caption("前々日と前日の差枚パターンから、翌日の勝率（反発しやすいか、据え置かれやすいか）を検証します。")
            
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
                    翌日勝率=('target', 'mean'),
                    平均翌日差枚=('next_diff', 'mean'),
                    サンプル数=('target', 'count')
                ).reset_index().sort_values('2日間トレンド')
                
                col_t1, col_t2 = st.columns([1, 1.2])
                with col_t1:
                    st.dataframe(
                        t2_stats,
                        column_config={
                            "翌日勝率": st.column_config.ProgressColumn("翌日勝率", format="%.1f%%", min_value=0, max_value=1),
                            "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                with col_t2:
                    chart_t2 = alt.Chart(t2_stats).mark_bar().encode(
                        x=alt.X('2日間トレンド', title='2日間の成績パターン'),
                        y=alt.Y('平均翌日差枚', title='平均翌日差枚 (枚)'),
                        color=alt.condition(alt.datum.平均翌日差枚 > 0, alt.value("#FF7043"), alt.value("#42A5F5")),
                        tooltip=['2日間トレンド', alt.Tooltip('翌日勝率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数']
                    ).interactive()
                    st.altair_chart(chart_t2, use_container_width=True)

                if shop_col:
                    st.divider()
                    st.markdown("**🏬 店舗別: 2日間トレンド別翌日成績の比較**")
                    
                    t2_patterns = sorted(trend2d_df['2日間トレンド'].unique())
                    selected_t2_pattern = st.selectbox(
                        "比較する2日間のパターンを選択",
                        t2_patterns,
                        index=0,
                        key="select_t2_pattern"
                    )
                    
                    target_t2_df = trend2d_df[trend2d_df['2日間トレンド'] == selected_t2_pattern]
                    if not target_t2_df.empty:
                        shop_t2_stats = target_t2_df.groupby(shop_col).agg(
                            翌日勝率=('target', 'mean'),
                            平均翌日差枚=('next_diff', 'mean'),
                            サンプル数=('target', 'count')
                        ).reset_index().sort_values('平均翌日差枚', ascending=False)
                        
                        st.dataframe(
                            shop_t2_stats,
                            column_config={
                                shop_col: st.column_config.TextColumn("店舗名"),
                                "翌日勝率": st.column_config.ProgressColumn("翌日勝率", format="%.1f%%", min_value=0, max_value=1),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                                "サンプル数": st.column_config.NumberColumn("サンプル数", format="%d 台")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
            else:
                st.info("前々日の差枚データが不足しています。")

        with tab4:
            st.markdown("**🔍 連続マイナス台の「上げリセット」検証**")
            st.caption("何日間マイナスが続くと「上げリセット（設定変更）」されやすくなるのか、店舗ごとの見切りラインを検証します。")
            
            if '連続マイナス日数' in reg_df.columns:
                reset_df = reg_df.copy()
                
                def classify_cons_minus(d):
                    d = int(d)
                    if d == 0: return "① 0日 (前日プラス)"
                    elif d == 1: return "② 1日マイナス"
                    elif d == 2: return "③ 2日連続マイナス"
                    elif d == 3: return "④ 3日連続マイナス"
                    elif d >= 4: return "⑤ 4日以上連続マイナス"
                    return "不明"
                    
                reset_df['マイナス継続状況'] = reset_df['連続マイナス日数'].apply(classify_cons_minus)
                
                r_stats = reset_df.groupby('マイナス継続状況').agg(
                    翌日勝率=('target', 'mean'),
                    平均翌日差枚=('next_diff', 'mean'),
                    サンプル数=('target', 'count')
                ).reset_index().sort_values('マイナス継続状況')
                
                st.dataframe(
                    r_stats,
                    column_config={
                        "翌日勝率": st.column_config.ProgressColumn("翌日勝率", format="%.1f%%", min_value=0, max_value=1),
                        "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                    },
                    hide_index=True,
                    use_container_width=True
                )

                if shop_col:
                    st.divider()
                    st.markdown("**🏬 店舗別: 3日以上連続マイナス台の『上げリセット期待度』**")
                    
                    target_reset_df = reset_df[reset_df['連続マイナス日数'] >= 3]
                    if not target_reset_df.empty:
                        shop_reset_stats = target_reset_df.groupby(shop_col).agg(
                            翌日勝率=('target', 'mean'),
                            平均翌日差枚=('next_diff', 'mean'),
                            サンプル数=('target', 'count')
                        ).reset_index().sort_values('翌日勝率', ascending=False)
                        
                        st.dataframe(
                            shop_reset_stats,
                            column_config={
                                shop_col: st.column_config.TextColumn("店舗名"),
                                "翌日勝率": st.column_config.ProgressColumn("リセット(上げ)期待度", format="%.1f%%", min_value=0, max_value=1),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                                "サンプル数": st.column_config.NumberColumn("サンプル数", format="%d 台")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("3日以上連続マイナスのデータがまだありません。")
            else:
                st.info("連続マイナス日数のデータがありません。")

    # --- 6. 特徴量重要度 (Feature Importance) ---
    if df_importance is not None and not df_importance.empty:
        st.divider()
        st.subheader("🧠 AIが重視したポイント (特徴量重要度)")
        st.caption("AIが明日の勝敗を予測する上で、どのデータ（特徴量）を一番重要視したかを示します。")
        
        feature_name_map = {
            '累計ゲーム': '前日: 累計ゲーム数',
            'REG確率': '前日: REG確率',
            'BIG確率': '前日: BIG確率',
            '差枚': '前日: 差枚数',
            '末尾番号': '台番号: 末尾',
            'weekday': '日付: 曜日',
            'weekday_avg_diff': '店舗: 曜日平均差枚',
            'mean_7days_diff': '台: 直近7日平均差枚',
            'mean_14days_diff': '台: 直近14日平均差枚',
            'mean_30days_diff': '台: 直近30日平均差枚',
            '連続マイナス日数': '台: 連続マイナス日数',
            'machine_code': '機種',
            'shop_code': '店舗',
            'reg_ratio': '前日: REG比率',
            'is_corner': '配置: 角台',
            'neighbor_avg_diff': '配置: 両隣の平均差枚',
            'event_avg_diff': 'イベント: 平均差枚',
            'prev_最終ゲーム': '前々日: 最終ゲーム数',
            'event_code': 'イベント: 種類',
            'event_rank_score': 'イベント: ランク',
            'prev_差枚': '前々日: 差枚数',
            'prev_REG確率': '前々日: REG確率',
            'prev_累計ゲーム': '前々日: 累計ゲーム数',
            'shop_avg_diff': '店舗: 当日平均差枚',
            'island_avg_diff': '島: 当日平均差枚'
        }
        
        display_importance = df_importance.copy()
        display_importance['特徴量名'] = display_importance['feature'].map(lambda x: feature_name_map.get(x, x))
        
        chart_imp = alt.Chart(display_importance).mark_bar(color='#AB47BC').encode(
            x=alt.X('importance:Q', title='重要度スコア'),
            y=alt.Y('特徴量名:N', title='特徴量', sort='-x'),
            tooltip=['特徴量名', 'importance']
        ).properties(height=500).interactive()
        
        st.altair_chart(chart_imp, use_container_width=True)

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
    st.subheader("✏️ イベント編集")
    
    edit_target_uid = st.selectbox("編集するイベントを選択", df_display['uid'].unique(), key="edit_target")
    if edit_target_uid:
        target_row = df_display[df_display['uid'] == edit_target_uid].iloc[0]
        
        with st.form("edit_event_form"):
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                edit_shop = st.text_input("店舗名", value=target_row['店名'])
                try:
                    default_date = pd.to_datetime(target_row['イベント日付']).date()
                except:
                    default_date = pd.Timestamp.now().date()
                edit_date = st.date_input("イベント日付", value=default_date, key="edit_date")
            with e_col2:
                edit_name = st.text_input("イベント名", value=target_row['イベント名'])
                rank_options = ["S", "A", "B", "C"]
                current_rank = target_row.get('イベントランク', 'A')
                idx = rank_options.index(current_rank) if current_rank in rank_options else 1
                edit_rank = st.selectbox("イベントの強さ", rank_options, index=idx)
                
            if st.form_submit_button("更新を保存", type="primary"):
                if backend.update_shop_event(target_row['店名'], default_date, target_row['イベント名'], edit_shop, edit_date, edit_name, edit_rank):
                    st.success("イベントを更新しました！")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("イベントの更新に失敗しました。")

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

# --- ページ描画関数: 島マスター管理 ---
def render_island_master_page(df_raw):
    st.header("🗺️ 島マスター管理 (角・並び設定)")
    st.caption("台が属する「島（列）」を登録することで、AIが正確な『角台』や『通路を跨がない隣台（並び）』を認識できるようになり、精度が劇的に向上します。")
    
    df_island = backend.load_island_master()
    
    with st.expander("📝 新しい島（列）を登録", expanded=True):
        st.info("💡 **登録のコツ**: ハイフン（範囲）とカンマ（区切り）を組み合わせて、複雑な形でも指定できます。\n\n例1: `501-510` (一般的な10台島)\n例2: `785-786, 880-885` (特殊な並びや飛び番)")
        with st.form("island_form", clear_on_submit=True):
            shops = []
            if not df_raw.empty:
                shop_col = '店名' if '店名' in df_raw.columns else '店舗名'
                if shop_col in df_raw.columns:
                    shops = list(df_raw[shop_col].unique())
            input_shop = st.selectbox("店舗名", shops)
            input_island = st.text_input("島名 (例: マイジャグA列)", placeholder="マイジャグA列")
            input_rule = st.text_input("対象台番号 (例: 501-510 または 786, 880-885)", placeholder="501-510, 786, 880")
            
            submitted = st.form_submit_button("島を登録", type="primary")
            if submitted:
                if not input_island:
                    st.error("島名を入力してください。")
                elif not input_rule:
                    st.error("対象台番号を入力してください。")
                else:
                    if backend.save_island_master(input_shop, input_island, input_rule):
                        st.success(f"{input_shop}の島マスターを登録しました！")
                        st.cache_data.clear()
                        st.rerun()

    if not df_island.empty:
        st.subheader("📋 登録済みの島一覧")
        
        def get_display_rule(r):
            rule = r.get('台番号ルール', '')
            if pd.notna(rule) and str(rule).strip() != '':
                return str(rule)
            s = r.get('開始台番号', '')
            e = r.get('終了台番号', '')
            return f"{s}〜{e}"
            
        df_island['対象台番号'] = df_island.apply(get_display_rule, axis=1)
        df_island['uid_label'] = df_island['店名'].astype(str) + " | " + df_island['島名'].astype(str) + " (" + df_island['対象台番号'].astype(str) + ")"
        st.dataframe(df_island[['店名', '島名', '対象台番号']], use_container_width=True, hide_index=True)
        
        with st.form("delete_island_form"):
            target = st.selectbox("削除する島を選択", df_island['登録日時'].unique(), format_func=lambda x: df_island[df_island['登録日時']==x].iloc[0]['uid_label'])
            if st.form_submit_button("削除"):
                if backend.delete_island_master(target):
                    st.success("削除しました。")
                    st.cache_data.clear()
                    st.rerun()

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
                    except: default_date = pd.Timestamp.now().date()
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
        ["店舗別詳細データ", "全店分析サマリー", "AI傾向分析 (勝利の法則)", "精度検証 (答え合わせ)", "島マスター管理", "イベント管理", "💰 マイ収支管理"]
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
    df_island = backend.load_island_master()

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
        df, df_verify, df_importance = backend.run_analysis(df_raw, df_events, df_island, hyperparams)
    
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
        render_summary_page(df, df_raw, shop_col, df_events)
    elif page == "精度検証 (答え合わせ)":
        # 予測ログをロードして渡す
        df_log = backend.load_prediction_log()
        render_verification_page(df_log, df_raw)
    elif page == "AI傾向分析 (勝利の法則)":
        render_feature_analysis_page(df_verify, df_importance)
    elif page == "島マスター管理":
        render_island_master_page(df_raw)
    elif page == "イベント管理":
        render_event_management_page()
    elif page == "💰 マイ収支管理":
        render_my_balance_page(df_raw)
    else:
        render_shop_detail_page(df, df_raw, shop_col, df_events, df_verify)

if __name__ == "__main__":
    main()