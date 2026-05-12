import streamlit as st # type: ignore
import pandas as pd
import datetime
import backend

def render_weekly_schedule_page(df_raw, df_events, df_island, shop_hyperparams):
    st.header("📅 3日間スケジュール予測")
    st.caption("指定した日から向こう3日間の各店舗の営業予測（平均差枚・期待度）をシミュレーションし、稼働予定を立てやすくします。")
    
    # ユーザーが自由に開始日を選べるようにする
    start_date = st.date_input("シミュレーション開始日", value=pd.Timestamp.now(tz='Asia/Tokyo').date())
    
    st.info("💡 **使い方:**\n「シミュレーション開始」ボタンを押すと、AIが指定日から3日間の予測を順次実行します。曜日ごとの店癖やイベント情報を考慮して、どの日にどの店舗に行くべきかのスケジュール作成に役立ててください。\n※ 予測には少し時間がかかります（数十秒〜1分程度）。")
    
    if st.button("🚀 3日間の一括予測シミュレーションを開始", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        shop_col = '店名' if '店名' in df_raw.columns else ('店舗名' if '店舗名' in df_raw.columns else None)
        
        if not shop_col:
            st.error("店舗データが見つかりません。")
            return
            
        for i in range(3):
            target_date = start_date + datetime.timedelta(days=i)
            status_text.text(f"⏳ {target_date.strftime('%Y-%m-%d')} の予測を実行中... ({i+1}/3)")
            
            # 推論実行（既存の run_analysis を再利用）
            df_pred, _, _ = backend.run_analysis(
                df_raw, _df_events=df_events, _df_island=df_island, 
                shop_hyperparams=shop_hyperparams, target_date=target_date
            )
            
            if not df_pred.empty and '予測差枚数' in df_pred.columns and 'prediction_score' in df_pred.columns:
                if 'sueoki_score' in df_pred.columns:
                    df_pred['max_score'] = df_pred[['prediction_score', 'sueoki_score']].max(axis=1)
                else:
                    df_pred['max_score'] = df_pred['prediction_score']
                    
                # 店舗ごとの平均予測差枚を計算
                shop_daily = df_pred.groupby(shop_col).agg(
                    平均予測差枚=('予測差枚数', 'mean'),
                    平均期待度=('max_score', 'mean')
                ).reset_index()
                
                # --- AI予測の平滑化問題を補正 (過去の実績トレンドを合成) ---
                if not df_raw.empty and shop_col in df_raw.columns:
                    df_raw_temp = df_raw.copy()
                    df_raw_temp['対象日付'] = pd.to_datetime(df_raw_temp['対象日付'], errors='coerce')
                    df_raw_temp['曜日'] = df_raw_temp['対象日付'].dt.dayofweek
                    df_raw_temp['末尾'] = df_raw_temp['対象日付'].dt.day % 10
                    def classify_period(d):
                        if d <= 7: return '月初'
                        elif d >= 25: return '月末'
                        else: return '中旬'
                    df_raw_temp['月内期間'] = df_raw_temp['対象日付'].dt.day.apply(classify_period)
                    
                    shop_daily_act = df_raw_temp.groupby([shop_col, '対象日付', '曜日', '末尾', '月内期間']).agg(
                        店舗平均差枚=('差枚', 'mean')
                    ).reset_index()
                    
                    for idx, row in shop_daily.iterrows():
                        s_name = row[shop_col]
                        shop_data = shop_daily_act[shop_daily_act[shop_col] == s_name]
                        if not shop_data.empty:
                            curr_wd = target_date.weekday()
                            curr_digit = target_date.day % 10
                            curr_period = classify_period(target_date.day)
                            
                            wd_diff = shop_data[shop_data['曜日'] == curr_wd]['店舗平均差枚'].mean()
                            digit_diff = shop_data[shop_data['末尾'] == curr_digit]['店舗平均差枚'].mean()
                            period_diff = shop_data[shop_data['月内期間'] == curr_period]['店舗平均差枚'].mean()
                            
                            valid_trends = {}
                            if pd.notna(wd_diff): valid_trends['weekday'] = wd_diff
                            if pd.notna(digit_diff): valid_trends['digit'] = digit_diff
                            if pd.notna(period_diff): valid_trends['period'] = period_diff
                            
                            trend_base = np.nan
                            if valid_trends:
                                strongest_trend_name = max(valid_trends, key=lambda k: abs(valid_trends[k]))
                                strongest_trend_value = valid_trends[strongest_trend_name]
                                
                                if abs(strongest_trend_value) >= 75: # Threshold for strong trend
                                    trend_base = strongest_trend_value
                                else:
                                    trend_base = sum(valid_trends.values()) / len(valid_trends)
                                
                            if pd.notna(trend_base) and pd.notna(row['平均予測差枚']):
                                shop_daily.at[idx, '平均予測差枚'] = (row['平均予測差枚'] * 0.3) + (trend_base * 0.7)
                
                shop_daily['予測対象日'] = target_date.strftime('%m/%d')
                shop_daily['曜日'] = ["月", "火", "水", "木", "金", "土", "日"][target_date.weekday()]
                shop_daily['日付(表示用)'] = shop_daily['予測対象日'] + "(" + shop_daily['曜日'] + ")"
                
                # イベント情報の取得と付与
                if df_events is not None and not df_events.empty and 'イベント日付' in df_events.columns:
                    # イベント日付と target_date を比較
                    ev_today = df_events[df_events['イベント日付'].dt.date == target_date]
                    if not ev_today.empty:
                        # 同じ店舗で複数のイベントがある場合は結合
                        ev_summary = ev_today.groupby('店名')['イベント名'].apply(lambda x: ' / '.join(x)).reset_index()
                        shop_daily = pd.merge(shop_daily, ev_summary, left_on=shop_col, right_on='店名', how='left')
                        shop_daily['イベント名'] = shop_daily['イベント名'].fillna('-')
                        if '店名_y' in shop_daily.columns:
                            shop_daily = shop_daily.drop(columns=['店名_y'])
                    else:
                        shop_daily['イベント名'] = '-'
                else:
                    shop_daily['イベント名'] = '-'
                    
                results.append(shop_daily)
                
            progress_bar.progress((i + 1) / 3)
            
        status_text.text("✅ すべての予測が完了しました！")
        
        if results:
            df_results = pd.concat(results, ignore_index=True)
            
            st.divider()
            st.subheader("📊 店舗別 予測平均差枚スケジュール")
            st.caption("数値がプラス（緑色）の日は店舗全体が還元傾向、マイナス（赤色）の日は回収傾向の予測です。")
            
            # ピボットテーブルにして、縦軸を店舗、横軸を日付にする（平均差枚）
            pivot_diff = df_results.pivot(index=shop_col, columns='日付(表示用)', values='平均予測差枚').round(0)
            
            # カラム（日付）を正しい順序に並び替え
            cols_order = []
            for i in range(3):
                d = start_date + datetime.timedelta(days=i)
                cols_order.append(d.strftime('%m/%d') + "(" + ["月", "火", "水", "木", "金", "土", "日"][d.weekday()] + ")")
            
            valid_cols = [c for c in cols_order if c in pivot_diff.columns]
            pivot_diff = pivot_diff[valid_cols]
            
            # 条件付き書式で色付け (マイナスは赤、プラスは緑)
            st.dataframe(
                pivot_diff.style.background_gradient(cmap='RdYlGn', axis=None, vmin=-250, vmax=250).format("{:.0f} 枚"),
                use_container_width=True
            )
            
            st.divider()
            st.subheader("📝 日別 イベント詳細")
            
            display_df = df_results[['日付(表示用)', shop_col, '平均予測差枚', 'イベント名']].copy()
            display_df = display_df.sort_values(['日付(表示用)', '平均予測差枚'], ascending=[True, False])
            st.dataframe(display_df, use_container_width=True, hide_index=True, column_config={"平均予測差枚": st.column_config.NumberColumn("平均差枚", format="%.0f 枚")})