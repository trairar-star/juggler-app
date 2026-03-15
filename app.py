import os
import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import matplotlib.pyplot as plt # type: ignore
import altair as alt # type: ignore

# バックエンド処理をインポート
import backend
from views import shop_detail_page
from utils import get_confidence_indicator

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
            平均スコア=('prediction_score', 'mean'),
            推奨台数=('prediction_score', lambda x: (x >= 0.70).sum()),
            全台数=('台番号', 'nunique')
        ).reset_index()
        
        # 平均スコアが高い順にソート
        shop_stats = shop_stats.sort_values('平均スコア', ascending=False)
        
        st.dataframe(
            shop_stats,
            column_config={
                shop_col: st.column_config.TextColumn("店舗"),
                "平均スコア": st.column_config.ProgressColumn("期待度", min_value=0, max_value=1.0, format="%.2f", help="店舗全体の平均的な設定5以上確率"),
                "推奨台数": st.column_config.NumberColumn("推奨", format="%d台", help="AI期待度が70%以上の台数"),
                "全台数": st.column_config.NumberColumn("全台", format="%d台"),
            },
            use_container_width=True,
            hide_index=True
        )
        st.divider()
        
        # --- 店舗の質 vs 量 分析 (散布図) ---
        st.subheader("📊 店舗の質 vs 量 分析")
        st.caption("横軸が「推奨台数（量）」、縦軸が「平均設定5以上確率（質）」です。右上に位置するほど優良店と判断できます。")

        if not shop_stats.empty:
            # 散布図用に店舗名をインデックスに設定
            chart_data = shop_stats.set_index(shop_col)
            st.scatter_chart(
                chart_data,
                x='推奨台数',
                y='平均スコア',
                size='全台数',
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
                # --- 低回転ノイズの除外 ---
                # 分母が膨らんで確率が下がるのを防ぐため、1000G未満の未稼働・即ヤメ台は集計から除外する
                trend_df = trend_df[trend_df['累計ゲーム'] >= 1000].copy()
                
                trend_df['年月'] = trend_df['対象日付'].dt.strftime('%Y-%m')
                
                # 機種別の設定5基準(REGまたは合算)で高設定挙動を定義
                specs = backend.get_machine_specs()
                spec_reg = trend_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
                spec_tot = trend_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
                trend_df['合算確率'] = (trend_df['BIG'] + trend_df['REG']) / trend_df['累計ゲーム'].replace(0, np.nan)
                
                # 高設定の判定には「3000G以上回っていること」を条件に加え、低回転での上振れノイズを排除
                trend_df['高設定'] = ((trend_df['累計ゲーム'] >= 3000) & ((trend_df['REG確率'] >= spec_reg) | (trend_df['合算確率'] >= spec_tot))).astype(int)

                trend_stats = trend_df.groupby(['年月', shop_col]).agg(
                    高設定投入率=('高設定', 'mean'),
                    平均差枚=('差枚', 'mean'),
                    集計台数=('台番号', 'count')
                ).reset_index()

                trend_chart = alt.Chart(trend_stats).mark_line(point=True, strokeWidth=3).encode(
                    x=alt.X('年月', title='年月'),
                    y=alt.Y('高設定投入率', title='高設定投入率 (設定5基準)', axis=alt.Axis(format='%')),
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
                    高設定率=('高設定', 'mean'),
                    集計台数=('台番号', 'count')
                ).reset_index().sort_values('高設定率', ascending=False)
                
                st.dataframe(
                    latest_stats,
                    column_config={
                        shop_col: st.column_config.TextColumn("店舗"),
                        "平均差枚": st.column_config.NumberColumn("差枚", format="%+d枚", help="平均差枚数"),
                        "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=1),
                        "集計台数": st.column_config.NumberColumn("台数", format="%d台")
                    },
                    use_container_width=True,
                    hide_index=True
                )

# --- ページ描画関数: 精度検証 (答え合わせ) ---
def render_verification_page(df_pred_log, df_verify, df_predict, df_raw):
    st.header("✅ 精度検証 (保存データの答え合わせ)")
    st.caption("あなたが保存した過去の「予測結果ログ」と、実際の結果データを照合して、当時のAIの精度を検証します。")

    # --- 検証・評価設定 (サイドバー) ---
    with st.sidebar.expander("🎯 検証・評価設定", expanded=False):
        st.caption("設定5近似度を計算する際の減点ルールを調整できます。")
        penalty_reg = st.slider("REG 1回不足ごとの減点", min_value=0, max_value=50, value=15, step=1)
        penalty_big = st.slider("BIG 1回不足ごとの減点", min_value=0, max_value=50, value=5, step=1)
        low_g_penalty = st.slider("低稼働(1000G未満)の最大減点率(%)", min_value=0, max_value=100, value=30, step=5)

    if df_pred_log.empty:
        st.warning("保存された予測結果ログがありません。サイドバーの「予測結果をログ保存」ボタンから予測を保存してください。")
        return

    # --- 予測ログの実行日時フィルター ---
    if '実行日時' in df_pred_log.columns:
        df_pred_log['実行日時'] = pd.to_datetime(df_pred_log['実行日時'], errors='coerce')
        valid_dates = df_pred_log['実行日時'].dropna()
        if not valid_dates.empty:
            min_log_date = valid_dates.min().date()
            max_log_date = valid_dates.max().date()
            
            with st.sidebar.expander("📅 予測保存日で絞り込み (バージョン比較)", expanded=False):
                st.caption("AIが予測を保存した日時で絞り込みます。設定4基準時代と設定5基準時代の成績を分けて確認できます。")
                
                if 'ai_version' in df_pred_log.columns:
                    df_pred_log['ai_version'] = df_pred_log['ai_version'].replace('', 'v1.0 (記録なし)').fillna('v1.0 (記録なし)')
                    versions = ['すべて'] + sorted(list(df_pred_log['ai_version'].astype(str).unique()))
                    selected_version = st.selectbox("AIバージョンで絞り込み", versions)
                    if selected_version != 'すべて':
                        df_pred_log = df_pred_log[df_pred_log['ai_version'] == selected_version]

                date_range = st.date_input(
                    "保存日の範囲",
                    value=(min_log_date, max_log_date),
                    min_value=min_log_date,
                    max_value=max_log_date
                )
                if len(date_range) == 2:
                    start_d, end_d = date_range
                    df_pred_log = df_pred_log[(df_pred_log['実行日時'].dt.date >= start_d) & (df_pred_log['実行日時'].dt.date <= end_d)]
                    
            if df_pred_log.empty:
                st.warning("指定された期間の予測結果ログがありません。")
                return

    if df_verify.empty or 'next_diff' not in df_verify.columns:
        st.warning("分析可能な実データ（結果）がありません。")
        return

    # --- データ準備 ---
    if '予測対象日' in df_pred_log.columns:
        df_pred_log['予測対象日'] = pd.to_datetime(df_pred_log['予測対象日'], errors='coerce')
    df_pred_log['対象日付'] = pd.to_datetime(df_pred_log['対象日付'], errors='coerce')
    
    if '予測対象日' in df_pred_log.columns:
        df_pred_log['予測対象日_merge'] = df_pred_log['予測対象日'].fillna(df_pred_log['対象日付'] + pd.Timedelta(days=1))
    else:
        df_pred_log['予測対象日_merge'] = df_pred_log['対象日付'] + pd.Timedelta(days=1)
        
    # 店舗カラム名の統一
    shop_col = '店名' if '店名' in df_verify.columns else ('店舗名' if '店舗名' in df_verify.columns else '店名')
    shop_col_pred = '店名' if '店名' in df_pred_log.columns else '店舗名'
    if shop_col != shop_col_pred:
        df_pred_log = df_pred_log.rename(columns={shop_col_pred: shop_col})
        
    # 台番号を文字列化して型を統一（結合ミス防止）
    df_pred_log['台番号'] = df_pred_log['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # df_verify と df_predict を結合して全台の特徴量ベースを作る
    full_feature_df = pd.concat([df_verify, df_predict], ignore_index=True)
    if '台番号' in full_feature_df.columns:
        full_feature_df['台番号'] = full_feature_df['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)

    if 'next_date' in full_feature_df.columns:
        full_feature_df['予測対象日_merge'] = pd.to_datetime(full_feature_df['next_date'], errors='coerce')
    else:
        full_feature_df['予測対象日_merge'] = pd.to_datetime(full_feature_df['対象日付'], errors='coerce') + pd.Timedelta(days=1)

    # 保存ログをベースに左外部結合（データ落ち防止）
    base_df = pd.merge(
        df_pred_log,
        full_feature_df,
        on=['予測対象日_merge', shop_col, '台番号'],
        how='left',
        suffixes=('_saved', '_current')
    )

    if base_df.empty:
        st.info("保存された予測に対して、まだ結果データ（翌日の稼働データ）が登録されていないか、一致するデータがありません。")
        return
        
    # 古いログのフォールバック用に元の対象日付を保存
    if '対象日付_saved' in base_df.columns:
        base_df['予測ベース日'] = base_df['対象日付_saved']
    elif '対象日付' in base_df.columns:
        base_df['予測ベース日'] = base_df['対象日付']
        
    # 以降の集計や表示のために対象日付を予測対象日に上書きする
    base_df['対象日付'] = base_df['予測対象日_merge']

    # 保存当時のスコアと予測結果を使用
    if 'prediction_score_saved' in base_df.columns:
        base_df['prediction_score'] = base_df['prediction_score_saved']
    if '予測差枚数_saved' in base_df.columns:
        base_df['予測差枚数'] = base_df['予測差枚数_saved']
    if '機種名_saved' in base_df.columns:
        base_df['機種名'] = base_df['機種名_saved']
    if '予測信頼度_saved' in base_df.columns:
        base_df['予測信頼度'] = base_df['予測信頼度_saved']
    elif '予測信頼度_current' in base_df.columns:
        base_df['予測信頼度'] = base_df['予測信頼度_current']
    else:
        base_df['予測信頼度'] = "-"
        
    # --- 実際の成績を df_raw から直接取得 (未稼働台のデータ落ちを防ぐ) ---
    if '台番号' in df_raw.columns:
        df_raw_temp = df_raw.copy()
        df_raw_temp['台番号'] = df_raw_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_raw_temp['対象日付'] = pd.to_datetime(df_raw_temp['対象日付'], errors='coerce')
    else:
        df_raw_temp = pd.DataFrame()

    shop_operating_dates = {}
    if not df_raw_temp.empty and shop_col in df_raw_temp.columns:
        for shop in df_raw_temp[shop_col].unique():
            shop_dates = sorted(df_raw_temp[df_raw_temp[shop_col] == shop]['対象日付'].dropna().unique())
            shop_operating_dates[shop] = shop_dates

    def get_actual_result(row):
        shop = row.get(shop_col)
        target_date = row.get('対象日付') # 予測対象日に上書き済み
        base_date = row.get('予測ベース日') # 古いログのフォールバック用
        machine_no = row.get('台番号')
        
        actual_date = pd.NaT
        if pd.notna(target_date) and shop in shop_operating_dates:
            for d in shop_operating_dates[shop]:
                if d >= target_date:
                    actual_date = d
                    break
        elif pd.notna(base_date) and shop in shop_operating_dates:
            for d in shop_operating_dates[shop]:
                if d > base_date:
                    actual_date = d
                    break
                    
        if pd.isna(actual_date) or df_raw_temp.empty:
            return pd.Series([np.nan, np.nan, np.nan, np.nan, pd.NaT])
            
        target_row = df_raw_temp[
            (df_raw_temp[shop_col] == shop) & 
            (df_raw_temp['対象日付'] == actual_date) & 
            (df_raw_temp['台番号'] == machine_no)
        ]
        
        if not target_row.empty:
            tr = target_row.iloc[0]
            return pd.Series([tr.get('差枚', 0), tr.get('BIG', 0), tr.get('REG', 0), tr.get('累計ゲーム', 0), actual_date])
        else:
            # 営業日だが該当台のデータがない = 未稼働(0G)とみなす
            return pd.Series([0, 0, 0, 0, actual_date])
            
    results = base_df.apply(get_actual_result, axis=1)
    base_df[['差枚_actual', '結果_BIG', '結果_REG', '結果_累計ゲーム', '実際の稼働日']] = results
    
    # --- BIG/REG確率分母の計算 ---
    cum_g = pd.to_numeric(base_df['結果_累計ゲーム'], errors='coerce').fillna(0)
    big_c = pd.to_numeric(base_df['結果_BIG'], errors='coerce').fillna(0)
    reg_c = pd.to_numeric(base_df['結果_REG'], errors='coerce').fillna(0)
    base_df['結果_BIG確率分母'] = np.where(big_c > 0, cum_g / big_c, 0)
    base_df['結果_REG確率分母'] = np.where(reg_c > 0, cum_g / reg_c, 0)

    # 特徴量の名前を _current から元に戻す（弱点分析などで使用するため）
    for col in base_df.columns.copy():
        if col.endswith('_current'):
            orig_col = col.replace('_current', '')
            if orig_col not in ['prediction_score', '予測差枚数', '機種名', '予測信頼度', 'おすすめ度', '根拠', 'next_diff', 'next_BIG', 'next_REG', 'next_累計ゲーム']:
                base_df[orig_col] = base_df[col]

    base_df = base_df.dropna(subset=['差枚_actual', 'prediction_score']).copy()
    
    # --- 設定5近似度の算出 (ボーナス回数の精査) ---
    specs = backend.get_machine_specs()
    def evaluate_setting5(row):
        g = row.get('結果_累計ゲーム', 0)
        act_b = row.get('結果_BIG', 0)
        act_r = row.get('結果_REG', 0)
        if pd.isna(g) or g <= 0: return pd.Series([0, 0, 0, 0, 0])
        machine = row.get('機種名', '')
        matched_spec = backend.get_matched_spec_key(machine, specs)
        p_b, p_r = 1/259.0, 1/255.0 # デフォルト
        if matched_spec and "設定5" in specs[matched_spec]:
            s5 = specs[matched_spec]["設定5"]
            if "BIG" in s5: p_b = 1.0 / s5["BIG"]
            if "REG" in s5: p_r = 1.0 / s5["REG"]
        exp_b, exp_r = g * p_b, g * p_r
        diff_b, diff_r = act_b - exp_b, act_r - exp_r
        
        # 期待回数からの「不足回数」をベースに減点する方式
        score_r = max(0, 80 + (diff_r * penalty_reg if diff_r < 0 else 0))
        score_b = max(0, 20 + (diff_b * penalty_big if diff_b < 0 else 0))
        
        total_score = score_r + score_b
        # 極端な低稼働（1000G未満）の場合のみ指定の減点率を適用
        if g < 1000:
            total_score *= (1 - ((1000 - g) / 1000) * (low_g_penalty / 100.0))
        return pd.Series([total_score, exp_b, exp_r, diff_b, diff_r])
        
    eval_df = base_df.apply(evaluate_setting5, axis=1)
    base_df[['設定5近似度', '期待BIG', '期待REG', 'BIG不足分', 'REG不足分']] = eval_df
    
    if base_df.empty:
        st.info("まだ結果が判明している予測がありません。")
        return

    # --- 店舗フィルター ---
    selected_shop = '全店舗'
    if shop_col in base_df.columns:
        shops = ['全店舗'] + sorted(list(base_df[shop_col].unique()))
        selected_shop = st.selectbox("分析対象の店舗を選択", shops)

    if selected_shop == '全店舗':
        merged_df = base_df.copy()
        st.subheader("📊 AIモデル バックテスト通算成績 (全店舗)")
    else:
        merged_df = base_df[base_df[shop_col] == selected_shop].copy()
        st.subheader(f"📊 AIモデル バックテスト通算成績 ({selected_shop})")

    if merged_df.empty:
        st.info("選択された店舗の分析データがありません。")
        return

    # --- 1. 全体成績 (KPI) & 円グラフ ---
    specs = backend.get_machine_specs()
    spec_reg_val = merged_df['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
    spec_tot_val = merged_df['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
    
    m_cum_g = pd.to_numeric(merged_df['結果_累計ゲーム'], errors='coerce').fillna(0)
    m_big_c = pd.to_numeric(merged_df['結果_BIG'], errors='coerce').fillna(0)
    m_reg_c = pd.to_numeric(merged_df['結果_REG'], errors='coerce').fillna(0)
    merged_df['結果_合算確率分母'] = np.where((m_big_c + m_reg_c) > 0, m_cum_g / (m_big_c + m_reg_c), 0)
    merged_df['is_high_setting'] = (((merged_df['結果_REG確率分母'] > 0) & (merged_df['結果_REG確率分母'] <= spec_reg_val)) | ((merged_df['結果_合算確率分母'] > 0) & (merged_df['結果_合算確率分母'] <= spec_tot_val))).astype(int)

    total_count = len(merged_df)
    high_set_count = merged_df['is_high_setting'].sum()
    low_set_count = total_count - high_set_count
    high_setting_rate = high_set_count / total_count if total_count > 0 else 0
    win_count = (merged_df['差枚_actual'] > 0).sum()
    win_rate = win_count / total_count if total_count > 0 else 0
    avg_diff = merged_df['差枚_actual'].mean()
    total_diff = merged_df['差枚_actual'].sum()
    
    col_kpi, col_pie = st.columns([2, 1])
    
    with col_kpi:
        k1, k2, k5 = st.columns(3)
        k1.metric("検証台数", f"{total_count} 台")
        k2.metric("高設定率", f"{high_setting_rate:.1%}")
        k5.metric("勝率(差枚)", f"{win_rate:.1%}")
        
        k3, k4 = st.columns(2)
        k3.metric("平均差枚", f"{int(avg_diff):+d} 枚")
        k4.metric("合計収支", f"{int(total_diff):+d} 枚")
        
    with col_pie:
        # 勝敗円グラフ (Altair)
        pie_data = pd.DataFrame({
            'Category': ['高設定', '低設定'],
            'Count': [high_set_count, low_set_count]
        })
        
        pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=35).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Category", type="nominal", 
                            scale=alt.Scale(domain=['高設定', '低設定'], range=['#FF4B4B', '#4B4BFF']),
                            legend=alt.Legend(title="設定挙動", orient="bottom")),
            tooltip=['Category', 'Count']
        ).properties(height=200)
        
        st.altair_chart(pie_chart, use_container_width=True)

    # 日別推移データをAI評価より先に計算する
    daily_stats = merged_df.groupby('対象日付').agg(
        high_setting_rate=('is_high_setting', 'mean'),
        total_profit=('差枚_actual', 'sum'),
        avg_s5_score=('設定5近似度', 'mean'),
        count=('台番号', 'count')
    ).reset_index().sort_values('対象日付')

    # --- AI振り返りレポート ---
    st.divider()
    st.subheader("🤖 AIの振り返りレポート (過去の自分との比較)")
    st.caption("最新の予測結果（設定5近似度）を過去の平均的なパフォーマンスと比較し、AIが自身の成長や調子を分析します。")
    
    avg_s5_score = merged_df['設定5近似度'].mean()
    avg_g = merged_df['結果_累計ゲーム'].mean()
    avg_diff_r = merged_df['REG不足分'].mean()
    avg_diff_b = merged_df['BIG不足分'].mean()
    
    # データ不足の考慮 (検証台数不足 or 学習データ不足)
    low_rel_count = (merged_df['予測信頼度'] == '🔻低').sum() if '予測信頼度' in merged_df.columns else 0
    low_rel_rate = low_rel_count / total_count if total_count > 0 else 0
    comment_prefix = ""
    if total_count < 5:
        comment_prefix = f"⚠️ **【検証台数不足】** 今回対象となった推奨台が **{total_count}台** と少ないため、たまたまのヒキの影響を強く受けています。点数は参考程度にご覧ください。\n\n"
    elif low_rel_rate >= 0.5:
        comment_prefix = f"⚠️ **【学習データ不足】** 今回の推奨台は過去データが少ない台が多いため、傾向を完全に把握しきれていません。\n\n"

    # 全体評価（期間全体の総評）
    if pd.isna(avg_g) or avg_g < 2000:
        overall_comment = f"全体として推奨台の平均回転数が **{int(avg_g if not pd.isna(avg_g) else 0)}G** と少なく、試行回数不足です。もう少し稼働がある状況で検証したいですね。"
        mood = "🤔"
    elif avg_s5_score >= 80:
        overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と大成功レベルです！本物の高設定を的確に見抜けています！"
        mood = "🌟"
    elif avg_s5_score >= 60:
        if avg_diff_r < 0:
            overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** でまずまずですが、REGが平均 {abs(avg_diff_r):.1f}回 不足しています。低〜中間設定の上振れに助けられている部分もありそうです。"
        else:
            overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** で優秀です！中身もしっかり高設定挙動を示しています。"
        mood = "👍"
    elif avg_s5_score >= 40:
        if avg_g < 4000:
            overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と惜しい結果です。平均回転数が {int(avg_g)}G と少なめなので、下振れの可能性もあります。"
        else:
            overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と反省点が残ります。低〜中間設定が混ざっている可能性が高いです。"
        mood = "💦"
    else:
        overall_comment = f"全体の平均設定5近似度は **{avg_s5_score:.1f}点** と惨敗です…。過去の傾向が変わった可能性があるので、最近のデータで学習し直すか、店選びを見直す余地があります。"
        mood = "😭"

    # 過去の自分との比較ロジック
    has_comparison = False
    if len(daily_stats) >= 2:
        latest_date = daily_stats['対象日付'].max()
        latest_stat = daily_stats.iloc[-1]
        past_stats = daily_stats.iloc[:-1]
        
        # 直近の過去（最大14日分）を「最近の実力」とする
        recent_past = past_stats.tail(14)
        
        # 台数（サンプル数）を考慮した加重平均を計算する
        total_past_score = (recent_past['avg_s5_score'] * recent_past['count']).sum()
        total_past_count = recent_past['count'].sum()
        past_avg_score = total_past_score / total_past_count if total_past_count > 0 else 0
        
        latest_score = latest_stat['avg_s5_score']
        score_diff = latest_score - past_avg_score
        
        has_comparison = True
        latest_date_str = latest_date.strftime('%m/%d')
        
        if total_past_count < 10:
             comparison_comment = f"最新 ({latest_date_str}) の設定5近似度は **{latest_score:.1f}点** でした。比較対象となる過去の検証台数が少なく（計{int(total_past_count)}台）、AIの成長や調子を正しく評価するにはまだデータが不足しています。"
             mood_cmp = "🤔"
        elif latest_stat['count'] < 5:
             comparison_comment = f"最新 ({latest_date_str}) の設定5近似度は **{latest_score:.1f}点** でした。直近の平均 ({past_avg_score:.1f}点) と比較したいところですが、今回の検証台数が{int(latest_stat['count'])}台と少ないため、たまたまのブレが大きい可能性があります。"
             mood_cmp = "🤔"
        elif score_diff >= 5:
             comparison_comment = f"最新 ({latest_date_str}) の設定5近似度は **{latest_score:.1f}点** でした！直近の平均 ({past_avg_score:.1f}点) より **{score_diff:+.1f}点** も向上しており、予測精度が上がっています！日々学習して賢くなっているのを感じますね！"
             mood_cmp = "📈"
        elif score_diff <= -5:
             comparison_comment = f"最新 ({latest_date_str}) の設定5近似度は **{latest_score:.1f}点** でした…。直近の平均 ({past_avg_score:.1f}点) より **{score_diff:+.1f}点** 下がっています。少し調子を落としているか、お店の設定配分のクセが変わった（フェイクが増えた等）可能性があります。"
             mood_cmp = "📉"
        else:
             comparison_comment = f"最新 ({latest_date_str}) の設定5近似度は **{latest_score:.1f}点** でした。直近の平均 ({past_avg_score:.1f}点) とほぼ同水準をキープしており、安定した予測ができています。"
             mood_cmp = "⚖️"

    if has_comparison:
        final_comment = f"{mood_cmp} **最近の調子 (過去の自分との比較):**\n{comparison_comment}\n\n{mood} **全体の総評:**\n{overall_comment}"
    else:
        final_comment = f"{mood} **全体の総評:**\n{overall_comment}\n\n※比較対象となる過去の推移データが不足しています。"

    st.info(f"{comment_prefix}{final_comment}\n\n※全体平均 REG過不足: **{avg_diff_r:+.1f}回** / BIG過不足: **{avg_diff_b:+.1f}回**")
    
    # --- 特に優秀だった台トップ3 ---
    if not merged_df.empty:
        top3_df = merged_df[merged_df['設定5近似度'] > 0].sort_values('設定5近似度', ascending=False).head(3)
        if not top3_df.empty:
            st.markdown("##### 🌟 特に高設定挙動だった推奨台 トップ3")
            for _, row in top3_df.iterrows():
                shop = row.get(shop_col, '')
                m_num = row.get('台番号', '')
                m_name = row.get('機種名', '')
                score = row.get('設定5近似度', 0)
                diff = row.get('差枚_actual', 0)
                g = row.get('結果_累計ゲーム', 0)
                r = row.get('結果_REG', 0)
                b = row.get('結果_BIG', 0)
                date_str = row['対象日付'].strftime('%m/%d') if pd.notna(row.get('対象日付')) else ""
                st.markdown(f"- **{score:.1f}点** : {date_str} {shop} #{m_num} {m_name} ({int(g)}G BIG{int(b)} REG{int(r)} / {int(diff):+d}枚)")

    # --- 2. 時系列推移 (勝率 & 収支) ---
    st.subheader("📈 日別の推移")
    
    if not daily_stats.empty:
        daily_stats['date_str'] = daily_stats['対象日付'].dt.strftime('%m/%d')
        
        tab_prof, tab_s5, tab_ver = st.tabs(["💰 高設定率・収支推移", "🎯 設定5近似度 推移", "📅 予測保存日ごとの精度推移"])
        with tab_prof:
            base_chart = alt.Chart(daily_stats).encode(x=alt.X('date_str', title='予測対象日', sort=None))
            bar_chart = base_chart.mark_bar(opacity=0.6).encode(
                y=alt.Y('total_profit', title='日別収支 (枚)'), color=alt.condition(alt.datum.total_profit > 0, alt.value("#FF4B4B"), alt.value("#4B4BFF")), tooltip=['date_str', alt.Tooltip('total_profit', format='+d'), 'count']
            )
            line_chart = base_chart.mark_line(point=True, color='#FFA726', strokeWidth=3).encode(
                y=alt.Y('high_setting_rate', title='高設定率 (%)', axis=alt.Axis(format='%')), tooltip=['date_str', alt.Tooltip('high_setting_rate', format='.1%')]
            )
            st.altair_chart(alt.layer(bar_chart, line_chart).resolve_scale(y='independent'), use_container_width=True)
        with tab_s5:
            base_chart_s5 = alt.Chart(daily_stats).encode(x=alt.X('date_str', title='予測対象日', sort=None))
            line_s5 = base_chart_s5.mark_line(point=True, color='#AB47BC', strokeWidth=3).encode(
                y=alt.Y('avg_s5_score', title='設定5近似度 (平均点)', scale=alt.Scale(domain=[0, 100])), tooltip=['date_str', alt.Tooltip('avg_s5_score', format='.1f', title='設定5近似度'), 'count']
            )
            st.altair_chart(line_s5, use_container_width=True)
            st.caption("※点数が高いほど、推奨台が実際に設定5以上の確率でBIG/REGを引けていたことを示します。(予測対象日ベース)")
            
        with tab_ver:
            if '実行日時' in merged_df.columns:
                merged_df['実行日時'] = pd.to_datetime(merged_df['実行日時'], errors='coerce')
                merged_df['実行日'] = merged_df['実行日時'].dt.date
                exec_stats = merged_df.groupby('実行日').agg(
                    avg_s5_score=('設定5近似度', 'mean'),
                    high_setting_rate=('is_high_setting', 'mean'),
                    count=('台番号', 'count')
                ).reset_index().dropna(subset=['実行日'])
                
                if not exec_stats.empty:
                    exec_stats['exec_date_str'] = exec_stats['実行日'].apply(lambda x: x.strftime('%m/%d'))
                    
                    base_chart_exec = alt.Chart(exec_stats).encode(x=alt.X('exec_date_str', title='予測保存日 (実行日)', sort=None))
                    
                    # 棒グラフ: 高設定率 (左軸)
                    bar_exec = base_chart_exec.mark_bar(opacity=0.6, color='#42A5F5').encode(
                        y=alt.Y('high_setting_rate', title='高設定率 (%)', axis=alt.Axis(format='%')),
                        tooltip=['exec_date_str', alt.Tooltip('high_setting_rate', format='.1%'), 'count']
                    )
                    
                    # 折れ線: 設定5近似度 (右軸)
                    line_exec = base_chart_exec.mark_line(point=True, color='#AB47BC', strokeWidth=3).encode(
                        y=alt.Y('avg_s5_score', title='設定5近似度 (平均点)', scale=alt.Scale(domain=[0, 100])),
                        tooltip=['exec_date_str', alt.Tooltip('avg_s5_score', format='.1f', title='設定5近似度'), 'count']
                    )
                    
                    st.altair_chart(alt.layer(bar_exec, line_exec).resolve_scale(y='independent'), use_container_width=True)
                    st.caption("※ AIが予測を保存した「実行日」ごとの精度推移です。AIの設定変更(設定4基準→設定5基準など)による成長を確認できます。")
                else:
                    st.info("予測保存日のデータがありません。")
            else:
                st.info("実行日時のデータがありません。")

    # --- 2. 確率帯別 精度分析 ---
    st.subheader("📈 確率帯別 精度分析")
    if 'prediction_score' in merged_df.columns:
        def get_prob_band(score):
            if score >= 0.85: return '85%以上'
            elif score >= 0.70: return '70%〜84%'
            elif score >= 0.50: return '50%〜69%'
            elif score >= 0.30: return '30%〜49%'
            else: return '30%未満'
            
        merged_df['確率帯'] = merged_df['prediction_score'].apply(get_prob_band)
        
        # 期間絞り込みUIの追加
        period_options = {"直近1ヶ月": 30, "直近3ヶ月": 90, "直近半年": 180, "全期間": None}
        selected_period = st.radio("集計対象の期間 (AIの現在の実力を測るため直近絞り込み推奨)", list(period_options.keys()), index=0, horizontal=True)
        
        if period_options[selected_period] is not None:
            latest_date = merged_df['対象日付'].max()
            cutoff_date = latest_date - pd.Timedelta(days=period_options[selected_period])
            prob_analysis_df = merged_df[merged_df['対象日付'] >= cutoff_date].copy()
        else:
            prob_analysis_df = merged_df.copy()
            
        if prob_analysis_df.empty:
            st.warning(f"指定された期間（{selected_period}）のデータがありません。")
        else:
            # --- 期待度スコアのヒストグラム ---
            st.markdown(f"**📊 期待度スコア (予測スコア) の分布と正解率 ({selected_period})**")
            
            # 5%刻みの代表値(左端)に丸めて集計
            prob_analysis_df['score_bin_left'] = (prob_analysis_df['prediction_score'] // 0.05) * 0.05
            hist_stats = prob_analysis_df.groupby('score_bin_left').agg(
                count=('台番号', 'count'),
                high_setting_rate=('is_high_setting', 'mean')
            ).reset_index()
            
            base_hist = alt.Chart(hist_stats).encode(
                x=alt.X('score_bin_left:O', title='AI期待度 (5%刻み)', axis=alt.Axis(format='%'))
            )
            bar_hist = base_hist.mark_bar(color='#42A5F5', opacity=0.8).encode(
                y=alt.Y('count:Q', title='検証台数'),
                tooltip=[alt.Tooltip('score_bin_left:Q', format='.0%', title='スコア帯(下限)'), alt.Tooltip('count:Q', title='台数')]
            )
            line_hist = base_hist.mark_line(color='#FF7043', point=True, strokeWidth=3).encode(
                y=alt.Y('high_setting_rate:Q', title='実際の高設定率', axis=alt.Axis(format='%')),
                tooltip=[alt.Tooltip('score_bin_left:Q', format='.0%', title='スコア帯(下限)'), alt.Tooltip('high_setting_rate:Q', format='.1%', title='実際の高設定率')]
            )
            st.altair_chart(alt.layer(bar_hist, line_hist).resolve_scale(y='independent').properties(height=300), use_container_width=True)

            # AIチューニングアドバイス
            score_mean = prob_analysis_df['prediction_score'].mean()
            high_score_df = prob_analysis_df[prob_analysis_df['prediction_score'] >= 0.70]
            high_score_accuracy = high_score_df['is_high_setting'].mean() if not high_score_df.empty else 0
            period_high_setting_rate = prob_analysis_df['is_high_setting'].mean()
            
            total_eval_count = len(prob_analysis_df)
            high_score_count = len(high_score_df)
            
            advices = []
            needs_tuning = False

            if total_eval_count < 30:
                advices.append(f"⚠️ **全体的なデータ不足**: 現在の集計期間における検証データが **{total_eval_count}台** と少なく、たまたまのヒキによるブレの影響を強く受けています。チューニングを変更する前に、もう少しデータ（最低30台程度）が貯まるのを待つことをおすすめします。")
                needs_tuning = True
            else:
                if score_mean > period_high_setting_rate + 0.15:
                    advices.append("全体的にAIの評価が**甘すぎ（期待度が高すぎ）**る傾向があります。棒グラフが右側に寄りすぎている場合、サイドバーの `葉の数 (num_leaves)` を下げる（例: 10〜12）か、`学習率` を下げてより厳格に学習させてみてください。")
                    needs_tuning = True
                elif score_mean < period_high_setting_rate - 0.15:
                    advices.append("全体的にAIの評価が**慎重すぎ（期待度が低すぎ）**る傾向があります。グラフが左側に偏っている場合、サイドバーの `学習回数 (n_estimators)` を増やす（例: 400〜500）か、`葉の数` を少し上げて、パターンをより多く覚えさせてみてください。")
                    needs_tuning = True
                    
            if high_score_count > 0:
                if high_score_count < 10:
                    advices.append(f"⚠️ **推奨台のデータ不足**: 期待度70%以上の台が **{high_score_count}台** しかありません。ブレが大きいため以下の「正解率」の評価は参考程度にしてください。")
                    
                if high_score_accuracy < 0.40:
                    advices.append("期待度70%以上の台（本来のおすすめ台）の**正解率が低く、過学習（たまたまのノイズを必勝法と勘違い）**を起こしている可能性があります。`深さ制限 (max_depth)` を 3 などに下げて、シンプルな条件だけを覚えさせてみてください。")
                    needs_tuning = True
                elif high_score_accuracy > 0.60:
                    advices.append("期待度70%以上の台の正解率が非常に高く、**高評価の台はしっかり結果を出せています**。折れ線グラフが右肩上がりになっていれば大成功です！")
                    
            if not needs_tuning and total_eval_count >= 30:
                advices.append("期待度と実際の高設定率のバランスが取れており、**現在のパラメータ設定は非常に良好**です！この設定のまま運用を続けることをおすすめします。")

            with st.expander("🤖 ヒストグラムから見る AI チューニングアドバイス", expanded=True):
                st.markdown("分布の偏りと実際の正解率（折れ線グラフ）のズレを分析した結果、以下の設定調整をおすすめします：")
                for adv in advices:
                    st.markdown(f"- {adv}")
                    
            rank_stats = prob_analysis_df.groupby('確率帯').agg(
                台数=('台番号', 'count'),
                高設定率=('is_high_setting', 'mean'),
                勝率=('差枚_actual', lambda x: (x > 0).mean()),
                平均差枚=('差枚_actual', 'mean'),
                合計差枚=('差枚_actual', 'sum')
            ).reset_index()
            
            # ソート順序固定
            rank_order = {'85%以上': 1, '70%〜84%': 2, '50%〜69%': 3, '30%〜49%': 4, '30%未満': 5}
            rank_stats['sort'] = rank_stats['確率帯'].map(rank_order).fillna(99)
            rank_stats = rank_stats.sort_values('sort').drop('sort', axis=1)
            
            rank_stats['信頼度'] = rank_stats['台数'].apply(get_confidence_indicator)
            st.dataframe(
                rank_stats,
                column_config={
                    "確率帯": st.column_config.TextColumn("期待度"),
                    "台数": st.column_config.NumberColumn("台数", format="%d台", help="検証数"),
                    "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=1),
                    "勝率": st.column_config.ProgressColumn("勝率(差枚)", format="%.1f%%", min_value=0, max_value=1),
                    "平均差枚": st.column_config.NumberColumn("平均", format="%+d枚", help="平均結果(差枚)"),
                    "合計差枚": st.column_config.NumberColumn("合計", format="%+d枚", help="合計収支(差枚)"),
                    "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                },
                use_container_width=True,
                hide_index=True
            )
        
    # --- 4. AIの弱点分析 (騙された台の共通点) ---
    st.divider()
    st.subheader("🧠 AIの弱点分析 (騙された台の共通点)")
    if selected_shop == '全店舗':
        st.caption("AIが予測を大きく外した「期待はずれ台」と「逃したお宝台」が、どのような特徴を持っていたかを全体平均と比較して分析します。AIのクセや弱点を把握するのに役立ちます。")
    else:
        st.caption(f"【{selected_shop}】において、AIが予測を大きく外した台の特徴を店舗平均と比較して分析します。店舗ごとのAIのクセや弱点を把握するのに役立ちます。")

    display_df = merged_df.copy() # このセクションで使うDF
    bad_pred_df = display_df[(display_df['prediction_score'] >= 0.70) & (display_df['差枚_actual'] <= -1000)].copy()
    missed_df = display_df[(display_df['prediction_score'] <= 0.40) & (display_df['差枚_actual'] >= 2000)].copy()

    if bad_pred_df.empty or missed_df.empty:
        st.info("分析に必要な「期待はずれ台」または「逃したお宝台」のサンプルが不足しています。")
    else:
        # 特徴量の定義
        features_to_analyze = {
            '累計ゲーム': "前日: 累計ゲーム",
            'REG分母': "前日: REG確率(分母)",
            '差枚': "前日: 差枚",
            'mean_7days_diff': "台: 過去7日平均差枚",
            'win_rate_7days': "台: 過去7日高設定率",
            '連続マイナス日数': "台: 連続マイナス日数",
            'neighbor_avg_diff': "配置: 両隣の平均差枚",
            'is_corner': "配置: 角台の割合",
            'event_rank_score': "イベント: ランクスコア"
        }
        
        # REG分母を計算
        for df_ in [display_df, bad_pred_df, missed_df]:
            if 'REG確率' in df_.columns:
                df_['REG分母'] = df_['REG確率'].apply(lambda x: 1/x if x > 0 else 9999)
            else:
                df_['REG分母'] = 9999

        analysis_results = []
        for f_key, f_name in features_to_analyze.items():
            if f_key in display_df.columns:
                # is_cornerは割合(%)、他は平均値
                if f_key == 'is_corner':
                    avg_all = display_df[f_key].mean() * 100
                    avg_bad = bad_pred_df[f_key].mean() * 100
                    avg_missed = missed_df[f_key].mean() * 100
                else:
                    avg_all = display_df[f_key].mean()
                    avg_bad = bad_pred_df[f_key].mean()
                    avg_missed = missed_df[f_key].mean()

                analysis_results.append({
                    "特徴": f_name,
                    "期待はずれ台": avg_bad,
                    "逃したお宝台": avg_missed,
                    "全体平均": avg_all,
                })

        if analysis_results:
            analysis_df = pd.DataFrame(analysis_results)
            
            # --- 解説の生成 ---
            explanation_bad, explanation_missed = [], []
            
            try:
                # 期待はずれ台の解説
                bad_diff = analysis_df[analysis_df['特徴'] == '前日: 差枚']['期待はずれ台'].iloc[0]
                if bad_diff > 300: explanation_bad.append("前日勝っている台の「据え置き」を期待しすぎている")
                bad_games = analysis_df[analysis_df['特徴'] == '前日: 累計ゲーム']['期待はずれ台'].iloc[0]
                all_games = analysis_df[analysis_df['特徴'] == '前日: 累計ゲーム']['全体平均'].iloc[0]
                if bad_games < all_games * 0.9: explanation_bad.append("回転数が少ない台の数値を信用しすぎている")

                # 逃したお宝台の解説
                missed_diff = analysis_df[analysis_df['特徴'] == '前日: 差枚']['逃したお宝台'].iloc[0]
                if missed_diff < -500: explanation_missed.append("前日大きく凹んだ台の「反発」を読み切れていない")
                missed_minus_days = analysis_df[analysis_df['特徴'] == '台: 連続マイナス日数']['逃したお宝台'].iloc[0]
                if missed_minus_days > 2: explanation_missed.append("連続凹み台の「上げリセット」を見逃している")
            except (IndexError, KeyError):
                pass # データが足りない場合はスキップ

            # 表示
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.markdown("##### 📉 期待はずれ台の傾向")
                if explanation_bad:
                    for item in explanation_bad: st.markdown(f"- {item}傾向があります。")
                else: st.markdown("特に目立った傾向はありません。")
            with col_exp2:
                st.markdown("##### 📈 逃したお宝台の傾向")
                if explanation_missed:
                    for item in explanation_missed: st.markdown(f"- {item}傾向があります。")
                else: st.markdown("特に目立った傾向はありません。")

            st.dataframe(
                analysis_df,
                column_config={
                    "期待はずれ台": st.column_config.NumberColumn(format="%.1f"),
                    "逃したお宝台": st.column_config.NumberColumn(format="%.1f"),
                    "全体平均": st.column_config.NumberColumn(format="%.1f"),
                },
                hide_index=True,
                use_container_width=True
            )

    # --- 5. 大外れ（ワースト予測）の分析 ---
    st.divider()
    st.subheader("⚠️ AIの予測が大外れした台 (ワーストランキング)")
    st.caption("AIが高く評価したのに大きく負けてしまった台（期待はずれ）や、低評価だったのに大勝ちした台のワーストランキングです。")
    # 表示用に整理
    display_df = merged_df.copy()
    display_df['結果判定'] = display_df['差枚_actual'].apply(lambda x: 'Win 🔴' if x > 0 else 'Lose 🔵')
    display_df = display_df.sort_values('対象日付', ascending=False)
    
    if 'prediction_score' in display_df.columns:
        display_df['予想設定5以上確率'] = (display_df['prediction_score'] * 100).astype(int)
    else:
        display_df['予想設定5以上確率'] = 0

    cols = ['対象日付', shop_col, '台番号', '機種名', '予想設定5以上確率', '設定5近似度', '差枚_actual', '結果_累計ゲーム', '結果_BIG', '結果_BIG確率分母', '結果_REG', '結果_REG確率分母', 'REG不足分']
    config_dict = {
        "対象日付": st.column_config.DateColumn("予測対象日", format="MM/DD"),
        "予想設定5以上確率": st.column_config.ProgressColumn("期待度", format="%d%%", min_value=0, max_value=100, help="AIが予測する設定5以上の確率"),
        "設定5近似度": st.column_config.ProgressColumn("近似度", format="%d点", min_value=0, max_value=100, help="設定5近似度"),
        "差枚_actual": st.column_config.NumberColumn("差枚", format="%+d"),
        "結果_累計ゲーム": st.column_config.NumberColumn("総G数", format="%dG"),
        "結果_BIG": st.column_config.NumberColumn("BIG", format="%d"),
        "結果_BIG確率分母": st.column_config.NumberColumn("B確率", format="1/%d"),
        "結果_REG": st.column_config.NumberColumn("REG", format="%d"),
        "結果_REG確率分母": st.column_config.NumberColumn("R確率", format="1/%d"),
        "REG不足分": st.column_config.NumberColumn("R不足", format="%+.1f", help="REG過不足"),
    }

    if 'prediction_score' in merged_df.columns:
        bad_pred_df = display_df[(display_df['prediction_score'] >= 0.70) & (display_df['差枚_actual'] <= -1000)].copy()
        missed_df = display_df[(display_df['prediction_score'] <= 0.40) & (display_df['差枚_actual'] >= 2000)].copy()
        
        tab_b1, tab_b2 = st.tabs(["📉 期待はずれ (高評価で大負け)", "📈 逃したお宝台 (低評価で大勝ち)"])
        
        with tab_b1:
            if bad_pred_df.empty: st.success("現在、大きく期待を裏切った台はありません！")
            else: st.dataframe(bad_pred_df[cols].sort_values('差枚_actual'), column_config=config_dict, use_container_width=True, hide_index=True)
                
        with tab_b2:
            if missed_df.empty: st.success("現在、大きく見逃した台はありません！")
            else: st.dataframe(missed_df[cols].sort_values('差枚_actual', ascending=False), column_config=config_dict, use_container_width=True, hide_index=True)

    # --- 6. 全履歴データ (バックテスト結果) ---
    st.divider()
    st.subheader("📝 全履歴データ (バックテスト結果)")
    
    band_options = ['すべて', '85%以上', '70%〜84%', '50%〜69%', '30%〜49%', '30%未満']
    selected_band = st.selectbox("表示する期待度（確率帯）を選択", band_options, index=0)
    
    history_display_df = display_df.copy()
    if selected_band != 'すべて':
        if '確率帯' in history_display_df.columns:
            history_display_df = history_display_df[history_display_df['確率帯'] == selected_band]

    st.dataframe(
        history_display_df[cols],
        column_config=config_dict,
        use_container_width=True,
        hide_index=True
    )

# --- ページ描画関数: 予測 vs 実際 ランキング比較 ---
def render_ranking_comparison_page(df_pred_log, df_verify, df_predict, df_raw):
    st.header("🏆 予測 vs 実際 ランキング")
    st.caption("指定した日付・店舗における「AIの予測ランキング」と「実際の店のランキング」を比較します。AIの推奨台が実際のランキングにどれくらい食い込んでいるかを確認できます。")

    if df_pred_log.empty:
        st.warning("保存された予測結果ログがありません。サイドバーの「予測結果をログ保存」ボタンから予測を保存してください。")
        return

    if df_verify.empty or 'next_diff' not in df_verify.columns:
        st.warning("分析可能な実データ（結果）がありません。")
        return

    # --- データ準備 ---
    if '予測対象日' in df_pred_log.columns:
        df_pred_log['予測対象日'] = pd.to_datetime(df_pred_log['予測対象日'], errors='coerce')
    df_pred_log['対象日付'] = pd.to_datetime(df_pred_log['対象日付'], errors='coerce')
    
    if '予測対象日' in df_pred_log.columns:
        df_pred_log['予測対象日_merge'] = df_pred_log['予測対象日'].fillna(df_pred_log['対象日付'] + pd.Timedelta(days=1))
    else:
        df_pred_log['予測対象日_merge'] = df_pred_log['対象日付'] + pd.Timedelta(days=1)
        
    # 店舗カラム名の統一
    shop_col = '店名' if '店名' in df_verify.columns else ('店舗名' if '店舗名' in df_verify.columns else '店名')
    shop_col_pred = '店名' if '店名' in df_pred_log.columns else '店舗名'
    if shop_col != shop_col_pred:
        df_pred_log = df_pred_log.rename(columns={shop_col_pred: shop_col})
        
    # 台番号を文字列化して型を統一（結合ミス防止）
    df_pred_log['台番号'] = df_pred_log['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # df_verify と df_predict を結合して全台の特徴量ベースを作る
    full_feature_df = pd.concat([df_verify, df_predict], ignore_index=True)
    if '台番号' in full_feature_df.columns:
        full_feature_df['台番号'] = full_feature_df['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)

    if 'next_date' in full_feature_df.columns:
        full_feature_df['予測対象日_merge'] = pd.to_datetime(full_feature_df['next_date'], errors='coerce')
    else:
        full_feature_df['予測対象日_merge'] = pd.to_datetime(full_feature_df['対象日付'], errors='coerce') + pd.Timedelta(days=1)

    # 保存ログをベースに左外部結合（データ落ち防止）
    base_df = pd.merge(
        df_pred_log,
        full_feature_df,
        on=['予測対象日_merge', shop_col, '台番号'],
        how='left',
        suffixes=('_saved', '_current')
    )

    if base_df.empty:
        st.info("保存された予測に対して、まだ結果データ（翌日の稼働データ）が登録されていないか、一致するデータがありません。")
        return
        
    # 古いログのフォールバック用に元の対象日付を保存
    if '対象日付_saved' in base_df.columns:
        base_df['予測ベース日'] = base_df['対象日付_saved']
    elif '対象日付' in base_df.columns:
        base_df['予測ベース日'] = base_df['対象日付']
        
    # 以降の集計や表示のために対象日付を予測対象日に上書きする
    base_df['対象日付'] = base_df['予測対象日_merge']

    # 保存当時のスコアと予測結果を使用
    if 'prediction_score_saved' in base_df.columns:
        base_df['prediction_score'] = base_df['prediction_score_saved']
    if '予測差枚数_saved' in base_df.columns:
        base_df['予測差枚数'] = base_df['予測差枚数_saved']
    if '機種名_saved' in base_df.columns:
        base_df['機種名'] = base_df['機種名_saved']
        
    # --- 実際の成績を df_raw から直接取得 (未稼働台のデータ落ちを防ぐ) ---
    if '台番号' in df_raw.columns:
        df_raw_temp = df_raw.copy()
        df_raw_temp['台番号'] = df_raw_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_raw_temp['対象日付'] = pd.to_datetime(df_raw_temp['対象日付'], errors='coerce')
    else:
        df_raw_temp = pd.DataFrame()

    shop_operating_dates = {}
    if not df_raw_temp.empty and shop_col in df_raw_temp.columns:
        for shop in df_raw_temp[shop_col].unique():
            shop_dates = sorted(df_raw_temp[df_raw_temp[shop_col] == shop]['対象日付'].dropna().unique())
            shop_operating_dates[shop] = shop_dates

    def get_actual_result(row):
        shop = row.get(shop_col)
        target_date = row.get('対象日付') 
        base_date = row.get('予測ベース日') 
        machine_no = row.get('台番号')
        
        actual_date = pd.NaT
        if pd.notna(target_date) and shop in shop_operating_dates:
            for d in shop_operating_dates[shop]:
                if d >= target_date:
                    actual_date = d
                    break
        elif pd.notna(base_date) and shop in shop_operating_dates:
            for d in shop_operating_dates[shop]:
                if d > base_date:
                    actual_date = d
                    break
                    
        if pd.isna(actual_date) or df_raw_temp.empty:
            return pd.Series([np.nan, np.nan, np.nan, np.nan, pd.NaT])
            
        target_row = df_raw_temp[
            (df_raw_temp[shop_col] == shop) & 
            (df_raw_temp['対象日付'] == actual_date) & 
            (df_raw_temp['台番号'] == machine_no)
        ]
        
        if not target_row.empty:
            tr = target_row.iloc[0]
            return pd.Series([tr.get('差枚', 0), tr.get('BIG', 0), tr.get('REG', 0), tr.get('累計ゲーム', 0), actual_date])
        else:
            return pd.Series([0, 0, 0, 0, actual_date])
            
    results = base_df.apply(get_actual_result, axis=1)
    base_df[['差枚_actual', '結果_BIG', '結果_REG', '結果_累計ゲーム', '実際の稼働日']] = results
    
    cum_g = pd.to_numeric(base_df['結果_累計ゲーム'], errors='coerce').fillna(0)
    big_c = pd.to_numeric(base_df['結果_BIG'], errors='coerce').fillna(0)
    reg_c = pd.to_numeric(base_df['結果_REG'], errors='coerce').fillna(0)
    base_df['結果_合算確率分母'] = np.where((big_c + reg_c) > 0, cum_g / (big_c + reg_c), 0)
    base_df['結果_BIG確率分母'] = np.where(big_c > 0, cum_g / big_c, 0)
    base_df['結果_REG確率分母'] = np.where(reg_c > 0, cum_g / reg_c, 0)

    merged_df = base_df.dropna(subset=['差枚_actual', 'prediction_score']).copy()

    if merged_df.empty:
        st.info("まだ結果が判明している予測がありません。")
        return

    if '対象日付' in merged_df.columns and not merged_df.empty:
        available_dates = sorted(merged_df['対象日付'].dropna().dt.date.unique(), reverse=True)
        if available_dates:
            selected_date = st.selectbox("📅 比較する対象日付を選択", available_dates, key="compare_date")
            
            shops_in_date = sorted(merged_df[merged_df['対象日付'].dt.date == selected_date][shop_col].unique())
            if shops_in_date:
                compare_shop = st.selectbox("🏬 比較する店舗を選択", shops_in_date, key="compare_shop")
            else:
                compare_shop = None
            
            if compare_shop:
                # --- 機種フィルター ---
                machines_in_date = ['すべての機種'] + sorted(merged_df[(merged_df['対象日付'].dt.date == selected_date) & (merged_df[shop_col] == compare_shop)]['機種名'].dropna().unique().tolist())
                selected_machine = st.selectbox("🎰 比較する機種を選択", machines_in_date, key="compare_machine")

                # --- 全体実績: ランキングトップ3獲得回数 (店舗別) ---
                if selected_machine != 'すべての機種':
                    eval_days = merged_df[(merged_df[shop_col] == compare_shop) & (merged_df['機種名'] == selected_machine)][['実際の稼働日', shop_col]].dropna().drop_duplicates()
                    shop_machine_label = f"【{compare_shop} - {selected_machine}】"
                else:
                    eval_days = merged_df[merged_df[shop_col] == compare_shop][['実際の稼働日', shop_col]].dropna().drop_duplicates()
                    shop_machine_label = f"【{compare_shop}】"
                    
                top3_count = 0
                total_eval_days = len(eval_days)
                
                if total_eval_days > 0:
                    raw_subset = pd.merge(df_raw_temp, eval_days, left_on=['対象日付', shop_col], right_on=['実際の稼働日', shop_col], how='inner')
                    if selected_machine != 'すべての機種':
                        raw_subset = raw_subset[raw_subset['機種名'] == selected_machine]
                    raw_subset = raw_subset.dropna(subset=['差枚'])
                    if not raw_subset.empty:
                        top3_machines = raw_subset.sort_values('差枚', ascending=False).groupby(['対象日付', shop_col]).head(3)
                        top3_machines = top3_machines[['対象日付', shop_col, '台番号']]
                        
                        top3_machines = top3_machines.rename(columns={'対象日付': '実際の稼働日'})
                        match_df = pd.merge(merged_df, top3_machines, on=['実際の稼働日', shop_col, '台番号'], how='inner')
                        top3_count = len(match_df.drop_duplicates(subset=['実際の稼働日', shop_col]))
                        
                    st.info(f"👑 **{shop_machine_label}AI推奨台のトップ3獲得実績**: 過去 {total_eval_days} 回中 **{top3_count} 回** (獲得率: {top3_count/total_eval_days:.1%})\n※AIが推奨した台（Top10）が、その日の指定条件の差枚ランキングでトップ3に入った回数です。")

                rank_metric = st.radio("📊 実際のランキング基準", ["差枚", "合算確率", "REG確率"], horizontal=True, help="合算確率とREG確率は、総ゲーム数3000G以上の台のみを対象とします。")

                col_pred, col_actual = st.columns(2)
                
                # 左: AI予測ランキング (Top 10)
                pred_df_day = merged_df[(merged_df['対象日付'].dt.date == selected_date) & (merged_df[shop_col] == compare_shop)].copy()
                if selected_machine != 'すべての機種':
                    pred_df_day = pred_df_day[pred_df_day['機種名'] == selected_machine]
                pred_df_day = pred_df_day.sort_values('prediction_score', ascending=False).head(10)
                
                # 右: 実際の差枚ランキング
                target_ts = pd.Timestamp(selected_date)
                actual_date = pd.NaT
                if compare_shop in shop_operating_dates:
                    for d in shop_operating_dates[compare_shop]:
                        if d >= target_ts:
                            actual_date = d
                            break
                            
                if pd.notna(actual_date) and not df_raw_temp.empty:
                    actual_df_day = df_raw_temp[
                        (df_raw_temp[shop_col] == compare_shop) & 
                        (df_raw_temp['対象日付'] == actual_date)
                    ].copy()
                    if selected_machine != 'すべての機種':
                        actual_df_day = actual_df_day[actual_df_day['機種名'] == selected_machine]
                else:
                    actual_df_day = pd.DataFrame()

                # --- 実際のランキングTop10を事前に計算 (照合用) ---
                actual_top10_machines = []
                if not actual_df_day.empty:
                    act_g = pd.to_numeric(actual_df_day['累計ゲーム'], errors='coerce').fillna(0)
                    act_b = pd.to_numeric(actual_df_day['BIG'], errors='coerce').fillna(0)
                    act_r = pd.to_numeric(actual_df_day['REG'], errors='coerce').fillna(0)
                    actual_df_day['合算回数'] = act_b + act_r
                    actual_df_day['BIG確率分母'] = np.where(act_b > 0, act_g / act_b, 0).astype(int)
                    actual_df_day['REG確率分母'] = np.where(act_r > 0, act_g / act_r, 0).astype(int)
                    actual_df_day['合算確率分母'] = np.where(actual_df_day['合算回数'] > 0, act_g / actual_df_day['合算回数'], 0).astype(int)
                    
                    if rank_metric == "差枚":
                        actual_df_day = actual_df_day.sort_values('差枚', ascending=False).head(10)
                    else:
                        actual_df_day = actual_df_day[actual_df_day['累計ゲーム'] >= 3000]
                        if rank_metric == "合算確率":
                            actual_df_day = actual_df_day[actual_df_day['合算確率分母'] > 0].sort_values('合算確率分母', ascending=True).head(10)
                        else:
                            actual_df_day = actual_df_day[actual_df_day['REG確率分母'] > 0].sort_values('REG確率分母', ascending=True).head(10)
                            
                    actual_top10_machines = actual_df_day['台番号'].tolist()

                with col_pred:
                    if selected_machine != 'すべての機種':
                        st.markdown(f"##### 🤖 AI予測ランキング ({selected_machine} Top 10)")
                    else:
                        st.markdown("##### 🤖 AI予測ランキング (Top 10)")
                    if pred_df_day.empty:
                        st.info("この日の予測ログがありません。")
                    else:
                        if 'prediction_score' in pred_df_day.columns:
                            pred_df_day['予想設定5以上確率'] = (pred_df_day['prediction_score'] * 100).astype(int)
                        else:
                            pred_df_day['予想設定5以上確率'] = 0

                        # トップ10ランクインのマーク
                        pred_df_day['的中'] = pred_df_day['台番号'].apply(lambda x: '🎯' if x in actual_top10_machines else '')

                        display_cols_pred = ['台番号', '機種名', '予想設定5以上確率', '差枚_actual', '結果_累計ゲーム', '結果_BIG確率分母', '結果_REG確率分母', '結果_合算確率分母', '的中']
                        
                        def highlight_positive(row):
                            if row.get('的中', '') == '🎯':
                                return ['background-color: rgba(255, 215, 0, 0.3)'] * len(row) # 的中なら少し強めの黄色
                            elif pd.notna(row['差枚_actual']) and row['差枚_actual'] > 0:
                                return ['background-color: rgba(255, 75, 75, 0.2)'] * len(row)
                            return [''] * len(row)
                            
                        styled_pred_df = pred_df_day[display_cols_pred].style.apply(highlight_positive, axis=1)

                        st.dataframe(
                            styled_pred_df,
                            column_config={
                                "台番号": st.column_config.TextColumn("台番号"),
                                "機種名": st.column_config.TextColumn("機種", width="small"),
                                "予想設定5以上確率": st.column_config.ProgressColumn("期待度", format="%d%%", min_value=0, max_value=100),
                                "差枚_actual": st.column_config.NumberColumn("結果差枚", format="%+d"),
                                "結果_累計ゲーム": st.column_config.NumberColumn("総G", format="%d"),
                                "結果_BIG確率分母": st.column_config.NumberColumn("BB", format="1/%d"),
                                "結果_REG確率分母": st.column_config.NumberColumn("RB", format="1/%d"),
                                "結果_合算確率分母": st.column_config.NumberColumn("合算", format="1/%d"),
                                "的中": st.column_config.TextColumn("的中", width="small"),
                            },
                            hide_index=True,
                            use_container_width=True
                        )

                with col_actual:
                    date_str_actual = actual_date.strftime('%Y-%m-%d') if pd.notna(actual_date) else "不明"
                    if selected_machine != 'すべての機種':
                        st.markdown(f"##### 🎰 実際の{rank_metric}ランキング ({selected_machine} Top 10)")
                    else:
                        st.markdown(f"##### 🎰 実際の{rank_metric}ランキング (店全体 Top 10)")
                    st.caption(f"※実稼働日: {date_str_actual}")
                    if actual_df_day.empty:
                        st.info("条件を満たす結果データがありません。")
                        else:
                            actual_df_day = actual_df_day.reset_index(drop=True)
                            actual_df_day.index = actual_df_day.index + 1
                            
                            def get_rank_medal(rank):
                                if rank == 1: return '🥇 1位'
                                elif rank == 2: return '🥈 2位'
                                elif rank == 3: return '🥉 3位'
                                else: return f'{rank}位'
                                
                            actual_df_day['順位'] = actual_df_day.index.map(get_rank_medal)
                            
                            # AIが推奨していた台にはマークをつける
                            pred_machines = pred_df_day['台番号'].tolist()
                            actual_df_day['AI推奨'] = actual_df_day['台番号'].apply(lambda x: '🎯' if x in pred_machines else '')
                            
                            display_cols = ['順位', '台番号', '機種名', '差枚', '累計ゲーム', 'BIG確率分母', 'REG確率分母', '合算確率分母', 'AI推奨']
                            
                            def highlight_top3(row):
                                if '1位' in row['順位']:
                                    return ['background-color: rgba(255, 215, 0, 0.2)'] * len(row)
                                elif '2位' in row['順位']:
                                    return ['background-color: rgba(192, 192, 192, 0.2)'] * len(row)
                                elif '3位' in row['順位']:
                                    return ['background-color: rgba(205, 127, 50, 0.2)'] * len(row)
                                return [''] * len(row)
                                
                            styled_df = actual_df_day[display_cols].style.apply(highlight_top3, axis=1)
                            
                            st.dataframe(
                                styled_df,
                                column_config={
                                    "順位": st.column_config.TextColumn("順位"),
                                    "台番号": st.column_config.TextColumn("台番号"),
                                    "機種名": st.column_config.TextColumn("機種", width="small"),
                                    "差枚": st.column_config.NumberColumn("差枚", format="%+d"),
                                    "累計ゲーム": st.column_config.NumberColumn("総G", format="%d"),
                                    "BIG確率分母": st.column_config.NumberColumn("BB", format="1/%d"),
                                    "REG確率分母": st.column_config.NumberColumn("RB", format="1/%d"),
                                    "合算確率分母": st.column_config.NumberColumn("合算", format="1/%d"),
                                    "AI推奨": st.column_config.TextColumn("推奨", width="small"),
                                },
                                hide_index=True,
                                use_container_width=True
                            )

# --- ページ描画関数: AI学習データ分析 (勝利の法則) ---
def render_feature_analysis_page(df_train, df_importance=None, df_events=None):
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
    selected_shop = '全店舗'
    if shop_col:
        shops = ['全店舗'] + sorted(list(base_analysis_df[shop_col].unique()))
        selected_shop = st.selectbox("分析対象の店舗を選択", shops)

    if selected_shop == '全店舗':
        analysis_df = base_analysis_df.copy()
        st.caption("過去の全データから、「翌日高設定挙動になった台」の傾向を分析し、高設定が入りやすい台の特徴を可視化します。")
    else:
        analysis_df = base_analysis_df[base_analysis_df[shop_col] == selected_shop].copy()
        st.caption(f"【{selected_shop}】の過去データから、この店で高設定が入りやすい台の特徴を可視化します。")

    # --- 曜日フィルター ---
    if 'target_weekday' in analysis_df.columns:
        day_options = ["すべての曜日", "平日 (月〜金)", "週末 (土・日)", "月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"]
        selected_day = st.selectbox("分析対象の曜日を選択", day_options)
        
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
    
    # --- 1. REG確率別の翌日高設定率 (最重要) ---
    st.subheader("📊 REG確率と高設定据え置きの関係")
    st.caption("「前日のREG確率が良い台は、翌日も高設定のまま（据え置き）になるのか？」を検証します。")
    
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
            高設定率=('target', 'mean'),
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

    # --- 3. 前日差枚と高設定率 ---
    with col2:
        st.markdown("**📉 前日差枚と翌日の高設定率**")
        # 差枚をビン分割
        d_bins = [-10000, -2000, -500, 500, 2000, 10000]
        d_labels = ['大負け', '負け', 'トントン', '勝ち', '大勝ち']
        analysis_df['差枚区間'] = pd.cut(analysis_df['差枚'], bins=d_bins, labels=d_labels)
        
        d_stats = analysis_df.groupby('差枚区間', observed=True)['target'].mean().reset_index()
        
        chart_d = alt.Chart(d_stats).mark_bar(color='#FFA726').encode(
            x=alt.X('差枚区間', title='前日の結果', sort=None),
            y=alt.Y('target', title='翌日高設定率', axis=alt.Axis(format='%'))
        )
        st.altair_chart(chart_d, use_container_width=True)

    # --- 4. イベントランク別の設定投入傾向 ---
    st.divider()
    st.subheader("🎉 イベントランク別の設定投入傾向")
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
        event_df['高設定挙動'] = event_df.get('is_win', (event_df['REG分母'] <= 260).astype(int))
        
        event_stats = event_df.groupby('イベントランク').agg(
            高設定投入率=('高設定挙動', 'mean'),
            平均差枚=('差枚', 'mean'),
            サンプル数=('台番号', 'count')
        ).reset_index()
        event_stats['信頼度'] = event_stats['サンプル数'].apply(get_confidence_indicator)
        
        # 並び替え順の指定 (S -> A -> B -> C -> 通常日)
        rank_order = {'S': 1, 'A': 2, 'B': 3, 'C': 4, '通常日': 5}
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
                st.altair_chart(chart_e, use_container_width=True)
            
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
                    use_container_width=True
                )
        else:
            st.info("イベントランクが登録されたデータがまだありません。サイドバーからイベントを登録すると傾向が表示されます。")
    else:
        st.info("イベントデータが登録されていないか、結合に失敗しました。")

    # --- 5. 予測日ベースの曜日・末尾別の傾向 ---
    st.divider()
    st.subheader("📅 予測日ベースの傾向 (曜日・末尾)")
    st.caption(f"指定した回転数（{min_g}G）以上回っている台を対象に、「予測日の曜日」や「台の末尾番号」ごとの成績を比較します。店舗のクセを見抜くのに役立ちます。")

    if not reg_df.empty:
        tab_wd, tab_end = st.tabs(["📅 曜日別の傾向", "🔢 末尾番号の傾向"])

        with tab_wd:
            if 'target_weekday' in reg_df.columns:
                weekdays_map = {0: '月曜', 1: '火曜', 2: '水曜', 3: '木曜', 4: '金曜', 5: '土曜', 6: '日曜'}
                wd_df = reg_df.dropna(subset=['target_weekday']).copy()
                wd_df['曜日'] = wd_df['target_weekday'].map(weekdays_map)

                wd_stats = wd_df.groupby(['target_weekday', '曜日']).agg(
                    高設定率=('target', 'mean'),
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
                    st.altair_chart(chart_wd, use_container_width=True)
                with col_w2:
                    st.dataframe(
                        wd_stats[['曜日', '高設定率', '平均翌日差枚', 'サンプル数', '信頼度']],
                        column_config={
                            "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=1),
                            "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
            else:
                st.info("曜日データがありません。")

        with tab_end:
            if '末尾番号' in reg_df.columns:
                end_df = reg_df.dropna(subset=['末尾番号']).copy()
                end_stats = end_df.groupby('末尾番号').agg(
                    高設定率=('target', 'mean'),
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
                    st.altair_chart(chart_end, use_container_width=True)
                with col_e2:
                    st.dataframe(
                        end_stats[['末尾番号', '高設定率', '平均翌日差枚', 'サンプル数', '信頼度']],
                        column_config={
                            "高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=1),
                            "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
            else:
                st.info("末尾番号データがありません。")

    # --- 6. 特殊パターンの検証 (REG先行・大凹み・大勝) ---
    st.divider()
    st.subheader("🕵️‍♂️ 特殊パターンの検証 (REG先行・凹み反発・大勝のその後)")
    st.caption(f"指定した回転数（{min_g}G）以上回っている台を対象に、スロット特有のパターンの翌日の成績を調査します。")

    if not reg_df.empty:
        chart_metric = st.radio("📊 グラフの表示指標", ["平均翌日差枚", "翌日高設定率"], horizontal=True)
        y_field = "平均翌日差枚" if chart_metric == "平均翌日差枚" else "翌日高設定率"
        y_title = "平均翌日差枚 (枚)" if chart_metric == "平均翌日差枚" else "翌日高設定率"
        y_format = "" if chart_metric == "平均翌日差枚" else "%"
        y_axis = alt.Axis(format=y_format, title=y_title) if y_format else alt.Axis(title=y_title)
        color_cond = alt.condition(alt.datum.平均翌日差枚 > 0, alt.value("#FF7043"), alt.value("#42A5F5")) if chart_metric == "平均翌日差枚" else alt.value("#AB47BC")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["REG先行", "大凹み・大勝", "2日間トレンド", "上げリセット", "安定度vs一撃", "BB極端欠損"])
        
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
                    翌日高設定率=('target', 'mean'),
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
                        use_container_width=True
                    )
                with col_rl2:
                    chart_rl = alt.Chart(rl_stats).mark_bar().encode(
                        x=alt.X('REG先行分類', title='REG先行の分類'),
                        y=alt.Y(y_field, axis=y_axis),
                        color=color_cond,
                        tooltip=['REG先行分類', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                    ).interactive()
                    st.altair_chart(chart_rl, use_container_width=True)

                if shop_col and selected_shop == '全店舗':
                    st.divider()
                    st.markdown("**🏬 店舗別: 完全不発台 (REG先行&BB欠損マイナス) の翌日成績**")
                    st.caption("このパターンが発生したとき、どの店舗が一番底上げ（上げ狙い成功）しやすいかを比較します。")
                    target_reg_df = reg_lead_df[reg_lead_df['REG先行分類'] == "① REG先行 & 差枚マイナス (BB欠損・完全不発)"]
                    if not target_reg_df.empty:
                        shop_reg_stats = target_reg_df.groupby(shop_col).agg(
                            翌日高設定率=('target', 'mean'),
                            平均翌日差枚=('next_diff', 'mean'),
                            サンプル数=('target', 'count')
                        ).reset_index().sort_values('平均翌日差枚', ascending=False)
                        shop_reg_stats['信頼度'] = shop_reg_stats['サンプル数'].apply(get_confidence_indicator)
                        
                        st.dataframe(
                            shop_reg_stats,
                            column_config={
                                shop_col: st.column_config.TextColumn("店舗名"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("現在、該当するパターンのデータはありません。")
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
                翌日高設定率=('target', 'mean'),
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
                    use_container_width=True
                )
            with col_dp2:
                chart_dp = alt.Chart(dp_stats).mark_bar().encode(
                    x=alt.X('前日結果', title='前日差枚'),
                    y=alt.Y(y_field, axis=y_axis),
                    color=color_cond,
                    tooltip=['前日結果', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                ).interactive()
                st.altair_chart(chart_dp, use_container_width=True)

            if shop_col and selected_shop == '全店舗':
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
                        翌日高設定率=('target', 'mean'),
                        平均翌日差枚=('next_diff', 'mean'),
                        サンプル数=('target', 'count')
                    ).reset_index().sort_values('平均翌日差枚', ascending=False)
                    shop_dp_stats['信頼度'] = shop_dp_stats['サンプル数'].apply(get_confidence_indicator)
                    
                    st.dataframe(
                        shop_dp_stats,
                        column_config={
                            shop_col: st.column_config.TextColumn("店舗名"),
                            "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
                            "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                            "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                            "サンプル数": st.column_config.NumberColumn("サンプル数", format="%d 台")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("現在、該当するパターンのデータはありません。")

            st.divider()
            st.markdown("**📉 大凹み台の「総ゲーム数」別 反発期待度**")
            st.caption("前日大きく凹んだ(-1000枚以下)台について、あまり回されずに放置された台と、タコ粘りされて凹んだ台で翌日の反発(上げ)期待度がどう違うか検証します。")
            
            big_lose_df = diff_pat_df[diff_pat_df['差枚'] <= -1000].copy()
            if not big_lose_df.empty:
                g_bins2 = [0, 3000, 5000, 7000, 15000]
                g_labels2 = ['① ~3000G (放置)', '② 3000~5000G', '③ 5000~7000G', '④ 7000G~ (タコ粘り)']
                
                big_lose_df['G数区間'] = pd.cut(big_lose_df['累計ゲーム'], bins=g_bins2, labels=g_labels2)
                
                bl_stats = big_lose_df.groupby('G数区間', observed=True).agg(
                    翌日高設定率=('target', 'mean'),
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
                        use_container_width=True
                    )
                with col_bl2:
                    chart_bl = alt.Chart(bl_stats).mark_bar().encode(
                        x=alt.X('G数区間', title='前日の総ゲーム数'),
                        y=alt.Y(y_field, axis=y_axis),
                        color=color_cond,
                        tooltip=['G数区間', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                    ).interactive()
                    st.altair_chart(chart_bl, use_container_width=True)
                
                if shop_col and selected_shop == '全店舗':
                    st.divider()
                    st.markdown("**🏬 店舗別: タコ粘り大凹み台 (7000G~ & -1000枚以下) の翌日成績**")
                    st.caption("たくさん回されて大きく負けた台に対して、翌日しっかり「お詫び（上げ・据え置き）」をしてくれる店舗を比較します。")
                    
                    target_bl_df = big_lose_df[big_lose_df['G数区間'] == '④ 7000G~ (タコ粘り)']
                    if not target_bl_df.empty:
                        shop_bl_stats = target_bl_df.groupby(shop_col).agg(
                            翌日高設定率=('target', 'mean'),
                            平均翌日差枚=('next_diff', 'mean'),
                            サンプル数=('target', 'count')
                        ).reset_index().sort_values('翌日高設定率', ascending=False)
                        shop_bl_stats['信頼度'] = shop_bl_stats['サンプル数'].apply(get_confidence_indicator)
                        
                        st.dataframe(
                            shop_bl_stats,
                            column_config={
                                shop_col: st.column_config.TextColumn("店舗名"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                                "サンプル数": st.column_config.NumberColumn("サンプル数", format="%d 台")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("現在、該当するパターンのデータはありません。")
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
                    翌日高設定率=('target', 'mean'),
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
                        use_container_width=True
                    )
                with col_t2:
                    chart_t2 = alt.Chart(t2_stats).mark_bar().encode(
                        x=alt.X('2日間トレンド', title='2日間の成績パターン'),
                        y=alt.Y(y_field, axis=y_axis),
                        color=color_cond,
                        tooltip=['2日間トレンド', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                    ).interactive()
                    st.altair_chart(chart_t2, use_container_width=True)

                if shop_col and selected_shop == '全店舗':
                    st.divider()
                    st.markdown("**🏬 店舗別: 2日間トレンド別 翌日成績の比較**")
                    
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
                            翌日高設定率=('target', 'mean'),
                            平均翌日差枚=('next_diff', 'mean'),
                            サンプル数=('target', 'count')
                        ).reset_index().sort_values('平均翌日差枚', ascending=False)
                        shop_t2_stats['信頼度'] = shop_t2_stats['サンプル数'].apply(get_confidence_indicator)
                        
                        st.dataframe(
                            shop_t2_stats,
                            column_config={
                                shop_col: st.column_config.TextColumn("店舗名"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                                "サンプル数": st.column_config.NumberColumn("サンプル数", format="%d 台")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("現在、該当するパターンのデータはありません。")
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
                    翌日高設定率=('target', 'mean'),
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
                        use_container_width=True
                    )
                with col_r2:
                    chart_r = alt.Chart(r_stats).mark_bar().encode(
                        x=alt.X('マイナス継続状況', title='マイナス継続状況'),
                        y=alt.Y(y_field, axis=y_axis),
                        color=color_cond,
                        tooltip=['マイナス継続状況', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                    ).interactive()
                    st.altair_chart(chart_r, use_container_width=True)

                if shop_col and selected_shop == '全店舗':
                    st.divider()
                    st.markdown("**🏬 店舗別: 3日以上連続マイナス台の『上げリセット期待度』**")
                    
                    target_reset_df = reset_df[reset_df['連続マイナス日数'] >= 3]
                    if not target_reset_df.empty:
                        shop_reset_stats = target_reset_df.groupby(shop_col).agg(
                            翌日高設定率=('target', 'mean'),
                            平均翌日差枚=('next_diff', 'mean'),
                            サンプル数=('target', 'count')
                        ).reset_index().sort_values('翌日高設定率', ascending=False)
                        shop_reset_stats['信頼度'] = shop_reset_stats['サンプル数'].apply(get_confidence_indicator)
                        
                        st.dataframe(
                            shop_reset_stats,
                            column_config={
                                shop_col: st.column_config.TextColumn("店舗名"),
                                "翌日高設定率": st.column_config.ProgressColumn("リセット(上げ)期待度", format="%.1f%%", min_value=0, max_value=1),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
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
                    翌日高設定率=('target', 'mean'),
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
                        use_container_width=True
                    )
                with col_s2:
                    chart_stab = alt.Chart(stab_stats).mark_bar().encode(
                        x=alt.X('安定度分類', title='台の性質'),
                        y=alt.Y(y_field, axis=y_axis),
                        color=color_cond,
                        tooltip=['安定度分類', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                    ).interactive()
                    st.altair_chart(chart_stab, use_container_width=True)

                if shop_col and selected_shop == '全店舗':
                    st.divider()
                    st.markdown("**🏬 店舗別: 「一撃・荒波台」の翌日成績（据え置きか回収か）**")
                    st.caption("一撃で出た台をそのまま据え置く店か、しっかり回収する店かを比較します。")
                    
                    target_stab_df = stab_df[stab_df['安定度分類'] == "② 一撃・荒波台 (週間+500枚以上 & 高設定率50%未満)"]
                    if not target_stab_df.empty:
                        shop_stab_stats = target_stab_df.groupby(shop_col).agg(
                            翌日高設定率=('target', 'mean'),
                            平均翌日差枚=('next_diff', 'mean'),
                            サンプル数=('target', 'count')
                        ).reset_index().sort_values('翌日高設定率', ascending=False)
                        shop_stab_stats['信頼度'] = shop_stab_stats['サンプル数'].apply(get_confidence_indicator)
                        
                        st.dataframe(
                            shop_stab_stats,
                            column_config={
                                shop_col: st.column_config.TextColumn("店舗名"),
                                "翌日高設定率": st.column_config.ProgressColumn("据え置き期待度", format="%.1f%%", min_value=0, max_value=1),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                                "サンプル数": st.column_config.NumberColumn("サンプル数", format="%d 台")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("該当するデータがありません。")
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
                    翌日高設定率=('target', 'mean'),
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
                        use_container_width=True
                    )
                with col_bb2:
                    chart_bb = alt.Chart(bb_stats).mark_bar().encode(
                        x=alt.X('BB欠損分類', title='BB欠損の度合い'),
                        y=alt.Y(y_field, axis=y_axis),
                        color=color_cond,
                        tooltip=['BB欠損分類', alt.Tooltip('翌日高設定率', format='.1%'), alt.Tooltip('平均翌日差枚', format='+.0f'), 'サンプル数', '信頼度']
                    ).interactive()
                    st.altair_chart(chart_bb, use_container_width=True)

                if shop_col and selected_shop == '全店舗':
                    st.divider()
                    st.markdown("**🏬 店舗別: 超不発台 (BIG 1/400以下 & REG高設定基準) の翌日成績**")
                    st.caption("BIGが全く引けなかった高設定挙動台に対して、翌日どの店舗が一番上げ(または据え置き)をしてくれるかを比較します。")
                    
                    target_bb_df = bb_def_df[bb_def_df['BB欠損分類'] == "① BIG 1/400以下 & REGは高設定基準 (超不発台)"]
                    if not target_bb_df.empty:
                        shop_bb_stats = target_bb_df.groupby(shop_col).agg(
                            翌日高設定率=('target', 'mean'),
                            平均翌日差枚=('next_diff', 'mean'),
                            サンプル数=('target', 'count')
                        ).reset_index().sort_values('平均翌日差枚', ascending=False)
                        shop_bb_stats['信頼度'] = shop_bb_stats['サンプル数'].apply(get_confidence_indicator)
                        
                        st.dataframe(
                            shop_bb_stats,
                            column_config={
                                shop_col: st.column_config.TextColumn("店舗名"),
                                "翌日高設定率": st.column_config.ProgressColumn("翌日高設定率", format="%.1f%%", min_value=0, max_value=1),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)"),
                                "平均翌日差枚": st.column_config.NumberColumn("平均翌日差枚", format="%+d 枚"),
                                "サンプル数": st.column_config.NumberColumn("サンプル数", format="%d 台")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("現在、該当するパターンのデータはありません。")
            else:
                st.info("BIG確率のデータがありません。")

    # --- 7. 特徴量重要度 (Feature Importance) ---
    if df_importance is not None and not df_importance.empty:
        if 'shop_name' in df_importance.columns:
            display_importance = df_importance[df_importance['shop_name'] == selected_shop].copy()
        else:
            display_importance = df_importance.copy() if selected_shop == '全店舗' else pd.DataFrame()
            
        if not display_importance.empty:
            display_importance = display_importance.sort_values('importance', ascending=False)
            st.divider()
            st.subheader("🧠 AIが重視したポイント (特徴量重要度)")
            if selected_shop == '全店舗':
                st.caption("AIが対象日の勝敗を予測する上で、どのデータ（特徴量）を一番重要視したかを示します。（全店舗共通のモデル）")
            else:
                st.caption(f"AIが【{selected_shop}】の台を予測する際に、どのデータを一番重要視しているかを示します。（店舗専用の分析モデル）")
            
            feature_name_map = {
                '累計ゲーム': '前日: 累計ゲーム数',
                'REG確率': '前日: REG確率',
                'BIG確率': '前日: BIG確率',
                '差枚': '前日: 差枚数',
                '末尾番号': '台番号: 末尾',
                'target_weekday': '予測日: 曜日',
                'target_date_end_digit': '予測日: 日付末尾 (7のつく日等)',
                'weekday_avg_diff': '店舗: 曜日平均差枚',
                'mean_7days_diff': '台: 直近7日平均差枚',
                'mean_14days_diff': '台: 直近14日平均差枚',
                'mean_30days_diff': '台: 直近30日平均差枚',
                'win_rate_7days': '台: 直近7日間高設定率 (一撃排除用)',
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
            
            display_importance['特徴量名'] = display_importance['feature'].map(lambda x: feature_name_map.get(x, x))
            
            chart_imp = alt.Chart(display_importance).mark_bar(color='#AB47BC').encode(
                x=alt.X('importance:Q', title='重要度スコア'),
                y=alt.Y('特徴量名:N', title='特徴量', sort='-x', axis=alt.Axis(labelLimit=0)),
                tooltip=['特徴量名', 'importance']
            ).properties(height=500).interactive()
            
            st.altair_chart(chart_imp, use_container_width=True)

            # 多角的 重視ポイント比較表 (全店舗選択時のみ表示)
            if selected_shop == '全店舗' and 'shop_name' in df_importance.columns:
                st.divider()
                st.subheader("🏢 多角的 AI重視ポイント比較表")
                st.caption("様々な切り口（店舗別・曜日別・イベント有無別）で、AIがどのデータを重視しているか（上位5つ）を一覧で比較します。")
                
                tab_shop, tab_wd, tab_ev = st.tabs(["🏬 店舗別", "📅 曜日別", "🎉 イベント有無別"])
                
                with tab_shop:
                    if 'category' in df_importance.columns:
                        shop_imp_df = df_importance[df_importance['category'] == '店舗'].copy()
                    else:
                        shop_imp_df = df_importance[df_importance['shop_name'] != '全店舗'].copy()
                        
                    if not shop_imp_df.empty:
                        shop_imp_df['特徴量名'] = shop_imp_df['feature'].map(lambda x: feature_name_map.get(x, x))
                        shop_imp_df['rank'] = shop_imp_df.groupby('shop_name')['importance'].rank(method='first', ascending=False)
                        top5_df = shop_imp_df[shop_imp_df['rank'] <= 5].copy()
                        pivot_df = top5_df.pivot(index='rank', columns='shop_name', values='特徴量名')
                        pivot_df.index = [f"第{int(i)}位" for i in pivot_df.index]
                        pivot_df.columns.name = None
                        st.dataframe(pivot_df, use_container_width=True)
                    else:
                        st.info("店舗別の比較データがありません。")

                with tab_wd:
                    if 'category' in df_importance.columns:
                        wd_imp_df = df_importance[df_importance['category'] == '曜日'].copy()
                        if not wd_imp_df.empty:
                            wd_imp_df['特徴量名'] = wd_imp_df['feature'].map(lambda x: feature_name_map.get(x, x))
                            wd_imp_df['rank'] = wd_imp_df.groupby('shop_name')['importance'].rank(method='first', ascending=False)
                            top5_wd_df = wd_imp_df[wd_imp_df['rank'] <= 5].copy()
                            pivot_wd_df = top5_wd_df.pivot(index='rank', columns='shop_name', values='特徴量名')
                            pivot_wd_df.index = [f"第{int(i)}位" for i in pivot_wd_df.index]
                            pivot_wd_df.columns.name = None
                            
                            # カラムを月曜から順に並び替え
                            wd_order = ['月曜', '火曜', '水曜', '木曜', '金曜', '土曜', '日曜']
                            cols = [c for c in wd_order if c in pivot_wd_df.columns]
                            st.dataframe(pivot_wd_df[cols], use_container_width=True)
                        else:
                            st.info("曜日別の比較データがありません。")

                with tab_ev:
                    if 'category' in df_importance.columns:
                        ev_imp_df = df_importance[df_importance['category'] == 'イベント'].copy()
                        if not ev_imp_df.empty:
                            ev_imp_df['特徴量名'] = ev_imp_df['feature'].map(lambda x: feature_name_map.get(x, x))
                            ev_imp_df['rank'] = ev_imp_df.groupby('shop_name')['importance'].rank(method='first', ascending=False)
                            top5_ev_df = ev_imp_df[ev_imp_df['rank'] <= 5].copy()
                            pivot_ev_df = top5_ev_df.pivot(index='rank', columns='shop_name', values='特徴量名')
                            pivot_ev_df.index = [f"第{int(i)}位" for i in pivot_ev_df.index]
                            pivot_ev_df.columns.name = None
                            st.dataframe(pivot_ev_df, use_container_width=True)
                        else:
                            st.info("イベント有無別の比較データがありません。")

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
                    default_date = pd.Timestamp.now(tz='Asia/Tokyo').date()
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
            try:
                correct_password = st.secrets.get("app_password", "1234")
            except Exception:
                correct_password = "1234"
            if password == correct_password:
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("パスワードが違います。")
        return

    # タイトルや設定は共通
    st.title("🎰 スロット予測ビューアー")
    
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "店舗別詳細データ"
        
    pages = ["店舗別詳細データ", "全店分析サマリー", "AI傾向分析 (勝利の法則)", "精度検証 (答え合わせ)", "🏆 予測 vs 実際 ランキング", "島マスター管理", "イベント管理", "💰 マイ収支管理"]
    
    # --- ページ切り替えメニュー (サイドバーの一番上) ---
    page = st.sidebar.radio(
        "メニュー", 
        pages,
        index=pages.index(st.session_state["current_page"]) if st.session_state["current_page"] in pages else 0
    )
    
    if page != st.session_state["current_page"]:
        st.session_state["current_page"] = page
        # スマホ幅 (768px以下) の場合のみサイドバーを閉じるJSを発火
        st.components.v1.html(
            """
            <script>
                if (window.parent.innerWidth <= 768) {
                    var btn = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]') || window.parent.document.querySelector('button[kind="header"]');
                    if (btn) { btn.click(); }
                }
            </script>
            """,
            width=0, height=0
        )
        
    st.sidebar.divider()

    # --- 予測対象日の選択 (サイドバー) ---
    predict_target_date = st.sidebar.date_input(
        "📅 予測対象日",
        value=pd.Timestamp.now(tz='Asia/Tokyo').date(),
        help="指定した日付の予測を行います。過去日を指定した場合は、その前日までのデータのみを使用して予測を再現します。"
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
                reg_date = st.date_input("日付", pd.Timestamp.now(tz='Asia/Tokyo').date())
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
        hp_train_months = st.slider("学習データ期間 (直近〇ヶ月)", 1, 12, 3, step=1, help="店長スイッチ対策や処理落ちを防ぐため、直近のデータのみで学習させます。")
        hp_n_estimators = st.slider("学習回数 (n_estimators)", 50, 1000, 300, step=50, help="値を大きくすると学習量が増えますが、時間がかかり過学習のリスクもあります。")
        hp_learning_rate = st.slider("学習率 (learning_rate)", 0.01, 0.3, 0.03, step=0.01, help="値を小さくすると丁寧に学習しますが、回数を増やす必要があります。")
        hp_num_leaves = st.slider("葉の数 (num_leaves)", 10, 127, 15, step=1, help="モデルの複雑さ。スロットのようなノイズが多いデータは小さめ(15〜20)がおすすめです。")
        hp_max_depth = st.slider("深さ制限 (max_depth)", -1, 15, 4, step=1, help="木の深さの上限。ノイズ対策として3〜7程度に制限するのがおすすめです。-1は無制限。")
        
        hyperparams = {
            'train_months': hp_train_months,
            'n_estimators': hp_n_estimators,
            'learning_rate': hp_learning_rate,
            'num_leaves': hp_num_leaves,
            'max_depth': hp_max_depth
        }

    # 分析実行
    with st.spinner("AIがデータを分析し、予測を生成しています..."):
        # キャッシュキーとしてデータの長さを利用（簡易的）
        df, df_verify, df_importance = backend.run_analysis(df_raw, df_events, df_island, hyperparams, target_date=predict_target_date)
    
    if df.empty:
        st.warning(f"指定された予測対象日（{predict_target_date.strftime('%Y-%m-%d')}）以前の分析可能なデータがありません。")
        return

    # 日付表示
    if '対象日付' in df.columns:
        target_date = df['対象日付'].max()
        st.caption(f"📈 過去データ集計期間: 〜 {target_date.strftime('%Y-%m-%d')} (これまでの全履歴の傾向から {predict_target_date.strftime('%Y-%m-%d')} の結果を予測しています)")
    
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

    with st.spinner(f"⏳ 「{page}」の画面を構築しています... しばらくお待ちください。"):
        if page == "全店分析サマリー":
            render_summary_page(df, df_raw, shop_col, df_events)
        elif page == "精度検証 (答え合わせ)":
            df_pred_log = backend.load_prediction_log()
            render_verification_page(df_pred_log, df_verify, df, df_raw)
        elif page == "🏆 予測 vs 実際 ランキング":
            df_pred_log = backend.load_prediction_log()
            render_ranking_comparison_page(df_pred_log, df_verify, df, df_raw)
        elif page == "AI傾向分析 (勝利の法則)":
            render_feature_analysis_page(df_verify, df_importance, df_events)
        elif page == "島マスター管理":
            render_island_master_page(df_raw)
        elif page == "イベント管理":
            render_event_management_page()
        elif page == "💰 マイ収支管理":
            render_my_balance_page(df_raw)
        else:
            shop_detail_page.render_shop_detail_page(df, df_raw, shop_col, df_events, df_verify)

if __name__ == "__main__":
    main()