import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import backend

def render_calendar_compare_page(df_raw, df_predict, target_date):
    st.header("🗺️ 店舗間ヒートマップ比較")
    st.caption("各店舗の直近の出しぶり（平均差枚や高設定率）をヒートマップで比較し、「全体的に還元している熱い店」を直感的に探せます。")

    if df_raw.empty:
        st.warning("比較するデータがありません。")
        return

    shop_col = '店名' if '店名' in df_raw.columns else ('店舗名' if '店舗名' in df_raw.columns else None)
    date_col = '対象日付'

    if not shop_col or date_col not in df_raw.columns:
        st.error("店舗または日付のデータが見つかりません。")
        return
        
    # --- 1. 明日の店舗期待度ランキング ---
    st.subheader(f"🔮 【{target_date.strftime('%Y-%m-%d')}】のAI店舗推奨度")
    if not df_predict.empty and shop_col in df_predict.columns and 'prediction_score' in df_predict.columns:
        if 'sueoki_score' in df_predict.columns:
            df_predict['max_score'] = df_predict[['prediction_score', 'sueoki_score']].max(axis=1)
        else:
            df_predict['max_score'] = df_predict['prediction_score']
            
        agg_dict = {
            '平均期待度': ('max_score', 'mean'),
            '高期待台数': ('max_score', lambda x: (x >= 0.30).sum()),
            '全台数': ('台番号', 'nunique')
        }
        if '予測差枚数' in df_predict.columns:
            agg_dict['予測平均差枚'] = ('予測差枚数', 'mean')
            
        shop_pred_stats = df_predict.groupby(shop_col).agg(**agg_dict).reset_index().sort_values('平均期待度', ascending=False)
        
        def get_eval_badge(row):
            return backend.classify_shop_eval(row.get('予測平均差枚'), row.get('全台数', 50), is_prediction=True)
            
        shop_pred_stats['営業予測'] = shop_pred_stats.apply(get_eval_badge, axis=1)
        
        if '予測平均差枚' not in shop_pred_stats.columns:
            shop_pred_stats['予測平均差枚'] = np.nan
        
        if not shop_pred_stats.empty:
            st.dataframe(
                shop_pred_stats[[shop_col, '営業予測', '予測平均差枚', '平均期待度', '高期待台数', '全台数']],
                column_config={
                    shop_col: st.column_config.TextColumn("店舗名"),
                    "営業予測": st.column_config.TextColumn("AI営業予測"),
                    "予測平均差枚": st.column_config.NumberColumn("予測平均差枚", format="%+d枚"),
                    "平均期待度": st.column_config.ProgressColumn("店舗全体の平均期待度", format="%.2f", min_value=0, max_value=1.0),
                    "高期待台数": st.column_config.NumberColumn("推奨台(30%以上)", format="%d台"),
                    "全台数": st.column_config.NumberColumn("集計台数", format="%d台")
                },
                width="stretch",
                hide_index=True
            )
    else:
        st.info("予測データがありません。")

    st.divider()

    # --- 2. 過去の実績ヒートマップ ---
    st.subheader("🔥 過去の実績ヒートマップ")
    
    # データの準備
    df_history = df_raw.copy()
    df_history[date_col] = pd.to_datetime(df_history[date_col], errors='coerce')
    df_history = df_history.dropna(subset=[date_col])
    
    if df_history.empty:
        st.warning("履歴データがありません。")
        return
        
    df_history['年月'] = df_history[date_col].dt.strftime('%Y-%m')
    available_months = sorted(df_history['年月'].unique(), reverse=True)
    
    if not available_months:
        st.warning("有効な履歴データがありません。")
        return

    period_options = ["直近30日"] + available_months
    selected_period = st.selectbox("📅 表示する期間を選択", period_options, index=0)
    
    if selected_period == "直近30日":
        max_date = df_history[date_col].max()
        cutoff_date = max_date - pd.Timedelta(days=30)
        df_recent = df_history[df_history[date_col] > cutoff_date].copy()
    else:
        df_recent = df_history[df_history['年月'] == selected_period].copy()
    
    if df_recent.empty:
        st.warning("指定された期間のデータがありません。")
        return
    
    # 高設定フラグの計算 (機種ごとの合算やREGで判定)
    specs = backend.get_machine_specs()
    def is_high(row):
        g = pd.to_numeric(row.get('累計ゲーム', 0), errors='coerce')
        if g < 3000:
            return np.nan
        
        b = pd.to_numeric(row.get('BIG', 0), errors='coerce')
        r = pd.to_numeric(row.get('REG', 0), errors='coerce')
        
        machine = row.get('機種名', '')
        matched_spec = backend.get_matched_spec_key(machine, specs)
        p_r_5 = 1.0 / specs[matched_spec].get("設定5", {"REG": 260.0})["REG"] if matched_spec else 1/260.0
        p_t_5 = 1.0 / specs[matched_spec].get("設定5", {"合算": 128.0})["合算"] if matched_spec else 1/128.0
        p_r_3 = 1.0 / specs[matched_spec].get("設定3", {"REG": 300.0})["REG"] if matched_spec else 1/300.0
        p_r_1 = 1.0 / specs[matched_spec].get("設定1", {"REG": 400.0})["REG"] if matched_spec else 1/400.0
        
        r_prob = r / g if g > 0 else 0
        t_prob = (b + r) / g if g > 0 else 0
        
        import math
        exp_r1 = g * p_r_1
        std_r1 = math.sqrt(g * p_r_1 * (1.0 - p_r_1)) if g > 0 else 0
        z_score = (r - exp_r1) / std_r1 if std_r1 > 0 else 0
        
        return 1 if r_prob >= p_r_5 or (t_prob >= p_t_5 and r_prob >= p_r_3) or z_score >= 1.64 else 0
        
    df_recent['is_high'] = df_recent.apply(is_high, axis=1)

    df_recent['表示日'] = df_recent[date_col].dt.strftime('%m/%d')
    
    # --- UI: 指標と機種の選択 ---
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        metric_choice = st.radio("表示する指標", ["REG確率", "平均差枚", "高設定率"], horizontal=True)
    with col_m2:
        all_machines = ["(全機種平均)"] + sorted(list(df_recent['機種名'].dropna().unique()))
        selected_machine = st.selectbox("機種で絞り込み (REG確率のみ)", all_machines)

    # 機種が選択されている場合、指標は強制的にREG確率にする
    if selected_machine != "(全機種平均)":
        metric_choice = "REG確率"
        st.caption(f"※機種「{selected_machine}」が選択されているため、指標はREG確率で表示されます。")
    
    if metric_choice == "平均差枚":
        pivot_df = df_recent.pivot_table(index=shop_col, columns='表示日', values='差枚', aggfunc='mean')
        fmt = "{:+.0f}"
        cmap = "RdYlBu_r" # 赤(プラス)〜青(マイナス)
        vmin, vmax = -300, 300
    elif metric_choice == "高設定率":
        pivot_df = df_recent.pivot_table(index=shop_col, columns='表示日', values='is_high', aggfunc='mean')
        pivot_df = pivot_df * 100 # %表記にするため
        fmt = "{:.1f}%"
        cmap = "Greens" # 緑が濃いほど高設定率が高い
        vmin, vmax = 0, 30
    else: # REG確率
        source_df = df_recent.copy()
        if selected_machine != "(全機種平均)":
            source_df = source_df[source_df['機種名'] == selected_machine]

        source_df['REG確率分母'] = np.where(
            pd.to_numeric(source_df['REG'], errors='coerce').fillna(0) > 0,
            pd.to_numeric(source_df['累計ゲーム'], errors='coerce').fillna(0) / pd.to_numeric(source_df['REG'], errors='coerce').fillna(0),
            np.nan
        )
        pivot_df = source_df.pivot_table(index=shop_col, columns='表示日', values='REG確率分母', aggfunc='mean')
        
        # 全店舗が表示されるように reindex を追加
        all_shops_in_period = df_recent[shop_col].unique()
        pivot_df = pivot_df.reindex(all_shops_in_period)

        fmt = "1/{:.0f}"
        cmap = "RdYlBu_r"
        vmin, vmax = 200, 400
        
    # 日付順に列をソート
    pivot_df = pivot_df[sorted(pivot_df.columns)]
    
    st.caption("※色が濃い（赤い/緑色）ほど優秀な営業日だったことを示します。横に見れば『店舗の好不調の波』が、縦に見れば『その日一番強かった店』がわかります。")
    
    # Pandas Stylerを使ってヒートマップ化
    styled_pivot = pivot_df.style.background_gradient(cmap=cmap, axis=None, vmin=vmin, vmax=vmax, text_color_threshold=0.5).format(fmt, na_rep="-")
    
    st.dataframe(
        styled_pivot,
        width="stretch",
        height=400
    )

    # --- 新規追加: 回収日の一点突破（優良店）ランキング ---
    st.divider()
    st.subheader("💎 【優良店発掘】回収日の「一点突破」投入状況 (直近90日)")
    st.caption("過去90日間で、**実際に店舗全体が「回収日」だった日**（店舗全体の平均差枚から判定）において、高設定（設定5基準）がどれくらい投入されていたかを店舗ごとに比較します。回収日でもしっかり見せ台を用意してくれる優良店と、一切設定を使わない極悪店を見極めることができます。")

    df_raw_eval = df_raw.copy()
    df_raw_eval['対象日付'] = pd.to_datetime(df_raw_eval['対象日付'], errors='coerce').dt.normalize()
    
    max_d = df_raw_eval['対象日付'].max()
    if pd.notna(max_d):
        df_raw_eval = df_raw_eval[df_raw_eval['対象日付'] >= (max_d - pd.Timedelta(days=90))]
        
        # ヒートマップ用に使った高設定判定ロジック(is_high)を流用
        df_raw_eval['is_high'] = df_raw_eval.apply(is_high, axis=1)
        
        daily_stats = df_raw_eval.groupby([shop_col, '対象日付']).agg(
            総台数=('台番号', 'nunique'),
            高設定有効数=('is_high', 'count'),
            高設定台数=('is_high', 'sum'),
            平均差枚=('差枚', 'mean'),
            合計REG=('REG', 'sum'),
            合計ゲーム数=('累計ゲーム', 'sum')
        ).reset_index()
        
        daily_stats['営業状態'] = daily_stats.apply(lambda r: backend.classify_shop_eval(r['平均差枚'], r['総台数'], is_prediction=False), axis=1)
        cold_days = daily_stats[daily_stats['営業状態'] == "🥶 回収日"].copy()
        
        if not cold_days.empty:
            cold_summary = cold_days.groupby(shop_col).agg(
                回収日数=('対象日付', 'count'),
                総台数=('総台数', 'sum'),
                高設定有効数=('高設定有効数', 'sum'),
                高設定台数=('高設定台数', 'sum'),
                回収日平均差枚=('平均差枚', 'mean'),
                合計REG=('合計REG', 'sum'),
                合計ゲーム数=('合計ゲーム数', 'sum')
            ).reset_index()
            
            cold_summary['回収日高設定率'] = np.where(cold_summary['高設定有効数'] > 0, (cold_summary['高設定台数'] / cold_summary['高設定有効数']) * 100, 0.0)
            cold_summary['1日平均高設定台数'] = cold_summary['高設定台数'] / cold_summary['回収日数']
            cold_summary['回収日平均REG確率'] = cold_summary.apply(lambda r: f"1/{int(r['合計ゲーム数'] / r['合計REG'])}" if r['合計REG'] > 0 and r['合計ゲーム数'] > 0 else "-", axis=1)
            
            def get_betapin_badge(rate):
                if rate >= 3.0: return "💎 優良 (見せ台あり)"
                elif rate >= 1.5: return "🟡 普通 (マグレ混じり)"
                else: return "⚠️ 完全ベタピン"
                
            cold_summary['回収日評価'] = cold_summary['回収日高設定率'].apply(get_betapin_badge)
            cold_summary = cold_summary.sort_values('回収日高設定率', ascending=False)
            
            st.dataframe(
                cold_summary[[shop_col, '回収日評価', '回収日数', '回収日平均差枚', '回収日平均REG確率', '1日平均高設定台数', '回収日高設定率']],
                column_config={
                    shop_col: st.column_config.TextColumn("店舗名"),
                    "回収日評価": st.column_config.TextColumn("店舗評価", help="回収日における高設定率が3%以上なら優良店、1.5%未満は完全ベタピン店と判定します。"),
                    "回収日数": st.column_config.NumberColumn("ド回収日 発生数", format="%d日"),
                    "回収日平均差枚": st.column_config.NumberColumn("回収日 平均差枚", format="%+d 枚"),
                    "回収日平均REG確率": st.column_config.TextColumn("回収日 REG確率"),
                    "1日平均高設定台数": st.column_config.NumberColumn("1日あたり高設定", format="%.1f 台", help="回収日1日あたり、平均して何台の高設定(設定5基準)が投入されていたか"),
                    "回収日高設定率": st.column_config.ProgressColumn("高設定率", format="%.1f%%", min_value=0, max_value=10.0)
                },
                width="stretch",
                hide_index=True
            )
        else:
            st.info("過去90日間にAIが「回収日」と判定した営業日がありません。")