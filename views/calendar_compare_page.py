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
        shop_pred_stats = df_predict.groupby(shop_col).agg(
            平均期待度=('prediction_score', 'mean'),
            高期待台数=('prediction_score', lambda x: (x >= 0.65).sum()),
            全台数=('台番号', 'nunique')
        ).reset_index().sort_values('平均期待度', ascending=False)
        
        def get_eval_badge(score):
            if pd.isna(score): return "⚖️ 通常営業"
            elif score >= 0.20: return "🔥 還元予測"
            elif score < 0.10: return "🥶 回収警戒"
            else: return "⚖️ 通常営業"
            
        shop_pred_stats['営業予測'] = shop_pred_stats['平均期待度'].apply(get_eval_badge)
        
        if not shop_pred_stats.empty:
            st.dataframe(
                shop_pred_stats[[shop_col, '営業予測', '平均期待度', '高期待台数', '全台数']],
                column_config={
                    shop_col: st.column_config.TextColumn("店舗名"),
                    "営業予測": st.column_config.TextColumn("AI営業予測"),
                    "平均期待度": st.column_config.ProgressColumn("店舗全体の平均期待度", format="%.2f", min_value=0, max_value=1.0),
                    "高期待台数": st.column_config.NumberColumn("推奨台(65%以上)", format="%d台"),
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

    selected_month = st.selectbox("📅 表示する対象月を選択", available_months, index=0)
    
    df_recent = df_history[df_history['年月'] == selected_month].copy()
    
    if df_recent.empty:
        st.warning("指定された月のデータがありません。")
        return

    df_recent['表示日'] = df_recent[date_col].dt.strftime('%m/%d')
    
    # 高設定フラグの計算 (機種ごとの合算やREGで判定)
    specs = backend.get_machine_specs()
    def is_high(row):
        g = pd.to_numeric(row.get('累計ゲーム', 0), errors='coerce')
        b = pd.to_numeric(row.get('BIG', 0), errors='coerce')
        r = pd.to_numeric(row.get('REG', 0), errors='coerce')
        diff = pd.to_numeric(row.get('差枚', 0), errors='coerce')
        
        valid_play = (g >= 3000) or (g < 3000 and (diff <= -750 or diff >= 750))
        if not valid_play:
            return np.nan
        
        machine = row.get('機種名', '')
        matched_spec = backend.get_matched_spec_key(machine, specs)
        p_r_5 = 1.0 / specs[matched_spec].get("設定5", {"REG": 260.0})["REG"] if matched_spec else 1/260.0
        p_t_5 = 1.0 / specs[matched_spec].get("設定5", {"合算": 128.0})["合算"] if matched_spec else 1/128.0
        
        r_prob = r / g if g > 0 else 0
        t_prob = (b + r) / g if g > 0 else 0
        
        if r_prob >= p_r_5 or t_prob >= p_t_5: return 1
        return 0
        
    df_recent['is_high'] = df_recent.apply(is_high, axis=1)

    metric_choice = st.radio("表示する指標", ["平均差枚", "高設定率"], horizontal=True)
    
    if metric_choice == "平均差枚":
        pivot_df = df_recent.pivot_table(index=shop_col, columns='表示日', values='差枚', aggfunc='mean')
        fmt = "{:+.0f}"
        cmap = "RdYlBu_r" # 赤(プラス)〜青(マイナス)
        vmin, vmax = -300, 300
    else:
        pivot_df = df_recent.pivot_table(index=shop_col, columns='表示日', values='is_high', aggfunc='mean')
        pivot_df = pivot_df * 100 # %表記にするため
        fmt = "{:.1f}%"
        cmap = "Greens" # 緑が濃いほど高設定率が高い
        vmin, vmax = 0, 30
        
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