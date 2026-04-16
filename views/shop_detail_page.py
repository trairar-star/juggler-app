import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import altair as alt # type: ignore

# バックエンド処理をインポート
import backend
from utils import get_confidence_indicator

def _display_machine_detail_expander(row, index, shop_col, selected_shop, df_raw, df_events, specs, df_importance=None):
    """店舗別詳細ページで、ランキング上位台の詳細情報を表示するExpanderを描画する"""
    shop_name = row.get(shop_col, '')
    machine_name = row.get('機種名', '')
    machine_no = row.get('台番号', 'Unknown')
    
    # This function now renders the *content* of an expander, not the expander itself.
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

    c1, c2 = st.columns(2)
    with c1: st.metric("総回転", format_val(row.get('累計ゲーム', '-')))
    with c2: st.metric("週間差枚", f"{int(row.get('mean_7days_diff', 0)):+d}枚")
    
    c3, c4 = st.columns(2)
    with c3: st.metric("BIG", format_val(row.get('BIG', '-')))
    with c4: st.metric("REG", format_val(row.get('REG', '-')))
    
    c5, c6, c7, c8 = st.columns(4)
    
    try:
        games_val = float(row.get('累計ゲーム', 0))
        big_val = float(row.get('BIG', 0))
        reg_val = float(row.get('REG', 0))
        total_prob = (big_val + reg_val) / games_val if games_val > 0 else 0
    except:
        total_prob = 0
        
    with c5: st.metric("合算確率", format_prob(total_prob))
    with c6: st.metric("BIG確率", format_prob(row.get('BIG確率', 0)))
    with c7: st.metric("REG確率", format_prob(row.get('REG確率', 0)))
    with c8: 
        if '推定ぶどう確率' in row and pd.notna(row['推定ぶどう確率']):
            st.metric("推定🍇確率", f"1/{row['推定ぶどう確率']:.2f}")
        else:
            st.metric("推定🍇確率", "-")
    
    # --- 新規追加: 店舗ごとの重要特徴量トップ5の動的表示 ---
    if df_importance is not None and not df_importance.empty:
        imp_shop = df_importance[df_importance['shop_name'] == selected_shop].sort_values('importance', ascending=False)
        if not imp_shop.empty:
            feature_name_map = {
                '累計ゲーム': '前日 累計G数', 'REG確率': '前日 REG確率', 'BIG確率': '前日 BIG確率',
                '差枚': '前日 差枚数', '末尾番号': '台番号末尾', 'target_weekday': '予測日 曜日',
                'target_date_end_digit': '日付末尾', 'weekday_avg_diff': '店舗 曜日平均', 'weekday_high_rate': '店舗 曜日高設定率', 'mean_7days_reg_prob': '台 7日平均REG確率',
                'mean_7days_diff': '台 直近7日平均', 'median_7days_diff': '台 7日中央値', 'win_rate_7days': '台 7日間高設定率', 'plus_rate_7days': '台 7日間勝率',
                'mean_7days_games': '台 直近7日平均G数',
                '連続マイナス日数': '連続凹み日数', '連続プラス日数': '連続勝ち日数', '連続低稼働日数': '連続放置日数', 'is_prev_no_play': '前日 稼働なし',
                'machine_code': '機種', 'shop_code': '店舗',
                'reg_ratio': '前日 REG比率', 'is_corner': '角台フラグ', 'is_main_corner': 'メイン角フラグ', 'is_main_island': '目立つ島フラグ', 'is_wall_island': '壁側島フラグ',
                'neighbor_avg_diff': '両隣 平均差枚', 'neighbor_positive_count': '両隣 プラス台数',
                'is_neighbor_high_reg': '両隣 REG高設定水準', 'neighbor_reg_reliability_score': '両隣 REG信頼度スコア', 'neighbor_high_setting_count': '両隣 高設定示唆台数',
                'event_avg_diff': 'イベント 平均差枚', 'event_high_rate': 'イベント 高設定率', 'event_code': 'イベント 種類', 'event_rank_score': 'イベント ランク',
                'prev_event_rank_score': '前日(特日)ランク',
                'relative_games_ratio': '台 相対稼働率', 'is_new_machine': '新台フラグ', 'is_moved_machine': '配置変更フラグ',
                'shop_7days_avg_diff': '店舗 直近7日平均',
                'prev_shop_daily_avg_diff': '店舗 前日平均差枚',
                'prev_推定ぶどう確率': '前日 ぶどう確率', 'shop_avg_games': '店舗 平均稼働G数', 'shop_abandon_rate': '店舗 見切り割合',
                'event_x_machine_avg_diff': 'イベント×機種 差枚', 'event_x_machine_high_rate': 'イベント×機種 高設定率', 'event_x_end_digit_avg_diff': 'イベント×末尾 差枚',
                'cons_minus_total_diff': '連続凹み 吸込み量', 'machine_no_30days_avg_diff': '場所 30日平均', 'machine_no_30days_high_rate': '場所 30日高設定率', 'std_7days_diff': '台 7日差枚の標準偏差(荒れ具合)',
                'is_beginning_of_month': '月初フラグ', 'is_end_of_month': '月末フラグ', 'is_pension_day': '年金支給日フラグ',
                'shop_monthly_cumulative_diff': '店舗 月間累計差枚', 'prev_bonus_balance': '前日 BB/RB偏り', 'prev_unlucky_gap': '前日 不発度合い',
                'is_prev_up_trend_and_high_reg': '複合: 前日右肩上がり&高REG', 'is_prev_low_reg_and_good_diff': '複合: 前日低REG&差枚プラス',
                'prev_reg_reliability_score': '複合: 前日REG信頼度スコア',
                'is_low_play_high_reg': '複合: 低稼働&高設定挙動', 'is_hot_wd_and_heavy_lose': '複合: 還元曜&週間大凹み',
                'trend_v_recovery': '波 V字反発(負→勝)',
                'trend_cont_lose': '波 連続凹み(負→負)',
                'trend_cont_win': '波 連続据え(勝→勝)',
                'trend_down_rebound': '波 上げ戻し(勝→負)',
                'shop_pred_diff_7d_avg': '店舗 AI予測7日平均', 'predicted_diff': 'AI予測 差枚数'
            }
            
            st.markdown(f"**🌟 この店舗のAI評価 決定要因 (トップ10):**")
            st.caption("AIはこの店舗の傾向として、以下の10個の指標を特に重視しています。")
            
            top10 = imp_shop.head(10)
            cols1 = st.columns(5)
            cols2 = st.columns(5)
            all_cols = cols1 + cols2
            
            for idx, (_, imp_row) in enumerate(top10.iterrows()):
                f_key = imp_row['feature']
                f_name = feature_name_map.get(f_key, f_key)
                corr = imp_row.get('correlation', 0)
                val = row.get(f_key, '-')
                
                # 特徴量の種類に応じて、直感的な表現に変える
                if '確率' in f_name and '比率' not in f_name:
                    corr_str = "🔼 確率が良いほど+" if corr >= 0 else "🔽 確率が悪いほど+"
                elif '差枚' in f_name or '吸込み' in f_name:
                    corr_str = "🔼 出ているほど+" if corr >= 0 else "🔽 凹んでいるほど+"
                elif 'ゲーム' in f_name or 'G数' in f_name:
                    corr_str = "🔼 回されているほど+" if corr >= 0 else "🔽 放置されているほど+"
                elif '日数' in f_name:
                    corr_str = "🔼 続くほど+" if corr >= 0 else "🔽 少ないほど+"
                elif f_key.startswith('is_') or 'フラグ' in f_name or ('日' in f_name and isinstance(val, (int, float)) and val in [0,1]):
                    corr_str = "🔼 該当すれば+" if corr >= 0 else "🔽 該当しない方が+"
                else:
                    corr_str = "🔼 大きいほど+" if corr >= 0 else "🔽 小さいほど+"
                
                if isinstance(val, (int, float)) and not pd.isna(val):
                    if f_key.startswith('is_') or 'フラグ' in f_name or ('日' in f_name and val in [0,1]):
                        val_str = "あり" if val == 1 else "なし"
                    elif '確率' in f_name and val > 0 and val < 1: val_str = f"1/{int(1/val)}"
                    elif '差枚' in f_name or '吸込み' in f_name: val_str = f"{int(val):+d}枚"
                    elif 'ゲーム' in f_name or 'G数' in f_name: val_str = f"{int(val)}G"
                    elif '割合' in f_name or '率' in f_name: val_str = f"{val*100:.1f}%" if val <= 1.0 else f"{val:.1f}%"
                    else: val_str = str(int(val)) if float(val).is_integer() else f"{val:.2f}"
                else: val_str = str(val)
                    
                with all_cols[idx]:
                    # 🔽はデフォルトで赤になるため、inverseを指定して緑色にする
                    d_color = "normal" if corr >= 0 else "inverse"
                    st.metric(f_name, val_str, delta=corr_str, delta_color=d_color)

    # --- 新規追加: AI評価用 詳細特徴量データ ---
    st.markdown("**🔍 その他のサブ特徴量データ:**")
    st.caption("根拠の文章に現れない加点の理由を推測するための追加データです。")
    c_f1, c_f2, c_f3, c_f4 = st.columns(4)
    with c_f1:
        st.metric("前々日差枚", f"{int(row.get('prev_差枚', 0)):+d}枚" if pd.notna(row.get('prev_差枚')) else "-")
        st.metric("両隣平均差枚", f"{int(row.get('neighbor_avg_diff', 0)):+d}枚" if pd.notna(row.get('neighbor_avg_diff')) else "-")
    with c_f2:
        prev_r = row.get('prev_REG確率', 0)
        st.metric("前々日REG", f"1/{int(1/prev_r)}" if pd.notna(prev_r) and prev_r > 0 else "-")
        st.metric("島平均差枚", f"{int(row.get('island_avg_diff', 0)):+d}枚" if pd.notna(row.get('island_avg_diff')) else "-")
    with c_f3:
        st.metric("連続凹み日数", f"{int(row.get('連続マイナス日数', 0))}日" if pd.notna(row.get('連続マイナス日数')) else "-")
        st.metric("機種30日平均", f"{int(row.get('machine_30days_avg_diff', 0)):+d}枚" if pd.notna(row.get('machine_30days_avg_diff')) else "-")
    with c_f4:
        st.metric("相対稼働率", f"{row.get('relative_games_ratio', 1.0):.2f}倍" if pd.notna(row.get('relative_games_ratio')) else "-")
        st.metric("店舗7日平均", f"{int(row.get('shop_7days_avg_diff', 0)):+d}枚" if pd.notna(row.get('shop_7days_avg_diff')) else "-")

    matched_spec_key = backend.get_matched_spec_key(machine_name, specs)
    
    if matched_spec_key:
        st.markdown(f"**📚 {matched_spec_key} スペック目安:**")
        if matched_spec_key == "ジャグラー（デフォルト）":
            st.warning("⚠️ **注意:** この機種はスペックが未登録のため、デフォルト値で代用して分析しています。")
        spec_data = {k: v for k, v in specs[matched_spec_key].items() if k.startswith("設定")}
        spec_df = pd.DataFrame(spec_data).T
        st.dataframe(spec_df.style.format(formatter="1/{:.2f}"), width="stretch")
    
    # --- AIのやめどきアドバイス ---
    score = float(row.get('prediction_score', 0))
    reliability = row.get('予測信頼度', '🔸中')
    
    base_budget = 10000
    base_hamari = 250
    
    if score >= 0.4:
        base_budget += 15000; base_hamari += 250
    elif score >= 0.3:
        base_budget += 10000; base_hamari += 150
    elif score >= 0.2:
        base_budget += 5000; base_hamari += 50
        
    if '高' in reliability:
        base_budget += 5000; base_hamari += 100
    elif '低' in reliability:
        base_budget -= 5000; base_hamari -= 50
        
    base_budget = max(5000, min(40000, base_budget))
    base_hamari = max(150, min(600, base_hamari))
    
    reg_threshold = 300.0
    if matched_spec_key and "設定4" in specs[matched_spec_key]:
        reg_threshold = specs[matched_spec_key]["設定4"].get("REG", 300.0)

    st.info(f"**🤖 当日の稼働・やめどきアドバイス**\n\n"
            f"- **おすすめ予算**: 約 **{base_budget:,} 円**\n"
            f"- **ヤメ時ハマり目安**: **{base_hamari} G** ハマったら一旦冷静に。\n"
            f"- **REG確率の目安**: 当日の総回転数 1000G〜1500G の時点で、REG確率が **1/{int(reg_threshold * 1.15)}** より悪ければ撤退を推奨します。")

    # --- 過去の同曜日成績 ---
    target_wd = row.get('target_weekday', row.get('weekday', -1))
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
                same_wd_df = history_subset[history_subset['wd'] == target_wd].copy()
                
                # 低回転ノイズを除外 (1000G未満は集計から外す)
                same_wd_df = same_wd_df[same_wd_df['累計ゲーム'] >= 1000].copy()
                
                if not same_wd_df.empty:
                    same_wd_df['合算確率'] = (same_wd_df['BIG'] + same_wd_df['REG']) / same_wd_df['累計ゲーム'].replace(0, np.nan)
                    spec_reg_p = 1.0 / specs[matched_spec_key].get('設定5', {"REG": 260.0})["REG"] if matched_spec_key else (1/260)
                    spec_tot_p = 1.0 / specs[matched_spec_key].get('設定5', {"合算": 128.0})["合算"] if matched_spec_key else (1/128)
                    
                    count = len(same_wd_df)
                    avg_diff = same_wd_df['差枚'].mean()
                    
                    # 高設定率は3000G以上の台のみを母数として計算
                    high_valid_mask = same_wd_df['累計ゲーム'] >= 3000
                    high_mask = high_valid_mask & ((same_wd_df['REG確率'] >= spec_reg_p) | (same_wd_df['合算確率'] >= spec_tot_p))
                    
                    high_valid_count = high_valid_mask.sum()
                    win_rate = (high_mask.sum() / high_valid_count * 100) if high_valid_count > 0 else 0.0
                    
                    avg_reg = same_wd_df['REG'].mean()
                    
                    sw1, sw2 = st.columns(2)
                    with sw1: st.metric("集計回数", f"{count} 回")
                    with sw2: st.metric("高設定率", f"{win_rate:.1f} %")
                    
                    sw3, sw4 = st.columns(2)
                    with sw3: st.metric("平均差枚", f"{int(avg_diff):+d} 枚")
                    with sw4: st.metric("平均REG", f"{avg_reg:.1f} 回")
                else:
                    st.caption(f"※ 過去に{wd_name}曜日のデータが存在しません。")

    if 'prediction_score' in row:
        st.progress(float(row['prediction_score']), text=f"設定5以上確率: {float(row['prediction_score']) * 100:.1f}%")
        st.markdown("**🎯 AI推定の設定期待度 (擬似分布):**")
        score = float(row['prediction_score'])
        
        games = float(row.get('累計ゲーム', 0))
        big_count = float(row.get('BIG', 0))
        reg_count = float(row.get('REG', 0))
        
        likelihoods = [1.0] * 6
        
        if matched_spec_key and games > 0:
            import math
            ms = specs[matched_spec_key]
            s1 = ms.get("設定1", {"BIG": 280.0, "REG": 400.0})
            s4 = ms.get("設定4", {"BIG": 260.0, "REG": 300.0})
            s5 = ms.get("設定5", s4)
            s6 = ms.get("設定6", s5)
            
            full_specs = {1: s1, 4: s4, 5: s5, 6: s6}
            
            for s in [2, 3]:
                full_specs[s] = {}
                for k in ["BIG", "REG"]:
                    p1 = 1.0 / s1.get(k, 300.0)
                    p4 = 1.0 / s4.get(k, 300.0)
                    p_s = p1 + (p4 - p1) * (s - 1) / 3.0
                    full_specs[s][k] = 1.0 / p_s if p_s > 0 else 999.0
            
            log_L = []
            for i in range(1, 7):
                p_big = 1.0 / full_specs[i]["BIG"]
                p_reg = 1.0 / full_specs[i]["REG"]
                exp_big = games * p_big
                exp_reg = games * p_reg
                ll_big = big_count * math.log(exp_big) - exp_big if exp_big > 0 else 0
                ll_reg = reg_count * math.log(exp_reg) - exp_reg if exp_reg > 0 else 0
                log_L.append(ll_big + ll_reg)
            
            max_ll = max(log_L)
            likelihoods = [math.exp(ll - max_ll) for ll in log_L]
        
        l_low, l_high = likelihoods[0:3], likelihoods[3:6]
        sum_low, sum_high = sum(l_low), sum(l_high)
        p_ai_low, p_ai_high = max(0, 1.0 - score), max(0, score)
        
        sizes = [p_ai_low * (l / sum_low) if sum_low > 0 else p_ai_low / 3.0 for l in l_low] + \
                [p_ai_high * (h / sum_high) if sum_high > 0 else p_ai_high / 3.0 for h in l_high]

        labels, colors = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5', 'Set 6'], ['#cfd8dc', '#b0bec5', '#90a4ae', '#fff59d', '#ffcc80', '#ffab91']
        
        pie_df = pd.DataFrame({'設定': labels, '確率': sizes, '色': colors})
        pie_df = pie_df[pie_df['確率'] > 0] # 0%のものは除外
        
        pie_chart = alt.Chart(pie_df).mark_arc(innerRadius=20).encode(
            theta=alt.Theta(field="確率", type="quantitative"),
            color=alt.Color(field="設定", type="nominal", scale=alt.Scale(domain=labels, range=colors), legend=alt.Legend(title="推定設定", orient="right")),
            tooltip=['設定', alt.Tooltip('確率', format='.1%')]
        ).properties(height=250)
        
        st.altair_chart(pie_chart, width="stretch")

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

# --- ページ描画関数: 店舗別詳細データ ---
def render_shop_detail_page(df, df_raw, shop_col, df_events=None, df_train=None, df_pred_log=None, df_importance=None):
    st.header("🏪 店舗別 詳細データ & 傾向分析")
    
    # 店舗・機種選択 (メイン画面上部)
    col_filter1, col_filter2 = st.columns(2)
    
    selected_shop = '全て'
    if shop_col in df.columns:
        shops = ['全て'] + list(df[shop_col].unique())
        
        default_index = 0
        saved_shop = st.session_state.get("global_selected_shop", "全て")
        if saved_shop in ["全店舗", "店舗を選択してください"]:
            saved_shop = "全て"
        if saved_shop in shops:
            default_index = shops.index(saved_shop)
            
        selected_shop = col_filter1.selectbox("店舗名を選択", shops, index=default_index, key="shop_detail_shop")
        st.session_state["global_selected_shop"] = selected_shop
        
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

    # 機種スペックの読み込み (スコープ全体で使用するため上部で定義)
    specs = backend.get_machine_specs()

    # --- 店癖トップ5の抽出と明日の候補台へのマッピング ---
    top_trends_df = None
    worst_trends_df = None
    base_win_rate = 0
    
    if df_train is not None and not df_train.empty and shop_col in df_train.columns:
        all_trends_dict = backend._calculate_shop_trends(df_train, shop_col, specs)

        # 画面表示用変数の設定 (選択された店舗がある場合)
        if selected_shop != '全て' and selected_shop in all_trends_dict:
            base_win_rate = all_trends_dict[selected_shop]['base_win_rate']
            top_trends_df = all_trends_dict[selected_shop]['top_df']
            worst_trends_df = all_trends_dict[selected_shop]['worst_df']

    tab_pred, tab_trend = st.tabs(["🔮 明日の予測ランキング", "🔬 過去の傾向分析 (勝利の法則)"])

    with tab_pred:
        # --- 🎊 周年記念イベント アラート & カウントダウン ---
        if selected_shop == '全て' and df_events is not None and not df_events.empty:
            pred_date = pd.NaT
            if 'next_date' in df.columns:
                pred_date = df['next_date'].dropna().max()
            elif '対象日付' in df.columns:
                pred_date = df['対象日付'].dropna().max() + pd.Timedelta(days=1)
                
            if pd.notna(pred_date):
                current_date = pred_date.date()
                
                # --- 周年イベントの毎年のループ処理 ---
                valid_events = df_events.copy()
                anniversary_events = valid_events[valid_events['イベント名'].astype(str).str.contains('周年')].copy()
                
                if not anniversary_events.empty:
                    expanded_list = []
                    for _, ev in anniversary_events.iterrows():
                        ev_date = ev['イベント日付'].date()
                        try: d_this = ev_date.replace(year=current_date.year)
                        except ValueError: d_this = (ev['イベント日付'] + pd.offsets.DateOffset(years=(current_date.year - ev['イベント日付'].year))).date()
                        ev_this = ev.copy(); ev_this['イベント日付'] = pd.Timestamp(d_this); expanded_list.append(ev_this)
                        
                        try: d_next = ev_date.replace(year=current_date.year + 1)
                        except ValueError: d_next = (ev['イベント日付'] + pd.offsets.DateOffset(years=(current_date.year + 1 - ev['イベント日付'].year))).date()
                        ev_next = ev.copy(); ev_next['イベント日付'] = pd.Timestamp(d_next); expanded_list.append(ev_next)
                        
                    valid_events = pd.concat([valid_events, pd.DataFrame(expanded_list)], ignore_index=True)
                    valid_events = valid_events.drop_duplicates(subset=['店名', 'イベント日付', 'イベント名'])

                # 今日〜30日後までの周年イベントを取得
                future_events = valid_events[
                    (valid_events['イベント日付'].dt.date >= current_date) & 
                    (valid_events['イベント日付'].dt.date <= current_date + pd.Timedelta(days=30)) &
                    ((valid_events['イベントランク'].astype(str) == 'SS (周年)') | (valid_events['イベント名'].astype(str).str.contains('周年|リニューアル|グランド')))
                ].copy()
                
                if not future_events.empty:
                    future_events['days_left'] = (future_events['イベント日付'].dt.date - current_date).dt.days
                    future_events = future_events.sort_values('days_left')
                    
                    today_events = future_events[future_events['days_left'] == 0]
                    upcoming_events = future_events[future_events['days_left'] > 0]
                    
                    if not today_events.empty:
                        html_str = f"""
                        <div style="background-color: #ffebee; border: 2px solid #f44336; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <h3 style="color: #c62828; margin-top: 0;">🎊 【特報】本日 特大イベント開催！ ({pred_date.strftime('%m/%d')})</h3>
                            <p style="color: #b71c1c; font-size: 1.1em; font-weight: bold; margin-bottom: 10px;">本日は以下の店舗で「周年・リニューアル」クラスの超特大イベントが予定されています！お見逃しなく！</p>
                            <ul style="color: #b71c1c; font-size: 1.1em; font-weight: bold; margin-bottom: 0;">
                        """
                        for _, r in today_events.iterrows():
                            html_str += f"<li>{r['店名']}：{r['イベント名']}</li>"
                        html_str += "</ul></div>"
                        st.markdown(html_str, unsafe_allow_html=True)
                        
                    if not upcoming_events.empty:
                        html_str = f"""
                        <div style="background-color: #fff8e1; border: 2px solid #ffb300; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <h4 style="color: #f57f17; margin-top: 0; margin-bottom: 10px;">⏳ 開催間近！ 特大イベント カウントダウン</h4>
                            <ul style="color: #e65100; font-size: 1.05em; font-weight: bold; margin-bottom: 0; padding-left: 20px;">
                        """
                        for _, r in upcoming_events.iterrows():
                            days = r['days_left']
                            date_str = r['イベント日付'].strftime('%m/%d')
                            html_str += f"<li>{r['店名']}：{r['イベント名']} ({date_str}) ･･･ <span style='color: #d84315; font-size: 1.2em;'>あと {days} 日！</span></li>"
                        html_str += "</ul></div>"
                        st.markdown(html_str, unsafe_allow_html=True)

        # --- 👑 本日のおすすめ店舗 ピックアップ ---
        if selected_shop == '全て' and shop_col in df.columns and '予測差枚数' in df.columns:
            shop_mean_diff = df.groupby(shop_col)['予測差枚数'].mean() if not df.empty else pd.Series()
            if not shop_mean_diff.empty:
                best_shop = shop_mean_diff.idxmax()
                best_diff = shop_mean_diff.max()
                
                if best_diff >= 100:
                    st.success(f"👑 **本日のおすすめ店舗**: **{best_shop}** (予測店舗平均: **+{int(best_diff)}枚** / 🔥還元予測)\n\n系列・登録店舗の中で最も出玉（差枚）が甘く、全体的に還元される可能性が高いとAIが推奨しています！")
                elif best_diff >= 0:
                    st.info(f"👍 **本日の注目店舗**: **{best_shop}** (予測店舗平均: **+{int(best_diff)}枚** / ⚖️通常営業)\n\n突出した還元予測の店舗はありませんが、登録店舗の中では一番差枚が甘いと予測されています。")
                else:
                    st.warning(f"🥶 **全店舗 回収警戒**: 本日は全店舗の予測平均差枚がマイナスで (一番マシな **{best_shop}** でも {int(best_diff)}枚)、厳しい戦いが予想されます。無理な勝負は避けるのが無難です。")

        # --- 前日データ欠損アラート ---
        if not df.empty and '対象日付' in df.columns:
            latest_data_date = df['対象日付'].max().date()
            pred_target_date = df['next_date'].max().date()
            
            if (pred_target_date - latest_data_date).days > 1:
                import datetime
                missing_date = pred_target_date - datetime.timedelta(days=1)
                st.warning(f"⚠️ **前日データ欠損**: {missing_date.strftime('%m/%d')} のデータがありません。AIは2日前のデータで予測しているため、精度が低下している可能性があります。")

        # --- � AI本日の立ち回りアドバイス (店舗個別) ---
        if selected_shop != '全て' and not df.empty:
            st.markdown("### 💬 AI本日の立ち回りアドバイス")
            advice_list = []
            
            # 予測対象日の取得
            pred_date = pd.NaT
            if 'next_date' in df.columns:
                pred_date = df['next_date'].dropna().max()
            elif '対象日付' in df.columns:
                pred_date = df['対象日付'].dropna().max() + pd.Timedelta(days=1)
                
            # --- 1. 基本設定配分 (ベース高設定率) ---
            if base_win_rate > 0:
                ratio = max(1, int(100 / base_win_rate))
                advice_list.append(f"📊 **基本の設定配分**: この店舗の通常営業時の高設定(設定5基準)の投入率は約 **{base_win_rate:.1f}%**（およそ **{ratio}台に1台** の割合）です。")

            # --- 2. 特定日・曜日の傾向 ---
            if pd.notna(pred_date) and not df_raw_shop.empty and '対象日付' in df_raw_shop.columns:
                target_wd = pred_date.dayofweek
                target_digit = pred_date.day % 10
                wd_str = ['月', '火', '水', '木', '金', '土', '日'][target_wd]
                
                # 特定日（日付末尾）の傾向
                digit_df = df_raw_shop[df_raw_shop['対象日付'].dt.day % 10 == target_digit]
                if not digit_df.empty:
                    digit_avg_diff = digit_df['差枚'].mean()
                    if digit_avg_diff > 50:
                        advice_list.append(f"🔢 **特定日の傾向 ({target_digit}のつく日)**: 本日は **{target_digit}のつく日** です。過去の同日は店舗平均 **+{int(digit_avg_diff)}枚** と甘く使われており、勝率が高まるチャンス日です！")
                    elif digit_avg_diff < -50:
                        advice_list.append(f"🚨 **特定日の傾向 ({target_digit}のつく日) [回収警戒]**: 本日は **{target_digit}のつく日** ですが、過去の傾向では店舗平均 **{int(digit_avg_diff)}枚** とかなり厳しめです。基本的には勝負を避けるべきですが、**AIが激アツと評価している上位台であれば試してみる価値はあります。** ただし、少しでも挙動が悪ければ即撤退を徹底してください。")

                # 曜日の傾向
                wd_df = df_raw_shop[df_raw_shop['対象日付'].dt.dayofweek == target_wd]
                if not wd_df.empty:
                    wd_avg_diff = wd_df['差枚'].mean()
                    if wd_avg_diff > 50:
                        advice_list.append(f"📅 **曜日の傾向 ({wd_str}曜)**: この店舗は **{wd_str}曜日** に出玉を出してくる傾向があります (過去平均 **+{int(wd_avg_diff)}枚**)。曜日別の狙い目として有効です。")
                    elif wd_avg_diff < -50:
                        advice_list.append(f"🚨 **曜日の傾向 ({wd_str}曜) [回収警戒]**: 過去のデータ上、**{wd_str}曜日** は回収傾向が強いです (過去平均 **{int(wd_avg_diff)}枚**)。基本的には稼働を控えるのが無難ですが、**AIの激アツ台に絞って攻める場合も、撤退基準は普段より厳しく設定してください。**")

            # --- 3. 客層レベルと後ヅモ難易度 ---
            if not df_raw_shop.empty and '累計ゲーム' in df_raw_shop.columns:
                avg_kado = df_raw_shop['累計ゲーム'].mean()
                if avg_kado >= 5000:
                    advice_list.append(f"👥 **客層レベル(後ヅモ)**: 平均稼働が **{int(avg_kado)}G** と高く、客層のレベルが非常に高い（または専業が多い）店舗です。優秀台は空きにくいため、朝イチの台選びが勝敗を分けます。")
                elif avg_kado <= 3500:
                    advice_list.append(f"👥 **客層レベル(後ヅモ)**: 平均稼働が **{int(avg_kado)}G** と低めです。ライトユーザーが多く見切りが早いため、夕方以降からでも履歴打ち（後ヅモ）できるチャンスが十分にあります。")

            # --- 4. 還元/回収モード判定 ---
            shop_7d_diff = df['shop_7days_avg_diff'].mean() if 'shop_7days_avg_diff' in df.columns else 0
            if shop_7d_diff > 100:
                advice_list.append(f"📈 **還元モード濃厚**: 直近1週間の店舗全体がプラス推移 (平均 **+{int(shop_7d_diff)}枚**) しており、ベース設定が高めです。積極的に攻める価値があります。")
            elif shop_7d_diff < -100:
                advice_list.append(f"📉 **回収モード警戒**: 直近1週間の店舗全体がマイナス推移 (平均 **{int(shop_7d_diff)}枚**) しており、設定状況は厳しめです。強い根拠がない台は早めの見切りを推奨します。")
                
            # イベント状況
            if 'イベント名' in df.columns:
                event_names = df['イベント名'].dropna().unique()
                event_names = [e for e in event_names if e != '通常' and str(e).strip() != '']
                if event_names:
                    ev_str = "、".join(event_names)
                    
                    past_ev_msg = ""
                    if df_events is not None and not df_events.empty and not df_raw_shop.empty:
                        past_ev_diffs = []
                        for en in event_names:
                            past_dates = df_events[(df_events['店名'] == selected_shop) & (df_events['イベント名'] == en) & (df_events['イベント日付'].dt.date != pred_date.date())]['イベント日付'].dt.date.unique()
                            if len(past_dates) > 0:
                                past_raw = df_raw_shop[df_raw_shop['対象日付'].dt.date.isin(past_dates)]
                                if not past_raw.empty:
                                    avg_d = past_raw.groupby('対象日付')['差枚'].mean().mean()
                                    past_ev_diffs.append(f"「{en}」(過去平均 **{int(avg_d):+d}枚**)")
                        if past_ev_diffs:
                            past_ev_msg = f" 過去の同イベント実績は {', '.join(past_ev_diffs)} となっており、当日の押し引きの参考にしてください。"
                            
                    advice_list.append(f"🎉 **イベント対象日**: 本日は「**{ev_str}**」が予定されています。{past_ev_msg}")

            # 店癖に基づく立ち回り
            if top_trends_df is not None and not top_trends_df.empty:
                hot_conditions = top_trends_df['条件'].tolist()
                cond_str = "、".join(hot_conditions)
                advice_list.append(f"🎯 **有効な店癖**: この店舗では『**{cond_str}**』が高設定になる傾向が強いです。台選びの際はこれらを最優先で意識してください。")
                
                if any("末尾" in c for c in hot_conditions):
                    advice_list.append("🔢 **末尾の意識**: 特定の末尾に設定を寄せる傾向があります。自分の台だけでなく、同じ末尾の別機種の挙動も常にチェックして「当たり末尾」を早期に察知しましょう。")
                if any("据え" in c or "勝ち" in c for c in hot_conditions):
                    advice_list.append("🔁 **据え置き狙い**: 前日出ている台をそのまま据え置く（または高設定を連日投入する）クセがあります。前日の優秀台は朝イチの有力候補です。")
                if any("負け" in c or "凹み" in c for c in hot_conditions):
                    advice_list.append("⤴️ **上げリセット狙い**: 前日や数日間大きく凹んでいる台に対して「お詫び（設定上げ）」をしてくる傾向があります。不発台のガックンや朝イチ挙動に注目です。")
                if any("角" in c for c in hot_conditions):
                    advice_list.append("🪑 **角台優遇**: 角台（または角周辺）を強くする傾向があります。迷ったら角寄りの台を選ぶのがベターです。")

            # 並び・島・特殊パターンに関するアドバイス (特徴量重要度から攻略ポイントを自動生成)
            if df_importance is not None and not df_importance.empty:
                imp_shop = df_importance[df_importance['shop_name'] == selected_shop]
                if not imp_shop.empty:
                    imp_shop_sorted = imp_shop.sort_values('importance', ascending=False).reset_index(drop=True)
                    top_features = imp_shop_sorted.head(8)['feature'].tolist()
                    
                    if '末尾番号' in imp_shop_sorted['feature'].values:
                        end_digit_rank = imp_shop_sorted[imp_shop_sorted['feature'] == '末尾番号'].index[0] + 1
                        total_features = len(imp_shop_sorted)
                        if end_digit_rank <= 3:
                            advice_list.append(f"🔥 **末尾の重要度 (極高)**: AI分析において「末尾番号」は全{total_features}指標中 **第{end_digit_rank}位** の超重要要素です！台選びの際は『当たり末尾』を探すことを最優先に立ち回ってください。")
                        elif end_digit_rank <= 10:
                            advice_list.append(f"🌟 **末尾の重要度 (高)**: AI分析において「末尾番号」は全{total_features}指標中 **第{end_digit_rank}位** と重視されています。周りの台の末尾の挙動には常に気を配りましょう。")
                        elif end_digit_rank <= 20:
                            advice_list.append(f"⚖️ **末尾の重要度 (中)**: AI分析において「末尾番号」は全{total_features}指標中 **第{end_digit_rank}位** です。意識はしつつも、末尾単体の根拠だけで粘るのは危険です。")
                        else:
                            advice_list.append(f"⚠️ **末尾の重要度 (低)**: AI分析において「末尾番号」は全{total_features}指標中 **第{end_digit_rank}位** とあまり重視されていません。この店舗では安易な末尾狙いは危険なため、他の要素を優先してください。")

                    if 'neighbor_avg_diff' in top_features:
                        advice_list.append("🤝 **並び・塊に注意**: AIの分析上、この店は「両隣の差枚（並び）」が設定予測に強く影響しています。自分の台が良くても両隣が死んでいればフェイクの可能性があり、逆に両隣が強ければ「3台並び」などの対象になっている可能性があります。")
                    if 'island_avg_diff' in top_features:
                        advice_list.append("🏝️ **全台系・列に注意**: 「島（列）全体の差枚」の重要度が高いため、列単位での全台系や半ヅキなどをやってくる可能性があります。周囲の活気をよく観察してください。")
                    if 'machine_no_30days_avg_diff' in top_features:
                        advice_list.append("📍 **定位置・看板台を意識**: 「台番号ごとの過去成績」が非常に重視されています。この店には『いつも設定が入りやすい特定の場所（看板台）』が存在する可能性が高いです。")
                    if 'cons_minus_total_diff' in top_features:
                        advice_list.append("📈 **強烈なお詫び・反発狙い**: 「連続マイナス中の合計吸い込み量」が重要視されています。単に凹んでいるだけでなく『客のヘイトが溜まっている（極端に吸い込んだ）台』への上げリセットを狙うのが有効です。")
                    if 'event_x_machine_avg_diff' in top_features:
                        advice_list.append("🎯 **特効機種の存在**: 「イベント×機種の強さ」が重視されています。普段の営業や特定のイベントで『露骨に甘くなる機種』が存在するクセがあります。")
                    if 'event_x_end_digit_avg_diff' in top_features:
                        advice_list.append("🔢 **当たり末尾の存在**: 「イベント×末尾の強さ」が重視されています。イベント日は『特定の末尾』に当たりを寄せてくる傾向が強いため、周りの状況（同じ末尾の挙動）を要チェックです。")
                    if 'prev_unlucky_gap' in top_features:
                        advice_list.append("🔄 **高設定の不発・据え置き**: 「前日の不発度合い（REGは引けたが差枚マイナス）」が重視されています。前日悔しい思いをした高設定挙動の不発台は、そのまま翌日も据え置かれるチャンス大です。")
                    if 'prev_bonus_balance' in top_features:
                        advice_list.append("⚖️ **BIG・REGの偏り反発**: 「前日のREG先行具合」が重視されています。REGに極端に偏って負けた台（BB欠損）は、翌日の狙い目として非常に優秀です。")

            # 警戒パターン
            if worst_trends_df is not None and not worst_trends_df.empty:
                cold_conditions = worst_trends_df['条件'].tolist()
                cold_str = "、".join(cold_conditions)
                advice_list.append(f"⚠️ **警戒パターン**: 逆に『**{cold_str}**』の条件に当てはまる台は回収（低設定）の危険性が高いため、手を出さないのが無難です。")

            # 機種別のアドバイス
            if '機種名' in df.columns:
                mac_advices = []
                # 1. 過去30日間で甘く使われている機種
                if 'machine_30days_avg_diff' in df.columns:
                    mac_stats = df.groupby('機種名').agg(
                        avg_diff=('machine_30days_avg_diff', 'mean'),
                        count=('台番号', 'count')
                    ).reset_index()
                    mac_stats = mac_stats[mac_stats['count'] >= 3] # サンプル不足（バラエティ等）を除外
                    mac_stats = mac_stats.dropna(subset=['avg_diff'])
                    if not mac_stats.empty:
                        best_mac_hist = mac_stats.loc[mac_stats['avg_diff'].idxmax()]
                        worst_mac_hist = mac_stats.loc[mac_stats['avg_diff'].idxmin()]
                        
                        if best_mac_hist['avg_diff'] > 150:
                            mac_advices.append(f"『**{best_mac_hist['機種名']}**』(過去30日平均 **+{int(best_mac_hist['avg_diff'])}枚**)")
                        if worst_mac_hist['avg_diff'] < -150:
                            advice_list.append(f"🧊 **冷遇機種に警戒**: 過去30日間のデータから、この店舗では『**{worst_mac_hist['機種名']}**』がかなり辛く使われています (平均 **{int(worst_mac_hist['avg_diff'])}枚**)。この機種を打つ際は特に慎重に立ち回ってください。")
                
                # 2. 明日AIが特に熱いと見ている機種
                if 'prediction_score' in df.columns:
                    mac_pred_stats = df.groupby('機種名').agg(avg_score=('prediction_score', 'mean'), count=('台番号', 'count')).reset_index()
                    mac_pred_stats = mac_pred_stats[mac_pred_stats['count'] >= 3]
                    mac_pred_stats = mac_pred_stats.dropna(subset=['avg_score'])
                    if not mac_pred_stats.empty:
                        best_mac_pred = mac_pred_stats.loc[mac_pred_stats['avg_score'].idxmax()]
                        if best_mac_pred['avg_score'] >= 0.50 and not any(best_mac_pred['機種名'] in adv for adv in mac_advices):
                            mac_advices.append(f"『**{best_mac_pred['機種名']}**』(明日のAI平均期待度 **{int(best_mac_pred['avg_score']*100)}%**)")
                                
                if mac_advices:
                    adv_str = " または ".join(mac_advices)
                    advice_list.append(f"🎰 **おすすめの狙い目機種**: 現在のデータとAIの予測から、**{adv_str}** に設定が入りやすい（ベースが高い）と推測されます。機種選びに迷ったらこのあたりから攻めるのがオススメです。")

            # 表示
            st.info("\n\n".join([f"- {adv}" for adv in advice_list]))
            st.divider()

        # --- 島（列）別 期待度ランキング (追加) ---
        if selected_shop != '全て' and 'island_id' in df.columns:
            island_pred_df = df[df['island_id'] != "Unknown"].copy()
            if not island_pred_df.empty:
                with st.expander("🏝️ 島（列）別 期待度ランキング", expanded=True):
                    st.caption("AIの予測スコアを島（列）ごとに平均し、「どの島全体に設定が入りそうか（還元島か回収島か）」をランキング化しています。")
                    island_pred_df['島名'] = island_pred_df['island_id'].apply(lambda x: str(x).split('_', 1)[1] if '_' in str(x) else str(x))
                    
                    isl_stats = island_pred_df.groupby('島名').agg(
                        平均期待度=('prediction_score', 'mean'),
                        激アツ台数=('prediction_score', lambda x: (x >= 0.30).sum()),
                        全台数=('台番号', 'count')
                    ).reset_index().sort_values('平均期待度', ascending=False)
                    
                    isl_stats['営業予測'] = isl_stats.apply(
                        lambda row: backend.classify_shop_eval(row.get('予測平均差枚'), row.get('全台数', 20), is_prediction=False).replace('営業', '島').replace('日', '島'), axis=1
                    )
                    isl_stats['平均期待度'] = isl_stats['平均期待度'] * 100
                    
                    st.dataframe(
                        isl_stats,
                        column_config={
                            "島名": st.column_config.TextColumn("島名"),
                            "営業予測": st.column_config.TextColumn("島全体の熱さ"),
                            "平均期待度": st.column_config.ProgressColumn("平均期待度", format="%.1f%%", min_value=0, max_value=100),
                            "激アツ台数": st.column_config.NumberColumn("推奨台(30%以上)", format="%d台"),
                            "全台数": st.column_config.NumberColumn("総台数", format="%d台")
                        },
                        hide_index=True,
                        width="stretch"
                    )

        # --- 店舗別 期待度ランキング (追加) ---
        if selected_shop == '全て' and shop_col in df.columns:
            with st.expander("🏬 店舗別 期待度ランキング", expanded=True):
                eval_period = st.radio("AI正答率・勝率の集計期間", ["直近1週間", "直近1ヶ月", "全期間"], index=0, horizontal=True, help="「直近1週間」でデータが0件になる場合は、期間を延ばして確認してください。")
                
                # 店舗ごとの集計
                shop_stats = df.groupby(shop_col).agg(
                    平均スコア=('prediction_score', 'mean'),
                    推奨台数=('prediction_score', lambda x: (x >= 0.30).sum()),
                    全台数=('台番号', 'nunique')
                ).reset_index()
                
                if '予測差枚数' in df.columns:
                    diff_stats = df.groupby(shop_col).agg(予測平均差枚=('予測差枚数', 'mean')).reset_index()
                    shop_stats = pd.merge(shop_stats, diff_stats, on=shop_col, how='left')
                else:
                    shop_stats['予測平均差枚'] = np.nan

                # --- 収集日数の計算 ---
                shop_days_map = {}
                if not df_raw.empty and shop_col in df_raw.columns and '対象日付' in df_raw.columns:
                    days_stats = df_raw.groupby(shop_col)['対象日付'].nunique().reset_index()
                    shop_days_map = dict(zip(days_stats[shop_col], days_stats['対象日付']))
                shop_stats['収集日数'] = shop_stats[shop_col].map(shop_days_map).fillna(0).astype(int)
                
                # --- ガチ予測ログベースのAI正答率・勝率計算 ---
                ai_accuracy_map = {}
                ai_win_rate_map = {}
                ai_acc_str_map = {}
                ai_win_str_map = {}
                
                if df_pred_log is not None and not df_pred_log.empty and not df_raw.empty:
                    df_pred_log_temp = df_pred_log.copy()
                    if '予測対象日' in df_pred_log_temp.columns:
                        df_pred_log_temp['予測対象日'] = pd.to_datetime(df_pred_log_temp['予測対象日'], errors='coerce')
                    df_pred_log_temp['対象日付'] = pd.to_datetime(df_pred_log_temp['対象日付'], errors='coerce')
                    
                    if '予測対象日' in df_pred_log_temp.columns:
                        df_pred_log_temp['予測対象日_merge'] = df_pred_log_temp['予測対象日'].fillna(df_pred_log_temp['対象日付'] + pd.Timedelta(days=1))
                    else:
                        df_pred_log_temp['予測対象日_merge'] = df_pred_log_temp['対象日付'] + pd.Timedelta(days=1)
                        
                    shop_col_pred = '店名' if '店名' in df_pred_log_temp.columns else '店舗名'
                    if shop_col != shop_col_pred:
                        df_pred_log_temp = df_pred_log_temp.rename(columns={shop_col_pred: shop_col})
                        
                    df_pred_log_temp['台番号'] = df_pred_log_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
                    
                    if '実行日時' in df_pred_log_temp.columns:
                        df_pred_log_temp = df_pred_log_temp.sort_values('実行日時', ascending=False).drop_duplicates(
                            subset=['予測対象日_merge', shop_col, '台番号'], keep='first'
                        )
        
                    df_raw_temp = df_raw.copy()
                    df_raw_temp['台番号'] = df_raw_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
                    df_raw_temp['対象日付'] = pd.to_datetime(df_raw_temp['対象日付'], errors='coerce')
                    
                    merged = pd.merge(
                        df_pred_log_temp,
                        df_raw_temp,
                        left_on=['予測対象日_merge', shop_col, '台番号'],
                        right_on=['対象日付', shop_col, '台番号'],
                        how='inner',
                        suffixes=('_pred', '_raw')
                    )
                    
                    if not merged.empty and 'prediction_score' in merged.columns:
                        if eval_period != "全期間":
                            days = 7 if eval_period == "直近1週間" else 30
                            max_date = merged['予測対象日_merge'].max()
                            cutoff_date = max_date - pd.Timedelta(days=days)
                            merged = merged[merged['予測対象日_merge'] > cutoff_date].copy()
                        
                        merged['prediction_score'] = pd.to_numeric(merged['prediction_score'], errors='coerce')
                        
                        # 店舗の規模（ジャグラー全台数）に応じて、評価対象とするトップ台数を動的に変動させる（約10%、最低3台〜最大10台）
                        shop_machine_counts = df.groupby(shop_col)['台番号'].nunique().to_dict()
                        merged['top_k_threshold'] = merged[shop_col].apply(lambda x: max(3, min(10, int(shop_machine_counts.get(x, 50) * 0.10))))
                        
                        merged['daily_rank'] = merged.groupby(['予測対象日_merge', shop_col])['prediction_score'].rank(method='first', ascending=False)
                        high_expect_df = merged[merged['daily_rank'] <= merged['top_k_threshold']].copy()
                        
                        if not high_expect_df.empty:
                            act_b = pd.to_numeric(high_expect_df['BIG'], errors='coerce').fillna(0)
                            act_r = pd.to_numeric(high_expect_df['REG'], errors='coerce').fillna(0)
                            act_g = pd.to_numeric(high_expect_df['累計ゲーム'], errors='coerce').fillna(0)
                            
                            spec_reg_val = high_expect_df['機種名_raw'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
                            spec_tot_val = high_expect_df['機種名_raw'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
                            
                            reg_prob_den = np.where(act_r > 0, act_g / act_r, 0)
                            tot_prob_den = np.where((act_b + act_r) > 0, act_g / (act_b + act_r), 0)
                            
                            high_expect_df['valid_high_play'] = (act_g >= 3000)
                            high_expect_df['is_high_setting'] = (
                                (((reg_prob_den > 0) & (reg_prob_den <= spec_reg_val)) | 
                                 ((tot_prob_den > 0) & (tot_prob_den <= spec_tot_val)))
                            ).astype(int)
                            
                            act_diff = pd.to_numeric(high_expect_df['差枚'], errors='coerce').fillna(0)
                            high_expect_df['valid_play'] = (act_g >= 3000) | ((act_g < 3000) & ((act_diff <= -750) | (act_diff >= 750)))
                            high_expect_df['valid_win'] = high_expect_df['valid_play'] & (act_diff > 0)
                            high_expect_df['valid_high'] = high_expect_df['valid_high_play'] & (high_expect_df['is_high_setting'] == 1)
        
                            acc_stats = high_expect_df.groupby(shop_col).agg(
                                正答数=('valid_high', 'sum'),
                                高設定有効数=('valid_high_play', 'sum'),
                                有効稼働数=('valid_play', 'sum'),
                                勝数=('valid_win', 'sum'),
                                サンプル数=('台番号', 'count')
                            ).reset_index()
                            acc_stats['勝率'] = np.where(acc_stats['有効稼働数'] > 0, acc_stats['勝数'] / acc_stats['有効稼働数'], 0.0)
                            acc_stats['正答率'] = np.where(acc_stats['高設定有効数'] > 0, acc_stats['正答数'] / acc_stats['高設定有効数'], 0.0)
                            
                            ai_accuracy_map = dict(zip(acc_stats[shop_col], acc_stats['正答率']))
                            ai_win_rate_map = dict(zip(acc_stats[shop_col], acc_stats['勝率']))
                            for _, r in acc_stats.iterrows():
                                shop = r[shop_col]
                                ai_acc_str_map[shop] = f"{r['正答率']*100:.1f}% ({int(r['正答数'])}/{int(r['高設定有効数'])}台)"
                                ai_win_str_map[shop] = f"{r['勝率']*100:.1f}% ({int(r['勝数'])}/{int(r['有効稼働数'])}台)"
                    
                shop_stats['AI正答率_数値'] = shop_stats[shop_col].map(ai_accuracy_map).fillna(0.0) * 100
                shop_stats['AI推奨台勝率_数値'] = shop_stats[shop_col].map(ai_win_rate_map).fillna(0.0) * 100
                
                shop_stats['AI正答率'] = shop_stats[shop_col].map(ai_acc_str_map).fillna("- (0/0台)")
                shop_stats['AI推奨台勝率'] = shop_stats[shop_col].map(ai_win_str_map).fillna("- (0/0台)")
                
                # --- AI実績に基づくペナルティ計算 ---
                def calc_sort_score(row):
                    score = row['平均スコア']
                    sample_count = 0
                    acc_str = str(row.get('AI正答率', ''))
                    if '/' in acc_str and '台)' in acc_str:
                        try:
                            sample_count = int(acc_str.split('/')[1].split('台)')[0])
                        except:
                            pass
        
                    if sample_count >= 5: # ガチ予測ログはサンプルが溜まりにくいため、5件以上でペナルティ評価
                        if row['AI正答率_数値'] > 0:
                            if row['AI正答率_数値'] < 30: score -= 0.15  # 30%未満なら重いペナルティ
                            elif row['AI正答率_数値'] < 40: score -= 0.05  # 40%未満なら軽いペナルティ
                        if row['AI推奨台勝率_数値'] > 0:
                            if row['AI推奨台勝率_数値'] < 30: score -= 0.15
                            elif row['AI推奨台勝率_数値'] < 40: score -= 0.05
                    return score
        
                shop_stats['ソート用スコア'] = shop_stats.apply(calc_sort_score, axis=1)
                
                # 営業予測バッジの追加
                def get_shop_eval_badge(row):
                    return backend.classify_shop_eval(row.get('予測平均差枚'), row.get('全台数', 50), is_prediction=True).replace('営業', '').replace('日', '')
                    
                shop_stats['営業予測'] = shop_stats.apply(get_shop_eval_badge, axis=1)
                
                # ソート用スコア（ペナルティ適用後）が高い順にソート
                shop_stats = shop_stats.sort_values('ソート用スコア', ascending=False)
                
                st.caption(f"※{eval_period}の**ガチ予測（保存ログ）に対する正答率と勝率**です。この数値が極端に低い（40%未満）店舗は、AIの予測が通用しにくい（フェイクが多い等）と判断し、ランキング順位を下げるペナルティを適用しています。（※サンプル5台以上で適用）")
                st.dataframe(
                    shop_stats[[shop_col, '営業予測', '予測平均差枚', '平均スコア', '推奨台数', '全台数', '収集日数', 'AI正答率', 'AI推奨台勝率']],
                    column_config={
                        shop_col: st.column_config.TextColumn("店舗", width="small"),
                        "営業予測": st.column_config.TextColumn("営業予測", width="small", help="AIの店舗全体期待度に基づく明日の予測"),
                        "予測平均差枚": st.column_config.NumberColumn("予測差枚", width="small", format="%+d枚", help="AIが予測する店舗全体の平均差枚"),
                        "平均スコア": st.column_config.ProgressColumn("期待度", width="small", min_value=0, max_value=1.0, format="%.2f", help="明日の店舗全体の平均的な設定5以上確率"),
                        "推奨台数": st.column_config.NumberColumn("高期待", width="small", format="%d台", help="AI期待度が30%以上の激アツ台数"),
                        "全台数": st.column_config.NumberColumn("全台", width="small", format="%d台"),
                        "収集日数": st.column_config.ProgressColumn("進捗", width="small", format="%d日/30日", min_value=0, max_value=30, help="AIの信頼度が最大になる30日分のデータ収集までの進捗です。"),
                        "AI正答率": st.column_config.TextColumn("正答率", width="small", help=f"{eval_period}にAIが推奨した台（上位約10%）が、実際に高設定挙動だった割合と台数です。この店でAIの予測がどれくらい通用するかを示します。"),
                        "AI推奨台勝率": st.column_config.TextColumn("勝率", width="small", help="AIが推奨した台（上位約10%）が、実際に差枚プラスで終わった割合と台数です。"),
                    },
                    hide_index=True,
                    width="stretch"
                )
            
        # --- 🚨 激アツ台 アラート ---
        if 'prediction_score' in df.columns:
            if '予測信頼度' in df.columns:
                super_hot_df = df[(df['prediction_score'] >= 0.40) & (df['予測信頼度'] != '🔻低')]
            else:
                super_hot_df = df[df['prediction_score'] >= 0.40]
                
            super_hot_df = super_hot_df.sort_values('prediction_score', ascending=False)
            
            if not super_hot_df.empty:
                html_str = f"""
                <div style="background-color: rgba(244, 67, 54, 0.1); border-left: 5px solid #f44336; padding: 10px 15px; border-radius: 5px; margin-bottom: 15px;">
                    <h4 style="color: #d32f2f; margin-top: 0; margin-bottom: 5px; font-size: 1.0rem;">🚨 激アツ台 発見！ ({len(super_hot_df)}台)</h4>
                    <p style="color: #b71c1c; margin-bottom: 5px; font-size: 0.85em;">期待度40%以上かつデータ信頼度が十分な、超・狙い目台です！最優先での確保をおすすめします。</p>
                    <ul style="color: #b71c1c; margin-bottom: 0; font-size: 0.85em;">
                """
                for _, r in super_hot_df.iterrows():
                    s_name = r.get(shop_col, '')
                    shop_prefix = f"【{s_name}】 " if selected_shop == '全て' else ""
                    html_str += f"<li>{shop_prefix}<b>#{r.get('台番号')} {r.get('機種名')}</b> (期待度: <b>{int(r.get('prediction_score', 0)*100)}%</b>)</li>"
                html_str += "</ul></div>"
                st.markdown(html_str, unsafe_allow_html=True)
    
        # --- メインコンテンツ: 予測ランキング表示 (上部に配置) ---
        with st.expander("🏆 予測期待度ランキング", expanded=True):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                display_mode = st.radio("表示件数", ["厳選推奨台 (店舗別 上位10%)", "Top 10", "Top 20", "すべて"], horizontal=True)
            with col_d2:
                min_score_filter = st.slider("表示する最低期待度 (%)", min_value=0, max_value=100, value=0, step=5, help="ここで設定した期待度以上の台のみを表示します。全体的にスコアが低い日は下げてみてください。", key="min_score_filter")
        
            sort_cols = []
            if 'prediction_score' in df.columns: sort_cols.append('prediction_score')
            
            ascending_list = [False] * len(sort_cols)
            
            if sort_cols:
                df_sorted = df.sort_values(by=sort_cols, ascending=ascending_list).reset_index(drop=True)
            else:
                df_sorted = df
        
            if 'prediction_score' in df_sorted.columns:
                df_sorted['予想設定5以上確率'] = (df_sorted['prediction_score'] * 100).astype(int)
        
            # --- ランク変動の計算 (前日差枚順位との比較) ---
            if 'prediction_score' in df_sorted.columns and '差枚' in df_sorted.columns:
                df_sorted['AI順位_num'] = df_sorted['prediction_score'].rank(method='min', ascending=False).fillna(999).astype(int)
                df_sorted['前日差枚順位_num'] = df_sorted['差枚'].rank(method='min', ascending=False).fillna(999).astype(int)
                
                def format_rank_change(row):
                    ai_r = row['AI順位_num']
                    prev_r = row['前日差枚順位_num']
                    if ai_r == 999: return "-"
                    if prev_r == 999: return f"{ai_r}位"
                    diff = prev_r - ai_r
                    if diff > 0: return f"{ai_r}位 (🔼+{diff})"
                    elif diff < 0: return f"{ai_r}位 (🔻{diff})"
                    else: return f"{ai_r}位 (➖)"
                    
                df_sorted['AI順位'] = df_sorted.apply(format_rank_change, axis=1)
            else:
                df_sorted['AI順位'] = "-"
        
            # 表示件数の絞り込み
            if display_mode == "厳選推奨台 (店舗別 上位10%)":
                if shop_col in df_sorted.columns:
                    df_list = []
                    for shop_name, group in df_sorted.groupby(shop_col):
                        valid_group = group[group['prediction_score'] >= 0.10] if 'prediction_score' in group.columns else group
                        df_list.append(valid_group.head(max(3, int(len(group) * 0.10))))
                    if df_list:
                        df_display = pd.concat(df_list, ignore_index=True)
                        if sort_cols:
                            df_display = df_display.sort_values(by=sort_cols, ascending=ascending_list).reset_index(drop=True)
                    else:
                        df_display = pd.DataFrame(columns=df_sorted.columns)
                else:
                    limit = max(3, int(len(df_sorted) * 0.10))
                    valid_group = df_sorted[df_sorted['prediction_score'] >= 0.10] if 'prediction_score' in df_sorted.columns else df_sorted
                    df_display = valid_group.head(limit)
            elif display_mode == "Top 10":
                df_display = df_sorted.head(10)
            elif display_mode == "Top 20":
                df_display = df_sorted.head(20)
            else:
                df_display = df_sorted
                
            if df_display.empty:
                st.info("推奨台がありません。")
                
            # --- スライダーによる期待度フィルターの適用 ---
            if not df_display.empty and '予想設定5以上確率' in df_display.columns and min_score_filter > 0:
                df_display = df_display[df_display['予想設定5以上確率'] >= min_score_filter]
                if df_display.empty:
                    st.warning(f"期待度が {min_score_filter}% 以上の台はありません。スライダーを下げてみてください。")
        
            # 常に「店名」を表示するようにカラムを厳選
            base_cols = ['AI順位', shop_col, '台番号', '機種名', '店癖マッチ', '予測信頼度', '予想設定5以上確率']
            display_cols = [c for c in base_cols if c in df_display.columns]

            # データフレームの表示設定 (Pandas Stylerを使って緑いバーを描画)
            styled_display = df_display[display_cols]
            if '予想設定5以上確率' in display_cols:
                styled_display = styled_display.style.bar(subset=['予想設定5以上確率'], color='rgba(76, 175, 80, 0.6)', vmin=0, vmax=100)
        
            st.dataframe(
                styled_display,
                column_config={
                    "AI順位": st.column_config.TextColumn("順位", width="small", help="AIの予測順位です。()内は前日の差枚ランキングからの順位変動を示します。"),
                    shop_col: st.column_config.TextColumn("店舗", width="small"),
                    "台番号": st.column_config.TextColumn("No.", width="small"),
                    "機種名": st.column_config.TextColumn("機種", width="small"),
                        "店癖マッチ": st.column_config.TextColumn("店癖", width="small", help="AIが検知した激アツ(🔥)や警戒(⚠️)の条件"),
                    "予測信頼度": st.column_config.TextColumn("信頼度", width="small", help="対象台の過去データ量に基づく予測の信頼度 (🔼高:30日~ / 🔸中:14~29日 / 🔻低:1~13日)"),
                    "予想設定5以上確率": st.column_config.NumberColumn("期待度", format="%d%%", width="small", help="AIが予測する設定5以上の確率"),
                },
                width="stretch",
                hide_index=True
            )
    
        # --- 詳細分析: 上位台の根拠とスペック ---
        if selected_shop != '全て' and not df_display.empty:
            st.markdown("### 🧐 推奨台の詳細データ・根拠")
            st.caption("AIが高く評価した推奨台について、判断根拠と詳細数値を表示します。")
    
            for i, row in df_display.iterrows():
                shop_name = row.get(shop_col, '')
                machine_no = row.get('台番号', 'Unknown')
                machine_name = row.get('機種名', '')
                prob_val = row.get('予想設定5以上確率', 0)
                
                label_prefix = f"【{shop_name}】 " if selected_shop == '全て' else ""
                label = f"{label_prefix}#{machine_no} {machine_name} ({prob_val}%)"
                
                with st.expander(label, expanded=(i == 0)):
                    _display_machine_detail_expander(row, i, shop_col, selected_shop, df_raw, df_events, specs, df_importance)
                    
            st.info("💡 **なぜこの台の期待度が高くなったか、もっと詳しく知りたいですか？**\nサイドバーの「🤖 AIチャット相談」を開き、データアナリストに「〇〇番台の評価理由を詳しく分析して」と質問すると、AIが全データをもとに論理的に解説してくれます！")

    with tab_trend:
        if selected_shop == '全て':
            st.info("👆 上部のメニューで特定の「店舗」を選択すると、その店舗の傾向分析（勝利の法則）が表示されます。")
        else:
            from views import feature_analysis_page
            feature_analysis_page.render_feature_analysis_page(df_train, df_importance, df_events, df_raw, shop_col, pre_selected_shop=selected_shop)