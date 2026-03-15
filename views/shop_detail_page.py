import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import matplotlib.pyplot as plt # type: ignore
import altair as alt # type: ignore

# バックエンド処理をインポート
import backend
from utils import get_confidence_indicator

def _display_machine_detail_expander(row, index, shop_col, selected_shop, df_raw, df_events, specs):
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
    with c1: st.metric("累計ゲーム", format_val(row.get('累計ゲーム', '-')))
    with c2: st.metric("週間平均差枚", f"{int(row.get('mean_7days_diff', 0)):+d}枚")
    
    c3, c4 = st.columns(2)
    with c3: st.metric("BIG回数", format_val(row.get('BIG', '-')))
    with c4: st.metric("REG回数", format_val(row.get('REG', '-')))
    
    c5, c6 = st.columns(2)
    with c5: st.metric("BIG確率", format_prob(row.get('BIG確率', 0)))
    with c6: st.metric("REG確率", format_prob(row.get('REG確率', 0)))
    
    matched_spec_key = backend.get_matched_spec_key(machine_name, specs)
    
    if matched_spec_key:
        st.markdown(f"**📚 {matched_spec_key} スペック目安:**")
        if matched_spec_key == "ジャグラー（デフォルト）":
            st.warning("⚠️ **注意:** この機種はスペックが未登録のため、デフォルト値で代用して分析しています。")
        spec_df = pd.DataFrame(specs[matched_spec_key]).T
        st.dataframe(spec_df.style.format(formatter="1/{:.1f}"), use_container_width=True)
    
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
                    # 3000G以上を要求
                    win_rate = ((same_wd_df['累計ゲーム'] >= 3000) & ((same_wd_df['REG確率'] >= spec_reg_p) | (same_wd_df['合算確率'] >= spec_tot_p))).mean() * 100
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

        labels, colors, explode = ['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5', 'Set 6'], ['#cfd8dc', '#b0bec5', '#90a4ae', '#fff59d', '#ffcc80', '#ffab91'], (0, 0, 0, 0, 0.05, 0.1)
        fig, ax = plt.subplots(figsize=(6, 3)); fig.patch.set_alpha(0); ax.patch.set_alpha(0)
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct=lambda p: f'{p:.0f}%' if p >= 1.0 else '', startangle=90, counterclock=False, colors=colors, explode=explode, textprops={'fontsize': 8})
        plt.setp(autotexts, weight="bold", color="black")
        st.pyplot(fig); plt.close(fig)

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

    # 機種スペックの読み込み (スコープ全体で使用するため上部で定義)
    specs = backend.get_machine_specs()

    # --- 店癖トップ5の抽出と明日の候補台へのマッピング ---
    top_trends_df = None
    worst_trends_df = None # 警戒条件用DFを追加
    base_win_rate = 0
    if df_train is not None and not df_train.empty and selected_shop != '全て':
        train_shop = df_train[df_train[shop_col] == selected_shop]
        if len(train_shop) > 0:
            base_win_rate = train_shop['target'].mean()
            trends = []
            
            if 'is_corner' in train_shop.columns:
                subset = train_shop[train_shop['is_corner'] == 1]
                if len(subset) >= 5: trends.append({"id": "corner", "条件": "角台", "高設定率": subset['target'].mean(), "サンプル": len(subset)})
            if 'REG' in train_shop.columns and 'BIG' in train_shop.columns and 'REG確率' in train_shop.columns:
                spec_reg_5 = train_shop['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
                subset = train_shop[(train_shop['REG'] > train_shop['BIG']) & (train_shop['REG確率'] >= spec_reg_5)]
                if len(subset) >= 5: trends.append({"id": "reg_lead", "条件": "REG先行・BB欠損 (高設定不発狙い)", "高設定率": subset['target'].mean(), "サンプル": len(subset)})
                if 'BIG確率' in train_shop.columns:
                    train_shop_tmp = train_shop.copy()
                    train_shop_tmp['BIG分母'] = train_shop_tmp['BIG確率'].apply(lambda x: 1/x if x > 0 else 9999)
                    subset_bb = train_shop_tmp[(train_shop_tmp['BIG分母'] >= 400) & (train_shop_tmp['REG確率'] >= spec_reg_5)]
                    if len(subset_bb) >= 5: trends.append({"id": "bb_deficit", "条件": "超不発台 (BIG 1/400以下 & REG高設定)", "高設定率": subset_bb['target'].mean(), "サンプル": len(subset_bb)})
            if '連続マイナス日数' in train_shop.columns:
                subset = train_shop[train_shop['連続マイナス日数'] >= 3]
                if len(subset) >= 5: trends.append({"id": "cons_minus", "条件": "3日以上連続凹み (上げリセット狙い)", "高設定率": subset['target'].mean(), "サンプル": len(subset)})
            if '差枚' in train_shop.columns:
                subset = train_shop[train_shop['差枚'] <= -1000]
                if len(subset) >= 5: trends.append({"id": "prev_lose", "条件": "前日大負け (-1000枚以下) からの反発", "高設定率": subset['target'].mean(), "サンプル": len(subset)})
                if '累計ゲーム' in train_shop.columns:
                    subset_taco = train_shop[(train_shop['差枚'] <= -1000) & (train_shop['累計ゲーム'] >= 7000)]
                    if len(subset_taco) >= 5: trends.append({"id": "taco_lose", "条件": "タコ粘り大凹み (7000G~ & -1000枚以下)", "高設定率": subset_taco['target'].mean(), "サンプル": len(subset_taco)})
                subset = train_shop[train_shop['差枚'] >= 1000]
                if len(subset) >= 5: trends.append({"id": "prev_win", "条件": "前日大勝ち (+1000枚以上) の据え置き", "高設定率": subset['target'].mean(), "サンプル": len(subset)})
                if 'is_win' in train_shop.columns:
                    subset = train_shop[(train_shop['差枚'] >= 1000) & (train_shop['is_win'] == 1)]
                    if len(subset) >= 5: trends.append({"id": "prev_win_reg", "条件": "前日大勝ち (+1000枚以上) & 高設定挙動の据え置き", "高設定率": subset['target'].mean(), "サンプル": len(subset)})
                else:
                    subset = train_shop[train_shop['差枚'] >= 1000]
                    if len(subset) >= 5: trends.append({"id": "prev_win", "条件": "前日大勝ち (+1000枚以上) の据え置き", "高設定率": subset['target'].mean(), "サンプル": len(subset)})
            if 'prev_差枚' in train_shop.columns and '差枚' in train_shop.columns:
                subset_v = train_shop[(train_shop['prev_差枚'] < 0) & (train_shop['差枚'] >= 0)]
                if len(subset_v) >= 5: trends.append({"id": "v_recovery", "条件": "V字反発 (前々日負け → 前日勝ち)", "高設定率": subset_v['target'].mean(), "サンプル": len(subset_v)})
                
                subset_cont_lose = train_shop[(train_shop['prev_差枚'] <= -1000) & (train_shop['差枚'] <= -1000)]
                if len(subset_cont_lose) >= 5: trends.append({"id": "cont_big_lose", "条件": "連続大負け (-1000枚以下2日連続)", "高設定率": subset_cont_lose['target'].mean(), "サンプル": len(subset_cont_lose)})
            if 'target_date_end_digit' in train_shop.columns:
                for d in [0, 5, 7]:
                    subset = train_shop[train_shop['target_date_end_digit'] == d]
                    if len(subset) >= 5: trends.append({"id": f"day_{d}", "条件": f"{d}のつく日 (予測日)", "高設定率": subset['target'].mean(), "サンプル": len(subset)})
            if '末尾番号' in train_shop.columns:
                best_m, best_wr, best_count = -1, 0, 0
                for m in range(10):
                    subset = train_shop[train_shop['末尾番号'] == m]
                    if len(subset) >= 10:
                        wr = subset['target'].mean()
                        if wr > best_wr: best_m, best_wr, best_count = m, wr, len(subset)
                if best_m != -1: trends.append({"id": f"end_{int(best_m)}", "条件": f"末尾【{int(best_m)}】", "高設定率": best_wr, "サンプル": best_count})

            # --- 警戒条件の定義 ---
            if '差枚' in train_shop.columns and 'REG確率' in train_shop.columns:
                subset = train_shop[(train_shop['差枚'] >= 2000) & (train_shop['REG確率'] < (1/350))]
                if len(subset) >= 5: trends.append({"id": "big_win_reaction", "条件": "大勝ち(+2000枚以上) & REG確率悪", "高設定率": subset['target'].mean(), "サンプル": len(subset)})
            if 'mean_7days_diff' in train_shop.columns and 'win_rate_7days' in train_shop.columns:
                subset = train_shop[(train_shop['mean_7days_diff'] >= 500) & (train_shop['win_rate_7days'] < 0.5)]
                if len(subset) >= 5: trends.append({"id": "one_hit_reaction", "条件": "一撃荒波台 (週間+500枚以上 & 高設定率50%未満)", "高設定率": subset['target'].mean(), "サンプル": len(subset)})

            df['店癖マッチ'] = ""
            if trends:
                all_trends_df = pd.DataFrame(trends)
                all_trends_df['通常時との差'] = (all_trends_df['高設定率'] - base_win_rate) * 100

                # 勝率が高いトップ3 (プラス評価)
                top_trends_df = all_trends_df[all_trends_df['通常時との差'] > 5].sort_values('高設定率', ascending=False).head(3)
                
                # 勝率が低いトップ2 (マイナス評価)
                worst_trends_df = all_trends_df[all_trends_df['通常時との差'] < -5].sort_values('高設定率', ascending=True).head(2)

                top_ids = top_trends_df['id'].tolist()
                worst_ids = worst_trends_df['id'].tolist()
                
                def get_matched_trends(row):
                    matched_hot = []
                    if "corner" in top_ids and row.get('is_corner') == 1: matched_hot.append("角")
                    if "reg_lead" in top_ids and row.get('REG', 0) > row.get('BIG', 0): matched_hot.append("BB欠損・不発")
                    if "bb_deficit" in top_ids:
                        b_p = row.get('BIG確率', 0)
                        b_d = 1 / b_p if b_p > 0 else 9999
                        sp_r5 = 1.0 / specs[backend.get_matched_spec_key(row.get('機種名', ''), specs)].get('設定5', {"REG": 260.0})["REG"]
                        if b_d >= 400 and row.get('REG確率', 0) >= sp_r5: matched_hot.append("超不発")
                    if "cons_minus" in top_ids and row.get('連続マイナス日数', 0) >= 3: matched_hot.append("連凹")
                    if "taco_lose" in top_ids and row.get('差枚', 0) <= -1000 and row.get('累計ゲーム', 0) >= 7000: matched_hot.append("タコ粘りお詫び")
                    if "prev_lose" in top_ids and row.get('差枚', 0) <= -1000: matched_hot.append("負反発")
                    if "prev_win" in top_ids and row.get('差枚', 0) >= 1000: matched_hot.append("勝据え")
                    if "v_recovery" in top_ids and row.get('prev_差枚', 0) < 0 and row.get('差枚', -1) >= 0: matched_hot.append("V字反発")
                    if "cont_big_lose" in top_ids and row.get('prev_差枚', 0) <= -1000 and row.get('差枚', 0) <= -1000: matched_hot.append("連大凹み")
                    if "prev_win_reg" in top_ids and row.get('差枚', 0) >= 1000 and row.get('is_win', 0) == 1: matched_hot.append("高設定据え")
                    for tid in top_ids:
                        if tid.startswith("day_") and 'target_date_end_digit' in row:
                            if row['target_date_end_digit'] == int(tid.split("_")[1]): matched_hot.append(f"{int(tid.split('_')[1])}のつく日")
                        elif tid.startswith("end_") and row.get('末尾番号') == int(tid.split("_")[1]): matched_hot.append(f"末尾{int(tid.split('_')[1])}")
                    
                    matched_cold = []
                    if "big_win_reaction" in worst_ids and row.get('差枚', 0) >= 2000 and row.get('REG確率', 1) < (1/350): matched_cold.append("大勝反動")
                    if "one_hit_reaction" in worst_ids and row.get('mean_7days_diff', 0) >= 500 and row.get('win_rate_7days', 1) < 0.5: matched_cold.append("一撃反動")

                    hot_str = "🔥" + " ".join(matched_hot) if matched_hot else ""
                    cold_str = "⚠️" + " ".join(matched_cold) if matched_cold else ""
                    
                    return f"{hot_str} {cold_str}".strip()
                df['店癖マッチ'] = df.apply(get_matched_trends, axis=1)

                # --- 激アツ/警戒条件によるスコア加減算 ---
                def apply_bonus_penalty(row):
                    score = row.get('prediction_score', 0)
                    match_str = row.get('店癖マッチ', '')
                    
                    if '🔥' in match_str:
                        hot_part = match_str.split('🔥')[1].split('⚠️')[0].strip()
                        bonus = 0.02 * len(hot_part.split())
                        bonus = min(0.10, bonus) # ボーナスの最大値を +0.10 (10%) に制限
                        score = min(1.0, score + bonus) # 上限は1.0 (100%)
                    
                    if '⚠️' in match_str:
                        cold_part = match_str.split('⚠️')[1].strip()
                        penalty = 0.05 * len(cold_part.split()) # ペナルティは少し重く
                        penalty = min(0.15, penalty) # ペナルティの最大値を -0.15 (15%) に制限
                        score = max(0.0, score - penalty) # 下限は0.0 (0%)

                    return score
                
                if 'prediction_score' in df.columns:
                    df['prediction_score'] = df.apply(apply_bonus_penalty, axis=1)

                # --- 根拠に店舗固有の店癖（末尾や特定日など）を追記 ---
                def append_trend_reasons(row):
                    reason = row.get('根拠', '')
                    if pd.isna(reason): reason = ""
                    else: reason = str(reason)
                    
                    match_str = row.get('店癖マッチ', '')
                    if pd.isna(match_str): match_str = ""
                    else: match_str = str(match_str)
                    
                    add_reasons = []
                    if '🔥' in match_str:
                        hot_part = match_str.split('🔥')[1].split('⚠️')[0].strip()
                        for h in hot_part.split():
                            if h.startswith("末尾"):
                                add_reasons.append(f"【🎯店癖】過去の傾向から、この店舗で特に勝率が高い『{h}』に合致しています。")
                            elif h.endswith("のつく日"):
                                add_reasons.append(f"【🎯店癖】過去の傾向から、この店舗が還元している『{h}』に合致しています。")
                            elif h == "角":
                                add_reasons.append("【🎯店癖】過去の傾向から、この店舗で設定が入りやすい『角台』に合致しています。")
                            elif h == "BB欠損・不発":
                                add_reasons.append("【🎯店癖】過去の傾向から、この店舗で上げられやすい『REG先行のBB欠損台（不発台）』に合致しています。")
                            elif h == "超不発":
                                add_reasons.append("【🎯店癖】過去の傾向から、この店舗で反発（上げ/据え置き）されやすい『BIG極端欠損の超不発台』に合致しています。")
                            elif h == "連凹":
                                add_reasons.append("【🎯店癖】過去の傾向から、この店舗で上げリセットされやすい『連続凹み台』に合致しています。")
                            elif h == "タコ粘りお詫び":
                                add_reasons.append("【🎯店癖】過去の傾向から、この店舗でしっかりお詫び（上げ/据え置き）されやすい『タコ粘り大凹み台』に合致しています。")
                            elif h == "負反発":
                                add_reasons.append("【🎯店癖】過去の傾向から、この店舗で反発（底上げ）されやすい『前日大負け台』に合致しています。")
                            elif h == "勝据え":
                                add_reasons.append("【🎯店癖】過去の傾向から、この店舗で据え置かれやすい『前日大勝ち台』に合致しています。")
                            elif h == "V字反発":
                                add_reasons.append("【🎯店癖】過去の傾向から、この店舗で好調ウェーブが継続しやすい『V字反発の波(前々日負け→前日勝ち)』に合致しています。")
                            elif h == "連大凹み":
                                add_reasons.append("【🎯店癖】過去の傾向から、この店舗で強烈な底上げ（お詫び）が期待できる『2日連続大負けの波』に合致しています。")
                            elif h == "高設定据え":
                                add_reasons.append("【🎯店癖】過去の傾向から、この店舗で据え置かれやすい『高設定挙動の大勝ち台』に合致しています。")
                                
                    if '⚠️' in match_str:
                        cold_part = match_str.split('⚠️')[1].strip()
                        for c in cold_part.split():
                            if c == "大勝反動":
                                add_reasons.append("【⚠️警戒】大勝後のREG確率が悪い台です。過去の傾向から反動（回収）の危険性が高いため注意してください。")
                            elif c == "一撃反動":
                                add_reasons.append("【⚠️警戒】一撃で出た荒波台です。過去の傾向から据え置きされにくく回収される危険性が高いため注意してください。")
                                
                    if add_reasons:
                        return (reason + " " + " ".join(add_reasons)).strip()
                    return reason
                
                if '根拠' in df.columns:
                    df['根拠'] = df.apply(append_trend_reasons, axis=1)

    # --- メインコンテンツ: ランキング表示 (上部に配置) ---
    st.subheader("🏆 予測期待度ランキング (Top 10)")

    sort_cols = []
    if 'prediction_score' in df.columns: sort_cols.append('prediction_score')
    
    ascending_list = [False] * len(sort_cols)
    
    if sort_cols:
        df_sorted = df.sort_values(by=sort_cols, ascending=ascending_list).reset_index(drop=True)
    else:
        df_sorted = df

    if 'prediction_score' in df_sorted.columns:
        df_sorted['予想設定5以上確率'] = (df_sorted['prediction_score'] * 100).astype(int)

    # トップ10に絞る
    df_top10 = df_sorted.head(10)

    # スマホで見やすいようにカラムを厳選（「全て」の店が選ばれている時だけ「店名」を表示）
    base_cols = ['台番号', '機種名', '店癖マッチ', '予測信頼度', '予想設定5以上確率']
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
            "店癖マッチ": st.column_config.TextColumn("店癖", width="medium", help="AIが検知した激アツ(🔥)や警戒(⚠️)の条件"),
            "予測信頼度": st.column_config.TextColumn("信頼度", width="small", help="対象台の過去データ量に基づく予測の信頼度 (🔼高:30日~ / 🔸中:14~29日 / 🔻低:1~13日)"),
            "予想設定5以上確率": st.column_config.ProgressColumn("期待度", format="%d%%", min_value=0, max_value=100, width="small", help="AIが予測する設定5以上の確率"),
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
        prob_val = row.get('予想設定5以上確率', 0)
        
        label_prefix = f"【{shop_name}】 " if selected_shop == '全て' else ""
        label = f"{label_prefix}#{machine_no} {machine_name} (設定5以上確率: {prob_val}%)"
        
        with st.expander(label, expanded=(i == 0)):
            _display_machine_detail_expander(row, i, shop_col, selected_shop, df_raw, df_events, specs)

    # --- 店舗別 傾向分析 (下部に移動) ---
    if selected_shop != '全て' and not df_raw_shop.empty:
        st.divider()
        st.subheader(f"📅 {selected_shop} の傾向分析")
        st.caption("過去データに基づく、この店舗のイベント日や曜日ごとの平均差枚数です。")
        
        # --- 🤖 AIが発見した店癖/警戒条件 ---
        if top_trends_df is not None or worst_trends_df is not None:
            st.markdown(f"**🤖 AIが発見した {selected_shop} の店癖/警戒条件**")
            
            if top_trends_df is not None and not top_trends_df.empty:
                st.caption("AIが過去データから見つけた、この店舗で特に翌日に高設定が入りやすい『激アツ条件 (🔥)』です。")
                top_trends_df['信頼度'] = top_trends_df['サンプル'].apply(get_confidence_indicator)
                st.dataframe(
                    top_trends_df,
                            column_config={
                                "条件": st.column_config.TextColumn("激アツ条件"),
                                "高設定率": st.column_config.ProgressColumn("高設定率", format="%.2f", min_value=0, max_value=1, help="条件合致時の高設定率"),
                                "通常時との差": st.column_config.NumberColumn("差分", format="%+.1fpt", help="通常時との高設定率の差"),
                                "サンプル": st.column_config.NumberColumn("台数", format="%d台", help="サンプル数"),
                                "信頼度": st.column_config.TextColumn("信頼", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                            },
                            hide_index=True,
                            use_container_width=True
                        )

            if worst_trends_df is not None and not worst_trends_df.empty:
                st.caption("AIが過去データから見つけた、この店舗で特に翌日に高設定が入りにくい『警戒条件 (⚠️)』です。")
                worst_trends_df['信頼度'] = worst_trends_df['サンプル'].apply(get_confidence_indicator)
                st.dataframe(
                    worst_trends_df,
                        column_config={
                            "条件": st.column_config.TextColumn("警戒条件"),
                            "高設定率": st.column_config.ProgressColumn("高設定率", format="%.2f", min_value=0, max_value=1, help="条件合致時の高設定率"),
                            "通常時との差": st.column_config.NumberColumn("差分", format="%+.1fpt", help="通常時との高設定率の差"),
                            "サンプル": st.column_config.NumberColumn("台数", format="%d台", help="サンプル数"),
                            "信頼度": st.column_config.TextColumn("信頼", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
            st.caption(f"※この店舗の通常時の平均高設定率は **{base_win_rate:.1%}** です。")
        
        # スマホ対応: 縦に並べる
        
        # --- 低回転ノイズの除外 ---
        # 実態に即した分析を行うため、1000G未満の台（未稼働/即ヤメ）は集計対象（分母）から除外する
        viz_df = df_raw_shop[df_raw_shop['累計ゲーム'] >= 1000].copy()
        
        viz_df['合算確率'] = (viz_df['BIG'] + viz_df['REG']) / viz_df['累計ゲーム'].replace(0, np.nan)
        spec_reg = viz_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
        spec_tot = viz_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
        
        # 高設定判定（分子）には3000G以上の試行回数を要求し、低回転の上振れを防ぐ
        viz_df['高設定'] = ((viz_df['累計ゲーム'] >= 3000) & ((viz_df['REG確率'] >= spec_reg) | (viz_df['合算確率'] >= spec_tot))).astype(int)

        chart_metric_shop = st.radio("📊 グラフの表示指標", ["平均差枚", "高設定率"], horizontal=True, key="shop_detail_metric")
        y_col = "差枚" if chart_metric_shop == "平均差枚" else "高設定"
        bar_color1 = "#FF4B4B" if chart_metric_shop == "平均差枚" else "#AB47BC"
        bar_color2 = "#4B4BFF" if chart_metric_shop == "平均差枚" else "#AB47BC"

        if '日付要素' in viz_df.columns and not viz_df['日付要素'].isnull().all():
            st.markdown(f"**🔥 イベント別 {chart_metric_shop}**")
            event_summary = viz_df.groupby('日付要素')[y_col].mean().sort_values(ascending=False)
            st.bar_chart(event_summary, color=bar_color1)
        
        st.markdown(f"**📅 曜日別 {chart_metric_shop}**")
        if '曜日' not in viz_df.columns and '対象日付' in viz_df.columns:
            day_map = {'Monday': '月', 'Tuesday': '火', 'Wednesday': '水', 'Thursday': '木', 'Friday': '金', 'Saturday': '土', 'Sunday': '日'}
            viz_df['曜日'] = viz_df['対象日付'].dt.day_name().map(day_map)
        
        if '曜日' in viz_df.columns:
            weekday_shop_stats = viz_df.groupby('曜日')[y_col].mean().sort_values(ascending=False)
            st.bar_chart(weekday_shop_stats, color=bar_color2)
        
        # --- 月間トレンド分析 (月初・月末) ---
        st.divider()
        st.subheader("🗓️ 月間トレンド (月初・月末の傾向)")
        st.caption("過去データにおける、日付（1日〜31日）ごとの平均差枚数や高設定率です。")
        
        trend_df = viz_df.copy()
        if '対象日付' in trend_df.columns:
            trend_df['day'] = trend_df['対象日付'].dt.day
            
            def classify_period(d):
                if d <= 7: return '月初 (1-7日)'
                elif d >= 25: return '月末 (25日-)'
                else: return '中旬 (8-24日)'
            
            trend_df['period'] = trend_df['day'].apply(classify_period)
            period_stats = trend_df.groupby('period')[y_col].mean()
            
            # スマホ対応: 少し狭いがmetricは自動調整されるのでそのまま
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
                    
                    machine_rank = period_df.groupby('機種名').agg(
                        平均差枚=('差枚', 'mean'),
                        高設定率=('高設定', 'mean'),
                        設置台数=('台番号', 'nunique')
                    ).sort_values('高設定率', ascending=False).reset_index()
                    machine_rank['信頼度'] = machine_rank['設置台数'].apply(get_confidence_indicator)
                    
                    st.dataframe(
                        machine_rank,
                        column_config={
                            "平均差枚": st.column_config.NumberColumn(format="%+d 枚"),
                            "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                            "設置台数": st.column_config.NumberColumn(format="%d 台"),
                            "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    st.markdown(f"🔢 **{selected_period} の末尾番号傾向 (0-9)**")
                    if '末尾番号' in period_df.columns:
                        digit_rank = period_df.groupby('末尾番号').agg(
                            平均差枚=('差枚', 'mean'),
                            高設定率=('高設定', 'mean'),
                            サンプル数=('差枚', 'count')
                        ).sort_index().reset_index()
                        digit_rank['信頼度'] = digit_rank['サンプル数'].apply(get_confidence_indicator)
                        
                        st.bar_chart(digit_rank.set_index('末尾番号')[y_col], color="#29b6f6" if chart_metric_shop == "平均差枚" else "#AB47BC")
                        st.dataframe(
                            digit_rank.style.background_gradient(subset=['平均差枚'], cmap='RdYlGn', vmin=-300, vmax=300),
                            column_config={
                                "平均差枚": st.column_config.NumberColumn(format="%+d 枚"),
                                "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                                "サンプル数": st.column_config.NumberColumn(format="%d 件"),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                            },
                            use_container_width=True
                        )
                        
                    st.markdown(f"📅 **{selected_period} の曜日別傾向**")
                    if '曜日' in period_df.columns:
                        wd_rank = period_df.groupby('曜日').agg(
                            平均差枚=('差枚', 'mean'),
                            高設定率=('高設定', 'mean'),
                            サンプル数=('差枚', 'count')
                        ).reset_index()
                        day_order = {'月': 1, '火': 2, '水': 3, '木': 4, '金': 5, '土': 6, '日': 7}
                        wd_rank['sort'] = wd_rank['曜日'].map(day_order).fillna(99)
                        wd_rank = wd_rank.sort_values('sort').drop(columns=['sort'])
                        wd_rank['信頼度'] = wd_rank['サンプル数'].apply(get_confidence_indicator)
                        
                        st.bar_chart(wd_rank.set_index('曜日')[y_col], color="#4B4BFF" if chart_metric_shop == "平均差枚" else "#AB47BC")
                        st.dataframe(
                            wd_rank.style.background_gradient(subset=['平均差枚'], cmap='RdYlGn', vmin=-300, vmax=300),
                            column_config={
                                "平均差枚": st.column_config.NumberColumn(format="%+d 枚"),
                                "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                                "サンプル数": st.column_config.NumberColumn(format="%d 件"),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                            },
                            hide_index=True,
                            use_container_width=True
                        )

                    if '日付要素' in period_df.columns and not period_df['日付要素'].isnull().all():
                        st.markdown(f"🔥 **{selected_period} のイベント別傾向**")
                        ev_rank = period_df.groupby('日付要素').agg(
                            平均差枚=('差枚', 'mean'),
                            高設定率=('高設定', 'mean'),
                            サンプル数=('差枚', 'count')
                        ).reset_index().sort_values(y_col, ascending=False)
                        ev_rank['信頼度'] = ev_rank['サンプル数'].apply(get_confidence_indicator)
                        
                        st.bar_chart(ev_rank.set_index('日付要素')[y_col], color="#FF4B4B" if chart_metric_shop == "平均差枚" else "#AB47BC")
                        st.dataframe(
                            ev_rank.style.background_gradient(subset=['平均差枚'], cmap='RdYlGn', vmin=-300, vmax=300),
                            column_config={
                                "平均差枚": st.column_config.NumberColumn(format="%+d 枚"),
                                "高設定率": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
                                "サンプル数": st.column_config.NumberColumn(format="%d 件"),
                                "信頼度": st.column_config.TextColumn("信頼度", help="データのサンプル量に基づく信頼度 (🔼高:30件~ / 🔸中:10件~ / 🔻低:~9件)")
                            },
                            hide_index=True,
                            use_container_width=True
                        )

            st.markdown(f"**📅 日付別 {chart_metric_shop}推移**")
            day_stats = trend_df.groupby('day')[y_col].mean()
            st.bar_chart(day_stats, color="#00E676" if chart_metric_shop == "平均差枚" else "#AB47BC")

    # --- 視覚化: 機種ごとの分析 ---
    st.divider()
    st.subheader("📊 機種別 平均設定5以上確率")
    
    if '機種名' in df.columns and 'prediction_score' in df.columns:
        # 機種ごとの平均設定5以上確率(%)を算出
        machine_stats = (df.groupby("機種名")["prediction_score"].mean() * 100).sort_values(ascending=False)
        
        # 棒グラフで表示
        st.bar_chart(machine_stats, color="#FF4B4B")
        
        st.caption("※ 各機種の予想設定5以上確率の平均値です")