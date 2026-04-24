import pandas as pd
import numpy as np
from utils import get_matched_spec_key

def calculate_shop_trends(df_train, shop_col, specs):
    all_trends_dict = {}
    for s in df_train[shop_col].unique():
        train_shop = df_train[df_train[shop_col] == s]
        if len(train_shop) == 0: continue
        
        def get_high_rate(subset):
            valid = subset[subset['next_累計ゲーム'] >= 3000]
            if len(valid) > 0:
                return valid['target'].mean() * 100
            return 0.0
            
        s_base_win_rate = get_high_rate(train_shop)
        trends = []
        
        if 'is_corner' in train_shop.columns:
            subset = train_shop[train_shop['is_corner'] == 1]
            if len(subset) >= 5: trends.append({"id": "corner", "条件": "角台", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'REG' in train_shop.columns and 'BIG' in train_shop.columns and 'REG確率' in train_shop.columns:
            spec_reg_5 = train_shop['機種名'].apply(lambda x: 1.0 / specs[get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
            subset = train_shop[(train_shop['REG'] > train_shop['BIG']) & (train_shop['REG確率'] >= spec_reg_5)]
            if len(subset) >= 5: trends.append({"id": "reg_lead", "条件": "REG先行・BB欠損 (高設定不発狙い)", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
            if 'BIG確率' in train_shop.columns:
                train_shop_tmp = train_shop.copy()
                train_shop_tmp['BIG分母'] = train_shop_tmp['BIG確率'].apply(lambda x: 1/x if x > 0 else 9999)
                subset_bb = train_shop_tmp[(train_shop_tmp['BIG分母'] >= 400) & (train_shop_tmp['REG確率'] >= spec_reg_5)]
                if len(subset_bb) >= 5: trends.append({"id": "bb_deficit", "条件": "超不発台 (BIG 1/400以下 & REG高設定)", "高設定率": get_high_rate(subset_bb), "サンプル": len(subset_bb)})
        if '連続マイナス日数' in train_shop.columns:
            subset = train_shop[train_shop['連続マイナス日数'] >= 3]
            if len(subset) >= 5: trends.append({"id": "cons_minus", "条件": "3日以上連続凹み (上げリセット狙い)", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'cons_high_reg_days' in train_shop.columns:
            subset = train_shop[train_shop['cons_high_reg_days'] >= 2]
            if len(subset) >= 5: trends.append({"id": "cons_high_reg", "条件": "連続高REG (2日以上高設定挙動の据え置き)", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if '差枚' in train_shop.columns:
            subset = train_shop[train_shop['差枚'] <= -1000]
            if len(subset) >= 5: trends.append({"id": "prev_lose", "条件": "前日大負け (-1000枚以下) からの反発", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
            if '累計ゲーム' in train_shop.columns:
                subset_taco = train_shop[(train_shop['差枚'] <= -1000) & (train_shop['累計ゲーム'] >= 7000)]
                if len(subset_taco) >= 5: trends.append({"id": "taco_lose", "条件": "タコ粘り大凹み (7000G~ & -1000枚以下)", "高設定率": get_high_rate(subset_taco), "サンプル": len(subset_taco)})
            subset = train_shop[train_shop['差枚'] >= 1000]
            if len(subset) >= 5: trends.append({"id": "prev_win", "条件": "前日大勝ち (+1000枚以上) の据え置き", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
            if 'is_win' in train_shop.columns:
                subset = train_shop[(train_shop['差枚'] >= 1000) & (train_shop['is_win'] == 1)]
                if len(subset) >= 5: trends.append({"id": "prev_win_reg", "条件": "前日大勝ち (+1000枚以上) & 高設定挙動の据え置き", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
            else:
                subset = train_shop[train_shop['差枚'] >= 1000]
                if len(subset) >= 5: trends.append({"id": "prev_win", "条件": "前日大勝ち (+1000枚以上) の据え置き", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'prev_差枚' in train_shop.columns and '差枚' in train_shop.columns:
            subset_v = train_shop[(train_shop['prev_差枚'] < 0) & (train_shop['差枚'] >= 0)]
            if len(subset_v) >= 5: trends.append({"id": "v_recovery", "条件": "V字反発 (前々日負け → 前日勝ち)", "高設定率": get_high_rate(subset_v), "サンプル": len(subset_v)})
            subset_cont_lose = train_shop[(train_shop['prev_差枚'] <= -1000) & (train_shop['差枚'] <= -1000)]
            if len(subset_cont_lose) >= 5: trends.append({"id": "cont_big_lose", "条件": "連続大負け (-1000枚以下2日連続)", "高設定率": get_high_rate(subset_cont_lose), "サンプル": len(subset_cont_lose)})
        if 'target_date_end_digit' in train_shop.columns:
            for d in range(10):
                subset = train_shop[train_shop['target_date_end_digit'] == d]
                if len(subset) >= 5: trends.append({"id": f"day_{d}", "条件": f"{d}のつく日", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'target_weekday' in train_shop.columns:
            for wd, wd_name in enumerate(["月", "火", "水", "木", "金", "土", "日"]):
                subset = train_shop[train_shop['target_weekday'] == wd]
                if len(subset) >= 10: trends.append({"id": f"wd_{wd}", "条件": f"{wd_name}曜日", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if '機種名' in train_shop.columns:
            for mac in train_shop['機種名'].unique():
                subset = train_shop[train_shop['機種名'] == mac]
                if len(subset) >= 10: trends.append({"id": f"mac_{mac}", "条件": f"機種:{mac}", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if '末尾番号' in train_shop.columns:
            for m in range(10):
                subset = train_shop[train_shop['末尾番号'] == m]
                if len(subset) >= 10: trends.append({"id": f"end_{m}", "条件": f"末尾【{m}】", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if '累計ゲーム' in train_shop.columns:
            subset = train_shop[train_shop['累計ゲーム'] >= 8000]
            if len(subset) >= 5: trends.append({"id": "high_kado_reaction", "条件": "前日超高稼働 (8000G~)", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'REG確率' in train_shop.columns and '累計ゲーム' in train_shop.columns:
            spec_reg_5 = train_shop['機種名'].apply(lambda x: 1.0 / specs[get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
            subset = train_shop[(train_shop['累計ゲーム'] >= 5000) & (train_shop['REG確率'] >= spec_reg_5)]
            if len(subset) >= 5: trends.append({"id": "high_setting_reaction", "条件": "前日高設定挙重", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'prev_差枚' in train_shop.columns and '差枚' in train_shop.columns:
            subset = train_shop[(train_shop['prev_差枚'] >= 500) & (train_shop['差枚'] >= 500)]
            if len(subset) >= 5: trends.append({"id": "cons_win_reaction", "条件": "連勝中 (2日連続+500枚~)", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if '差枚' in train_shop.columns and 'REG確率' in train_shop.columns:
            subset = train_shop[(train_shop['差枚'] >= 2000) & (train_shop['REG確率'] < (1/350))]
            if len(subset) >= 5: trends.append({"id": "big_win_reaction", "条件": "大勝ち(+2000枚以上) & REG確率悪", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'mean_7days_diff' in train_shop.columns and 'win_rate_7days' in train_shop.columns:
            subset = train_shop[(train_shop['mean_7days_diff'] >= 500) & (train_shop['win_rate_7days'] < 0.5)]
            if len(subset) >= 5: trends.append({"id": "one_hit_reaction", "条件": "一撃荒波台 (週間+500枚以上 & 高設定率50%未満)", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        
        s_top_trends_df = None
        s_worst_trends_df = None
        if trends:
            all_trends_df = pd.DataFrame(trends)
            all_trends_df['通常時との差'] = (all_trends_df['高設定率'] - s_base_win_rate)
            s_top_trends_df = all_trends_df[all_trends_df['通常時との差'] > 5].sort_values('高設定率', ascending=False).head(3)
            s_worst_trends_df = all_trends_df[all_trends_df['通常時との差'] < -5].sort_values('高設定率', ascending=True).head(3)
        all_trends_dict[s] = {
            'base_win_rate': s_base_win_rate,
            'top_ids': s_top_trends_df['id'].tolist() if s_top_trends_df is not None else [],
            'worst_ids': s_worst_trends_df['id'].tolist() if s_worst_trends_df is not None else [],
            'trend_diffs': dict(zip(all_trends_df['id'], all_trends_df['通常時との差'])) if trends else {},
            'trend_win_rates': dict(zip(all_trends_df['id'], all_trends_df['高設定率'])) if trends else {},
            'top_df': s_top_trends_df,
            'worst_df': s_worst_trends_df
        }
    return all_trends_dict

def apply_trends_to_row(row, all_trends_dict, shop_col, specs):
    s = row.get(shop_col)
    if s not in all_trends_dict:
        row['店癖マッチ'] = ""
        return row
        
    status = row.get('営業状態', '⚖️ 通常営業')
    t_info = all_trends_dict[s].get(status)
    
    # 該当ステータスの実績が全くない場合は「全体」にフォールバック
    if not t_info or (not t_info['top_ids'] and not t_info['worst_ids']):
        t_info = all_trends_dict[s].get("全体")
        status_label = "全体"
    else:
        status_label = status.replace("🔥 ", "").replace("🥶 ", "").replace("⚖️ ", "")
        
    if not t_info:
        row['店癖マッチ'] = ""
        return row
        
    top_ids = t_info['top_ids']
    worst_ids = t_info['worst_ids']
    
    matched_hot = []
    matched_hot_ids = []
    if "corner" in top_ids and row.get('is_corner') == 1: matched_hot.append("角"); matched_hot_ids.append("corner")
    if "reg_lead" in top_ids and row.get('REG', 0) > row.get('BIG', 0): matched_hot.append("BB欠損・不発"); matched_hot_ids.append("reg_lead")
    if "bb_deficit" in top_ids:
        b_p = row.get('BIG確率', 0)
        b_d = 1 / b_p if b_p > 0 else 9999
        sp_r5 = 1.0 / specs[get_matched_spec_key(row.get('機種名', ''), specs)].get('設定5', {"REG": 260.0})["REG"]
        if b_d >= 400 and row.get('REG確率', 0) >= sp_r5: matched_hot.append("超不発"); matched_hot_ids.append("bb_deficit")
    if "cons_minus" in top_ids and row.get('連続マイナス日数', 0) >= 3: matched_hot.append("連凹"); matched_hot_ids.append("cons_minus")
    if "cons_high_reg" in top_ids and row.get('cons_high_reg_days', 0) >= 2: matched_hot.append("連高REG"); matched_hot_ids.append("cons_high_reg")
    if "taco_lose" in top_ids and row.get('差枚', 0) <= -1000 and row.get('累計ゲーム', 0) >= 7000: matched_hot.append("タコ粘りお詫び"); matched_hot_ids.append("taco_lose")
    if "prev_lose" in top_ids and row.get('差枚', 0) <= -1000: matched_hot.append("負反発"); matched_hot_ids.append("prev_lose")
    if "prev_win" in top_ids and row.get('差枚', 0) >= 1000: matched_hot.append("勝据え"); matched_hot_ids.append("prev_win")
    if "v_recovery" in top_ids and row.get('prev_差枚', 0) < 0 and row.get('差枚', -1) >= 0: matched_hot.append("V字反発"); matched_hot_ids.append("v_recovery")
    if "cont_big_lose" in top_ids and row.get('prev_差枚', 0) <= -1000 and row.get('差枚', 0) <= -1000: matched_hot.append("連大凹み"); matched_hot_ids.append("cont_big_lose")
    if "prev_win_reg" in top_ids and row.get('差枚', 0) >= 1000 and row.get('is_win', 0) == 1: matched_hot.append("高設定据え"); matched_hot_ids.append("prev_win_reg")
    if "high_kado_reaction" in top_ids and row.get('累計ゲーム', 0) >= 8000: matched_hot.append("高稼働据え置き"); matched_hot_ids.append("high_kado_reaction")
    if "cons_win_reaction" in top_ids and row.get('prev_差枚', 0) >= 500 and row.get('差枚', 0) >= 500: matched_hot.append("連勝据え置き"); matched_hot_ids.append("cons_win_reaction")
    if "main_corner" in top_ids and row.get('is_main_corner') == 1: matched_hot.append("メイン角"); matched_hot_ids.append("main_corner")
    if "main_island" in top_ids and row.get('is_main_island') == 1: matched_hot.append("目立つ島"); matched_hot_ids.append("main_island")
    if "wall_island" in top_ids and row.get('is_wall_island') == 1: matched_hot.append("壁側・死に島"); matched_hot_ids.append("wall_island")
    for tid in top_ids:
        if tid.startswith("day_") and 'target_date_end_digit' in row:
            if row['target_date_end_digit'] == int(tid.split("_")[1]): matched_hot.append(f"{int(tid.split('_')[1])}のつく日"); matched_hot_ids.append(tid)
        elif tid.startswith("end_") and row.get('末尾番号') == int(tid.split("_")[1]): matched_hot.append(f"末尾{int(tid.split('_')[1])}"); matched_hot_ids.append(tid)
        elif tid.startswith("wd_") and row.get('target_weekday') == int(tid.split("_")[1]):
            wd_names = ["月", "火", "水", "木", "金", "土", "日"]
            matched_hot.append(f"{wd_names[int(tid.split('_')[1])]}曜日"); matched_hot_ids.append(tid)
        elif tid.startswith("mac_") and row.get('機種名') == tid.split("_")[1]:
            matched_hot.append(f"看板機種"); matched_hot_ids.append(tid)
    
    matched_cold = []
    matched_cold_ids = []
    if "big_win_reaction" in worst_ids and row.get('差枚', 0) >= 2000 and row.get('REG確率', 1) < (1/350): matched_cold.append("大勝反動"); matched_cold_ids.append("big_win_reaction")
    if "cons_high_reg" in worst_ids and row.get('cons_high_reg_days', 0) >= 2: matched_cold.append("連高REG反動"); matched_cold_ids.append("cons_high_reg")
    if "one_hit_reaction" in worst_ids and row.get('mean_7days_diff', 0) >= 500 and row.get('win_rate_7days', 1) < 0.5: matched_cold.append("一撃反動"); matched_cold_ids.append("one_hit_reaction")
    if "high_kado_reaction" in worst_ids and row.get('累計ゲーム', 0) >= 8000: matched_cold.append("高稼働反動"); matched_cold_ids.append("high_kado_reaction")
    if "cons_win_reaction" in worst_ids and row.get('prev_差枚', 0) >= 500 and row.get('差枚', 0) >= 500: matched_cold.append("連勝ストップ"); matched_cold_ids.append("cons_win_reaction")
    if "main_corner" in worst_ids and row.get('is_main_corner') == 1: matched_cold.append("メイン角(見せ台フェイク)"); matched_cold_ids.append("main_corner")
    if "main_island" in worst_ids and row.get('is_main_island') == 1: matched_cold.append("目立つ島(回収用)"); matched_cold_ids.append("main_island")
    if "wall_island" in worst_ids and row.get('is_wall_island') == 1: matched_cold.append("壁側(冷遇)"); matched_cold_ids.append("wall_island")
    
    for tid in worst_ids:
        if tid.startswith("day_") and 'target_date_end_digit' in row:
            if row['target_date_end_digit'] == int(tid.split("_")[1]): matched_cold.append(f"{int(tid.split('_')[1])}のつく日(冷遇)"); matched_cold_ids.append(tid)
        elif tid.startswith("end_") and row.get('末尾番号') == int(tid.split("_")[1]): matched_cold.append(f"末尾{int(tid.split('_')[1])}(冷遇)"); matched_cold_ids.append(tid)
        elif tid.startswith("wd_") and row.get('target_weekday') == int(tid.split("_")[1]):
            wd_names = ["月", "火", "水", "木", "金", "土", "日"]
            matched_cold.append(f"{wd_names[int(tid.split('_')[1])]}曜日(冷遇)"); matched_cold_ids.append(tid)
        elif tid.startswith("mac_") and row.get('機種名') == tid.split("_")[1]:
            matched_cold.append(f"冷遇機種"); matched_cold_ids.append(tid)

    fixed_hot = []
    fixed_cold = []
    mac_name = row.get('機種名', '')
    matched_key = get_matched_spec_key(mac_name, specs)
    if matched_key and matched_key in specs:
        spec_b1 = specs[matched_key].get('設定1', {}).get('BIG', 280.0)
        spec_b5 = specs[matched_key].get('設定5', {}).get('BIG', 260.0)
        spec_b6 = specs[matched_key].get('設定6', {}).get('BIG', 260.0)
        spec_r6 = specs[matched_key].get('設定6', {}).get('REG', 260.0)
        b_prob = row.get('BIG確率', 0)
        r_prob = row.get('REG確率', 0)
        games = row.get('累計ゲーム', 0)
        if games >= 5000 and b_prob > 0 and r_prob > 0:
            if (1.0 / b_prob) > spec_b1 and (1.0 / r_prob) > spec_r6:
                fixed_cold.append("中間設定濃厚")
            if (1.0 / b_prob) <= spec_b6:
                    fixed_hot.append("BB設定6以上")
            elif (1.0 / b_prob) <= spec_b5:
                    fixed_hot.append("BB設定5以上")
        if games >= 5000 and r_prob > 0:
            if (1.0 / r_prob) <= 200.0:
                fixed_hot.append("超REG突出")
                
    if "high_setting_reaction" in worst_ids and row.get('累計ゲーム', 0) >= 5000:
        sp_r5 = 1.0 / specs[matched_key].get('設定5', {"REG": 260.0})["REG"] if matched_key in specs else 1.0/260.0
        if row.get('REG確率', 0) >= sp_r5:
            matched_cold.append("高設定下げ"); matched_cold_ids.append("high_setting_reaction")
            
    if "high_setting_reaction" in top_ids and row.get('累計ゲーム', 0) >= 5000:
        sp_r5 = 1.0 / specs[matched_key].get('設定5', {"REG": 260.0})["REG"] if matched_key in specs else 1.0/260.0
        if row.get('REG確率', 0) >= sp_r5:
            matched_hot.append("高設定完全据え置き"); matched_hot_ids.append("high_setting_reaction")

    hot_str = "🔥" + " ".join(matched_hot + fixed_hot) if (matched_hot or fixed_hot) else ""
    cold_str = "⚠️" + " ".join(matched_cold + fixed_cold) if (matched_cold or fixed_cold) else ""
    
    match_str = f"{hot_str} {cold_str}".strip()
    row['店癖マッチ'] = match_str
    
    # スコアの再計算（AIの予測スコアと、強力な店癖の過去実績確率をブレンドする）
    score = row.get('prediction_score', 0)
    
    # 強い店癖（トップトレンド）の実績勝率を加味してスコアを底上げ
    top_win_rates = [t_info['trend_win_rates'].get(tid, 0) for tid in matched_hot_ids]
    if top_win_rates:
        max_trend_prob = max(top_win_rates) / 100.0
        # AIの評価が実績より低い場合、実績確率側に歩み寄らせる（中間の値をとる）
        if score < max_trend_prob:
            score = (score + max_trend_prob) / 2.0

    # 悪い店癖（ワーストトレンド）の実績勝率を加味してスコアを引き下げ
    worst_win_rates = [t_info['trend_win_rates'].get(tid, 0) for tid in matched_cold_ids]
    if worst_win_rates:
        min_trend_prob = min(worst_win_rates) / 100.0
        # AIの評価が実績より高い場合、下方に歩み寄らせる
        if score > min_trend_prob:
            score = (score + min_trend_prob) / 2.0
            
    row['prediction_score'] = score

    # 根拠の追記
    reason = str(row.get('根拠', ''))
    add_reasons = []
    for tid, h in zip(matched_hot_ids, matched_hot):
        w_rate = t_info['trend_win_rates'].get(tid, 0)
        diff_rate = t_info['trend_diffs'].get(tid, 0)
        rate_str = f"(実績:高設定率{w_rate:.1f}% / 通常より{diff_rate:+.1f}%)"
        
        t_prefix_de = f"この店舗の{status_label}で" if status_label != "全体" else "この店舗で"
        t_prefix_ha = f"この店舗の{status_label}は" if status_label != "全体" else "この店舗は"
        
        if tid.startswith("end_") or tid.startswith("day_"): 
            add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}『{h}』は高設定の期待度が大幅に上がります {rate_str}。")
        elif h == "角": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}設定が入りやすい『角台』に合致しています {rate_str}。")
        elif h == "BB欠損・不発": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}上げられやすい『REG先行のBB欠損台（不発台）』に合致しています {rate_str}。")
        elif h == "超不発": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}反発（上げ/据え置き）されやすい『BIG極端欠損の超不発台』に合致しています {rate_str}。")
        elif h == "連凹": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}上げリセットされやすい『連続凹み台』に合致しています {rate_str}。")
        elif h == "連高REG": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}高設定が連続しやすい『連続高REG台』に合致しています {rate_str}。")
        elif h == "タコ粘りお詫び": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}しっかりお詫び（上げ/据え置き）されやすい『タコ粘り大凹み台』に合致しています {rate_str}。")
        elif h == "負反発": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}反発（底上げ）されやすい『前日大負け台』に合致しています {rate_str}。")
        elif h == "勝据え": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}据え置かれやすい『前日大勝ち台』に合致しています {rate_str}。")
        elif h == "V字反発": add_reasons.append(f"【🎯店癖】過去の傾向から、好調ウェーブが継続しやすい『V字反発の波(前々日負け→前日勝ち)』に合致しています {rate_str}。")
        elif h == "連大凹み": add_reasons.append(f"【🎯店癖】過去の傾向から、強烈な底上げ（お詫び）が期待できる『2日連続大負けの波』に合致しています {rate_str}。")
        elif h == "高設定据え": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}据え置かれやすい『高設定挙動の大勝ち台』に合致しています {rate_str}。")
        elif h == "高稼働据え置き": add_reasons.append(f"【🎯店癖(安心)】過去の傾向から、{t_prefix_ha}『タコ粘りされた翌日でも高設定を据え置く(または入れ直す)』太っ腹な傾向があります {rate_str}。")
        elif h == "高設定完全据え置き": add_reasons.append(f"【🎯店癖(安心)】過去の傾向から、{t_prefix_ha}『前日高設定挙動の優秀台をそのまま据え置く』傾向が非常に強いです {rate_str}。")
        elif h == "連勝据え置き": add_reasons.append(f"【🎯店癖(波乗り)】過去の傾向から、{t_prefix_ha}『連勝中の台を回収せず、さらに出玉を伸ばさせる(据え置く)』傾向があります {rate_str}。")
        elif h == "メイン角": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_ha}『メイン通路側の角台』にしっかり設定を入れてアピールする傾向があります {rate_str}。")
        elif h == "目立つ島": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_ha}『メイン通路沿いの目立つ島』をベース高めに扱う傾向があります {rate_str}。")
        elif h == "壁側・死に島": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_ha}あえて『壁側の目立たない島』に当たりを隠すクセがあります {rate_str}。")
        elif h.endswith("曜日"): add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_ha}『{h}』に高設定を多く投入する還元傾向があります {rate_str}。")
        elif h == "看板機種": add_reasons.append(f"【🎯店癖】過去の傾向から、この機種は{t_prefix_ha}看板機種として非常に甘く扱われています {rate_str}。")

    if "BB設定6以上" in fixed_hot: add_reasons.append("【🎯期待】5000G以上回ってBIG確率が設定6を上回っています。REGが引けていなくてもベースが高設定である期待が持てます。")
    if "BB設定5以上" in fixed_hot: add_reasons.append("【🎯期待】5000G以上回ってBIG確率が設定5を上回っています。REGが引けていなくてもベースが高設定である期待が持てます。")
    if "超REG突出" in fixed_hot: add_reasons.append("【🎯激熱】5000G以上回ってREG確率が1/200より良い極端な優秀台です。設定6（またはそれ以上）の期待が非常に高いお宝台です。")

    for tid, c in zip(matched_cold_ids, matched_cold):
        w_rate = t_info['trend_win_rates'].get(tid, 0)
        diff_rate = t_info['trend_diffs'].get(tid, 0)
        rate_str = f"(実績:高設定率{w_rate:.1f}% / 通常より{diff_rate:+.1f}%)"
        
        t_prefix_de = f"この店舗の{status_label}で" if status_label != "全体" else "この店舗で"
        t_prefix_ha = f"この店舗の{status_label}は" if status_label != "全体" else "この店舗は"

        if c == "大勝反動": add_reasons.append(f"【⚠️警戒】大勝後のREG確率が悪い台です。過去の傾向から{t_prefix_de}反動（回収）の危険性が高いため注意してください {rate_str}。")
        elif c == "連高REG反動": add_reasons.append(f"【⚠️警戒】連続で高REGを記録している台ですが、過去の傾向から{t_prefix_de}連日据え置かれた翌日は回収される危険性が高いため注意してください {rate_str}。")
        elif c == "一撃反動": add_reasons.append(f"【⚠️警戒】一撃で出た荒波台です。過去の傾向から{t_prefix_de}据え置きされにくく回収される危険性が高いため注意してください {rate_str}。")
        elif c == "高稼働反動": add_reasons.append(f"【⚠️警戒】前日よく回された台ですが、過去の傾向から{t_prefix_de}タコ粘りされた翌日は設定が下げられる(回収される)危険性が高いため注意してください {rate_str}。")
        elif c == "高設定下げ": add_reasons.append(f"【⚠️警戒】前日は高設定挙動でしたが、過去の傾向から{t_prefix_de}優秀台の据え置きが少なく、設定が下げられる危険性が高いため注意してください {rate_str}。")
        elif c == "連勝ストップ": add_reasons.append(f"【⚠️警戒】連勝中の好調台ですが、過去の傾向から{t_prefix_de}連続プラスの翌日は回収される危険性が高いため注意してください {rate_str}。")
        elif c.startswith("メイン角"): add_reasons.append(f"【⚠️警戒】過去の傾向から、{t_prefix_ha}『メイン通路側の角台』をフェイク（低設定の誤爆待ち）として使う傾向が強いため注意してください {rate_str}。")
        elif c.startswith("目立つ島"): add_reasons.append(f"【⚠️警戒】過去の傾向から、{t_prefix_ha}『メイン通路沿いの島』を回収用（黙っても客が座るため）に使う傾向が強いため注意してください {rate_str}。")
        elif c.startswith("壁側"): add_reasons.append(f"【⚠️警戒】過去の傾向から、{t_prefix_ha}『壁側の目立たない島』には設定を入れない傾向が強いため注意してください {rate_str}。")
        elif c.endswith("のつく日(冷遇)"): add_reasons.append(f"【⚠️警戒】過去の傾向から、{t_prefix_de}『{c.replace('(冷遇)', '')}』は回収日(高設定率が低い)の傾向が強いため注意してください {rate_str}。")
        elif c.endswith("(冷遇)"): add_reasons.append(f"【⚠️警戒】過去の傾向から、{t_prefix_de}『{c.replace('(冷遇)', '')}』は高設定が入りにくい傾向が強いため注意してください {rate_str}。")
        elif c.endswith("曜日(冷遇)"): add_reasons.append(f"【⚠️警戒】過去の傾向から、{t_prefix_de}『{c.replace('(冷遇)', '')}』は回収傾向が強いため注意してください {rate_str}。")
        elif c == "冷遇機種": add_reasons.append(f"【⚠️警戒】過去の傾向から、この機種は{t_prefix_de}極めて辛く扱われている（冷遇されている）ため注意してください {rate_str}。")

    if "中間設定濃厚" in fixed_cold: add_reasons.append("【⚠️警戒】BB確率が設定1より悪く、かつREG確率が設定6に届いていません。中間設定の誤爆やフェイクの可能性が高いため、高設定狙いとしては危険です。")
    
    # --- 店舗の「癖の強さ（掴みやすさ）」判定 ---
    is_hard_to_predict = False
    if not t_info:
        is_hard_to_predict = True
    else:
        top_diffs = [t_info['trend_diffs'].get(tid, 0) for tid in t_info.get('top_ids', [])]
        max_diff = max(top_diffs) if top_diffs else 0
        if max_diff < 10.0:  # 一番強い癖でも、通常時より勝率が10%未満しか上がらない場合は「癖が弱い」と判定
            is_hard_to_predict = True
            
    # 予測スコアが一定以上（AIが推奨している）台にのみ、注意書きとして添える
    if is_hard_to_predict and score >= 0.40:
        add_reasons.append(f"【💡店舗傾向】過去の実績から、{t_prefix_ha}特定の条件(角台や凹み台など)に偏って設定を入れる『分かりやすい癖』が少なく、的を絞りにくい（散らして入れている）傾向があります。")

    if add_reasons:
        row['根拠'] = (reason + " " + " ".join(add_reasons)).strip()
        
    return row