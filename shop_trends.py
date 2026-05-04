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
        if 'is_corner_2' in train_shop.columns:
            subset = train_shop[train_shop['is_corner_2'] == 1]
            if len(subset) >= 5: trends.append({"id": "corner_2", "条件": "カド2", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'is_machine_border' in train_shop.columns:
            subset = train_shop[train_shop['is_machine_border'] == 1]
            if len(subset) >= 5: trends.append({"id": "machine_border", "条件": "機種またぎ (隣と機種が違う台)", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'is_zorome' in train_shop.columns:
            subset = train_shop[train_shop['is_zorome'] == 1]
            if len(subset) >= 5: trends.append({"id": "zorome", "条件": "ゾロ目台番号", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'is_main_corner' in train_shop.columns:
            subset = train_shop[train_shop['is_main_corner'] == 1]
            if len(subset) >= 5: trends.append({"id": "main_corner", "条件": "メイン角番", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'is_main_island' in train_shop.columns:
            subset = train_shop[train_shop['is_main_island'] == 1]
            if len(subset) >= 5: trends.append({"id": "main_island", "条件": "目立つ島 (メイン通路沿い)", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'is_wall_island' in train_shop.columns:
            subset = train_shop[train_shop['is_wall_island'] == 1]
            if len(subset) >= 5: trends.append({"id": "wall_island", "条件": "壁側・奥の島 (目立たない)", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
        if 'REG' in train_shop.columns and 'BIG' in train_shop.columns and 'is_prev_high_reg' in train_shop.columns:
            subset = train_shop[(train_shop['REG'] > train_shop['BIG']) & (train_shop['is_prev_high_reg'] == 1)]
            if len(subset) >= 5: trends.append({"id": "reg_lead", "条件": "REG先行・BB欠損 (高設定不発狙い)", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
            if 'BIG確率' in train_shop.columns:
                train_shop_tmp = train_shop.copy()
                train_shop_tmp['BIG分母'] = train_shop_tmp['BIG確率'].apply(lambda x: 1/x if x > 0 else 9999)
                subset_bb = train_shop_tmp[(train_shop_tmp['BIG分母'] >= 400) & (train_shop_tmp['is_prev_high_reg'] == 1)]
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
        if 'is_prev_high_reg' in train_shop.columns:
            subset = train_shop[train_shop['is_prev_high_reg'] == 1]
            if len(subset) >= 5: trends.append({"id": "high_setting_reaction", "条件": "前日高設定挙動", "高設定率": get_high_rate(subset), "サンプル": len(subset)})
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
    if "corner_2" in top_ids and row.get('is_corner_2') == 1: matched_hot.append("カド2"); matched_hot_ids.append("corner_2")
    if "machine_border" in top_ids and row.get('is_machine_border') == 1: matched_hot.append("機種またぎ"); matched_hot_ids.append("machine_border")
    if "zorome" in top_ids and row.get('is_zorome') == 1: matched_hot.append("ゾロ目台"); matched_hot_ids.append("zorome")
    if "reg_lead" in top_ids and row.get('REG', 0) > row.get('BIG', 0) and row.get('is_prev_high_reg', 0) == 1: matched_hot.append("BB欠損・不発"); matched_hot_ids.append("reg_lead")
    if "bb_deficit" in top_ids:
        b_p = row.get('BIG確率', 0)
        b_d = 1 / b_p if b_p > 0 else 9999
        if b_d >= 400 and row.get('is_prev_high_reg', 0) == 1: matched_hot.append("超不発"); matched_hot_ids.append("bb_deficit")
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
    if "corner_2" in worst_ids and row.get('is_corner_2') == 1: matched_cold.append("カド2(冷遇)"); matched_cold_ids.append("corner_2")
    if "machine_border" in worst_ids and row.get('is_machine_border') == 1: matched_cold.append("機種またぎ(冷遇)"); matched_cold_ids.append("machine_border")
    if "zorome" in worst_ids and row.get('is_zorome') == 1: matched_cold.append("ゾロ目台(冷遇)"); matched_cold_ids.append("zorome")
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
                
    if "high_setting_reaction" in worst_ids and row.get('is_prev_high_reg', 0) == 1:
        matched_cold.append("高設定下げ"); matched_cold_ids.append("high_setting_reaction")
            
    if "high_setting_reaction" in top_ids and row.get('is_prev_high_reg', 0) == 1:
        matched_hot.append("高設定完全据え置き"); matched_hot_ids.append("high_setting_reaction")

    hot_str = "🔥" + " ".join(matched_hot + fixed_hot) if (matched_hot or fixed_hot) else ""
    cold_str = "⚠️" + " ".join(matched_cold + fixed_cold) if (matched_cold or fixed_cold) else ""
    
    match_str = f"{hot_str} {cold_str}".strip()
    row['店癖マッチ'] = match_str
    
    # スコアの再計算（AIの予測スコアと、強力な店癖の過去実績確率をブレンドする）
    score = row.get('prediction_score', 0)
    sue_score = row.get('sueoki_score', 0)
    
    # 強い店癖（トップトレンド）の実績勝率を加味してスコアを底上げ
    top_win_rates = [t_info['trend_win_rates'].get(tid, 0) for tid in matched_hot_ids]
    if top_win_rates:
        max_trend_prob = max(top_win_rates) / 100.0
        # AIの評価が実績より低い場合、実績確率側に歩み寄らせる（中間の値をとる）
        if score < max_trend_prob:
            score = (score + max_trend_prob) / 2.0
        if sue_score < max_trend_prob:
            sue_score = (sue_score + max_trend_prob) / 2.0

    # 悪い店癖（ワーストトレンド）の実績勝率を加味してスコアを引き下げ
    worst_win_rates = [t_info['trend_win_rates'].get(tid, 0) for tid in matched_cold_ids]
    if worst_win_rates:
        min_trend_prob = min(worst_win_rates) / 100.0
        # AIの評価が実績より高い場合、下方に歩み寄らせる
        if score > min_trend_prob:
            score = (score + min_trend_prob) / 2.0
        if sue_score > min_trend_prob:
            sue_score = (sue_score + min_trend_prob) / 2.0
            
    row['prediction_score'] = score
    row['sueoki_score'] = sue_score

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
        elif h == "カド2": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}あえて角を避けた『カド2』に設定を入れる傾向があります {rate_str}。")
        elif h == "機種またぎ": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_de}『機種またぎ(隣と機種が違う台)』を起点に設定を入れる傾向があります {rate_str}。")
        elif h == "ゾロ目台": add_reasons.append(f"【🎯店癖】過去の傾向から、{t_prefix_ha}『ゾロ目台番号』に設定を入れてくる遊び心（語呂合わせ）の傾向があります {rate_str}。")
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
        elif c.startswith("カド2"): add_reasons.append(f"【⚠️警戒】過去の傾向から、{t_prefix_ha}『カド2』は冷遇されている傾向が強いため注意してください {rate_str}。")
        elif c.startswith("機種またぎ"): add_reasons.append(f"【⚠️警戒】過去の傾向から、{t_prefix_ha}『機種またぎ』の境界線は冷遇されている傾向が強いため注意してください {rate_str}。")
        elif c.startswith("ゾロ目台"): add_reasons.append(f"【⚠️警戒】過去の傾向から、{t_prefix_ha}『ゾロ目台番号』はフェイクや冷遇に使われる傾向が強いため注意してください {rate_str}。")
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
    if is_hard_to_predict and max(score, sue_score) >= 0.40:
        add_reasons.append(f"【💡店舗傾向】過去の実績から、{t_prefix_ha}特定の条件(角台や凹み台など)に偏って設定を入れる『分かりやすい癖』が少なく、的を絞りにくい（散らして入れている）傾向があります。")

    if add_reasons:
        row['根拠'] = (reason + " " + " ".join(add_reasons)).strip()
        
    return row

def analyze_sueoki_and_change_triggers(df_train, shop_name, shop_col='店名'):
    shop_df = df_train[df_train[shop_col] == shop_name].copy()
    if len(shop_df) < 50:
        return None
        
    # 1. 据え置き傾向の独立診断
    prev_high = shop_df[shop_df['is_prev_high_reg'] == 1]
    sueoki_rate = prev_high['target'].mean() if len(prev_high) > 0 else 0.0
    
    if sueoki_rate >= 0.30: sue_tendency = "強い"
    elif sueoki_rate >= 0.15: sue_tendency = "混在"
    else: sue_tendency = "弱い"

    # 2. 変更(上げ)トリガーの分析
    change_df = shop_df[(shop_df['is_prev_high_reg'] == 0) & (shop_df['target'] == 1)]
    base_change_df = shop_df[shop_df['is_prev_high_reg'] == 0]
    
    trigger_diff = "弱"
    trigger_wd = "影響なし"
    trigger_kado = "影響なし"
    master_judge = "ランダム"
    
    if len(change_df) > 0 and len(base_change_df) > 0:
        base_rate = len(change_df) / len(base_change_df)
        
        # 差枚 (連続マイナス日数 >= 2 or prev_差枚 <= -1000)
        base_diff_minus = base_change_df[(base_change_df.get('連続マイナス日数', 0) >= 2) | (base_change_df.get('prev_差枚', 0) <= -1000)]
        diff_minus = change_df[(change_df.get('連続マイナス日数', 0) >= 2) | (change_df.get('prev_差枚', 0) <= -1000)]
        rate_diff = len(diff_minus) / len(base_diff_minus) if len(base_diff_minus) > 0 else 0
        if rate_diff > base_rate * 2.0 and len(diff_minus) >= 3: trigger_diff = "強"
        elif rate_diff > base_rate * 1.5 and len(diff_minus) >= 2: trigger_diff = "中"

        # 曜日
        wd_counts = change_df.get('target_weekday', pd.Series(dtype=int)).value_counts()
        if not wd_counts.empty and wd_counts.iloc[0] >= len(change_df) * 0.3 and wd_counts.iloc[0] >= 3:
            trigger_wd = "影響あり"
            
        # 稼働率
        base_low_kado = base_change_df[base_change_df.get('prev_累計ゲーム', 0) < 2000]
        low_kado = change_df[change_df.get('prev_累計ゲーム', 0) < 2000]
        rate_low_kado = len(low_kado) / len(base_low_kado) if len(base_low_kado) > 0 else 0
        if rate_low_kado > base_rate * 1.5 and len(low_kado) >= 3: trigger_kado = "影響あり"
            
        if trigger_diff == "強" and trigger_wd == "影響あり": master_judge = "複合"
        elif trigger_diff in ["強", "中"]: master_judge = "差枚主導"
        elif trigger_wd == "影響あり": master_judge = "曜日主導"

    return {
        "sueoki_rate": sueoki_rate,
        "sue_tendency": sue_tendency,
        "trigger_diff": trigger_diff,
        "trigger_wd": trigger_wd,
        "trigger_kado": trigger_kado,
        "master_judge": master_judge
    }

def diagnose_allocation_types(df_train, shop_col, specs):
    """
    店舗の「配分型」を過去データから診断し、辞書形式で返す関数。
    誤認誘導型(中間多用)、島型/機種型(面配分)、単体型(点配分)、ローテーション型等のフラグを含む。
    """
    alloc_types = {}
    for s in df_train[shop_col].unique():
        shop_df = df_train[df_train[shop_col] == s].copy()
        if len(shop_df) < 100:
            alloc_types[s] = {"is_mislead": False, "is_point": False, "main_type": "不明", "messages": ["データ不足のため配分型を診断できません。"]}
            continue
            
        messages = []
        is_mislead = False
        is_point = False
        main_type = "不明"

        valid_df = shop_df[shop_df['累計ゲーム'] >= 3000].copy()
        if len(valid_df) < 50:
            alloc_types[s] = {"is_mislead": False, "is_point": False, "main_type": "不明", "messages": ["稼働データ不足のため配分型を診断できません。"]}
            continue

        # 1. 誤認誘導型 (ミスリード・中間多用) の判定
        valid_df['is_reg_good'] = valid_df['REG確率'] >= (1/300.0)
        reg_good_df = valid_df[valid_df['is_reg_good']]
        if not reg_good_df.empty:
            mislead_rate = len(reg_good_df[(reg_good_df['差枚'] >= -500) & (reg_good_df['差枚'] <= 1000)]) / len(reg_good_df)
            lose_rate_in_good_reg = len(reg_good_df[reg_good_df['差枚'] < 0]) / len(reg_good_df)
            
            if mislead_rate >= 0.50 or lose_rate_in_good_reg >= 0.35:
                is_mislead = True
                messages.append("🎭 **誤認誘導型 (ミスリード・中間多用)**\nREG確率は設定4水準を満たす台が多いですが、差枚が伴わない(勝てない)不発台が異常に多いです。「中間設定を多用して高設定(5,6)と誤認させる」配分思想の可能性が極めて高く、AIは**『REG単体での過大評価を防止(差枚の伴いを必須条件化)』**して予測を厳格化しています。")

        # 2. 面配分 (島型・機種型) vs 点配分 (単体型) の判定
        daily_island = valid_df.groupby(['対象日付', 'island_id'])['差枚'].mean().reset_index()
        daily_mac = valid_df.groupby(['対象日付', '機種名'])['差枚'].mean().reset_index()
        daily_shop = valid_df.groupby('対象日付')['差枚'].mean().reset_index().rename(columns={'差枚': 'shop_avg'})
        
        daily_island = pd.merge(daily_island, daily_shop, on='対象日付')
        daily_mac = pd.merge(daily_mac, daily_shop, on='対象日付')
        
        island_hit_rate = len(daily_island[(daily_island['差枚'] > 1000) & (daily_island['差枚'] > daily_island['shop_avg'] + 500)]) / len(daily_shop) if not daily_island.empty else 0
        mac_hit_rate = len(daily_mac[(daily_mac['差枚'] > 1000) & (daily_mac['差枚'] > daily_mac['shop_avg'] + 500)]) / len(daily_shop) if not daily_mac.empty else 0
        
        # --- 新規追加: 各機種散らし型（各機種イチ配分）の判定 ---
        mac_dispersion = valid_df.copy()
        mac_dispersion['is_hot_machine'] = mac_dispersion['差枚'] >= 1000
        
        dispersion_stats = mac_dispersion.groupby('対象日付').agg(
            total_active_macs=('機種名', 'nunique'),
            hot_macs=('機種名', lambda x: x[mac_dispersion.loc[x.index, 'is_hot_machine']].nunique()),
            total_hot_machines=('台番号', lambda x: mac_dispersion.loc[x.index, 'is_hot_machine'].sum())
        ).reset_index()
        dispersion_stats['mac_hit_coverage'] = np.where(dispersion_stats['total_active_macs'] > 0, dispersion_stats['hot_macs'] / dispersion_stats['total_active_macs'], 0)
        dispersion_stats['hot_per_mac'] = np.where(dispersion_stats['hot_macs'] > 0, dispersion_stats['total_hot_machines'] / dispersion_stats['hot_macs'], 0)
        each_mac_rate = ((dispersion_stats['mac_hit_coverage'] >= 0.40) & (dispersion_stats['hot_per_mac'] <= 2.5)).mean() if not dispersion_stats.empty else 0

        top_machines = valid_df.loc[valid_df.groupby('対象日付')['差枚'].idxmax()]
        top_machines = pd.merge(top_machines, daily_island.rename(columns={'差枚': 'island_avg'}), on=['対象日付', 'island_id'], how='left')
        point_hit_rate = len(top_machines[(top_machines['差枚'] >= 2500) & (top_machines['island_avg'] <= 0)]) / len(top_machines) if not top_machines.empty else 0

        if each_mac_rate >= 0.20 and each_mac_rate > island_hit_rate and each_mac_rate > mac_hit_rate:
            main_type = "各機種散らし型 (各機種イチ・ニ配分)"
            messages.append("🎯 **各機種散らし型 (各機種イチ・ニ配分)**\n特定の島や全台系を作るのではなく、「多くの機種に1〜2台ずつ当たり台を散らばらせる」傾向が強いです。島全体の強さに騙されず、自分が打っている機種の中にまだ当たり台(高設定)が見えていないかを重視して立ち回ってください。\n  └ 💡 **立ち回りアドバイス**: パイ（当たり）の奪い合いになります。すでに同じ機種の中に明らかな当たり台がある場合、自分が座っている台は『中間設定のフェイク』である危険性が高いため、強い警戒が必要です。")
            is_point = True # 周りの台(島全体)の挙動に引っ張られすぎないように単体型ベースで学習させる
        elif island_hit_rate >= 0.20:
            main_type = "島型 (面配分)"
            island_msg = "🏝️ **島型 (面配分)**\n同じ島(列)に合算やREGが似通う台が固まりやすく、島全体の平均が強くなる傾向があります。「周りの台の挙動(面)」が強力な判別要素になります。"
            
            # --- 🏝️ 島型のサブタイプ分析 (並び・ランダム・フェイク) ---
            hit_islands = daily_island[(daily_island['差枚'] > 1000) & (daily_island['差枚'] > daily_island['shop_avg'] + 500)]
            if not hit_islands.empty:
                hit_island_keys = hit_islands[['対象日付', 'island_id']].drop_duplicates()
                hit_island_data = pd.merge(valid_df, hit_island_keys, on=['対象日付', 'island_id'], how='inner')
                
                if 'is_reg_good' in hit_island_data.columns:
                    fake_in_island_rate = len(hit_island_data[(hit_island_data['is_reg_good']) & (hit_island_data['差枚'] <= 0)]) / len(hit_island_data)
                    avg_island_size = hit_island_data.groupby('island_id')['台番号'].nunique().mean()
                    
                    if fake_in_island_rate >= 0.25:
                        island_msg += "\n  └ ⚠️ **フェイク交じり**: 当たり島の中にも「REGだけ引けて差枚がマイナス」のフェイク台が多数混ざっています。島全体が全台高設定というわけではなく、誤認を誘う配分です。"
                        if avg_island_size < 5:
                            island_msg += "\n  └ 🚨 **少台数島の誤認注意**: 台数が少ない島では、1台の爆出しに平均差枚が引っ張られ、全台系に見えてしまうトラップ(フェイク島)が頻発しています。過去の島フェイク率に注意してください。"
                    elif fake_in_island_rate <= 0.10:
                         island_msg += "\n  └ 💎 **全台ベース高め**: 当たり島にはフェイクが少なく、全体的にしっかり出玉が伴う傾向があります。島が強ければ安心して攻められます。"
                         
                if '台番号' in hit_island_data.columns:
                    hit_island_data_n = hit_island_data.copy()
                    hit_island_data_n['台番号_num'] = pd.to_numeric(hit_island_data_n['台番号'], errors='coerce')
                    hit_island_data_n = hit_island_data_n.dropna(subset=['台番号_num']).sort_values(['対象日付', 'island_id', '台番号_num'])
                    
                    narabi_blocks = 0
                    total_hit_islands = 0
                    for (d, i_id), group in hit_island_data_n.groupby(['対象日付', 'island_id']):
                        total_hit_islands += 1
                        group['is_hot'] = group['差枚'] >= 500
                        group['block'] = (group['is_hot'] != group['is_hot'].shift()).cumsum()
                        hot_blocks_counts = group[group['is_hot']].groupby('block').size()
                        if not hot_blocks_counts.empty and hot_blocks_counts.max() >= 3:
                             narabi_blocks += 1
                             
                    if total_hit_islands > 0:
                        narabi_rate = narabi_blocks / total_hit_islands
                        if narabi_rate >= 0.40:
                            island_msg += "\n  └ 🤝 **塊・並び集中**: 島の中でも特に「3台以上の並び（塊）」で高設定が入る傾向が強いです。当たり島を見つけたら、両隣の挙動が良い場所を優先して狙ってください。"
                        elif narabi_rate <= 0.15:
                            island_msg += "\n  └ 🎲 **ランダム・散らし**: 島全体は強いですが、出ている台は島の中でランダムに散らばっています。「隣が出ているから」という根拠は通用しにくいため、単体の挙動を重視してください。\n     └ 💡 **立ち回りアドバイス**: 島内の当たり台数に上限がある可能性があるため、島内に他の爆出し台が複数ある場合は自分の台が罠である可能性も考慮し、少し警戒レベルを上げましょう。"

            messages.append(island_msg)
        elif mac_hit_rate >= 0.20:
            main_type = "機種型 (機種単位配分)"
            mac_msg = "🎰 **機種型 (機種単位配分)**\n特定機種だけが明確に強くなる「全台系・半列系」の塊を作る傾向があります。機種全体のベースの高さに注目してください。"
            
            # --- 🎰 機種型のサブタイプ分析 (全台系・半列/ランダム・フェイク) ---
            hit_macs = daily_mac[(daily_mac['差枚'] > 1000) & (daily_mac['差枚'] > daily_mac['shop_avg'] + 500)]
            if not hit_macs.empty:
                hit_mac_keys = hit_macs[['対象日付', '機種名']].drop_duplicates()
                hit_mac_data = pd.merge(valid_df, hit_mac_keys, on=['対象日付', '機種名'], how='inner')
                
                if 'is_reg_good' in hit_mac_data.columns:
                    fake_in_mac_rate = len(hit_mac_data[(hit_mac_data['is_reg_good']) & (hit_mac_data['差枚'] <= 0)]) / len(hit_mac_data)
                    if fake_in_mac_rate >= 0.25:
                        mac_msg += "\n  └ ⚠️ **フェイク交じり**: 当たり機種の中にも「REGだけ引けて差枚がマイナス」のフェイク台が多数混ざっています。「全台系」に見せかけた「1/2配分」等に注意してください。"
                    elif fake_in_mac_rate <= 0.10:
                        mac_msg += "\n  └ 💎 **完全全台系**: 当たり機種にはフェイクが少なく、全体的にしっかり出玉が伴う傾向があります。「全台系」の信頼度が非常に高いです。"

                if '台番号' in hit_mac_data.columns:
                    hit_mac_data_n = hit_mac_data.copy()
                    hit_mac_data_n['台番号_num'] = pd.to_numeric(hit_mac_data_n['台番号'], errors='coerce')
                    hit_mac_data_n = hit_mac_data_n.dropna(subset=['台番号_num']).sort_values(['対象日付', '機種名', '台番号_num'])
                    
                    narabi_blocks = 0
                    total_hit_macs = 0
                    for (d, m_name), group in hit_mac_data_n.groupby(['対象日付', '機種名']):
                        total_hit_macs += 1
                        group['is_hot'] = group['差枚'] >= 500
                        group['block'] = (group['is_hot'] != group['is_hot'].shift()).cumsum()
                        hot_blocks_counts = group[group['is_hot']].groupby('block').size()
                        if not hot_blocks_counts.empty and hot_blocks_counts.max() >= 3:
                             narabi_blocks += 1
                             
                    if total_hit_macs > 0:
                        narabi_rate = narabi_blocks / total_hit_macs
                        if narabi_rate >= 0.40:
                            mac_msg += "\n  └ 🤝 **塊・並び集中**: 当たり機種の中でも特に「固まって」設定が入る傾向があります。出ている台の隣を狙うのがセオリーです。"
                        elif narabi_rate <= 0.15:
                            mac_msg += "\n  └ 🎲 **ランダム・散らし**: 当たり機種の中でも、出ている台はランダムに散らばっています。「末尾」など別の法則が絡んでいる可能性があります。\n     └ 💡 **立ち回りアドバイス**: 対象機種であっても、他の台がすでに出切っている場合はフェイクの罠に注意して押し引きを判断してください。"
            
            messages.append(mac_msg)
        elif point_hit_rate >= 0.40:
            is_point = True
            main_type = "単体型 (点配分)"
            messages.append("📍 **単体型 (点配分)**\n島や機種全体は死んでいるのに、ポツンと1台だけ突出して出ている日が多いです。ヒキ依存や当て物(ピンポイント)の要素が強く、周りの状況はアテになりません。深追いは禁物です。\n  └ 💡 **立ち回りアドバイス**: 各島（列）に1〜2台しか当たりがない傾向が強いため、同じ島内にすでに別の大当たり台がある場合、自分の台はフェイクの罠である可能性が高まります。パイの奪い合いを意識した撤退判断をおすすめします。")
        else:
            main_type = "複合型 (散らし配分)"
            messages.append("🧩 **複合型 (散らし配分)**\n島・機種・単体が複雑に混ざっています。明確な「面」が形成されにくいため、複数の根拠(店癖や波)を掛け合わせて狙う必要があります。\n  └ 💡 **立ち回りアドバイス**: 特定の「全台系」や「島単位」の狙いだけに固執せず、広く視野を持つことが重要です。周りの状況に流されず、目の前の台の単体挙動や、AIが提示する複数の根拠（店癖など）が重なっている台を優先して押し引きを判断してください。")

        # --- 店舗全体の並び・塊傾向の事前計算 (ローテーション型や客層反応型のサブ分析用) ---
        narabi_rate_shop = 0
        if '台番号' in valid_df.columns:
            valid_df_n = valid_df.copy()
            valid_df_n['台番号_num'] = pd.to_numeric(valid_df_n['台番号'], errors='coerce')
            valid_df_n = valid_df_n.dropna(subset=['台番号_num']).sort_values(['対象日付', '台番号_num'])
            
            narabi_blocks_shop = 0
            total_active_days = valid_df_n['対象日付'].nunique()
            
            if total_active_days > 0:
                for d, group in valid_df_n.groupby('対象日付'):
                    group['is_hot'] = group['差枚'] >= 500
                    group['block'] = (group['is_hot'] != group['is_hot'].shift()).cumsum()
                    hot_blocks_counts = group[group['is_hot']].groupby('block').size()
                    if not hot_blocks_counts.empty and hot_blocks_counts.max() >= 3:
                         narabi_blocks_shop += 1
                narabi_rate_shop = narabi_blocks_shop / total_active_days

        # 3. ローテーション型
        age_rate = shop_df[(shop_df['連続マイナス日数'] >= 2) & (shop_df['累計ゲーム'] >= 3000)]['target'].mean()
        sue_rate = shop_df[(shop_df['is_prev_high_reg'] == 1) & (shop_df['累計ゲーム'] >= 3000)]['target'].mean()
        if pd.notna(age_rate) and pd.notna(sue_rate) and age_rate > (sue_rate * 1.2) and age_rate >= 0.15:
            rot_msg = "🔄 **ローテーション型 (循環配分)**\n毎日強い場所が日替わりで移動し、前日弱かった島・機種・凹み台に設定が入りやすいです。AIの「過去特徴→翌日の変更(上げ)予測」が非常に機能しやすい環境です。"
            if narabi_rate_shop >= 0.30:
                rot_msg += "\n  └ 🤝 **塊ローテ**: 凹み台の「隣」なども巻き込んで、3台以上の塊でローテーションしてくる傾向があります。狙い台の隣もチャンスです。"
            elif narabi_rate_shop <= 0.15:
                rot_msg += "\n  └ 🎲 **単体ローテ**: 塊にはならず、凹んだ台単体だけをピンポイントで上げてきます。周りの状況に流されないようにしてください。"
            rot_msg += "\n  └ 💡 **立ち回りアドバイス**: 昨日の当たり台や島を避け、数日間凹んでいる台や冷遇されていた島を優先して狙ってください。AIの「変更(上げ)期待度」が高い台を中心に攻めるのがセオリーです。"
            messages.append(rot_msg)
        
        # 4. 客層反応型
        low_kado_hit = shop_df[(shop_df['prev_累計ゲーム'] < 1500) & (shop_df['prev_累計ゲーム'] > 0)]['target'].mean()
        if pd.notna(low_kado_hit) and low_kado_hit >= 0.20:
            reac_msg = "👥 **客層反応型 (リアクティブ)**\n客に見切られて稼働が落ちた(空き台が増えた)台や島に対して、テコ入れで設定を入れてくる傾向が見られます。"
            if narabi_rate_shop >= 0.30:
                reac_msg += "\n  └ 🤝 **島ごとテコ入れ**: 低稼働になった列や島を、塊で一気に全台系・半列系にしてテコ入れしてくる傾向があります。"
            elif narabi_rate_shop <= 0.15:
                reac_msg += "\n  └ 🎲 **単体テコ入れ**: 低稼働の台の中で、ポツンと単体で設定を入れて稼働を煽ります。並びを意識する必要はありません。"
            reac_msg += "\n  └ 💡 **立ち回りアドバイス**: 前日あまり回されなかった「放置台」が狙い目になります。人が少ない島や、稼働が落ちている機種にチャンスが眠っている可能性が高いです。"
            messages.append(reac_msg)

        # --- 5. 具体的な狙い目 (ピンポイント投入傾向) の抽出 ---
        hekomi_threshold = -1000
        win_threshold = 1000
        low_kado_threshold = 2000
        
        if '差枚' in shop_df.columns:
            q_lose = shop_df['差枚'].quantile(0.15)
            hekomi_threshold = min(-500, int(round(q_lose / 500) * 500)) if pd.notna(q_lose) and q_lose < 0 else -1000
            q_win = shop_df['差枚'].quantile(0.85)
            win_threshold = max(500, int(round(q_win / 500) * 500)) if pd.notna(q_win) and q_win > 0 else 1000
            
        if '累計ゲーム' in shop_df.columns:
            avg_kado = shop_df['累計ゲーム'].mean()
            low_kado_threshold = max(1000, int((avg_kado * 0.5) // 500 * 500))

        if 'target' in valid_df.columns:
            all_hit_rate = valid_df['target'].mean()
            target_hints = []
            
            if 'is_corner' in valid_df.columns:
                corner_hit_rate = valid_df[valid_df['is_corner'] == 1]['target'].mean() if len(valid_df[valid_df['is_corner'] == 1]) > 0 else 0
                if corner_hit_rate > all_hit_rate * 1.5 and corner_hit_rate >= 0.15:
                    target_hints.append(f"「角台」 (高設定率 {corner_hit_rate*100:.1f}%)")
                    
            if 'prev_差枚' in valid_df.columns:
                hekomi_hit_rate = valid_df[valid_df['prev_差枚'] <= hekomi_threshold]['target'].mean() if len(valid_df[valid_df['prev_差枚'] <= hekomi_threshold]) > 0 else 0
                if hekomi_hit_rate > all_hit_rate * 1.5 and hekomi_hit_rate >= 0.15:
                    target_hints.append(f"「前日大凹み({hekomi_threshold}枚以下)からの上げ」 (高設定率 {hekomi_hit_rate*100:.1f}%)")
                    
                win_hit_rate = valid_df[valid_df['prev_差枚'] >= win_threshold]['target'].mean() if len(valid_df[valid_df['prev_差枚'] >= win_threshold]) > 0 else 0
                if win_hit_rate > all_hit_rate * 1.5 and win_hit_rate >= 0.15:
                    target_hints.append(f"「前日大勝ち(+{win_threshold}枚以上)の据え置き」 (高設定率 {win_hit_rate*100:.1f}%)")

            if 'prev_累計ゲーム' in valid_df.columns:
                low_kado_hit_rate = valid_df[valid_df['prev_累計ゲーム'] < low_kado_threshold]['target'].mean() if len(valid_df[valid_df['prev_累計ゲーム'] < low_kado_threshold]) > 0 else 0
                if low_kado_hit_rate > all_hit_rate * 1.5 and low_kado_hit_rate >= 0.15:
                    target_hints.append(f"「前日低稼働({low_kado_threshold}G未満)の放置台」 (高設定率 {low_kado_hit_rate*100:.1f}%)")
                    
            if 'neighbor_high_setting_count' in valid_df.columns:
                neighbor_hit_rate = valid_df[valid_df['neighbor_high_setting_count'] >= 1]['target'].mean() if len(valid_df[valid_df['neighbor_high_setting_count'] >= 1]) > 0 else 0
                if neighbor_hit_rate > all_hit_rate * 1.5 and neighbor_hit_rate >= 0.15:
                    target_hints.append(f"「両隣のどちらかが高設定挙動だった台(並び狙い)」 (高設定率 {neighbor_hit_rate*100:.1f}%)")
                    
            if 'is_prev_high_reg' in valid_df.columns:
                sue_hit_rate = valid_df[valid_df['is_prev_high_reg'] == 1]['target'].mean() if len(valid_df[valid_df['is_prev_high_reg'] == 1]) > 0 else 0
                if sue_hit_rate > all_hit_rate * 1.5 and sue_hit_rate >= 0.15:
                    target_hints.append(f"「前日高設定挙動の据え置き」 (高設定率 {sue_hit_rate*100:.1f}%)")
                    
            if 'is_main_island' in valid_df.columns:
                main_isl_hit_rate = valid_df[valid_df['is_main_island'] == 1]['target'].mean() if len(valid_df[valid_df['is_main_island'] == 1]) > 0 else 0
                if main_isl_hit_rate > all_hit_rate * 1.5 and main_isl_hit_rate >= 0.15:
                    target_hints.append(f"「メイン通路沿いの目立つ島」 (高設定率 {main_isl_hit_rate*100:.1f}%)")
            
            if '対象日付' in valid_df.columns and '末尾番号' in valid_df.columns and '差枚' in valid_df.columns:
                end_daily = valid_df.groupby(['対象日付', '末尾番号']).agg(末尾平均=('差枚', 'mean'), 台数=('台番号', 'count')).reset_index()
                end_daily = end_daily[end_daily['台数'] >= 3].dropna(subset=['末尾平均'])
                if not end_daily.empty:
                    idx_max = end_daily.groupby('対象日付')['末尾平均'].idxmax()
                    top_end = end_daily.loc[idx_max]
                    daily_shop_stats = valid_df.groupby('対象日付').agg(店舗平均=('差枚', 'mean')).reset_index()
                    merged_end = pd.merge(top_end, daily_shop_stats, on='対象日付')
                    hit_end_days = ((merged_end['末尾平均'] >= 500) & ((merged_end['末尾平均'] - merged_end['店舗平均']) >= 500)).sum()
                    total_active_days = valid_df['対象日付'].nunique()
                    hit_end_rate = hit_end_days / total_active_days if total_active_days > 0 else 0
                    if hit_end_rate >= 0.25:
                        target_hints.append(f"「日替わりの当たり末尾」 (当たり末尾が存在する日の割合 {hit_end_rate*100:.1f}%)")
                        
            if target_hints:
                hints_str = "\n  ・".join(target_hints)
                messages.append(f"🔍 **具体的なピンポイント狙い目 (後ヅモ・台選びのヒント)**\n過去のデータから、配分とは別に以下の条件を満たす台に設定が入りやすい傾向が見られます。\n  ・{hints_str}")
            else:
                messages.append(f"🔍 **具体的なピンポイント狙い目 (後ヅモ・台選びのヒント)**\n過去のデータからは、角台や特定末尾、据え置きといった分かりやすいピンポイントの投入傾向（明確な偏り）は見られませんでした。特定の場所に依存せず、周りの台の挙動や島・機種全体の強さを優先して判断してください。")

        alloc_types[s] = {
            "is_mislead": is_mislead,
            "is_point": is_point,
            "main_type": main_type,
            "messages": messages,
            "hekomi_threshold": hekomi_threshold,
            "win_threshold": win_threshold,
            "low_kado_threshold": low_kado_threshold
        }
        
    return alloc_types

def evaluate_sueoki_premise(df_raw_shop, target_date, df_events=None):
    """
    本日の「据え置き前提成立判定」を行う。
    YES / NO / 不明 とその理由を返す。
    """
    if df_raw_shop.empty:
        return "不明", "過去データが不足しているため判定できません。"
        
    target_dt = pd.to_datetime(target_date)
    shop_col = '店名' if '店名' in df_raw_shop.columns else ('店舗名' if '店舗名' in df_raw_shop.columns else None)
    if not shop_col:
        return "不明", "店舗名データがありません。"
        
    shop_name = df_raw_shop[shop_col].iloc[0]
    
    reasons = []
    is_no = False
    
    # 1. イベント日判定
    if df_events is not None and not df_events.empty:
        ev_today = df_events[(df_events['店名'] == shop_name) & (df_events['イベント日付'].dt.date == target_dt.date())]
        if not ev_today.empty:
            ev = ev_today.iloc[0]
            if ev.get('イベントランク') in ['SS (周年)', 'S', 'A']:
                is_no = True
                reasons.append(f"本日は強いイベント（{ev.get('イベント名')}）が予定されており、設定の入れ替え（変更）がメインになると予想されます。")

    # 2. 前日の回収状況（回収→還元の切り替え日か？）
    prev_date = target_dt - pd.Timedelta(days=1)
    df_prev = df_raw_shop[df_raw_shop['対象日付'].dt.date == prev_date.date()]
    if not df_prev.empty:
        prev_avg_diff = df_prev['差枚'].mean()
        if prev_avg_diff <= -150:
            is_no = True
            reasons.append(f"前日が強めの回収営業（店舗平均 {int(prev_avg_diff)}枚）であり、本日は還元（上げリセット）にシフトする可能性が高い切り替え日です。")

    if is_no:
        return "NO", " ".join(reasons)
    else:
        return "YES", "設定変更の強いトリガー（特日や極端な回収など）が検知されていないため、据え置き(昨日の続き)を期待できる前提が成立しています。"