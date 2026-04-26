import pandas as pd
import numpy as np
import re
import math
from utils import get_matched_spec_key, classify_shop_eval
from shop_trends import calculate_shop_trends, apply_trends_to_row
from config import MACHINE_SPECS

def postprocess_predictions(predict_df, train_df):
    specs = MACHINE_SPECS
    
    def apply_setting5_boost(row):
        score = row.get('prediction_score', 0)
        machine_name = row.get('機種名', '')
        reg_prob = row.get('REG確率', 0)
        games = row.get('累計ゲーム', 0)
        
        if reg_prob <= 0 or games < 3000:
            return score
            
        matched_spec_key = get_matched_spec_key(machine_name, specs)
                    
        if matched_spec_key and "設定5" in specs[matched_spec_key]:
            set5_reg_prob = 1.0 / specs[matched_spec_key]["設定5"]["REG"]
            if reg_prob >= set5_reg_prob:
                score = score + (1.0 - score) * 0.15 # 設定5以上なら残りの伸びしろの15%を加算
        return score

    if not predict_df.empty: predict_df['prediction_score'] = predict_df.apply(apply_setting5_boost, axis=1)
    if not train_df.empty: train_df['prediction_score'] = train_df.apply(apply_setting5_boost, axis=1)

    def apply_reliability_penalty(row):
        score = row.get('prediction_score', 0)
        hc = row.get('history_count', 1)
        games = row.get('累計ゲーム', 0)
        reg_prob = row.get('REG確率', 0)
        
        # 過去データが少ない場合は予測のブレが大きいためスコアを割り引く
        if hc < 14: score *= 0.8
        elif hc < 30: score *= 0.95
        
        # 前日の稼働が少ない場合の減点（ただし、高設定挙動を示している場合は減点を緩和する）
        is_good_reg = reg_prob >= (1.0 / 280.0) if reg_prob > 0 else False
        
        if games < 1000: score *= (0.85 if is_good_reg else 0.70)
        elif games < 2000: score *= (0.95 if is_good_reg else 0.85)
        elif games < 3000: score *= (1.0 if is_good_reg else 0.95)
        
        return score
        
    def get_reliability_mark(row):
        hc = row.get('history_count', 1)
        if hc < 14: return "🔻低"
        elif hc < 30: return "🔸中"
        return "🔼高"

    if not predict_df.empty: 
        predict_df['prediction_score'] = predict_df.apply(apply_reliability_penalty, axis=1)
        predict_df['予測信頼度'] = predict_df.apply(get_reliability_mark, axis=1)
    if not train_df.empty: 
        train_df['予測信頼度'] = train_df.apply(get_reliability_mark, axis=1)
        
    # --- 営業状態（還元/通常/回収）の事前付与と店癖の適用 ---
    shop_col = '店名' if '店名' in train_df.columns else ('店舗名' if '店舗名' in train_df.columns else None)
    if shop_col and not train_df.empty:
        def _add_shop_status(df_target, is_actual=False):
            if df_target.empty: return df_target
            diff_col = 'next_diff' if is_actual and 'next_diff' in df_target.columns else '予測差枚数'
            if diff_col not in df_target.columns:
                df_target['営業状態'] = "⚖️ 通常営業"
                return df_target
                
            shop_daily = df_target.groupby([shop_col, 'next_date']).agg(
                avg_diff=(diff_col, 'mean'),
                count=('台番号', 'nunique')
            ).reset_index()
            
            shop_daily['営業状態'] = shop_daily.apply(lambda r: classify_shop_eval(r['avg_diff'], r['count'], is_prediction=False), axis=1)
            return pd.merge(df_target, shop_daily[[shop_col, 'next_date', '営業状態']], on=[shop_col, 'next_date'], how='left')

        train_df = _add_shop_status(train_df, is_actual=True)
        if not predict_df.empty:
            predict_df = _add_shop_status(predict_df, is_actual=False)

        all_trends_dict = calculate_shop_trends(train_df, shop_col, specs)
        if not predict_df.empty:
            predict_df = predict_df.apply(lambda row: apply_trends_to_row(row, all_trends_dict, shop_col, specs), axis=1)
        if not train_df.empty:
            train_df = train_df.apply(lambda row: apply_trends_to_row(row, all_trends_dict, shop_col, specs), axis=1)

    # --- ダメ台ペナルティとメリハリ補正の適用 ---
    def apply_hopeless_penalty(row):
        score = row.get('prediction_score', 0)
        games = row.get('累計ゲーム', 0)
        reg_prob = row.get('REG確率', 0)
        diff = row.get('差枚', 0)
        win_rate_7d = row.get('win_rate_7days', 0)
        
        penalty_factor = 1.0
        reasons = []

        if games >= 3000 and reg_prob > 0 and (1.0 / reg_prob) >= 400 and diff < 0:
            penalty_factor *= 0.60
            reasons.append("【🔻大幅減点】前日しっかり回された上でREGが絶望的(1/400以下)かつマイナスです。低設定の放置台の可能性が高く、危険です。")
            
        if diff >= 2000 and reg_prob > 0 and (1.0 / reg_prob) >= 350:
            penalty_factor *= 0.60
            reasons.append("【🔻大幅減点】前日大勝していますがREG確率が悪く、低設定のまぐれ吹きの可能性が高いです。本日の反動(回収)に警戒してください。")
            
        if win_rate_7d == 0 and diff < 0 and games >= 1000:
            penalty_factor *= 0.70
            reasons.append("【🔻減点】過去1週間で高設定挙動がなく、店側が全く設定を入れていない(見捨てられている)可能性が高いです。")
            
        if games < 1000:
            is_good_reg = reg_prob >= (1.0 / 280.0) if reg_prob > 0 else False
            if is_good_reg:
                reasons.append("【💎期待】前日の回転数は少ないですが、高設定挙動を示しているためポテンシャルを評価しています。")
            else:
                reasons.append("【🔻減点】前日の総回転数が極端に少なく、データ不足のため期待度を少し割り引いています。")

        score *= penalty_factor
        
        if reasons:
            existing_reason = str(row.get('根拠', ''))
            if existing_reason == 'nan': existing_reason = ''
            new_reason = " ".join(reasons)
            if existing_reason and existing_reason != '-':
                row['根拠'] = (existing_reason + " " + new_reason).strip()
            else:
                row['根拠'] = new_reason
                
        row['prediction_score'] = score
        return row

    if not predict_df.empty:
        predict_df = predict_df.apply(apply_hopeless_penalty, axis=1)
    if not train_df.empty:
        train_df = train_df.apply(apply_hopeless_penalty, axis=1)

    # --- 回収日の「完全ベタピン店」判定の事前計算 ---
    shop_col_for_betapin = '店名' if '店名' in train_df.columns else ('店舗名' if '店舗名' in train_df.columns else None)
    betapin_shops = set()
    if shop_col_for_betapin and not train_df.empty:
        if '予測差枚数' in train_df.columns:
            shop_daily = train_df.groupby([shop_col_for_betapin, 'next_date']).agg(予測差枚数=('予測差枚数', 'mean'), 台数=('台番号', 'nunique')).reset_index()
            shop_daily['営業予測'] = shop_daily.apply(lambda r: classify_shop_eval(r['予測差枚数'], r['台数'], is_prediction=True), axis=1)
            cold_days_dates = shop_daily[shop_daily['営業予測'] == "🥶 回収日予測"]
            cold_days = pd.merge(train_df, cold_days_dates[[shop_col_for_betapin, 'next_date']], on=[shop_col_for_betapin, 'next_date'])
            if not cold_days.empty:
                cold_stats = cold_days.groupby(shop_col_for_betapin).agg(高設定率=('target', 'mean'), サンプル数=('target', 'count')).reset_index()
                betapin_shops = set(cold_stats[(cold_stats['サンプル数'] >= 100) & (cold_stats['高設定率'] < 0.015)][shop_col_for_betapin])

    # --- 店舗全体の空気感による最終補正とメッセージ付与 ---
    def apply_shop_mood_correction(df_target):
        shop_col = '店名' if '店名' in df_target.columns else ('店舗名' if '店舗名' in df_target.columns else None)
        if df_target.empty or not shop_col: return df_target
            
        if '予測差枚数' in df_target.columns:
            df_target['temp_shop_diff'] = df_target.groupby([shop_col, 'next_date'])['予測差枚数'].transform('mean')
        else:
            df_target['temp_shop_diff'] = np.nan
        
        def _correct(row):
            score = row.get('prediction_score', 0)
            s_raw_avg = row.get('temp_shop_avg', 0.15)
            s_diff = row.get('temp_shop_diff', 0)
            s_name = row.get(shop_col)
            reasons = []
            
            if s_raw_avg < 0.10 and (pd.isna(s_diff) or s_diff < 0):
                is_betapin = s_name in betapin_shops
                if is_betapin:
                    reasons.append(f"【⚠️絶対回収】過去のデータから、この店舗は回収日に高設定をほぼ100%使わない「完全ベタピン」の傾向が確認されています。")
                else:
                    if score >= 0.30:
                        reasons.append(f"【💎一点突破】店舗全体は回収傾向ですが、AIはこの台に確かな高設定の根拠(見せ台)があると確信して強く推奨しています。")
                    else:
                        reasons.append("【🚨フェイク警戒】店舗全体が回収傾向です。普段なら強い根拠がある台ですが、今日はフェイク（罠）として使われるリスクが高いため慎重に判断してください。")

            if reasons:
                existing_reason = str(row.get('根拠', ''))
                if existing_reason == 'nan': existing_reason = ''
                new_reason = " ".join(reasons)
                row['根拠'] = (existing_reason + " " + new_reason).strip() if existing_reason and existing_reason != '-' else new_reason
                
            return row
            
        res_df = df_target.apply(_correct, axis=1)
        res_df = res_df.drop(columns=['temp_shop_avg', 'temp_shop_diff'], errors='ignore')
        return res_df

    predict_df = apply_shop_mood_correction(predict_df)
    train_df = apply_shop_mood_correction(train_df)

    # --- 予測スコアから「前日の自己評価スコア(past_prediction_score)」を作成 ---
    predict_df['_orig_index'] = predict_df.index
    train_df['_orig_index'] = train_df.index
    
    predict_df['_uid'] = range(len(predict_df))
    train_df['_uid'] = range(len(train_df))
    
    predict_df['_is_predict'] = True
    train_df['_is_predict'] = False
    
    all_df = pd.concat([train_df, predict_df], ignore_index=True)
    if shop_col:
        all_df = all_df.sort_values([shop_col, '台番号', '対象日付']).reset_index(drop=True)
        all_df['past_prediction_score'] = all_df.groupby([shop_col, '台番号'])['prediction_score'].shift(1).fillna(0.0)
    else:
        all_df = all_df.sort_values(['台番号', '対象日付']).reset_index(drop=True)
        all_df['past_prediction_score'] = all_df.groupby('台番号')['prediction_score'].shift(1).fillna(0.0)
        
    train_df = all_df[all_df['_is_predict'] == False].sort_values('_uid').set_index('_orig_index').drop(columns=['_is_predict', '_uid'])
    predict_df = all_df[all_df['_is_predict'] == True].sort_values('_uid').set_index('_orig_index').drop(columns=['_is_predict', '_uid'])

    train_df.index.name = None
    predict_df.index.name = None

    def get_rating(score):
        if score >= 0.70: return 'A'
        elif score >= 0.55: return 'B'
        elif score >= 0.40: return 'C'
        elif score >= 0.25: return 'D'
        else: return 'E'

    def get_reason(row):
        comments, reasons = [], []
        score = row.get('prediction_score', 0)
        if score > 0.50: comments.append("【激アツ】AIの自信度が非常に高い(生確率で50%超え)です。")
        
        past_score = row.get('past_prediction_score', 0)
        diff = row.get('差枚', 0)
        
        if past_score >= 0.30:
            if diff <= -500:
                reasons.append(f"【AIリベンジ狙い】前日もAIが推奨(期待度{past_score*100:.0f}%)していましたが不発でした。高設定据え置きのリベンジが期待できます。")
            elif diff > 0:
                reasons.append(f"【AI推奨継続】前日もAIが推奨(期待度{past_score*100:.0f}%)しており、好調のまま今日も強い根拠を維持しています。")
        
        mean_7d = row.get('mean_7days_diff', 0)
        win_rate_7d = row.get('win_rate_7days', 0)
        reg_prob = row.get('REG確率', 0)
        is_win_flag = row.get('is_win', 0)
        games = row.get('累計ゲーム', 0)

        if mean_7d < -300:
            if diff <= -1000:
                if games >= 7000:
                    reasons.append(f"直近1週間(平均{int(mean_7d)}枚)不調な上、前日は{int(games)}Gもタコ粘りされて大凹みしており、強烈な**「お詫び(反発)」**が期待できます。")
                else:
                    reasons.append(f"直近1週間(平均{int(mean_7d)}枚)と前日が大きく凹んでおり、**「不調台の反発」**の可能性が高いです。")
            else: reasons.append(f"週間成績は不調(平均{int(mean_7d)}枚)ですが、AIは**「底打ち上昇」**を予測しています。")
        elif mean_7d > 500:
            if win_rate_7d >= 0.5 and is_win_flag == 1:
                reasons.append(f"直近1週間(平均+{int(mean_7d)}枚, 高設定率{win_rate_7d*100:.0f}%)と好調かつ、REG確率(1/{int(1/reg_prob) if reg_prob > 0 else '-'})も優秀で、**「高設定の据え置き」**が期待できます。")
            elif win_rate_7d >= 0.5:
                reasons.append(f"直近1週間(平均+{int(mean_7d)}枚, 高設定率{win_rate_7d*100:.0f}%)と安定して高設定が使われています。")
            elif diff >= 2000:
                reasons.append(f"週間平均はプラスですが、直近の一撃(+{int(diff)}枚)による影響が大きいです。一撃後の回収に警戒が必要です。")
        
        prev2_diff = row.get('prev_差枚')
        if pd.notna(prev2_diff):
            if prev2_diff <= -1000 and diff <= -1000:
                reasons.append("【波・推移】2日連続の大凹み(-1000枚以下)で、グラフ底からの強烈な反発(底上げ)サインが点灯しています。")
            elif prev2_diff < 0 and diff >= 0:
                reasons.append("【波・推移】前々日のマイナスから前日プラスへV字反発しており、右肩上がりの好調ウェーブ続伸に期待できます。")
            elif prev2_diff >= 1000 and diff >= 1000:
                if reg_prob >= (1/300):
                    reasons.append("【波・推移】2日連続の大勝(+1000枚以上)かつREG確率も優秀です。高設定の据え置きによる綺麗な右肩上がりグラフに期待できます。")
                else:
                    reasons.append("【波・推移】2日連続の大勝(+1000枚以上)ですが、REG確率が伴っていません。一撃の波が終わる可能性があり警戒が必要です。")

        cons_minus = row.get('連続マイナス日数', 0)
        if cons_minus >= 3:
            reasons.append(f"【特殊】現在{int(cons_minus)}日連続マイナス中です。店舗の「上げリセット(底上げ)」ターゲットになる可能性が高いです。")

        cons_low_util = row.get('連続低稼働日数', 0)
        if cons_low_util >= 3:
            reasons.append(f"【特殊】現在{int(cons_low_util)}日連続で放置(1500G未満)されています。店側の「稼働喚起のテコ入れ(見せ台)」のターゲットになる可能性があります。")

        if row.get('is_prev_no_play', 0) == 1:
            reasons.append("【特殊】前日は全く稼働しておらず(0G)、店側のアピール(設定変更・テコ入れ)のターゲットになる可能性があります。")

        shop_7d = row.get('shop_7days_avg_diff', 0)
        if shop_7d < -150:
            reasons.append(f"【店舗状況】店舗全体が直近1週間回収モード(平均{int(shop_7d)}枚)ですが、あえてこの台を推奨しています。")
        elif shop_7d > 150:
            reasons.append(f"【店舗状況】店舗全体が直近1週間還元モード(平均+{int(shop_7d)}枚)で、全体のベースアップに期待できます。")
            
        machine_name = row.get('機種名', '')
        matched_spec_key = get_matched_spec_key(machine_name, specs)

        prev_grape_raw = row.get('prev_推定ぶどう確率_raw', row.get('prev_推定ぶどう確率'))
        prev_games = row.get('prev_累計ゲーム', 0)
        if pd.notna(prev_grape_raw) and prev_grape_raw > 0 and prev_games >= 4000:
            spec_grape_5 = specs[matched_spec_key].get('設定5', {}).get('ぶどう', 5.9)
            if pd.notna(row.get('prev_推定ぶどう確率')) and prev_grape_raw <= spec_grape_5:
                reasons.append(f"【🍇小役優秀】前日は{int(prev_games)}G稼働で推定ぶどう確率が1/{prev_grape_raw:.2f}と優秀です。REG確率・差枚も伴っており、高設定の強い裏付け（裏取り）となっています。")

        mac_30d = row.get('machine_30days_avg_diff', 0)
        if mac_30d > 150:
            reasons.append(f"【機種優遇】過去30日間、この機種(平均+{int(mac_30d)}枚)は店舗から甘く使われている傾向があります。")
        elif mac_30d < -150:
            reasons.append(f"【機種冷遇】過去30日間、この機種(平均{int(mac_30d)}枚)は冷遇気味ですが、この台単体は評価されています。")

        if row.get('is_new_machine', 0) == 1: reasons.append("【新台】新台導入から1週間以内のため、店側のアピール(高設定投入)が期待できます。")
        if row.get('is_moved_machine', 0) == 1: reasons.append("【配置変更】配置変更(移動)から1週間以内のため、扱いが変化している可能性があります。")
        if row.get('is_low_play_high_reg', 0) == 1: reasons.append("【💎お宝台候補】前日はあまり回されていませんが、高設定挙動を示しており、そのまま据え置かれる(隠れ高設定)ポテンシャルがあります。")
        if row.get('is_hot_wd_and_heavy_lose', 0) == 1: reasons.append("【📈還元曜日×凹み反発】この店舗の強い曜日(還元日)と、直近1週間大きく凹んでいる条件が重なっており、絶好の上げリセット狙い目です。")
        if row.get('is_prev_up_trend_and_high_reg', 0) == 1: reasons.append("【📈右肩上がり据え置き】前日は差枚がプラスでREG確率も高設定水準です。優秀台の完全な据え置きが期待できます。")
        if row.get('is_prev_low_reg_and_good_diff', 0) == 1: reasons.append("【⚠️低REG・差枚プラス】前日は差枚がプラスですがREG確率が伴っていません。低設定の誤爆による回収リスクと、据え置きの両面をAIが評価しています。")

        big = row.get('BIG', 0)
        reg = row.get('REG', 0)
        is_setting5_over = False
        if matched_spec_key and "設定5" in specs[matched_spec_key] and reg_prob > 0:
            set5_reg_prob_threshold = 1.0 / specs[matched_spec_key]["設定5"]["REG"]
            set1_reg_prob_threshold = 1.0 / specs[matched_spec_key].get("設定1", {"REG": 400.0})["REG"]
            
            exp_r1 = games * set1_reg_prob_threshold
            std_r1 = math.sqrt(games * set1_reg_prob_threshold * (1.0 - set1_reg_prob_threshold))
            z_score = (reg - exp_r1) / std_r1 if std_r1 > 0 else 0
            
            if (games >= 5000 and reg_prob >= set5_reg_prob_threshold) or z_score >= 1.64:
                is_setting5_over = True

        spec_reg_5 = 1.0 / specs[matched_spec_key].get("設定5", {"REG": 260.0})["REG"] if matched_spec_key else 1/260.0
        spec_big_5 = 1.0 / specs[matched_spec_key].get("設定5", {"BIG": 260.0})["BIG"] if matched_spec_key else 1/260.0
        big_prob = big / games if games > 0 else 0
        big_denom = 1 / big_prob if big_prob > 0 else 9999

        if is_setting5_over:
            reasons.append(f"【🌟高設定挙動】5000G以上回って前日のREG確率が1/{int(1/reg_prob)}で、機種スペックの**「設定5以上」**の基準を満たしており、強く推奨されます。")
        elif reg > big and reg_prob >= spec_reg_5 and games >= 5000:            
            if big_denom >= 400:
                if diff <= 0: reasons.append(f"【超不発】5000G以上回ってBIG確率が1/{int(big_denom)}と極端に欠損していますが、REG確率は設定5以上(1/{int(1/reg_prob)})をキープしている超・狙い目台です。")
                else: reasons.append(f"【特殊】5000G以上回ってBIGが極端に引けていませんが(1/{int(big_denom)})、REG確率は設定5以上(1/{int(1/reg_prob)})の高設定挙動です。")
            else:
                if diff <= 0: reasons.append(f"【特殊】5000G以上回ってREG先行(BB欠損)で差枚が沈んでいる、狙い目の「高設定 不発台」です。(REG 1/{int(1/reg_prob)})")
                else: reasons.append(f"【特殊】5000G以上回ってREG先行かつREG確率が設定5以上(1/{int(1/reg_prob)})の「高設定台」です。")
        elif big >= reg and big_prob >= spec_big_5 and games >= 5000:
            reasons.append(f"【特殊】5000G以上回ってBIG先行(1/{int(big_denom)})でBIG確率が設定5以上をキープしています。BIGヒキ強台の据え置き狙いとして期待できます。")
        else:
            if reg_prob > (1/280) and games >= 5000: reasons.append(f"5000G以上回って前日のREG確率が**1/{int(1/reg_prob)}**と高設定水準です。")
            elif reg_prob > (1/350) and games >= 5000: reasons.append(f"5000G以上回ってREG確率(1/{int(1/reg_prob)})が悪くなく、粘る価値があります。")
            elif reg_prob > (1/280) and games >= 3000: reasons.append(f"3000G以上回って前日のREG確率が**1/{int(1/reg_prob)}**と高設定水準です。")
            elif reg_prob > (1/350) and games >= 3000: reasons.append(f"3000G以上回ってREG確率(1/{int(1/reg_prob)})が悪くなく、粘る価値があります。")
        
        e_avg = row.get('event_avg_diff', 0)
        if e_avg > 150: reasons.append(f"今日はイベント特定日(平均+{int(e_avg)}枚)のため期待値が高いです。")

        evt_name = row.get('イベント名', '通常')
        evt_score = row.get('event_rank_score', 0)
        if evt_name != '通常' and pd.notna(evt_name):
            evt_rank = row.get('イベントランク', '')
            rank_str = f"(ランク{evt_rank})" if evt_rank else ""
            if evt_score <= -5: reasons.append(f"【🚨極悪回収警戒】本日は複合イベント「{evt_name}」ですが、他機種やパチンコへの還元によるシワ寄せで、極めて強い回収(期待度スコア: {evt_score})が予測されます。")
            elif evt_score < 0: reasons.append(f"【⚠️回収警戒】本日は複合イベント「{evt_name}」ですが、対象外機種であるため回収に回される危険性が高いです。")
            elif evt_score > 0: reasons.append(f"店舗イベント「{evt_name}」{rank_str}対象日です(複合スコア: {evt_score})。")

        prev_evt_score = row.get('prev_event_rank_score', 0)
        if evt_score <= 0 and prev_evt_score > 0 and score >= 0.60: reasons.append("【特日翌日】前日は特日でしたが、AIは本日の「据え置き」または「入れ直し」を有力視しています。")

        w_avg = row.get('weekday_avg_diff', 0)
        if w_avg > 150:
            wd_name = ['月', '火', '水', '木', '金', '土', '日'][int(row['target_weekday'])] if 'target_weekday' in row and 0 <= row['target_weekday'] <= 6 else ''
            reasons.append(f"{wd_name}曜日はこの店の得意日(平均+{int(w_avg)}枚)です。")

        ev_label = f"イベント「{evt_name}」" if evt_name != '通常' and pd.notna(evt_name) else "通常営業日"
        evt_mac_avg = row.get('event_x_machine_avg_diff', 0)
        if evt_mac_avg > 200: reasons.append(f"【特効機種】過去の{ev_label}において、この機種は非常に甘く使われています(平均+{int(evt_mac_avg)}枚)。")
        elif evt_mac_avg < -300: reasons.append(f"【警戒機種】過去の{ev_label}において、この機種は回収傾向(平均{int(evt_mac_avg)}枚)ですが、AIはこの台単体を評価しています。")
            
        evt_end_avg = row.get('event_x_end_digit_avg_diff', 0)
        if evt_end_avg > 200:
            end_digit = int(row.get('末尾番号', -1))
            if end_digit != -1: reasons.append(f"【当たり末尾】過去の{ev_label}において、末尾『{end_digit}』は対象になりやすい強い傾向があります(平均+{int(evt_end_avg)}枚)。")

        if row.get('is_corner', 0) == 1: reasons.append("角台（設定優遇枠）のため期待大です。")
        
        neighbor_high_count = row.get('neighbor_high_setting_count', 0)
        if neighbor_high_count > 0: reasons.append(f"【🤝並び・塊】隣台のうち{int(neighbor_high_count)}台が高設定挙動を示しており、並びの対象になっている可能性が高いです。")
        elif row.get('is_neighbor_high_reg', 0) == 1: reasons.append("【🤝並び・塊】隣台の合算REG確率が高設定水準であり、強い「並び」の根拠となっています。")
            
        i_avg = row.get('island_avg_diff', 0)
        if i_avg > 400: reasons.append(f"所属する島全体が好調(平均+{int(i_avg)}枚)で、塊対象の可能性があります。")
        
        shop_reason = str(row.get('根拠', '')).strip()
        if shop_reason == 'nan': shop_reason = ''
        has_strong_reason = bool(reasons) or (shop_reason and shop_reason != '-')
        
        if reasons: comments.append(" ".join(reasons))
            
        if not has_strong_reason:
            if score >= 0.40: comments.append("目立った特徴（特定の店癖など）には合致していませんが、様々なデータの全体バランスからAIが非常に高く評価しています。")
            elif score >= 0.20: comments.append("目立った強い根拠はありませんが、複数の細かな要因（周りの状況や過去のわずかな傾向）の組み合わせにより、消去法的に期待度が底上げされています。")
            else: comments.append("特筆すべき強い根拠はありません。")
            
        base_reason = " ".join(comments)
        if shop_reason and shop_reason != '-':
            return f"{base_reason} {shop_reason}".strip()
        return base_reason

    if not predict_df.empty:
        predict_df['根拠'] = predict_df.apply(get_reason, axis=1)
        predict_df['おすすめ度'] = predict_df['prediction_score'].apply(get_rating)
        if '店名' in predict_df.columns:
            shop_mean = predict_df.groupby('店名')['prediction_score'].transform('mean')
            predict_df['店舗期待度'] = shop_mean.apply(get_rating)
        
    if not train_df.empty:
        train_df['根拠'] = train_df.apply(get_reason, axis=1)
        train_df['おすすめ度'] = train_df['prediction_score'].apply(get_rating)

    def highlight_reasons(text):
        if not isinstance(text, str): return text
        text = re.sub(r'(【[^】]+】)', r'**\1**', text)
        text = text.replace('**【🎯激熱】**', '**:red[【🎯激熱】]**')
        text = text.replace('**【激アツ】**', '**:red[【激アツ】]**')
        text = text.replace('**【🚨極悪回収警戒】**', '**:blue[【🚨極悪回収警戒】]**')
        text = text.replace('**【⚠️絶対回収】**', '**:blue[【⚠️絶対回収】]**')
        text = text.replace('**【🔻大幅減点】**', '**:blue[【🔻大幅減点】]**')
        text = text.replace('**【💎お宝台候補】**', '**:orange[【💎お宝台候補】]**')
        text = text.replace('**【💎一点突破】**', '**:orange[【💎一点突破】]**')
        text = text.replace('**【🌟高設定挙動】**', '**:orange[【🌟高設定挙動】]**')
        text = text.replace('**【超不発】**', '**:orange[【超不発】]**')
        text = text.replace('**【波・推移】**', '**:green[【波・推移】]**')
        text = text.replace('**【AIリベンジ狙い】**', '**:orange[【AIリベンジ狙い】]**')
        return text

    if not predict_df.empty and '根拠' in predict_df.columns:
        predict_df['根拠'] = predict_df['根拠'].apply(highlight_reasons)
    if not train_df.empty and '根拠' in train_df.columns:
        train_df['根拠'] = train_df['根拠'].apply(highlight_reasons)

    return predict_df, train_df