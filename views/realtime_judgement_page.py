import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import altair as alt # type: ignore
import math
import datetime
import backend

def render_realtime_judgement_page(df_pred_log):
    st.header("⏱️ リアルタイム設定判別")
    st.caption("ホールでの実践中、目の前の台の現在のデータから設定を推測します。事前のAI期待度と組み合わせることで、精度の高い「押し引き」のジャッジが可能です。")

    specs = backend.get_machine_specs()
    machine_list = [k for k in specs.keys() if k != "ジャグラー（デフォルト）"] + ["ジャグラー（デフォルト）"]
    
    # --- 入力セクション ---
    with st.container():
        st.subheader("📝 データ入力")
        
        # AI連携オプション
        use_ai = st.checkbox("🤖 AIの事前予測（期待度）と連携する", value=True)
        
        prior_high_prob = 0.5 
        selected_machine = machine_list[0]
        
        if use_ai:
            st.info("本日の予測データから、対象の台の「事前期待度」を自動取得します。")
            today_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d')
            
            if not df_pred_log.empty:
                df_today = df_pred_log.copy()
                if '予測対象日' in df_today.columns:
                    df_today['予測対象日_str'] = pd.to_datetime(df_today['予測対象日'], errors='coerce').dt.strftime('%Y-%m-%d')
                    df_today = df_today[df_today['予測対象日_str'] == today_str]
                else:
                    df_today = pd.DataFrame()
            else:
                df_today = pd.DataFrame()

            if not df_today.empty:
                shop_col = '店名' if '店名' in df_today.columns else '店舗名'
                shops = sorted(df_today[shop_col].dropna().unique())
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    sel_shop = st.selectbox("店舗を選択", [""] + shops)
                with col_s2:
                    if sel_shop:
                        df_shop = df_today[df_today[shop_col] == sel_shop]
                        mac_list = sorted(df_shop['台番号'].astype(str).unique())
                        sel_mac = st.selectbox("AI推奨台の台番号を選択", [""] + mac_list)
                    else:
                        sel_mac = st.selectbox("AI推奨台の台番号を選択", [""])
                
                if sel_shop and sel_mac:
                    target_row = df_today[(df_today[shop_col] == sel_shop) & (df_today['台番号'].astype(str) == str(sel_mac))].iloc[0]
                    prior_high_prob = float(target_row.get('prediction_score', 0.5))
                    mac_name = target_row.get('機種名', '')
                    matched = backend.get_matched_spec_key(mac_name, specs)
                    selected_machine = matched if matched else machine_list[0]
                        
                    st.success(f"✅ AI事前期待度 **{prior_high_prob*100:.1f}%** をセットしました！ ({selected_machine})")
                else:
                    st.warning("店舗と台番号を選択してください。リストにない場合はAI推奨台外（または手入力）になります。")
                    prior_high_prob = st.slider("手動で事前期待度(設定4,5,6の確率)を設定", 0.0, 1.0, 0.1, 0.05, format="%.2f")
                    selected_machine = st.selectbox("機種を選択", machine_list)
            else:
                st.warning("本日のAI予測ログが見つかりません。手動で設定してください。")
                prior_high_prob = st.slider("手動で事前期待度(設定4,5,6の確率)を設定", 0.0, 1.0, 0.1, 0.05, format="%.2f")
                selected_machine = st.selectbox("機種を選択", machine_list)
        else:
            selected_machine = st.selectbox("機種を選択", machine_list)
            prior_high_prob = st.slider("事前期待度（高設定が入っているベースの確率）", 0.0, 1.0, 0.15, 0.01, format="%.2f", help="イベントの強さや店長のクセを加味した、打ち始める前の期待度です。通常営業なら10%〜15%程度がリアルです。")

        col1, col2 = st.columns(2)
        with col1:
            g_count = st.number_input("現在の総回転数 (G)", min_value=0, value=3000, step=100)
            diff_coins = st.number_input("現在の差枚数 (枚) ※任意", value=0, step=100, help="ぶどう確率の逆算に使用します。不明な場合は0のままでOKです。")
        with col2:
            b_count = st.number_input("BIG回数", min_value=0, value=10, step=1)
            r_count = st.number_input("REG回数", min_value=0, value=10, step=1)
            
        st.divider()
        st.subheader("⏱️ 時間・期待値設定")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            current_time = st.time_input("現在時刻", pd.Timestamp.now(tz='Asia/Tokyo').time())
        with col_t2:
            close_time = st.time_input("閉店時刻", pd.to_datetime("22:45").time())
        with col_t3:
            speed = st.number_input("1時間の回転数 (G/h)", min_value=100, max_value=1000, value=750, step=50, help="一般的な速度は700〜800G/hです。")
            
    if g_count == 0:
        st.info("総回転数を入力してください。")
        return

    # --- 残りゲーム数の計算 ---
    curr_dt = datetime.datetime.combine(datetime.date.today(), current_time)
    close_dt = datetime.datetime.combine(datetime.date.today(), close_time)
    
    if close_time.hour < 12 and current_time.hour >= 12:
        close_dt += datetime.timedelta(days=1)
        
    if curr_dt < close_dt:
        remain_minutes = (close_dt - curr_dt).total_seconds() / 60.0
    else:
        remain_minutes = 0
        
    remain_games = int((remain_minutes / 60.0) * speed)
    if remain_games < 0: remain_games = 0

    # --- 推論計算 ---
    ms = specs[selected_machine]
    s1 = ms.get("設定1", {"BIG": 280.0, "REG": 400.0, "ぶどう": 6.0})
    s4 = ms.get("設定4", {"BIG": 260.0, "REG": 300.0, "ぶどう": 5.9})
    s5 = ms.get("設定5", s4)
    s6 = ms.get("設定6", s5)
    
    full_specs = {1: s1, 4: s4, 5: s5, 6: s6}
    for s in [2, 3]:
        full_specs[s] = {}
        for k in ["BIG", "REG", "ぶどう"]:
            p1 = 1.0 / s1.get(k, 300.0)
            p4 = 1.0 / s4.get(k, 300.0)
            p_s = p1 + (p4 - p1) * (s - 1) / 3.0
            full_specs[s][k] = 1.0 / p_s if p_s > 0 else 999.0
            
    use_grape = False
    grape_count = 0
    if diff_coins != 0 and g_count >= 1000:
        use_grape = True
        in_tokens = g_count * 3
        out_tokens = in_tokens + diff_coins
        bonus_out = b_count * ms.get('BIG獲得', 252) + r_count * ms.get('REG獲得', 96)
        other_out = g_count * 0.4900
        grape_out = out_tokens - bonus_out - other_out
        grape_count = max(0, grape_out / ms.get('ぶどう獲得', 7))
    
    # 事前確率の分配
    prior_probs = []
    for i in range(1, 7):
        if i >= 4: prior_probs.append(prior_high_prob / 3.0)
        else: prior_probs.append((1.0 - prior_high_prob) / 3.0)
            
    # 尤度計算
    log_likelihoods = []
    for i in range(1, 7):
        p_b, p_r = 1.0 / full_specs[i]["BIG"], 1.0 / full_specs[i]["REG"]
        exp_b, exp_r = g_count * p_b, g_count * p_r
        ll_b = b_count * math.log(exp_b) - exp_b if exp_b > 0 else 0
        ll_r = r_count * math.log(exp_r) - exp_r if exp_r > 0 else 0
        ll_g = 0
        if use_grape:
            p_g = 1.0 / full_specs[i]["ぶどう"]
            exp_g = g_count * p_g
            ll_g = grape_count * math.log(exp_g) - exp_g if exp_g > 0 else 0
        log_likelihoods.append(ll_b + ll_r + ll_g)
        
    # ベイズ推論 (事後確率)
    log_posteriors = [log_likelihoods[i] + math.log(prior_probs[i]) if prior_probs[i] > 0 else -float('inf') for i in range(6)]
    max_log_post = max(log_posteriors)
    posteriors_unnormalized = [math.exp(lp - max_log_post) for lp in log_posteriors]
    sum_post = sum(posteriors_unnormalized)
    posteriors = [p / sum_post for p in posteriors_unnormalized]
    
    # --- 各設定の1Gあたり期待差枚の計算 ---
    expected_diff_per_game = []
    big_out_val = ms.get('BIG獲得', 252)
    reg_out_val = ms.get('REG獲得', 96)
    grape_out_val = ms.get('ぶどう獲得', 7)
    
    for i in range(1, 7):
        p_b = 1.0 / full_specs[i]["BIG"]
        p_r = 1.0 / full_specs[i]["REG"]
        p_g = 1.0 / full_specs[i]["ぶどう"]
        
        out_b = p_b * big_out_val
        out_r = p_r * reg_out_val
        out_g = p_g * grape_out_val
        out_others = 0.485 # リプレイ・チェリー・ベルピエロ等の概算OUT
        
        total_out = out_b + out_r + out_g + out_others
        diff_per_game = total_out - 3.0
        expected_diff_per_game.append(diff_per_game)
        
    total_expected_diff = sum([posteriors[i] * expected_diff_per_game[i] * remain_games for i in range(6)])
    
    # --- 結果表示 ---
    st.divider()
    st.subheader("📊 判別結果")
    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        labels, colors = ['設定1', '設定2', '設定3', '設定4', '設定5', '設定6'], ['#cfd8dc', '#b0bec5', '#90a4ae', '#fff59d', '#ffcc80', '#ffab91']
        pie_df = pd.DataFrame({'設定': labels, '事後確率': posteriors, '色': colors})
        pie_df = pie_df[pie_df['事後確率'] > 0.001]
        pie_chart = alt.Chart(pie_df).mark_arc(innerRadius=40).encode(theta=alt.Theta(field="事後確率", type="quantitative"), color=alt.Color(field="設定", type="nominal", scale=alt.Scale(domain=labels, range=colors), legend=alt.Legend(title="設定")), tooltip=['設定', alt.Tooltip('事後確率', format='.1%')]).properties(height=300)
        st.altair_chart(pie_chart, use_container_width=True)
        
    with col_r2:
        high_prob, s56_prob = sum(posteriors[3:]), sum(posteriors[4:])
        st.metric("📈 高設定(4,5,6) 期待度", f"{high_prob*100:.1f}%")
        st.metric("🔥 設定5・6 期待度", f"{s56_prob*100:.1f}%")
        
        # 期待値の表示
        st.metric("💸 閉店までの期待収支", f"{int(total_expected_diff):+d} 枚", help=f"残り {int(remain_minutes)} 分 ({remain_games} G) として、各設定の確率と事後確率を加味して算出した理論上の期待差枚数です。")
        
        if use_grape:
            st.metric("🍇 推定ぶどう確率", f"1/{(g_count / grape_count if grape_count > 0 else 0):.2f}")
            
        st.markdown("### 💡 AIジャッジ")
        if remain_games <= 0:
            st.info("🕒 **【稼働終了】**\n閉店時間を過ぎているか、残り時間がありません。")
        elif total_expected_diff > 300:
            st.success("🟢 **【ガンガンいこうぜ】**\n期待値が十分にプラスです。閉店まで全ツッパを推奨します！")
        elif total_expected_diff > 0:
            st.info("🟡 **【様子見・続行】**\n期待値はプラス圏内です。次の判別ポイントまで続行を推奨します。")
        elif total_expected_diff > -150:
            st.warning("🟠 **【警戒】**\n期待値がマイナスに転じています。強い根拠がなければ撤退を視野に入れてください。")
        else:
            st.error("🔴 **【撤退推奨】**\n明らかなマイナス期待値です。傷が浅いうちにヤメることをおすすめします。")

    with st.expander("詳細な確率分布"):
        st.dataframe(pd.DataFrame({
            "設定": labels, 
            "事後確率": [f"{p*100:.1f}%" for p in posteriors], 
            "事前確率": [f"{p*100:.1f}%" for p in prior_probs], 
            "BIG確率": [f"1/{full_specs[i]['BIG']:.1f}" for i in range(1, 7)], 
            "REG確率": [f"1/{full_specs[i]['REG']:.1f}" for i in range(1, 7)],
            "機械割 (概算)": [f"{(expected_diff_per_game[i-1] + 3.0) / 3.0 * 100:.1f}%" for i in range(1, 7)],
            "期待差枚": [f"{int(expected_diff_per_game[i-1] * remain_games):+d}枚" for i in range(1, 7)]
        }), hide_index=True, width="stretch")