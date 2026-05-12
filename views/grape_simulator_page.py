import streamlit as st
import pandas as pd
import backend
import math
import altair as alt

def render_grape_simulator_page():
    st.header("🍇 ぶどう逆算シミュレーター")
    st.caption("現在のゲーム数、ボーナス回数、差枚数から、設定推測に役立つ「推定ぶどう確率」を逆算します。")

    specs = backend.get_machine_specs()
    # デフォルトスペックを除外してリスト化
    machine_list = [k for k in specs.keys() if k != "ジャグラー（デフォルト）"]
    default_idx = machine_list.index("マイジャグラーV") if "マイジャグラーV" in machine_list else 0

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 データ入力")
        selected_machine = st.selectbox("機種を選択", machine_list, index=default_idx)
        
        play_style = st.radio("打ち方 (小役取得率の目安)", ["チェリー狙い", "フル攻略", "適当押し"], horizontal=True, help="チェリー狙い: チェリーのみ取得\nフル攻略: ベル・ピエロ全取得\n適当押し: チェリーの一部取りこぼし")

        if play_style == "フル攻略":
            other_out_per_g = 0.4935
        elif play_style == "適当押し":
            other_out_per_g = 0.4412
        else: # チェリー狙い
            other_out_per_g = 0.4715
            
        use_1bet_loss = st.checkbox("ボーナス察知後の1枚掛け(ロス)を計算に含める", value=True, help="ボーナス1回につき1ゲームを1枚掛けとしてIN枚数を減算します。より厳密な逆算になります。")
        use_grape_nuki = st.checkbox("ブドウ抜きを考慮する (1枚掛け時のブドウ取得)", value=True, help="1枚掛け時にブドウ抜きを実践している場合、その獲得期待値分を考慮して精度を高めます。")

        games = st.number_input("総回転数 (G)", min_value=0, value=3000, step=100)
        
        c_b, c_r = st.columns(2)
        with c_b:
            big_count = st.number_input("BIG回数", min_value=0, value=10, step=1)
        with c_r:
            reg_count = st.number_input("REG回数", min_value=0, value=10, step=1)
            
        diff_coins = st.number_input("現在の差枚数 (枚)", value=0, step=100)

        ms = specs[selected_machine]
        big_out = ms.get('BIG獲得', 252)
        reg_out = ms.get('REG獲得', 96)
        grape_out_val = ms.get('ぶどう獲得', 7)

        st.markdown("---")
        st.subheader("💡 逆算結果")
        
        if games > 0:
            total_bonus = big_count + reg_count
            
            # 1枚掛けゲーム数の推定
            estimated_1bet_games = total_bonus if use_1bet_loss else 0
            
            # IN枚数の厳密な計算
            in_tokens = ((games - estimated_1bet_games) * 3) + (estimated_1bet_games * 1)
            out_tokens = in_tokens + diff_coins
            
            bonus_out = (big_count * big_out) + (reg_count * reg_out)
            
            # 1枚掛け時のブドウ抜きによる平均払出を加味
            if use_grape_nuki and estimated_1bet_games > 0:
                # 1枚掛け時のブドウ成立確率を仮に1/28とした場合の期待値
                grape_nuki_expected_out = estimated_1bet_games * (1/28.0) * grape_out_val
                bonus_out += grape_nuki_expected_out
                
            # 打ち方による小役（リプレイ・チェリー・ベル・ピエロ）の概算OUT
            other_out = (games - estimated_1bet_games) * other_out_per_g 
            
            grape_out = out_tokens - bonus_out - other_out
            grape_count = max(0, grape_out / grape_out_val)
            
            grape_prob = games / grape_count if grape_count > 0 else 0
            
            if grape_prob > 0:
                c_res1, c_res2, c_res3, c_res4 = st.columns(4)
                with c_res1:
                    st.metric("推定ぶどう確率", f"1/{grape_prob:.2f}")
                with c_res2:
                    st.metric("推定ぶどう回数", f"{int(grape_count)} 回")
                with c_res3:
                    total_bonus = big_count + reg_count
                    b_prob_str = f"1/{games/total_bonus:.1f}" if total_bonus > 0 else "-"
                    st.metric("ボーナス合算", b_prob_str)
                with c_res4:
                    r_prob_str = f"1/{games/reg_count:.1f}" if reg_count > 0 else "-"
                    st.metric("REG確率", r_prob_str)
                    
                # スペックに対する簡単な判定メッセージ
                spec_g_6 = ms.get("設定6", {}).get("ぶどう", 5.8)
                spec_g_1 = ms.get("設定1", {}).get("ぶどう", 6.1)
                if grape_prob <= spec_g_6:
                    st.success("🌟 **設定6の基準を上回る、超優秀なぶどう確率です！**")
                elif grape_prob <= spec_g_1:
                    st.info("👍 **設定1〜設定5の間のぶどう確率です。**")
                else:
                    st.warning("⚠️ **設定1の基準を下回るぶどう確率です。**")
                    
                # --- ベイズ推論 (設定推測) ---
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

                log_likelihoods = []
                for i in range(1, 7):
                    p_b, p_r, p_g = 1.0 / full_specs[i]["BIG"], 1.0 / full_specs[i]["REG"], 1.0 / full_specs[i].get("ぶどう", 6.0)
                    exp_b, exp_r, exp_g = games * p_b, games * p_r, games * p_g
                    
                    ll_b = big_count * math.log(exp_b) - exp_b if exp_b > 0 else 0
                    ll_r = reg_count * math.log(exp_r) - exp_r if exp_r > 0 else 0
                    ll_g = grape_count * math.log(exp_g) - exp_g if exp_g > 0 else 0
                    
                    log_likelihoods.append(ll_b + ll_r + ll_g)

                max_log_post = max(log_likelihoods)
                posteriors_unnormalized = [math.exp(max(-700, lp - max_log_post)) for lp in log_likelihoods]
                sum_post = sum(posteriors_unnormalized)
                posteriors = [p / sum_post for p in posteriors_unnormalized]
                
                st.markdown("---")
                st.markdown("#### 🔍 設定推測結果 (ボーナス＋推定ぶどう)")
                st.caption("入力されたボーナス回数と、逆算された推定ぶどう回数を用いて、ベイズ推定により現在の各設定の可能性（事後確率）を算出します。")
                
                col_chart1, col_chart2 = st.columns([1, 1])
                with col_chart1:
                    labels, colors = ['設定1', '設定2', '設定3', '設定4', '設定5', '設定6'], ['#cfd8dc', '#b0bec5', '#90a4ae', '#fff59d', '#ffcc80', '#ffab91']
                    pie_df = pd.DataFrame({'設定': labels, '事後確率': posteriors, '色': colors})
                    pie_df = pie_df[pie_df['事後確率'] > 0.001]
                    pie_chart = alt.Chart(pie_df).mark_arc(innerRadius=40).encode(theta=alt.Theta(field="事後確率", type="quantitative"), color=alt.Color(field="設定", type="nominal", scale=alt.Scale(domain=labels, range=colors), legend=alt.Legend(title="設定")), tooltip=['設定', alt.Tooltip('事後確率', format='.1%')]).properties(height=250)
                    st.altair_chart(pie_chart, use_container_width=True)
                with col_chart2:
                    high_prob, s56_prob = sum(posteriors[3:]), sum(posteriors[4:])
                    st.metric("📈 高設定(4,5,6) 期待度", f"{high_prob*100:.1f}%")
                    st.metric("🔥 設定5・6 期待度", f"{s56_prob*100:.1f}%")
                    
                    st.dataframe(pd.DataFrame({
                        "設定": labels, 
                        "推測確率": [f"{p*100:.1f}%" for p in posteriors], 
                    }), hide_index=True, use_container_width=True)

            else:
                st.error("入力データからぶどう確率を計算できませんでした。（差枚が極端にマイナスすぎる等）")
        else:
            st.info("総回転数を入力してください。")

    with col2:
        st.subheader(f"📚 {selected_machine} スペック表")
        
        spec_data = []
        for s_key in ["設定1", "設定2", "設定3", "設定4", "設定5", "設定6"]:
            if s_key in ms:
                s_info = ms[s_key]
                
                # 機械割の概算計算
                payout = "-"
                if 'BIG' in s_info and 'REG' in s_info and 'ぶどう' in s_info:
                    out_b = big_out / s_info['BIG'] if s_info['BIG'] > 0 else 0
                    out_r = reg_out / s_info['REG'] if s_info['REG'] > 0 else 0
                    out_g = grape_out_val / s_info['ぶどう'] if s_info['ぶどう'] > 0 else 0
                    out_other = other_out_per_g  # リプレイ・チェリー等
                    total_out = out_b + out_r + out_g + out_other
                    payout_pct = (total_out / 3.0) * 100
                    payout = f"{payout_pct:.1f}%"

                spec_data.append({
                    "設定": s_key, 
                    "機械割": payout,
                    "BIG": f"1/{s_info.get('BIG', 0):.1f}", 
                    "REG": f"1/{s_info.get('REG', 0):.1f}", 
                    "合算": f"1/{s_info.get('合算', 0):.1f}", 
                    "ぶどう": f"1/{s_info.get('ぶどう', 0):.2f}" if 'ぶどう' in s_info else "-"
                })
        
        if spec_data:
            df_spec = pd.DataFrame(spec_data)
            st.dataframe(df_spec, hide_index=True, use_container_width=True, column_config={"設定": st.column_config.TextColumn("設定"), "機械割": st.column_config.TextColumn("機械割(概算)"), "BIG": st.column_config.TextColumn("BIG"), "REG": st.column_config.TextColumn("REG"), "合算": st.column_config.TextColumn("合算"), "ぶどう": st.column_config.TextColumn("ぶどう")})
            
        st.caption(f"※機械割およびぶどう確率の逆算において、ぶどう以外の小役（リプレイ、チェリー等）のOUT枚数は、選択された打ち方に基づく概算（約{other_out_per_g:.4f}枚/G）を使用しているため、実際の数値とは多少の誤差が生じる場合があります。")