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

    def big_counter(label, key, step=1, min_val=0, help_text=None):
        if key not in st.session_state:
            st.session_state[key] = min_val if min_val is not None else 0

        help_icon = f" <span title='{help_text}'>ℹ️</span>" if help_text else ""
        st.markdown(f"<div style='font-size: 0.9em; font-weight: bold; margin-bottom: 5px;'>{label}{help_icon}</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        
        def dec():
            st.session_state[key] = max(min_val, st.session_state[key] - step) if min_val is not None else st.session_state[key] - step
        def inc():
            st.session_state[key] += step

        c1.button("➖", key=f"dec_{key}", on_click=dec, use_container_width=True)
        c2.number_input(label, min_value=min_val, step=step, key=key, label_visibility="collapsed")
        c3.button("➕", key=f"inc_{key}", on_click=inc, use_container_width=True)
        
        return st.session_state[key]

    def clear_inputs():
        keys_to_clear = [
            "rt_use_ai", "rt_sel_shop", "rt_sel_mac", "rt_prior_high_prob", "rt_selected_machine",
            "rt_b_count", "rt_r_count", "rt_reg_hamari", "rt_g_input_mode", "rt_g_count",
            "rt_total_prob_den", "rt_start_g", "rt_grape_input_mode", "rt_diff_coins",
            "rt_manual_grape_count", "rt_peak_drop", "rt_use_gassan", "rt_gassan_type",
            "rt_gassan_count", "rt_input_mode", "rt_gassan_g_input_mode", "rt_gassan_b",
            "rt_gassan_r", "rt_gassan_g", "rt_gassan_total_prob_den",
            "rt_gassan_g_calc_mode_individual", "rt_gassan_total_prob_den_for_individual",
            "rt_close_time", "rt_speed"
        ]
        for k in list(st.session_state.keys()):
            if k.startswith("g_g_") or k.startswith("g_b_") or k.startswith("g_r_") or k in keys_to_clear:
                del st.session_state[k]

    specs = backend.get_machine_specs()
    machine_list = [k for k in specs.keys() if k != "ジャグラー（デフォルト）"] + ["ジャグラー（デフォルト）"]
    
    default_machine_idx = 0
    if "マイジャグラーV" in machine_list:
        default_machine_idx = machine_list.index("マイジャグラーV")
    
    # --- 入力セクション ---
    with st.container():
        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            st.subheader("📝 データ入力")
        with col_h2:
            if st.button("🗑️ 入力をクリア"):
                clear_inputs()
                st.rerun()
        
        # AI連携オプション
        use_ai = st.checkbox("🤖 AIの事前予測（期待度）と連携する", value=True, key="rt_use_ai")
        
        prior_high_prob = 0.5 
        selected_machine = machine_list[0]
        sel_shop = ""
        df_today = pd.DataFrame()
        shop_col = '店名'
        
        if use_ai:
            st.info("本日の予測データから、対象の台の「事前期待度」を自動取得します。")
            today_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d')
            
            if not df_pred_log.empty:
                df_today = df_pred_log.copy()
                if '予測対象日' in df_today.columns:
                    df_today['予測対象日_str'] = pd.to_datetime(df_today['予測対象日'], errors='coerce').dt.strftime('%Y-%m-%d')
                    df_today = df_today[df_today['予測対象日_str'] == today_str]
                else:
                    pass # 空のまま

            if not df_today.empty:
                shop_col = '店名' if '店名' in df_today.columns else '店舗名'
                shops = sorted(df_today[shop_col].dropna().unique())
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    sel_shop = st.selectbox("店舗を選択", [""] + shops, key="rt_sel_shop")
                with col_s2:
                    if sel_shop:
                        df_shop = df_today[df_today[shop_col] == sel_shop]
                        mac_list = sorted(df_shop['台番号'].astype(str).unique())
                        sel_mac = st.selectbox("AI推奨台の台番号を選択", [""] + mac_list, key="rt_sel_mac")
                    else:
                        sel_mac = st.selectbox("AI推奨台の台番号を選択", [""], key="rt_sel_mac")
                
                if sel_shop and sel_mac:
                    target_row = df_today[(df_today[shop_col] == sel_shop) & (df_today['台番号'].astype(str) == str(sel_mac))].iloc[0]
                    prior_high_prob = float(target_row.get('prediction_score', 0.5))
                    mac_name = target_row.get('機種名', '')
                    matched = backend.get_matched_spec_key(mac_name, specs)
                    selected_machine = matched if matched else machine_list[0]
                    reason = target_row.get('根拠', '')
                        
                    st.success(f"✅ AI事前期待度 **{prior_high_prob*100:.1f}%** をセットしました！ ({selected_machine})")
                    if reason and reason != '-':
                        st.info(f"💡 **AI推奨根拠 (店癖など)**: {reason}")
                else:
                    st.warning("店舗と台番号を選択してください。リストにない場合はAI推奨台外（または手入力）になります。")
                    prior_high_prob = st.slider("手動で事前期待度(設定4,5,6の確率)を設定", 0.0, 1.0, 0.1, 0.05, format="%.2f", key="rt_prior_high_prob")
                    selected_machine = st.selectbox("機種を選択", machine_list, index=default_machine_idx, key="rt_selected_machine")
            else:
                st.warning("本日のAI予測ログが見つかりません。手動で設定してください。")
                prior_high_prob = st.slider("手動で事前期待度(設定4,5,6の確率)を設定", 0.0, 1.0, 0.1, 0.05, format="%.2f", key="rt_prior_high_prob")
                selected_machine = st.selectbox("機種を選択", machine_list, index=default_machine_idx, key="rt_selected_machine")
        else:
            selected_machine = st.selectbox("機種を選択", machine_list, index=default_machine_idx, key="rt_selected_machine")
            prior_high_prob = st.slider("事前期待度（高設定が入っているベースの確率）", 0.0, 1.0, 0.15, 0.01, format="%.2f", help="イベントの強さや店長のクセを加味した、打ち始める前の期待度です。通常営業なら10%〜15%程度がリアルです。", key="rt_prior_high_prob")

        col1, col2 = st.columns(2)
        # ボーナス回数を利用して総回転数を逆算するため、先に右カラム(col2)のボーナス入力を受け取ります
        with col2:
            b_count = big_counter("BIG回数", "rt_b_count", step=1)
            r_count = big_counter("REG回数", "rt_r_count", step=1)
            reg_hamari = big_counter("現在のREG間ハマり (G) ※任意", "rt_reg_hamari", step=50, help_text="ヤメ時判定に使用します。")

        with col1:
            g_input_mode = st.radio("総回転数の入力方法", ["直接入力", "合算確率から逆算"], horizontal=True, key="rt_g_input_mode")
            if g_input_mode == "直接入力":
                g_count = big_counter("現在の総回転数 (G)", "rt_g_count", step=50)
            else:
                total_prob_den = st.number_input("現在の合算確率 (1/◯)", min_value=1.0, value=150.0, step=0.1, help="データサイトの合算確率分母（例: 150.5）を入力してください。", key="rt_total_prob_den")
                g_count = int(total_prob_den * (b_count + r_count))
                st.info(f"💡 逆算された総回転数: **{g_count}G**")
                
            start_g = big_counter("打ち始めの総回転数 (G) ※任意", "rt_start_g", step=50, help_text="途中から座ってぶどうをカウントした場合は入力してください。カウントしたぶどうは「現在の総回転数 - 打ち始めの総回転数」で計算されます。")
            if start_g > g_count:
                st.warning("打ち始めの回転数が現在の総回転数を上回っています。0として計算します。")
                start_g = 0

            grape_input_mode = st.radio("ぶどうデータの入力方法", ["差枚から逆算", "直接入力 (カウント)"], horizontal=True, key="rt_grape_input_mode")
            if grape_input_mode == "差枚から逆算":
                diff_coins = big_counter("現在の差枚数 (枚) ※任意", "rt_diff_coins", step=100, min_val=None, help_text="ぶどう確率の逆算に使用します。不明な場合は0のままでOKです。")
                manual_grape_count = 0
            else:
                diff_coins = 0
                manual_grape_count = big_counter("カウントしたぶどう回数", "rt_manual_grape_count", step=1, help_text="カチカチくん等でカウントした実際のぶどう回数を入力してください。")

            peak_drop = big_counter("ピークからの差枚落ち (枚) ※任意", "rt_peak_drop", step=100, help_text="最高出玉から何枚減っているか。ヤメ時判定に使用します。")
            
        use_gassan = False
        gassan_g, gassan_b, gassan_r = 0, 0, 0
        
        shop_trend_text = ""
        suggested_mode = "指定なし"
        
        if use_ai and sel_shop and not df_today.empty:
            df_shop_today = df_today[df_today[shop_col] == sel_shop]
            if not df_shop_today.empty and '根拠' in df_shop_today.columns:
                all_reasons = "".join(df_shop_today['根拠'].dropna().astype(str).tolist())
                
                trends = []
                if "末尾" in all_reasons:
                    trends.append("🔢 **末尾**")
                    suggested_mode = "末尾"
                if "並び" in all_reasons or "両隣" in all_reasons:
                    trends.append("🤝 **並び (両隣)**")
                    suggested_mode = "並び"
                if "島" in all_reasons or "塊" in all_reasons or "列" in all_reasons:
                    trends.append("🏝️ **島 (列・塊)**")
                    suggested_mode = "島"
                    
                if trends:
                    shop_trend_text = " / ".join(trends) + " 傾向あり"

        st.divider()
        st.subheader("🤝 複数台合算モード (シマ判別)")
        st.caption("対象の複数台のデータを合算することで、大数の法則により設定が見抜きやすくなります。全台系や並び探しの最強ツールです。")
        
        if shop_trend_text:
            st.info(f"💡 **AIによる店舗傾向の分析**: {shop_trend_text}\n\n店舗のクセに合わせて、合算する対象を選択するとより正確な判別が可能です。")
            
        use_gassan = st.checkbox("複数台合算モードを使用する", value=False, key="rt_use_gassan")
        
        if use_gassan:
            mode_options = ["並び (両隣など)", "同じ末尾", "同じ島 (列・塊)", "その他 (手動)"]
            default_index = 0
            if suggested_mode == "末尾": default_index = 1
            elif suggested_mode == "島": default_index = 2
            
            gassan_type = st.radio("合算の対象", mode_options, index=default_index, horizontal=True, key="rt_gassan_type")
            
            gassan_count = st.number_input("合算する【他台】の台数", min_value=1, max_value=20, value=2, step=1, help="両隣なら「2」、左右2台ずつなら「4」を指定してください。", key="rt_gassan_count")
            
            input_mode = st.radio("入力方法", ["合計値を一括入力", "1台ずつ個別に入力"], horizontal=True, key="rt_input_mode")
            
            gassan_g, gassan_b, gassan_r = 0, 0, 0
            
            if input_mode == "合計値を一括入力":
                st.caption(f"合算対象とする他台({gassan_count}台分)の「合計データ」を入力してください。")
                
                gassan_g_input_mode = st.radio("他台の合計回転数の入力方法", ["直接入力", "合算確率から逆算"], horizontal=True, key="rt_gassan_g_input_mode")
                
                gc1, gc2, gc3 = st.columns(3)
                with gc2:
                    gassan_b = st.number_input("他台の 合計BIG回数", min_value=0, value=0, step=1, key="rt_gassan_b")
                with gc3:
                    gassan_r = st.number_input("他台の 合計REG回数", min_value=0, value=0, step=1, key="rt_gassan_r")
                with gc1:
                    if gassan_g_input_mode == "直接入力":
                        gassan_g = st.number_input("他台の 合計回転数 (G)", min_value=0, value=0, step=100, key="rt_gassan_g")
                    else:
                        gassan_total_prob_den = st.number_input("他台の 合算確率 (1/◯)", min_value=1.0, value=150.0, step=0.1, help="データサイトの合算確率分母（例: 150.5）を入力してください。", key="rt_gassan_total_prob_den")
                        gassan_g = int(gassan_total_prob_den * (gassan_b + gassan_r))
                        st.info(f"💡 逆算総回転数: **{gassan_g}G**")
            else:
                st.caption(f"他台({gassan_count}台)のデータを1台ずつ入力してください。（合計値は自動計算されます）")
                hc1, hc2, hc3, hc4 = st.columns([1, 2, 2, 2])
                
                # Initialize sums for individual inputs
                gassan_g_sum_from_individual_inputs = 0
                gassan_b_sum = 0
                gassan_r_sum = 0

                hc2.caption("回転数(G)")
                hc3.caption("BIG")
                hc4.caption("REG")
                for i in range(int(gassan_count)):
                    c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
                    with c1: st.markdown(f"<div style='padding-top:8px;'>他台{i+1}</div>", unsafe_allow_html=True)
                    with c2: tmp_g = st.number_input("G", min_value=0, value=0, step=100, key=f"g_g_{i}", label_visibility="collapsed") # 個別のG数入力
                    with c3: tmp_b = st.number_input("B", min_value=0, value=0, step=1, key=f"g_b_{i}", label_visibility="collapsed") # 個別のBIG入力
                    with c4: tmp_r = st.number_input("R", min_value=0, value=0, step=1, key=f"g_r_{i}", label_visibility="collapsed") # 個別のREG入力
                    gassan_g_sum_from_individual_inputs += tmp_g
                    gassan_b_sum += tmp_b
                    gassan_r_sum += tmp_r
                
                # Assign the summed values to gassan_b and gassan_r
                gassan_b = gassan_b_sum
                gassan_r = gassan_r_sum

                # Now, after summing up individual inputs, provide an option to calculate gassan_g from combined probability
                st.markdown("---")
                st.caption("他台の合計ゲーム数の計算方法を選択してください:")
                gassan_g_calc_mode = st.radio(
                    "他台の合計ゲーム数",
                    ["個別のG数入力の合計を使用", "合計BIG/REG回数と合算確率から逆算"],
                    horizontal=True,
                    key="rt_gassan_g_calc_mode_individual"
                )

                if gassan_g_calc_mode == "個別のG数入力の合計を使用":
                    gassan_g = gassan_g_sum_from_individual_inputs
                else: # "合計BIG/REG回数と合算確率から逆算"
                    if gassan_b + gassan_r == 0:
                        st.warning("合計BIG回数と合計REG回数が0のため、合算確率から総ゲーム数を逆算できません。個別のG数入力の合計を使用します。")
                        gassan_g = gassan_g_sum_from_individual_inputs # Fallback to direct sum
                    else:
                        gassan_total_prob_den_for_individual = st.number_input(
                            "他台の合計合算確率 (1/◯)",
                            min_value=1.0,
                            value=150.0,
                            step=0.1,
                            key="rt_gassan_total_prob_den_for_individual",
                            help="入力された合計BIG/REG回数とこの合算確率から、他台の合計ゲーム数を逆算します。"
                        )
                        gassan_g = int(gassan_total_prob_den_for_individual * (gassan_b + gassan_r))
                        st.info(f"💡 逆算された他台の合計総回転数: **{gassan_g}G**")

                if gassan_g > 0 or gassan_b > 0 or gassan_r > 0:
                    other_total_prob = (gassan_b + gassan_r) / gassan_g if gassan_g > 0 else 0
                    other_total_prob_str = f"1/{int(1/other_total_prob)}" if other_total_prob > 0 else "-"
                    st.info(f"💡 他台の合計: 総回転 **{gassan_g}G** / BIG **{gassan_b}回** / REG **{gassan_r}回** (合算確率: {other_total_prob_str})")
                    
            total_g_disp = g_count + gassan_g
            total_b_disp = b_count + gassan_b
            total_r_disp = r_count + gassan_r
            total_reg_prob = int(total_g_disp / total_r_disp) if total_r_disp > 0 else 0
            
            st.success(f"✅ **合算データ (自台＋他台{gassan_count}台)**: 総回転 **{total_g_disp}G** / BIG **{total_b_disp}回** / REG **{total_r_disp}回** (合算REG確率: 1/{total_reg_prob})")
            
        st.divider()
        st.subheader("⏱️ 時間・期待値設定")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            current_time = st.time_input("現在時刻", pd.Timestamp.now(tz='Asia/Tokyo').time())
        with col_t2:
            close_time = st.time_input("閉店時刻", pd.to_datetime("22:45").time(), key="rt_close_time")
        with col_t3:
            speed = st.number_input("1時間の回転数 (G/h)", min_value=100, max_value=1000, value=750, step=50, help="一般的な速度は700〜800G/hです。", key="rt_speed")
            
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

    total_g = g_count + gassan_g
    total_b = b_count + gassan_b
    total_r = r_count + gassan_r

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
            
    my_g = max(0, g_count - start_g)
            
    use_grape = False
    grape_count = 0
    grape_target_g = g_count
    if manual_grape_count > 0:
        use_grape = True
        grape_count = manual_grape_count
        grape_target_g = my_g
    elif diff_coins != 0 and g_count >= 1000:
        use_grape = True
        grape_target_g = g_count
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
        exp_b, exp_r = total_g * p_b, total_g * p_r
        ll_b = total_b * math.log(exp_b) - exp_b if exp_b > 0 else 0
        ll_r = total_r * math.log(exp_r) - exp_r if exp_r > 0 else 0
        ll_g = 0
        if use_grape:
            p_g = 1.0 / full_specs[i]["ぶどう"]
            exp_g = grape_target_g * p_g # ぶどう確率は自台の回転数のみをベースに計算
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
            grape_label = "🍇 実測ぶどう確率" if manual_grape_count > 0 else "🍇 推定ぶどう確率"
            st.metric(grape_label, f"1/{(grape_target_g / grape_count if grape_count > 0 else 0):.2f}")
            
        # --- ヤメ時ロジック（アラート） ---
        alerts = []
        if reg_hamari >= 600:
            alerts.append(f"⚠️ **REG間 {reg_hamari}G ハマり**: 判別要素として大きなマイナスです。低設定の可能性が高まっています。")
        elif reg_hamari >= 400:
            alerts.append(f"⚠️ **REG間 {reg_hamari}G ハマり**: 雲行きが怪しくなってきました。")
            
        if peak_drop >= 1500:
            alerts.append(f"🔴 **ピークから {peak_drop}枚 減少**: 危険水域（全飲まれレベル）です。高設定の可能性はかなり低くなっています。")
        elif peak_drop >= 1000:
            alerts.append(f"🟠 **ピークから {peak_drop}枚 減少**: 大きな下降トレンドです。強い根拠（事前期待度が高い、ぶどうが極端に良い等）がなければ撤退を強く推奨します。")
        elif peak_drop >= 500:
            alerts.append(f"🟡 **ピークから {peak_drop}枚 減少**: 波が下がり始めました。合算確率が落ちてきているなら、メダルがあるうちの『勝ち逃げ（利確）』も有効なヤメ時です。")
            
        st.markdown("### 💡 AIジャッジ")
        
        if alerts:
            st.error("\n".join(alerts))
            if total_expected_diff > 0:
                st.caption("※期待値はプラス計算ですが、上記の危険信号が点灯しているため、ヤメ時（撤退）を慎重に判断してください。")
        
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