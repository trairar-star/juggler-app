import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import altair as alt # type: ignore
import backend

def render_my_balance_page(df_raw):
    st.header("💰 マイ収支管理")
    st.caption("あなたの稼働実績（投資・回収・収支）を記録・分析します。")

    df_balance = backend.load_my_balance()

    # --- 1. 収支入力・編集フォーム ---
    with st.expander("📝 収支データを登録・編集", expanded=False):
        input_mode = st.radio("操作を選択", ["新規登録", "既存データの編集"], horizontal=True, label_visibility="collapsed")
        
        if input_mode == "新規登録":
            # 日付選択をフォーム外に配置し、変更時に即座に曜日を反映させる
            input_date = st.date_input("稼働日", pd.Timestamp.now(tz='Asia/Tokyo').date())
            weekdays = ['月', '火', '水', '木', '金', '土', '日']
            wd_str = weekdays[input_date.weekday()]
    
            with st.form("balance_form", clear_on_submit=True):
                col1, col2 = st.columns(2)
                with col1:
                    # 既存データから店名リストを取得
                    shops = []
                    if not df_raw.empty:
                        shop_col = '店名' if '店名' in df_raw.columns else '店舗名'
                        if shop_col in df_raw.columns:
                            shops = list(df_raw[shop_col].unique())
                    input_shop = st.selectbox("店舗名", shops + ["その他 (手入力)"])
                    if input_shop == "その他 (手入力)":
                        input_shop = st.text_input("店舗名を入力")
                    
                    input_number = st.text_input("台番号", placeholder="例: 123")
                    
                with col2:
                    # 機種名リスト
                    machines = []
                    if not df_raw.empty and '機種名' in df_raw.columns:
                        machines = list(df_raw['機種名'].dropna().unique())
                        
                    machine_options = machines + ["その他 (手入力)"]
                    default_idx = machine_options.index("マイジャグラーV") if "マイジャグラーV" in machine_options else 0
                    
                    input_machine = st.selectbox("機種名", machine_options, index=default_idx)
                    if input_machine == "その他 (手入力)":
                        input_machine = st.text_input("機種名を入力")
                
                # --- 追加: ボーナス入力 ---
                st.markdown("**🎰 データ入力 (任意)**")
                st.caption("打ち始めと終了時のデータを入力すると、自分が回した分のボーナス確率が自動計算され、AIチャットでの相性相談に利用されます。")
                s_c1, s_c2, s_c3 = st.columns(3)
                with s_c1: start_g = st.number_input("開始時 総回転 (G)", min_value=0, step=100)
                with s_c2: start_b = st.number_input("開始時 BIG回数", min_value=0, step=1)
                with s_c3: start_r = st.number_input("開始時 REG回数", min_value=0, step=1)
    
                e_c1, e_c2, e_c3 = st.columns(3)
                with e_c1: end_g = st.number_input("終了時 総回転 (G)", min_value=0, step=100)
                with e_c2: end_b = st.number_input("終了時 BIG回数", min_value=0, step=1)
                with e_c3: end_r = st.number_input("終了時 REG回数", min_value=0, step=1)
    
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1: input_invest = st.number_input("投資金額 (円)", min_value=0, step=1000)
                with c2: input_recovery = st.number_input("回収金額 (円)", min_value=0, step=1000)
                with c3: input_hours = st.number_input("稼働時間 (h)", min_value=0.0, step=0.5, value=0.0, help="0.5時間=30分として記録します。")

                st.metric("収支", f"{(input_recovery - input_invest):+d} 円")
                
                input_memo = st.text_area("メモ", value=f"【{wd_str}曜】", placeholder="設定示唆、挙動など", height=80)
                
                submitted = st.form_submit_button("登録する", type="primary")
                if submitted:
                    if not input_shop or not input_machine:
                        st.error("店舗名と機種名は必須です。")
                    else:
                        # メモにボーナス情報を自動結合して保存
                        final_memo = input_memo
                        if start_g > 0 or start_b > 0 or start_r > 0 or end_g > 0 or end_b > 0 or end_r > 0:
                            my_g = max(0, end_g - start_g) if end_g > 0 else 0
                            my_b = max(0, end_b - start_b) if end_b > 0 else 0
                            my_r = max(0, end_r - start_r) if end_r > 0 else 0
                            final_memo = f"【データ】自分稼働:{int(my_g)}G BIG:{int(my_b)} REG:{int(my_r)} (開始 {int(start_g)}G B{int(start_b)} R{int(start_r)} → 終了 {int(end_g)}G B{int(end_b)} R{int(end_r)})\n" + input_memo
    
                        if backend.save_my_balance(input_date, input_shop, input_machine, input_number, input_invest, input_recovery, input_hours, final_memo):
                            st.success("収支データを登録（または上書き更新）しました！")
                            backend.load_my_balance.clear()
                            st.rerun()
        else:
            if df_balance.empty or '登録日時' not in df_balance.columns:
                st.info("編集できる収支データがまだありません。")
            else:
                # 選択肢を「日付」および「登録日時」の降順（最新の入力が一番上）にする
                edit_choices_df = df_balance.copy()
                if '日付' in edit_choices_df.columns:
                    edit_choices_df = edit_choices_df.sort_values(['日付', '登録日時'], ascending=[False, False])
                else:
                    edit_choices_df = edit_choices_df.sort_values('登録日時', ascending=False)
                target_uids = edit_choices_df['登録日時'].unique()

                def format_balance_label(uid):
                    row = df_balance[df_balance['登録日時'] == uid].iloc[0]
                    d_str = row['日付'].strftime('%Y-%m-%d') if pd.notna(row['日付']) else "不明"
                    return f"{d_str} | {row['店名']} | {row['機種名']} ({row['収支']}円)"
                
                def on_edit_target_change():
                    """編集対象が変更されたときにフォームの値を更新するコールバック"""
                    target_uid = st.session_state.edit_balance_target
                    if target_uid:
                        try:
                            target_row = df_balance[df_balance['登録日時'] == target_uid].iloc[0]
                            st.session_state.eb_date = pd.to_datetime(target_row['日付']).date()
                            st.session_state.eb_shop = target_row['店名']
                            st.session_state.eb_num = str(target_row['台番号'])
                            st.session_state.eb_mac = target_row['機種名']
                            st.session_state.eb_inv = int(target_row['投資']) if pd.notna(target_row['投資']) and str(target_row['投資']).strip() != '' else 0
                            st.session_state.eb_rec = int(target_row['回収']) if pd.notna(target_row['回収']) and str(target_row['回収']).strip() != '' else 0
                            st.session_state.eb_hours = float(target_row.get('稼働時間', 0.0) if pd.notna(target_row.get('稼働時間')) and target_row.get('稼働時間') != '' else 0.0)
                            st.session_state.eb_memo = str(target_row.get('メモ', ''))
        
                            import re
                            m_old = re.search(r'総回転:(\d+)G BIG:(\d+) REG:(\d+)', st.session_state.eb_memo)
                            m_new = re.search(r'開始 (\d+)G B(\d+) R(\d+) → 終了 (\d+)G B(\d+) R(\d+)', st.session_state.eb_memo)
                            
                            if m_new:
                                st.session_state.eb_sg = int(m_new.group(1))
                                st.session_state.eb_sb = int(m_new.group(2))
                                st.session_state.eb_sr = int(m_new.group(3))
                                st.session_state.eb_eg = int(m_new.group(4))
                                st.session_state.eb_eb = int(m_new.group(5))
                                st.session_state.eb_er = int(m_new.group(6))
                            elif m_old:
                                st.session_state.eb_sg = 0; st.session_state.eb_sb = 0; st.session_state.eb_sr = 0
                                st.session_state.eb_eg = int(m_old.group(1))
                                st.session_state.eb_eb = int(m_old.group(2))
                                st.session_state.eb_er = int(m_old.group(3))
                            else:
                                st.session_state.eb_sg = 0; st.session_state.eb_sb = 0; st.session_state.eb_sr = 0
                                st.session_state.eb_eg = 0; st.session_state.eb_eb = 0; st.session_state.eb_er = 0
                        except (IndexError, KeyError, ValueError):
                            pass
        
                target_uid = st.selectbox(
                    "編集・削除するデータを選択", 
                    target_uids, 
                    format_func=format_balance_label, 
                    key="edit_balance_target",
                    on_change=on_edit_target_change
                )
                
                # 初回実行時にフォームの値を初期化
                if 'eb_date' not in st.session_state and not df_balance.empty:
                    on_edit_target_change()
                
                if target_uid:
                    with st.form("edit_balance_form"):
                        e_col1, e_col2 = st.columns(2)
                        with e_col1:
                            st.date_input("稼働日", key="eb_date")
                            st.text_input("店舗名", key="eb_shop")
                        with e_col2:
                            st.text_input("機種名", key="eb_mac")
                            st.text_input("台番号", key="eb_num")
        
                        st.markdown("**🎰 データ入力 (任意)**")
                        s_c1, s_c2, s_c3 = st.columns(3)
                        with s_c1: st.number_input("開始時 総回転 (G)", min_value=0, step=100, key="eb_sg")
                        with s_c2: st.number_input("開始時 BIG回数", min_value=0, step=1, key="eb_sb")
                        with s_c3: st.number_input("開始時 REG回数", min_value=0, step=1, key="eb_sr")
        
                        e_c1, e_c2, e_c3 = st.columns(3)
                        with e_c1: st.number_input("終了時 総回転 (G)", min_value=0, step=100, key="eb_eg")
                        with e_c2: st.number_input("終了時 BIG回数", min_value=0, step=1, key="eb_eb")
                        with e_c3: st.number_input("終了時 REG回数", min_value=0, step=1, key="eb_er")
        
                        st.markdown("---")
                        i_c1, i_c2, i_c3 = st.columns(3)
                        with i_c1: st.number_input("投資金額 (円)", min_value=0, step=1000, key="eb_inv")
                        with i_c2: st.number_input("回収金額 (円)", min_value=0, step=1000, key="eb_rec")
                        with i_c3: st.number_input("稼働時間 (h)", min_value=0.0, step=0.5, key="eb_hours")

                        st.text_area("メモ", height=80, key="eb_memo")
                        if st.form_submit_button("更新を保存", type="primary"):
                            if not st.session_state.eb_shop or not st.session_state.eb_mac:
                                st.error("店舗名と機種名は必須です。")
                            else:
                                final_memo = st.session_state.eb_memo
                                if st.session_state.eb_sg > 0 or st.session_state.eb_sb > 0 or st.session_state.eb_sr > 0 or st.session_state.eb_eg > 0 or st.session_state.eb_eb > 0 or st.session_state.eb_er > 0:
                                    import re
                                    final_memo = re.sub(r'【データ】総回転:\d+G BIG:\d+ REG:\d+\n?', '', final_memo)
                                    final_memo = re.sub(r'【データ】自分稼働:\d+G BIG:\d+ REG:\d+ \(開始 \d+G B\d+ R\d+ → 終了 \d+G B\d+ R\d+\)\n?', '', final_memo)
                                    
                                    my_g = max(0, st.session_state.eb_eg - st.session_state.eb_sg) if st.session_state.eb_eg > 0 else 0
                                    my_b = max(0, st.session_state.eb_eb - st.session_state.eb_sb) if st.session_state.eb_eb > 0 else 0
                                    my_r = max(0, st.session_state.eb_er - st.session_state.eb_sr) if st.session_state.eb_er > 0 else 0
                                    
                                    final_memo = f"【データ】自分稼働:{my_g}G BIG:{my_b} REG:{my_r} (開始 {st.session_state.eb_sg}G B{st.session_state.eb_sb} R{st.session_state.eb_sr} → 終了 {st.session_state.eb_eg}G B{st.session_state.eb_eb} R{st.session_state.eb_er})\n" + final_memo
            
                                if backend.update_my_balance(target_uid, st.session_state.eb_date, st.session_state.eb_shop, st.session_state.eb_mac, st.session_state.eb_num, st.session_state.eb_inv, st.session_state.eb_rec, st.session_state.eb_hours, final_memo):
                                    st.success("収支データを更新しました！")
                                    backend.load_my_balance.clear(); st.rerun()
                                else: st.error("更新に失敗しました。")
                    
                    with st.form("delete_balance_form"):
                        st.caption("※この操作は取り消せません")
                        if st.form_submit_button("このデータを削除", type="primary"):
                            if backend.delete_my_balance(target_uid):
                                st.success("削除しました！"); backend.load_my_balance.clear(); st.rerun()
                            else: st.error("削除に失敗しました。")

    # --- 2. 収支データの表示 ---
    
    if df_balance.empty:
        st.info("まだ収支データがありません。「収支データを登録」から記録をつけてみましょう。")
        return

    # --- 過去の予測ログから期待度を取得して結合 ---
    df_pred_log = backend.load_prediction_log()
    
    # 手動で再計算したキャッシュがあればログに追加
    if 'recalc_preds' in st.session_state and not st.session_state.recalc_preds.empty:
        if not df_pred_log.empty:
            df_pred_log = pd.concat([df_pred_log, st.session_state.recalc_preds], ignore_index=True)
        else:
            df_pred_log = st.session_state.recalc_preds.copy()
            
    if not df_pred_log.empty:
        df_pred_log_temp = df_pred_log.copy()
        if '予測対象日' in df_pred_log_temp.columns:
            df_pred_log_temp['予測対象日_merge'] = pd.to_datetime(df_pred_log_temp['予測対象日'], errors='coerce').fillna(pd.to_datetime(df_pred_log_temp['対象日付'], errors='coerce') + pd.Timedelta(days=1))
        else:
            df_pred_log_temp['予測対象日_merge'] = pd.to_datetime(df_pred_log_temp['対象日付'], errors='coerce') + pd.Timedelta(days=1)
            
        shop_col_pred = '店名' if '店名' in df_pred_log_temp.columns else '店舗名'
        if shop_col_pred != '店名':
            df_pred_log_temp = df_pred_log_temp.rename(columns={shop_col_pred: '店名'})
            
        df_pred_log_temp['台番号'] = df_pred_log_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        if '実行日時' in df_pred_log_temp.columns:
            df_pred_log_temp['実行日時'] = pd.to_datetime(df_pred_log_temp['実行日時'], errors='coerce')
            df_pred_log_temp = df_pred_log_temp.sort_values('実行日時', ascending=False).drop_duplicates(
                subset=['予測対象日_merge', '店名', '台番号'], keep='first'
            )
            
        df_balance['日付_merge'] = pd.to_datetime(df_balance['日付'])
        df_balance['台番号_str'] = df_balance['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
        df_pred_log_temp['prediction_score'] = pd.to_numeric(df_pred_log_temp['prediction_score'], errors='coerce')
        
        df_balance = pd.merge(df_balance, df_pred_log_temp[['予測対象日_merge', '店名', '台番号', 'prediction_score']], left_on=['日付_merge', '店名', '台番号_str'], right_on=['予測対象日_merge', '店名', '台番号'], how='left', suffixes=('', '_pred'))
        df_balance = df_balance.drop(columns=['日付_merge', '台番号_str', '予測対象日_merge', '台番号_pred'], errors='ignore')
    else:
        df_balance['prediction_score'] = np.nan

    df_balance['期待度_pct'] = df_balance['prediction_score'] * 100

    # --- 欠損している期待度を再計算する機能 ---
    missing_dates = df_balance[df_balance['prediction_score'].isna()]['日付'].dropna().dt.date.unique()
    if len(missing_dates) > 0 and not df_raw.empty:
        st.info(f"💡 事前期待度のログがない稼働データが {len(missing_dates)} 日分あります。過去の生データから当時の状況をシミュレーションしてAI期待度を補完できます。")
        if st.button("🔄 過去データから欠損している期待度を再計算する"):
            with st.spinner("AIが当時の状況を再現して期待度を計算中...（日数によっては時間がかかります）"):
                df_events = backend.load_shop_events()
                df_island = backend.load_island_master()
                shop_hp = st.session_state.get("shop_hyperparams", {})
                
                if 'recalc_preds' not in st.session_state:
                    st.session_state.recalc_preds = pd.DataFrame()
                    
                new_preds = []
                progress_bar = st.progress(0)
                for i, target_d in enumerate(missing_dates):
                    try:
                        df_pred, _, _ = backend.run_analysis(
                            df_raw, _df_events=df_events, _df_island=df_island, shop_hyperparams=shop_hp, target_date=target_d
                        )
                        if not df_pred.empty:
                            df_pred_temp = df_pred.copy()
                            df_pred_temp['予測対象日'] = pd.to_datetime(target_d)
                            shop_col_pred = '店名' if '店名' in df_pred_temp.columns else '店舗名'
                            if shop_col_pred != '店名':
                                df_pred_temp = df_pred_temp.rename(columns={shop_col_pred: '店名'})
                            df_pred_temp['台番号'] = df_pred_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
                            new_preds.append(df_pred_temp[['予測対象日', '店名', '台番号', 'prediction_score']])
                    except Exception:
                        pass
                    progress_bar.progress((i + 1) / len(missing_dates))
                
                if new_preds:
                    st.session_state.recalc_preds = pd.concat([st.session_state.recalc_preds, pd.concat(new_preds, ignore_index=True)], ignore_index=True)
                    st.rerun()

    # 日付ソート
    df_balance = df_balance.sort_values('日付', ascending=False)

    # KPI 計算
    total_balance = df_balance['収支'].sum()
    total_invest = df_balance['投資'].sum()
    total_recovery = df_balance['回収'].sum()
    if '稼働時間' not in df_balance.columns:
        df_balance['稼働時間'] = 0.0
    df_balance['稼働時間'] = pd.to_numeric(df_balance['稼働時間'], errors='coerce').fillna(0)
    total_hours = df_balance['稼働時間'].sum()
    hourly_wage = total_balance / total_hours if total_hours > 0 else 0
    win_count = (df_balance['収支'] > 0).sum()
    total_count = len(df_balance)
    win_rate = win_count / total_count if total_count > 0 else 0
    
    # 通算の自力合算の計算
    import re
    def _extract_sum_data(x, kind):
        if pd.isna(x): return 0
        m_new = re.search(r'自分稼働:(\d+)G BIG:(\d+) REG:(\d+)', str(x))
        if m_new:
            return int(m_new.group(1)) if kind == 'G' else int(m_new.group(2)) if kind == 'B' else int(m_new.group(3))
        m_old = re.search(r'総回転:(\d+)G BIG:(\d+) REG:(\d+)', str(x))
        if m_old:
            return int(m_old.group(1)) if kind == 'G' else int(m_old.group(2)) if kind == 'B' else int(m_old.group(3))
        return 0
        
    df_balance['自力G'] = df_balance['メモ'].apply(lambda x: _extract_sum_data(x, 'G'))
    df_balance['自力B'] = df_balance['メモ'].apply(lambda x: _extract_sum_data(x, 'B'))
    df_balance['自力R'] = df_balance['メモ'].apply(lambda x: _extract_sum_data(x, 'R'))
    
    total_my_g = df_balance['自力G'].sum()
    total_my_b = df_balance['自力B'].sum()
    total_my_r = df_balance['自力R'].sum()
    my_total_prob = f"1/{total_my_g / (total_my_b + total_my_r):.1f}" if (total_my_b + total_my_r) > 0 and total_my_g > 0 else "-"
    my_bb_prob = f"1/{total_my_g / total_my_b:.1f}" if total_my_b > 0 and total_my_g > 0 else "-"
    my_reg_prob = f"1/{total_my_g / total_my_r:.1f}" if total_my_r > 0 and total_my_g > 0 else "-"

    st.subheader("📊 通算成績")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("総収支", f"{total_balance:+d} 円", delta_color="normal")
    k2.metric("回収率", f"{(total_recovery/total_invest*100):.1f} %" if total_invest > 0 else "-")
    k3.metric("勝率", f"{win_rate:.1%}")
    k4.metric("稼働数", f"{total_count} 回")
    k5.metric("時給", f"{int(hourly_wage):,} 円/h" if total_hours > 0 else "-")
    
    avg_pred_score = df_balance['prediction_score'].mean() if 'prediction_score' in df_balance.columns else np.nan
    st.caption("稼働データ (メモ欄からの自動集計)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("総ゲーム数", f"{total_my_g:,} G")
    m2.metric("BB確率", my_bb_prob)
    m3.metric("REG確率", my_reg_prob)
    m4.metric("合算確率", my_total_prob)
    m5.metric("平均打台期待度", f"{avg_pred_score*100:.1f}%" if pd.notna(avg_pred_score) else "-")

    # --- 月別収支グラフ ---
    st.subheader("🗓️ 月別成績")
    df_balance['年月'] = df_balance['日付'].dt.strftime('%Y-%m')
    
    monthly_stats = df_balance.groupby('年月').agg(
        総収支=('収支', 'sum'),
        稼働数=('収支', 'count'),
        勝率=('収支', lambda x: (x > 0).mean() * 100),
        稼働時間=('稼働時間', 'sum'),
        自力G=('自力G', 'sum'),
        自力B=('自力B', 'sum'),
        自力R=('自力R', 'sum')
    ).reset_index()
    monthly_stats['累積収支'] = monthly_stats['総収支'].cumsum()
    monthly_stats['時給'] = np.where(monthly_stats['稼働時間'] > 0, monthly_stats['総収支'] / monthly_stats['稼働時間'], 0)
    monthly_stats['合算確率'] = monthly_stats.apply(lambda r: f"1/{r['自力G']/(r['自力B']+r['自力R']):.1f}" if (r['自力B']+r['自力R'])>0 and r['自力G']>0 else "-", axis=1)
    monthly_stats['BB確率'] = monthly_stats.apply(lambda r: f"1/{r['自力G']/r['自力B']:.1f}" if r['自力B']>0 and r['自力G']>0 else "-", axis=1)
    monthly_stats['REG確率'] = monthly_stats.apply(lambda r: f"1/{r['自力G']/r['自力R']:.1f}" if r['自力R']>0 and r['自力G']>0 else "-", axis=1)
    
    monthly_stats_chart = monthly_stats.rename(columns={'総収支': '収支'})
    base_m = alt.Chart(monthly_stats_chart).encode(x=alt.X('年月', title='年月'))
    bar_m = base_m.mark_bar(opacity=0.7).encode(
        y=alt.Y('収支', title='収支 (円)'),
        color=alt.condition(alt.datum.収支 > 0, alt.value("#ef5350"), alt.value("#42a5f5")),
        tooltip=['年月', alt.Tooltip('収支', format='+d')]
    )
    line_m = base_m.mark_line(color='#ffa726', point=True, strokeWidth=3).encode(
        y=alt.Y('累積収支', title='累積収支 (円)'),
        tooltip=['年月', alt.Tooltip('累積収支', format='+d')]
    )
    st.altair_chart(alt.layer(bar_m, line_m).resolve_scale(y='independent').interactive(), use_container_width=True)
    
    st.dataframe(
        monthly_stats.sort_values('年月', ascending=False)[['年月', '総収支', '時給', '勝率', '稼働数', '自力G', 'BB確率', 'REG確率', '合算確率']],
        column_config={
            "年月": st.column_config.TextColumn("年月"),
            "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
            "時給": st.column_config.NumberColumn("時給", format="%+d 円/h"),
            "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=100),
            "稼働数": st.column_config.NumberColumn("回数"),
            "自力G": st.column_config.NumberColumn("総G数", format="%d G"),
            "BB確率": st.column_config.TextColumn("BB確率"),
            "REG確率": st.column_config.TextColumn("REG確率"),
            "合算確率": st.column_config.TextColumn("合算確率"),
        },
        width="stretch",
        hide_index=True
    )

    # --- AI期待度別の成績比較 ---
    st.divider()
    st.subheader("🤖 AI期待度別の成績比較")
    st.caption("稼働した台の事前のAI期待度別に、収支や勝率を比較します。（※保存された予測ログがある稼働のみ集計されます）")

    def classify_pred_score(pct):
        if pd.isna(pct):
            return "不明 (ログなし)"
        elif pct >= 50:
            return "🔥 50%以上 (超激アツ)"
        elif pct >= 40:
            return "🔥 40%〜49% (激アツ)"
        elif pct >= 30:
            return "🌟 30%〜39% (チャンス)"
        elif pct >= 20:
            return "⚖️ 20%〜29% (通常)"
        elif pct >= 15:
            return "⚠️ 15%〜19% (やや危険)"
        elif pct >= 10:
            return "🥶 10%〜14% (危険)"
        else:
            return "💀 10%未満 (超危険)"

    df_balance['期待度区分'] = df_balance['期待度_pct'].apply(classify_pred_score)
    
    pred_rank = df_balance.groupby('期待度区分').agg(
        総収支=('収支', 'sum'),
        勝率=('収支', lambda x: (x > 0).mean() * 100),
        稼働数=('収支', 'count'),
        稼働時間=('稼働時間', 'sum'),
        平均期待度=('期待度_pct', 'mean')
    ).reset_index()
    
    order_map = {
        "🔥 50%以上 (超激アツ)": 1, 
        "🔥 40%〜49% (激アツ)": 2, 
        "🌟 30%〜39% (チャンス)": 3, 
        "⚖️ 20%〜29% (通常)": 4, 
        "⚠️ 15%〜19% (やや危険)": 5, 
        "🥶 10%〜14% (危険)": 6, 
        "💀 10%未満 (超危険)": 7, 
        "不明 (ログなし)": 8
    }
    pred_rank['sort'] = pred_rank['期待度区分'].map(order_map)
    pred_rank = pred_rank.sort_values('sort').drop(columns=['sort'])
    pred_rank['時給'] = np.where(pred_rank['稼働時間'] > 0, pred_rank['総収支'] / pred_rank['稼働時間'], 0)

    st.dataframe(
        pred_rank[['期待度区分', '総収支', '時給', '勝率', '稼働数', '平均期待度']],
        column_config={
            "期待度区分": st.column_config.TextColumn("AI期待度"),
            "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
            "時給": st.column_config.NumberColumn("時給", format="%+d 円/h"),
            "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=100),
            "稼働数": st.column_config.NumberColumn("回数"),
            "平均期待度": st.column_config.NumberColumn("平均期待度", format="%.1f%%"),
        },
        width="stretch",
        hide_index=True
    )

    # --- カテゴリ別 ランキング ---
    st.divider()
    st.subheader("🏆 カテゴリ別 成績ランキング")
    tab_shop, tab_mac, tab_machine_no = st.tabs(["🏬 店舗別", "🎰 機種別", "🎯 店舗・台番号別"])
    
    with tab_shop:
        shop_rank = df_balance.groupby('店名').agg(
            総収支=('収支', 'sum'),
            勝率=('収支', lambda x: (x > 0).mean() * 100),
            稼働数=('収支', 'count'),
            稼働時間=('稼働時間', 'sum'),
            平均期待度=('期待度_pct', 'mean')
        ).sort_values('総収支', ascending=False).reset_index()
        shop_rank['時給'] = np.where(shop_rank['稼働時間'] > 0, shop_rank['総収支'] / shop_rank['稼働時間'], 0)
        
        st.dataframe(
            shop_rank[['店名', '総収支', '時給', '勝率', '稼働数', '平均期待度']],
            column_config={
                "店名": st.column_config.TextColumn("店舗"),
                "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
                "時給": st.column_config.NumberColumn("時給", format="%+d 円/h"),
                "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=100),
                "稼働数": st.column_config.NumberColumn("回数"),
                "平均期待度": st.column_config.NumberColumn("平均期待度", format="%.1f%%"),
            },
            width="stretch",
            hide_index=True
        )

    with tab_mac:
        machine_rank = df_balance.groupby('機種名').agg(
            総収支=('収支', 'sum'),
            勝率=('収支', lambda x: (x > 0).mean() * 100),
            稼働数=('収支', 'count'),
            稼働時間=('稼働時間', 'sum'),
            平均期待度=('期待度_pct', 'mean')
        ).sort_values('総収支', ascending=False).reset_index()
        machine_rank['時給'] = np.where(machine_rank['稼働時間'] > 0, machine_rank['総収支'] / machine_rank['稼働時間'], 0)
        
        st.dataframe(
            machine_rank[['機種名', '総収支', '時給', '勝率', '稼働数', '平均期待度']],
            column_config={
                "機種名": st.column_config.TextColumn("機種"),
                "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
                "時給": st.column_config.NumberColumn("時給", format="%+d 円/h"),
                "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=100),
                "稼働数": st.column_config.NumberColumn("回数"),
                "平均期待度": st.column_config.NumberColumn("平均期待度", format="%.1f%%"),
            },
            width="stretch",
            hide_index=True
        )
        
    with tab_machine_no:
        df_balance['店舗_台番号'] = df_balance['店名'] + " #" + df_balance['台番号'].astype(str)
        machine_no_rank = df_balance.groupby('店舗_台番号').agg(
            総収支=('収支', 'sum'),
            勝率=('収支', lambda x: (x > 0).mean() * 100),
            稼働数=('収支', 'count'),
            稼働時間=('稼働時間', 'sum'),
            平均期待度=('期待度_pct', 'mean')
        ).sort_values('総収支', ascending=False).reset_index()
        machine_no_rank['時給'] = np.where(machine_no_rank['稼働時間'] > 0, machine_no_rank['総収支'] / machine_no_rank['稼働時間'], 0)
        
        st.dataframe(
            machine_no_rank[['店舗_台番号', '総収支', '時給', '勝率', '稼働数', '平均期待度']],
            column_config={
                "店舗_台番号": st.column_config.TextColumn("店舗・台番号"),
                "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
                "時給": st.column_config.NumberColumn("時給", format="%+d 円/h"),
                "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=100),
                "稼働数": st.column_config.NumberColumn("回数"),
                "平均期待度": st.column_config.NumberColumn("平均期待度", format="%.1f%%"),
            },
            width="stretch",
            hide_index=True
        )

    # --- 曜日別 成績 ---
    st.divider()
    st.subheader("📅 曜日別 成績")
    
    # 曜日カラムの作成
    weekdays_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
    df_balance['曜日_num'] = df_balance['日付'].dt.dayofweek
    df_balance['曜日'] = df_balance['曜日_num'].map(weekdays_map)
    
    # 集計
    weekday_rank = df_balance.groupby(['曜日_num', '曜日']).agg(
        総収支=('収支', 'sum'),
        勝率=('収支', lambda x: (x > 0).mean() * 100),
        稼働数=('収支', 'count'),
        稼働時間=('稼働時間', 'sum'),
        平均期待度=('期待度_pct', 'mean')
    ).reset_index().sort_values('曜日_num')
    weekday_rank['時給'] = np.where(weekday_rank['稼働時間'] > 0, weekday_rank['総収支'] / weekday_rank['稼働時間'], 0)
    
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        base_w = alt.Chart(weekday_rank).encode(x=alt.X('曜日', sort=[weekdays_map[i] for i in range(7)], title='曜日'))
        bar_w = base_w.mark_bar(opacity=0.7).encode(
            y=alt.Y('総収支', title='総収支 (円)'),
            color=alt.condition(alt.datum.総収支 > 0, alt.value("#ef5350"), alt.value("#42a5f5")),
            tooltip=['曜日', alt.Tooltip('総収支', format='+d')]
        )
        line_w = base_w.mark_line(color='#ab47bc', point=True, strokeWidth=3).encode(
            y=alt.Y('勝率', title='勝率 (%)', scale=alt.Scale(domain=[0, 100])),
            tooltip=['曜日', alt.Tooltip('勝率', format='.1f', title='勝率(%)')]
        )
        st.altair_chart(alt.layer(bar_w, line_w).resolve_scale(y='independent').interactive(), width="stretch")
        
    with col_w2:
        st.dataframe(
            weekday_rank[['曜日', '総収支', '時給', '勝率', '稼働数', '平均期待度']],
            column_config={
                "曜日": st.column_config.TextColumn("曜日"),
                "総収支": st.column_config.NumberColumn("Total", format="%+d 円"),
                "時給": st.column_config.NumberColumn("時給", format="%+d 円/h"),
                "勝率": st.column_config.ProgressColumn("勝率", format="%.1f%%", min_value=0, max_value=100),
                "稼働数": st.column_config.NumberColumn("回数"),
                "平均期待度": st.column_config.NumberColumn("平均期待度", format="%.1f%%"),
            },
            width="stretch",
            hide_index=True
        )

    # グラフ (日別収支と累積収支)
    st.subheader("📈 資産推移")
    chart_data = df_balance.sort_values('日付').copy()
    chart_data['累積収支'] = chart_data['収支'].cumsum()
    
    # 複合グラフ: 棒(日別) + 線(累積)
    base = alt.Chart(chart_data).encode(x=alt.X('日付', title='日付'))
    
    bar = base.mark_bar(opacity=0.3).encode(
        y=alt.Y('収支', title='日別収支 (円)'),
        color=alt.condition(alt.datum.収支 > 0, alt.value("#ef5350"), alt.value("#42a5f5")),
        tooltip=['日付', '店名', '機種名', '台番号', '収支', 'メモ']
    )
    
    line = base.mark_line(point=True, color='#ffa726').encode(
        y=alt.Y('累積収支', title='累積収支 (円)'),
        tooltip=['日付', '累積収支']
    )
    
    st.altair_chart(alt.layer(bar, line).resolve_scale(y='independent'), width="stretch")

    # テーブル表示
    st.subheader("📝 稼働履歴一覧")
    
    def extract_my_prob(memo_text):
        if pd.isna(memo_text): return None
        memo_str = str(memo_text)
        import re
        m_new = re.search(r'自分稼働:(\d+)G BIG:(\d+) REG:(\d+)', memo_str)
        if m_new:
            g, b, r = int(m_new.group(1)), int(m_new.group(2)), int(m_new.group(3))
            if b + r > 0 and g > 0: return f"1/{g / (b + r):.1f}"
            elif g > 0: return "1/--"
            return None
        m_old = re.search(r'総回転:(\d+)G BIG:(\d+) REG:(\d+)', memo_str)
        if m_old:
            g, b, r = int(m_old.group(1)), int(m_old.group(2)), int(m_old.group(3))
            if b + r > 0 and g > 0: return f"1/{g / (b + r):.1f}"
            elif g > 0: return "1/--"
            return None
        return None

    df_balance['自力合算'] = df_balance['メモ'].apply(extract_my_prob)

    display_cols = ['日付', '店名', '台番号', '機種名', '投資', '回収', '収支', '稼働時間', '期待度_pct', '自力合算', 'メモ']
    available_cols = [c for c in display_cols if c in df_balance.columns]
    styled_balance = df_balance[available_cols].style.bar(subset=['収支'], align='mid', color=['rgba(66, 165, 245, 0.5)', 'rgba(255, 112, 67, 0.5)'], vmin=-30000, vmax=30000)
    st.dataframe(
        styled_balance,
        column_config={
            "日付": st.column_config.DateColumn("日付", format="YYYY-MM-DD"),
            "投資": st.column_config.NumberColumn("投資", format="%d 円"),
            "回収": st.column_config.NumberColumn("回収", format="%d 円"),
            "収支": st.column_config.NumberColumn("収支", format="%+d 円"),
            "稼働時間": st.column_config.NumberColumn("時間", format="%.1f h"),
            "期待度_pct": st.column_config.NumberColumn("事前AI期待度", format="%.1f%%", help="打った台の事前のAI予測期待度(保存ログから取得)"),
            "自力合算": st.column_config.TextColumn("自力合算", help="メモ欄の入力から計算した自分が回した分の合算確率"),
        },
        width="stretch",
        hide_index=True
    )