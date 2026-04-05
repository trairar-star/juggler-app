import streamlit as st
import pandas as pd
import backend

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

def render_ai_chat_page(df_predict, df_raw, shop_col, df_events=None, df_importance=None):
    st.header("👩‍💼 ホール案内スタッフ（AI）")
    st.caption("受付スタッフ風のAIに、アプリの最新データをもとにした立ち回りの相談や店舗の傾向についてチャットで質問できます。")

    # --- スマホ向けUI調整 (CSS) ---
    st.markdown("""
        <style>
            /* チャットメッセージのフォントサイズと行間をスマホ向けに調整 */
            [data-testid="stChatMessage"] {
                font-size: 0.85rem;
            }
            [data-testid="stChatMessage"] p {
                margin-bottom: 0.4rem;
                line-height: 1.5;
            }
            /* チャット入力欄のフォントサイズ調整 */
            [data-testid="stChatInput"] textarea {
                font-size: 0.9rem;
            }
            /* AIの回答がリスト(箇条書き)の場合の余白調整 */
            [data-testid="stChatMessage"] ul, [data-testid="stChatMessage"] ol {
                padding-left: 1.2rem;
                margin-bottom: 0.4rem;
            }
            [data-testid="stChatMessage"] li {
                margin-bottom: 0.2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # ライブラリのインストール確認
    if not GENAI_AVAILABLE:
        import sys
        st.error("`google-generativeai` ライブラリがインストールされていません。")
        st.info("💡 仮想環境のズレが原因の可能性があります。以下のコマンドをコピーしてターミナルで実行し、現在の環境に直接インストールしてください。")
        st.code(f"{sys.executable} -m pip install google-generativeai", language="bash")
        return

    # APIキーの設定確認
    if "GEMINI_API_KEY" not in st.secrets:
        st.error("APIキーが設定されていません。`.streamlit/secrets.toml` に `GEMINI_API_KEY` を設定してください。")
        st.info("設定例:\n```toml\nGEMINI_API_KEY = \"AIzaSy...\"\n```")
        return

    # APIキーの初期化
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # 会話履歴の初期化
    if "gemini_messages" not in st.session_state:
        st.session_state.gemini_messages = []

    # --- 連携する店舗の選択 ---
    shops = ["店舗を選択してください"]
    if not df_predict.empty and shop_col in df_predict.columns:
        shops.extend(sorted(df_predict[shop_col].dropna().unique().tolist()))
    
    default_index = 0
    saved_shop = st.session_state.get("global_selected_shop", "店舗を選択してください")
    if saved_shop in shops:
        default_index = shops.index(saved_shop)

    selected_shop = st.selectbox("分析対象の店舗を選択", shops, index=default_index)

    if selected_shop != "店舗を選択してください":
        st.session_state["global_selected_shop"] = selected_shop
        st.success(f"【{selected_shop}】のデータをGeminiに連携する準備ができました！")
    else:
        st.info("店舗を選択すると、その店舗のデータをもとに具体的なアドバイスが可能になります。")

    st.divider()

    # --- チャット履歴の表示 ---
    for msg in st.session_state.gemini_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- ユーザーからの入力 ---
    if prompt := st.chat_input("質問を入力してください (例: 明日の〇〇店の狙い目は？ / 今日の立ち回りアドバイスをお願い)"):
        
        # ユーザーの質問を表示し、履歴に追加
        st.chat_message("user").markdown(prompt)
        st.session_state.gemini_messages.append({"role": "user", "content": prompt})

        # --- アプリ内のデータをGemini用に文字列化して準備 ---
        context_data = ""
        target_date_str = "不明"
        
        if selected_shop != "店舗を選択してください":
            
            # --- 予測対象日の取得 (サイドバーで選んだ日付と一致させる) ---
            if not df_predict.empty and 'next_date' in df_predict.columns:
                target_date_val = df_predict['next_date'].max()
            else:
                target_date_val = pd.Timestamp.now(tz='Asia/Tokyo').date()
            target_date_str = pd.to_datetime(target_date_val).strftime('%Y-%m-%d')
            
            # --- 1. 店舗の直近状況とイベント情報 ---
            if not df_raw.empty:
                shop_raw = df_raw[df_raw[shop_col] == selected_shop].copy()
                if '対象日付' in shop_raw.columns:
                    recent_date = shop_raw['対象日付'].max() - pd.Timedelta(days=7)
                    recent_data = shop_raw[shop_raw['対象日付'] >= recent_date]
                    if not recent_data.empty:
                        avg_diff = recent_data['差枚'].mean()
                        avg_games = recent_data['累計ゲーム'].mean()
                        context_data += f"\n【{selected_shop} の直近1週間の店舗状況】\n店舗平均差枚: 約 {int(avg_diff):+d} 枚\n平均回転数: 約 {int(avg_games)} G (※稼働が高いほど客層レベルが高く後ヅモが困難)\n"
            
            if df_events is not None and not df_events.empty:
                shop_events = df_events[(df_events['店名'] == selected_shop) & (df_events['イベント日付'].dt.strftime('%Y-%m-%d') == target_date_str)]
                if not shop_events.empty:
                    context_data += f"\n【{selected_shop} の {target_date_str} のイベント情報】\n"
                    for _, ev in shop_events.iterrows():
                        ev_name = ev.get('イベント名', '')
                        ev_rank = ev.get('イベントランク', '不明')
                        ev_type = ev.get('イベント種別', '全体')
                        ev_target = ev.get('対象機種', '指定なし')
                        context_data += f"・{ev_name} (ランク: {ev_rank}, 種別: {ev_type}, 対象: {ev_target})\n"

            # --- 1.5. 過去のイベント種別ごとのジャグラー実績 (シワ寄せ確認用) ---
            if df_events is not None and not df_events.empty and not df_raw.empty:
                shop_raw_ev = df_raw[df_raw[shop_col] == selected_shop].copy()
                shop_events_all = df_events[df_events['店名'] == selected_shop].copy()
                
                if not shop_events_all.empty and not shop_raw_ev.empty:
                    shop_events_all = shop_events_all.drop_duplicates(subset=['イベント日付'], keep='last')
                    shop_raw_ev['対象日付'] = pd.to_datetime(shop_raw_ev['対象日付'])
                    shop_events_all['イベント日付'] = pd.to_datetime(shop_events_all['イベント日付'])
                    
                    merged_ev = pd.merge(shop_raw_ev, shop_events_all[['イベント日付', 'イベントランク', '対象機種', 'イベント種別']], left_on='対象日付', right_on='イベント日付', how='inner')
                    
                    if not merged_ev.empty:
                        def classify_target(row):
                            rank = str(row.get('イベントランク', '通常日'))
                            if rank == '通常日': return '通常日'
                            
                            t_mac = str(row.get('対象機種', '指定なし'))
                            my_mac = str(row.get('機種名', ''))
                            e_type = str(row.get('イベント種別', '全体')).replace('スロット/全体', '全体')
                            
                            if e_type == '対象外(無効)': return '通常日'
                            if e_type == 'パチンコ専用': return 'パチンコ特日 (スロットへのシワ寄せ等)'
                            
                            if t_mac in ['指定なし', 'スロット全体', 'ジャグラー全体', '全体', 'nan', 'None']:
                                return 'スロット全体対象イベント'
                            if t_mac == 'ジャグラー以外 (パチスロ他機種)':
                                return 'ジャグラー以外対象 (シワ寄せ等)'
                            if my_mac in t_mac or t_mac in my_mac:
                                return '自身が対象機種'
                            return '自身が対象外機種'
                            
                        merged_ev['対象ステータス'] = merged_ev.apply(classify_target, axis=1)
                        target_stats = merged_ev[merged_ev['対象ステータス'] != '通常日'].groupby('対象ステータス').agg(
                            差枚=('差枚', 'mean'),
                            サンプル数=('台番号', 'count')
                        ).reset_index()
                        
                        target_stats = target_stats[target_stats['サンプル数'] >= 10]
                        
                        if not target_stats.empty:
                            context_data += f"\n【{selected_shop} の過去イベント時のジャグラー平均差枚 (シワ寄せ等の影響確認用)】\n"
                            for _, row in target_stats.iterrows():
                                context_data += f"・{row['対象ステータス']}: 平均 {int(row['差枚']):+d} 枚 (サンプル {row['サンプル数']}件)\n"

            # --- 2. AIが分析した店舗の店癖（設定投入傾向） ---
            if df_importance is not None and not df_importance.empty:
                imp_shop = df_importance[df_importance['shop_name'] == selected_shop].sort_values('importance', ascending=False)
                if not imp_shop.empty:
                    feature_map = {
                        'neighbor_avg_diff': '並び・塊の傾向 (両隣の差枚を重視)', 'island_avg_diff': '島・列ごとの強さ',
                        'machine_no_30days_avg_diff': '場所・特定の台番号の強さ (定位置)', 'cons_minus_total_diff': '連続マイナス台の上げリセット(お詫び)',
                        'event_x_machine_avg_diff': 'イベント時の特定機種の強さ', 'event_x_end_digit_avg_diff': 'イベント時の特定末尾の強さ',
                        'prev_unlucky_gap': '前日の不発台の据え置き', 'prev_bonus_balance': '前日のREG先行台(BB欠損)の据え置き・反発',
                        'target_weekday': '曜日ごとの出し方の違い', 'target_date_end_digit': '日付末尾(特定日)の強さ', 'is_corner': '角台の優遇'
                    }
                    top_feats = [feature_map[f] for f in imp_shop['feature'].tolist() if f in feature_map][:4]
                    if top_feats:
                        context_data += f"\n【{selected_shop} の設定投入のクセ (AI分析による重要項目)】\n・" + "\n・".join(top_feats) + "\n"

            # --- 3. 最近甘く使われている機種 ---
            if not df_predict.empty and 'machine_30days_avg_diff' in df_predict.columns and '機種名' in df_predict.columns:
                shop_pred = df_predict[df_predict[shop_col] == selected_shop]
                if not shop_pred.empty:
                    mac_stats = shop_pred.groupby('機種名')['machine_30days_avg_diff'].mean().reset_index()
                    mac_stats = mac_stats[mac_stats['machine_30days_avg_diff'] > 100].sort_values('machine_30days_avg_diff', ascending=False).head(3)
                    if not mac_stats.empty:
                        context_data += f"\n【{selected_shop} の好調機種 (直近30日の平均差枚が優秀)】\n"
                        for _, row in mac_stats.iterrows():
                            context_data += f"・{row['機種名']} (平均 +{int(row['machine_30days_avg_diff'])}枚)\n"

            # --- 3.5. 最近大きく凹んでいる台（上げリセット候補） ---
            if not df_predict.empty and '連続マイナス日数' in df_predict.columns and 'cons_minus_total_diff' in df_predict.columns:
                shop_pred = df_predict[df_predict[shop_col] == selected_shop]
                if not shop_pred.empty:
                    # 連続マイナス日数が3日以上で、合計吸い込みが多い台を抽出
                    target_h = shop_pred[shop_pred['連続マイナス日数'] >= 3].sort_values('cons_minus_total_diff', ascending=True).head(3)
                    if not target_h.empty:
                        context_data += f"\n【{selected_shop} の最近大きく凹んでいる台 (上げリセット候補)】\n"
                        for _, row in target_h.iterrows():
                            context_data += f"・台番号 {row['台番号']} ({row['機種名']}) - {int(row['連続マイナス日数'])}日連続マイナス / 合計 {int(row['cons_minus_total_diff'])}枚 吸い込み\n"

            # --- 4. 明日の予測データ (全台) ---
            if not df_predict.empty:
                shop_pred = df_predict[df_predict[shop_col] == selected_shop].sort_values('prediction_score', ascending=False)
                if not shop_pred.empty:
                    context_data += f"\n【{selected_shop} の {target_date_str} の予測データ (全台の期待度ランキング)】\n"
                    cols = ['台番号', '機種名', 'prediction_score', '根拠']
                    available_cols = [c for c in cols if c in shop_pred.columns]
                    
                    display_df = top_10[available_cols].copy()
                    display_df = shop_pred[available_cols].copy()
                    if 'prediction_score' in display_df.columns:
                        display_df['prediction_score'] = display_df['prediction_score'].apply(lambda x: f"{int(x*100)}%")
                    
                    context_data += display_df.to_markdown(index=False) + "\n"

                    # --- 4.5. 上位推奨台の直近3日間の個別履歴データ ---
                    if not df_raw.empty:
                        context_data += f"\n【上記推奨台のうち、上位10台の直近3日間の個別データ】\n"
                        top_macs = shop_pred.head(10)['台番号'].tolist()
                        shop_raw_hist = df_raw[df_raw[shop_col] == selected_shop].copy()
                        
                        if '対象日付' in shop_raw_hist.columns:
                            shop_raw_hist['対象日付'] = pd.to_datetime(shop_raw_hist['対象日付'])
                            # 予測日の前日を基準にする
                            base_date = pd.to_datetime(target_date_val) - pd.Timedelta(days=1)
                            cutoff_date = base_date - pd.Timedelta(days=3)
                            
                            hist_df = shop_raw_hist[(shop_raw_hist['対象日付'] > cutoff_date) & (shop_raw_hist['対象日付'] <= base_date)]
                            
                            for mac_num in top_macs:
                                mac_hist = hist_df[hist_df['台番号'].astype(str).str.replace(r'\.0$', '', regex=True) == str(mac_num).replace('.0', '')].sort_values('対象日付', ascending=False)
                                if not mac_hist.empty:
                                    context_data += f"・台番号 {mac_num} の履歴:\n"
                                    for _, row in mac_hist.iterrows():
                                        d_str = row['対象日付'].strftime('%m/%d')
                                        diff = row.get('差枚', 0)
                                        b = row.get('BIG', 0)
                                        r = row.get('REG', 0)
                                        g = row.get('累計ゲーム', 0)
                                        context_data += f"  [{d_str}] 差枚: {int(diff):+d}枚 / 総回転: {int(g)}G / BIG: {int(b)} / REG: {int(r)}\n"

            # --- 5. AIの過去予測の実績検証（直近1ヶ月の推奨台勝率） ---
            df_pred_log = backend.load_prediction_log()
            if not df_pred_log.empty and not df_raw.empty:
                try:
                    df_pred_log_temp = df_pred_log.copy()
                    if '予測対象日' in df_pred_log_temp.columns:
                        df_pred_log_temp['予測対象日_merge'] = pd.to_datetime(df_pred_log_temp['予測対象日'], errors='coerce').fillna(pd.to_datetime(df_pred_log_temp['対象日付'], errors='coerce') + pd.Timedelta(days=1))
                    else:
                        df_pred_log_temp['予測対象日_merge'] = pd.to_datetime(df_pred_log_temp['対象日付'], errors='coerce') + pd.Timedelta(days=1)
                    
                    shop_col_pred = '店名' if '店名' in df_pred_log_temp.columns else '店舗名'
                    df_pred_log_shop = df_pred_log_temp[df_pred_log_temp[shop_col_pred] == selected_shop].copy()
                    
                    if not df_pred_log_shop.empty:
                        df_pred_log_shop['台番号'] = df_pred_log_shop['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
                        if '実行日時' in df_pred_log_shop.columns:
                            df_pred_log_shop = df_pred_log_shop.sort_values('実行日時', ascending=False).drop_duplicates(subset=['予測対象日_merge', '台番号'], keep='first')
                            
                        df_raw_temp = df_raw[df_raw[shop_col] == selected_shop].copy()
                        df_raw_temp['台番号'] = df_raw_temp['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
                        df_raw_temp['対象日付'] = pd.to_datetime(df_raw_temp['対象日付'], errors='coerce')
                        
                        merged_acc = pd.merge(
                            df_pred_log_shop, df_raw_temp,
                            left_on=['予測対象日_merge', '台番号'], right_on=['対象日付', '台番号'],
                            how='inner', suffixes=('_pred', '_raw')
                        )
                        
                        if not merged_acc.empty and 'prediction_score' in merged_acc.columns:
                            cutoff_date = merged_acc['予測対象日_merge'].max() - pd.Timedelta(days=30)
                            merged_acc = merged_acc[merged_acc['予測対象日_merge'] > cutoff_date].copy()
                            merged_acc['prediction_score'] = pd.to_numeric(merged_acc['prediction_score'], errors='coerce')
                            
                            merged_acc['daily_rank'] = merged_acc.groupby('予測対象日_merge')['prediction_score'].rank(method='first', ascending=False)
                            top_k = max(3, min(10, int(df_raw_temp['台番号'].nunique() * 0.10)))
                            
                            high_expect_df = merged_acc[(merged_acc['daily_rank'] <= top_k) | (merged_acc['prediction_score'] >= 0.70)].copy()
                            
                            if not high_expect_df.empty:
                                act_g = pd.to_numeric(high_expect_df['累計ゲーム'], errors='coerce').fillna(0)
                                act_diff = pd.to_numeric(high_expect_df['差枚'], errors='coerce').fillna(0)
                                
                                high_expect_df['valid_play'] = (act_g >= 3000) | ((act_g < 3000) & ((act_diff <= -750) | (act_diff >= 750)))
                                high_expect_df['valid_win'] = high_expect_df['valid_play'] & (act_diff > 0)
                                
                                valid_count = high_expect_df['valid_play'].sum()
                                win_count = high_expect_df['valid_win'].sum()
                                win_rate = (win_count / valid_count * 100) if valid_count > 0 else 0
                                
                                context_data += f"\n【{selected_shop} のAI予測実績 (直近1ヶ月)】\n"
                                context_data += f"AI推奨台の勝率 (差枚プラス割合): {win_rate:.1f}% (有効稼働 {int(valid_count)}台中 {int(win_count)}台)\n"
                                if win_rate >= 50:
                                    context_data += "AI自己評価: 予測が店舗の傾向とよく噛み合っており、非常に信頼できる状態です。\n"
                                elif win_rate >= 35:
                                    context_data += "AI自己評価: 予測は標準的な精度です。店癖などの他要素も加味して狙い台を絞るのがおすすめです。\n"
                                else:
                                    context_data += "AI自己評価: 予測がフェイク設定等に騙されやすく、精度が低迷しています。稼働を控えるか、強イベント時のみに絞るなどの警戒が必要です。\n"
                except Exception:
                    pass

            # --- 6. 機種スペック情報 (リアルタイム判別相談用) ---
            specs = backend.get_machine_specs()
            context_data += "\n【主要機種の設定5目安 (リアルタイム判別相談用)】\n"
            for m_name in ['アイムジャグラーEX', 'マイジャグラーV', 'ゴーゴージャグラー3', 'ファンキージャグラー2KT', 'ハッピージャグラーVIII', 'ジャグラーガールズSS']:
                if m_name in specs and '設定5' in specs[m_name]:
                    s5_reg = specs[m_name]['設定5'].get('REG', 0)
                    s5_tot = specs[m_name]['設定5'].get('合算', 0)
                    context_data += f"・{m_name}: REG 1/{s5_reg:.1f}, 合算 1/{s5_tot:.1f}\n"

        # --- マイ収支データをAIに読み込ませる ---
        df_balance = backend.load_my_balance()
        if not df_balance.empty:
            # 通算成績のサマリーを追加
            total_balance = df_balance['収支'].sum()
            win_count = (df_balance['収支'] > 0).sum()
            total_count = len(df_balance)
            win_rate = (win_count / total_count * 100) if total_count > 0 else 0
            context_data += f"\n【あなたの通算成績】\n総稼働数: {total_count}回 / 勝率: {win_rate:.1f}% / 総収支: {total_balance:+d}円\n"

            # 直近10件の収支履歴をテキスト化して渡す
            recent_balance = df_balance.sort_values('日付', ascending=False).head(10)
            context_data += "\n【あなたの直近の稼働収支データ (最新10件)】\n"
            context_data += recent_balance[['日付', '店名', '機種名', '収支', 'メモ']].to_markdown(index=False) + "\n"

        # 過去の会話履歴をプロンプト用に構築
        history_text = ""
        for msg in st.session_state.gemini_messages[:-1]: # 最新の質問以外
            role_name = "ユーザー" if msg["role"] == "user" else "コンサルタント"
            history_text += f"{role_name}: {msg['content']}\n"

        # AIへの指示（システムプロンプト）の組み立て
        now_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y年%m月%d日 %H:%M')
        full_prompt = f"""
あなたはパチンコ・スロットホールに勤務する、丁寧で親切な受付案内スタッフのお姉さんです。
ユーザーをご来店されたお客様として扱い、上品で柔らかい敬語（「〜ですね」「〜いたしますね」など）を使ってアドバイスしてください。
偉そうな態度は絶対に避け、お客様に寄り添うように優しく案内する口調を徹底してください。
論理的なデータ分析を踏まえつつも、専門用語を使いすぎず分かりやすく説明してください。
回答は長くなりすぎないよう、箇条書きなどを活用して「要点だけを簡潔に」まとめてください。

現在日時: {now_str}
※提供されているデータは【{target_date_str}】の予測・イベント情報です。
お客様の質問が「今日」や「明日」といった曖昧な日付表現であっても、基本的には提供されている【{target_date_str}】のデータに関する相談と解釈して、その日付を基準に回答してください。

【このアプリの予測AIの評価ルール】
・「パチンコ専用」や「他機種」イベント時、システムは一旦「シワ寄せ回収リスク」としてマイナスのイベントスコアを与えます。
・しかし最終的な予測期待度は、そのスコアを踏まえて「過去その店舗が実際にどういう営業をしたか」をAI（機械学習）が学習して算出しています。
・そのため、一般的には危険なイベントでも「過去の傾向から見てこの店は出している」とAIが判断すれば高い期待度が出力されます。
・アドバイスの際は、「一般的には回収リスクがあるイベント」であることを前提にしつつも、最終的には「AIの予測データや店癖がそれに対してどういう答えを出しているか」を最も重視して回答してください。
・お客様が「パチンコや他機種のイベントの日の状況はどう？」と質問された場合、他機種自体の出玉状況ではなく「そのイベントのシワ寄せでジャグラー（スロット）が回収されているか、あるいは還元されているか」という情報を求めています。提供されている「過去イベント時のジャグラー平均差枚」のデータを見て、ジャグラーが回収されているか還元されているかを的確に回答してください。
・「AI予測実績（勝率や自己評価）」のデータが提供されている場合は、AI自身の実力が現在その店舗でどれくらい通用しているかを客観的に捉え、勝率が低ければ無理に推奨台を薦めず「今は予測が当たりにくい危険な状態なので様子見が無難」といった警告も行ってください。
・店舗の「強い末尾番号」や「強い曜日」のデータが提供されている場合は、それらも具体的な狙い目の根拠としてアドバイスに積極的に盛り込んでください。
・お客様が稼働中の台のデータ（機種名、現在の総回転数、BIG回数、REG回数など）を提示して相談された場合は、提供されている「主要機種の設定5目安」を基準にして現在の確率を自ら計算し、高設定の期待度や押し引き（ヤメ時）の的確なアドバイスを行ってください。
・お客様から特定の「台番号」について相談された場合、提供されている「予測データ (全台の期待度ランキング)」からその台の事前期待度と根拠を確認して回答してください。もし上位10台に含まれていれば「個別データ(直近3日間の履歴)」も交えて具体的にアドバイスし、逆に期待度が低い台（期待外れ・危険台）の場合はその旨と低評価の根拠を正直に伝えて撤退などの注意を促してください。

以下は現在のアプリ内のデータです。最新の店舗状況として参考にしてください。
{context_data if context_data else "現在選択されている店舗データはありません。"}

【これまでの会話履歴】
{history_text if history_text else "なし"}

ユーザーの最新の質問:
{prompt}
"""

        # --- Gemini APIの呼び出し ---
        with st.chat_message("assistant"):
            with st.spinner("Geminiが分析中..."):
                try:
                    # --- 利用可能なモデルをAPIから自動取得して最適なものを選択 ---
                    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    
                    target_model = None
                    # 優先順位1: 1.5 flash 系を名前の一部から探す
                    for m in available_models:
                        if '1.5-flash' in m:
                            target_model = m.replace('models/', '')
                            break
                    # 優先順位2: なければ 1.5 pro 系を探す
                    if not target_model:
                        for m in available_models:
                            if '1.5-pro' in m:
                                target_model = m.replace('models/', '')
                                break
                    # フォールバック: とにかく一番最初に見つかったモデルを強制的に使う
                    if not target_model and len(available_models) > 0:
                        target_model = available_models[0].replace('models/', '')
                    
                    if not target_model:
                        raise Exception(f"利用可能なGeminiモデルが見つかりませんでした。利用可能リスト: {available_models}")
                        
                    # 自動選択されたモデルで呼び出し
                    model = genai.GenerativeModel(target_model)
                    response = model.generate_content(full_prompt)
                    
                    # 回答の表示と履歴への追加
                    st.markdown(response.text)
                    st.session_state.gemini_messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"APIリクエスト中にエラーが発生しました: {e}")
                    
        # 履歴が長くなりすぎてAPIコストや制限に引っかかるのを防ぐ
        if len(st.session_state.gemini_messages) > 20:
            st.session_state.gemini_messages = st.session_state.gemini_messages[-20:]