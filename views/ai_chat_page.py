import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import backend
import time
from utils import get_valid_play_mask
from config import BASE_FEATURES, FEATURE_NAME_MAP

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# --- チャット履歴のローカル保存用設定 ---
CHAT_HISTORY_FILE = "ai_chat_history.json"

def load_chat_history():
    """ローカルファイルからチャット履歴を読み込む"""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_history(messages):
    """チャット履歴をローカルファイルに保存する"""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def render_ai_chat_page(df_predict, df_raw, shop_col, df_verify, df_events=None, df_importance=None, shop_hyperparams=None):
    # 会話履歴の初期化 (ファイルから復元)
    if "gemini_messages" not in st.session_state:
        st.session_state.gemini_messages = load_chat_history()

    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.header("👩‍💼 ホール案内スタッフ（AI）")
    with col_h2:
        if st.session_state.gemini_messages:
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ リセット", use_container_width=True, help="会話履歴を破棄します"):
                st.session_state.gemini_messages = []
                save_chat_history([])
                st.rerun()

    chat_mode = st.radio(
        "🗣️ 相談相手を選択", 
        ["👩‍💼 案内スタッフ (立ち回り・店舗選び)", "👨‍🔬 データアナリスト (AI予測モデルの評価・チューニング)"], 
        horizontal=True,
        help="「案内スタッフ」は日々の勝負の立ち回りを、「データアナリスト」はAIの設定パラメータ変更や精度改善のアドバイスを行います。"
    )

    st.caption("受付スタッフ風のAIに、アプリの最新データをもとにした立ち回りの相談や店舗の傾向についてチャットで質問できます。")

    # --- スマホ向けUI調整 (CSS) ---
    st.markdown("""
        <style>
            /* チャットメッセージのフォントサイズと行間をスマホ向けに調整 */
            [data-testid="stChatMessage"] {
                font-size: 0.75rem;
            }
            [data-testid="stChatMessage"] p {
                margin-bottom: 0.2rem;
                line-height: 1.3;
            }
            /* チャット入力欄のフォントサイズ調整 */
            [data-testid="stChatInput"] textarea {
                font-size: 0.9rem;
            }
            /* AIの回答がリスト(箇条書き)の場合の余白調整 */
            [data-testid="stChatMessage"] ul, [data-testid="stChatMessage"] ol {
                padding-left: 1.0rem;
                margin-bottom: 0.2rem;
            }
            [data-testid="stChatMessage"] li {
                margin-bottom: 0.1rem;
            }
            /* 下へスクロールするボタン */
            .scroll-down-btn {
                position: fixed;
                bottom: 150px;
                right: 20px;
                background-color: rgba(66, 165, 245, 0.8);
                color: white !important;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                text-align: center;
                line-height: 40px;
                font-size: 18px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                text-decoration: none;
                z-index: 9999;
                transition: background-color 0.3s;
            }
            .scroll-down-btn:hover {
                background-color: rgba(30, 136, 229, 1);
            }
        </style>
        <a href="#chat-bottom" class="scroll-down-btn" title="最新のメッセージへ">⬇️</a>
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
        if "データアナリスト" in chat_mode:
            st.warning("⚠️ データアナリストにAIモデルの精度評価やチューニングの相談をするには、上のプルダウンから**分析対象の店舗を選択**してください。")
        else:
            st.info("店舗を選択せずに質問すると、一般的な立ち回りの相談や「今日はどの店舗に行くべき？」といった店舗選びの相談ができます。")

    st.divider()
    
    # --- APIキーで利用可能なモデルを動的取得して選択肢を作成 ---
    @st.cache_data(ttl=3600)
    def get_available_models():
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        try:
            return [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except Exception:
            return []
            
    available_models = get_available_models()
    model_options = {}
    
    flash_15 = next((m for m in available_models if "1.5-flash" in m and "8b" not in m), None)
    pro_15 = next((m for m in available_models if "1.5-pro" in m), None)
    flash_25 = next((m for m in available_models if "2.5-flash" in m), None)
    
    if flash_15: model_options[f"{flash_15} (推奨・制限緩め)"] = flash_15
    if pro_15: model_options[f"{pro_15} (高性能)"] = pro_15
    if flash_25: model_options[f"{flash_25} (最新・制限厳しめ)"] = flash_25
        
    if not model_options and available_models:
        for m in available_models: model_options[m] = m
            
    if not model_options:
        model_options = {"gemini-1.5-flash-latest (推奨)": "gemini-1.5-flash-latest", "gemini-1.5-pro-latest": "gemini-1.5-pro-latest"}

    selected_model_label = st.selectbox(
        "🧠 使用するAIモデルを選択", list(model_options.keys()), index=0,
        help="現在お使いのAPIキーで実際に利用可能なモデルを自動取得して表示しています。"
    )
    target_model = model_options[selected_model_label]

    # --- チャット履歴の表示 ---
    for msg in st.session_state.gemini_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

    # --- ユーザーからの入力 ---
    if prompt := st.chat_input("質問を入力してください (例: 明日の〇〇店の狙い目は？ / 今日の立ち回りアドバイスをお願い)"):
        
        # ユーザーの質問を表示し、履歴に追加
        st.chat_message("user").markdown(prompt)
        st.session_state.gemini_messages.append({"role": "user", "content": prompt})
        save_chat_history(st.session_state.gemini_messages)

        # --- アプリ内のデータをGemini用に文字列化して準備 ---
        context_data = ""
        
        # --- カンニングなしテストを実行し、店舗の「スイートスポット」をAIに教える ---
        backtest_summary = ""
        if selected_shop != "店舗を選択してください" and not df_verify.empty:
            try:
                shop_df = df_verify[df_verify[shop_col] == selected_shop].copy()
                if len(shop_df) >= 50:
                    actual_features = [f for f in BASE_FEATURES if f in shop_df.columns]
                    cat_features = [f for f in ['machine_code', 'shop_code', 'event_code', 'target_weekday', 'target_date_end_digit'] if f in actual_features]
                    
                    shop_df['対象日付'] = pd.to_datetime(shop_df['対象日付'])
                    max_date = shop_df['対象日付'].max()
                    cutoff_date = max_date - pd.Timedelta(days=30)
                    train_data = shop_df[shop_df['対象日付'] <= cutoff_date].copy()
                    test_data = shop_df[shop_df['対象日付'] > cutoff_date].copy()

                    if len(train_data) >= 30 and len(test_data) >= 10:
                        X_train, y_train = train_data[actual_features], train_data['target']
                        X_test, y_test = test_data[actual_features], test_data['target']
                        
                        days_diff = (train_data['対象日付'].max() - train_data['対象日付']).dt.days
                        sample_weights = 0.995 ** days_diff
                        
                        params = shop_hyperparams.get(selected_shop, shop_hyperparams.get("デフォルト", {}))
                        
                        reg_model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
                        reg_model.fit(X_train, train_data['next_diff'], sample_weight=sample_weights, categorical_feature=cat_features)
                        
                        X_train_st = X_train.copy(); X_train_st['predicted_diff'] = reg_model.predict(X_train)
                        model = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1, **params, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
                        model.fit(X_train_st, y_train, sample_weight=sample_weights, categorical_feature=cat_features)
                        
                        X_test_st = X_test.copy(); X_test_st['predicted_diff'] = reg_model.predict(X_test)
                        preds = model.predict_proba(X_test_st)[:, 1]
                        test_data['pred_score'] = preds

                        test_data['valid_play'] = get_valid_play_mask(test_data['next_累計ゲーム'], test_data['next_diff'])
                        test_data['valid_win'] = test_data['valid_play'] & (pd.to_numeric(test_data['next_diff'], errors='coerce').fillna(0) > 0)
                        
                        def get_prob_band(score):
                            if score >= 0.50: return '50%以上'
                            elif score >= 0.40: return '40%〜49%'
                            elif score >= 0.30: return '30%〜39%'
                            elif score >= 0.20: return '20%〜29%'
                            else: return '20%未満'
                        test_data['確率帯'] = test_data['pred_score'].apply(get_prob_band)
                        
                        test_stats = test_data.groupby('確率帯').agg(有効稼働数=('valid_play', 'sum'), 勝数=('valid_win', 'sum'), 平均差枚=('next_diff', 'mean')).reset_index()
                        test_stats['勝率'] = np.where(test_stats['有効稼働数'] > 0, test_stats['勝数'] / test_stats['有効稼働数'] * 100, 0.0)
                        
                        test_stats_filtered = test_stats[test_stats['有効稼働数'] >= 5]
                        if not test_stats_filtered.empty:
                            best_band_row = test_stats_filtered.loc[test_stats_filtered['勝率'].idxmax()]
                            backtest_summary = f"\n【最重要・カンニングなしテストによるAI実力分析】\nこの店舗のAI予測では、期待度が「{best_band_row['確率帯']}」の台を狙うのが最も勝率が高く({best_band_row['勝率']:.1f}%)、平均差枚も{int(best_band_row['平均差枚']):+d}枚と優秀です。AIに相談する際は、この確率帯を狙うように指示すると効果的です。\n"
            except Exception:
                pass # エラー時は何もせずスキップ
        context_data += backtest_summary

        # --- 予測対象日の取得 (サイドバーで選んだ日付と一致させる) ---
        if not df_predict.empty and 'next_date' in df_predict.columns:
            target_date_val = df_predict['next_date'].max()
        else:
            target_date_val = pd.Timestamp.now(tz='Asia/Tokyo').date()
        target_date_str = pd.to_datetime(target_date_val).strftime('%Y-%m-%d')
        
        if selected_shop != "店舗を選択してください":
            
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
            
                    # 店舗全体の還元日/回収日の傾向
                    shop_daily_df = shop_raw.groupby('対象日付').agg(店舗平均差枚=('差枚', 'mean')).reset_index()
                    shop_daily_df['曜日'] = shop_daily_df['対象日付'].dt.dayofweek
                    shop_daily_df['末尾'] = shop_daily_df['対象日付'].dt.day % 10
                    
                    if df_events is not None and not df_events.empty:
                        events_shop = df_events[df_events['店名'] == selected_shop].drop_duplicates(subset=['イベント日付'], keep='last')
                        events_shop['イベント日付'] = pd.to_datetime(events_shop['イベント日付'])
                        shop_daily_df = pd.merge(shop_daily_df, events_shop[['イベント日付', 'イベントランク']], left_on='対象日付', right_on='イベント日付', how='left')
                        shop_daily_df['イベント有無'] = shop_daily_df['イベント日付'].notna().map({True: 'イベント日', False: '通常日'})
                        shop_daily_df['イベントランク'] = shop_daily_df['イベントランク'].fillna('通常営業')
                    else:
                        shop_daily_df['イベント有無'] = '通常日'
                        shop_daily_df['イベントランク'] = '通常営業'
                    
                    wd_stats = shop_daily_df.groupby('曜日')['店舗平均差枚'].mean()
                    digit_stats = shop_daily_df.groupby('末尾')['店舗平均差枚'].mean()
                    ev_stats = shop_daily_df.groupby('イベント有無')['店舗平均差枚'].mean()
                    rank_stats = shop_daily_df.groupby('イベントランク')['店舗平均差枚'].mean().sort_values(ascending=False)
                    
                    if not wd_stats.empty and not digit_stats.empty:
                        weekdays_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
                        best_wd = wd_stats.idxmax()
                        worst_wd = wd_stats.idxmin()
                        best_digit = digit_stats.idxmax()
                        worst_digit = digit_stats.idxmin()
                        
                        context_data += f"\n【{selected_shop} の全体的な還元/回収の傾向 (過去実績)】\n"
                        context_data += f"・最も甘い(還元)曜日: {weekdays_map[best_wd]}曜日 (平均 {int(wd_stats[best_wd]):+d}枚)\n"
                        context_data += f"・最も辛い(回収)曜日: {weekdays_map[worst_wd]}曜日 (平均 {int(wd_stats[worst_wd]):+d}枚)\n"
                        context_data += f"・最も甘い(還元)特定日: {best_digit}のつく日 (平均 {int(digit_stats[best_digit]):+d}枚)\n"
                        context_data += f"・最も辛い(回収)特定日: {worst_digit}のつく日 (平均 {int(digit_stats[worst_digit]):+d}枚)\n"
                        if 'イベント日' in ev_stats.index and '通常日' in ev_stats.index:
                            context_data += f"・イベント日の営業: イベント日 平均 {int(ev_stats['イベント日']):+d}枚 / 通常日 平均 {int(ev_stats['通常日']):+d}枚\n"
                            rank_strs = []
                            for r_name, r_val in rank_stats.items():
                                if r_name != '通常営業':
                                    rank_strs.append(f"{r_name}: {int(r_val):+d}枚")
                            if rank_strs:
                                context_data += f"  (イベントランク別実績: {', '.join(rank_strs)})\n"
            
            # --- 1.8. 日替わり特定パターンの投入頻度（末尾・機種・並び・角台） ---
            if not df_raw.empty and shop_col in df_raw.columns:
                shop_raw_pat = df_raw[df_raw[shop_col] == selected_shop].copy()
                if not shop_raw_pat.empty and '対象日付' in shop_raw_pat.columns and '差枚' in shop_raw_pat.columns:
                    shop_daily_stats = shop_raw_pat.groupby('対象日付').agg(店舗平均差枚=('差枚', 'mean')).reset_index()
                    total_days = len(shop_daily_stats)
                    
                    if total_days > 0:
                        pattern_str = f"\n【{selected_shop} の日替わり特定パターンの投入頻度 (全{total_days}日中)】\n"
                        
                        # 1. 当たり末尾
                        if '末尾番号' in shop_raw_pat.columns:
                            end_daily = shop_raw_pat.groupby(['対象日付', '末尾番号']).agg(末尾平均=('差枚', 'mean'), 台数=('台番号', 'count')).reset_index()
                            end_daily = end_daily[end_daily['台数'] >= 3]
                            end_daily = end_daily.dropna(subset=['末尾平均'])
                            if not end_daily.empty:
                                idx_max = end_daily.groupby('対象日付')['末尾平均'].idxmax()
                                top_end = end_daily.loc[idx_max]
                                merged_end = pd.merge(top_end, shop_daily_stats, on='対象日付')
                                hit_end = ((merged_end['末尾平均'] >= 500) & ((merged_end['末尾平均'] - merged_end['店舗平均差枚']) >= 500)).sum()
                                pattern_str += f"・日替わり当たり末尾 投入率: {(hit_end / total_days * 100):.1f}%\n"

                        # 2. 全台系(当たり機種)
                        if '機種名' in shop_raw_pat.columns:
                            mac_daily = shop_raw_pat.groupby(['対象日付', '機種名']).agg(機種平均=('差枚', 'mean'), 台数=('台番号', 'count')).reset_index()
                            mac_daily = mac_daily[mac_daily['台数'] >= 3]
                            mac_daily = mac_daily.dropna(subset=['機種平均'])
                            if not mac_daily.empty:
                                idx_max = mac_daily.groupby('対象日付')['機種平均'].idxmax()
                                top_mac = mac_daily.loc[idx_max]
                                merged_mac = pd.merge(top_mac, shop_daily_stats, on='対象日付')
                                hit_mac = ((merged_mac['機種平均'] >= 1000) & ((merged_mac['機種平均'] - merged_mac['店舗平均差枚']) >= 500)).sum()
                                pattern_str += f"・全台系(当たり機種) 投入率: {(hit_mac / total_days * 100):.1f}%\n"

                        # 3. 並び(塊)
                        if '台番号' in shop_raw_pat.columns:
                            n_df = shop_raw_pat[['対象日付', '台番号', '差枚']].copy()
                            n_df['台番号'] = pd.to_numeric(n_df['台番号'], errors='coerce')
                            n_df = n_df.dropna(subset=['台番号']).sort_values(['対象日付', '台番号'])
                            hit_narabi = 0
                            for _, group in n_df.groupby('対象日付'):
                                group['is_hot'] = group['差枚'] >= 1000
                                group['block'] = (group['is_hot'] != group['is_hot'].shift()).cumsum()
                                hot_blocks = group[group['is_hot']].groupby('block').size()
                                if not hot_blocks.empty and hot_blocks.max() >= 3:
                                    hit_narabi += 1
                            pattern_str += f"・3台以上の並び(塊) 投入率: {(hit_narabi / total_days * 100):.1f}%\n"

                        # 4. 角台優遇
                        if 'is_corner' in shop_raw_pat.columns:
                            corner_daily = shop_raw_pat[shop_raw_pat['is_corner'] == 1].groupby('対象日付').agg(角台平均=('差枚', 'mean')).reset_index()
                            merged_corner = pd.merge(corner_daily, shop_daily_stats, on='対象日付')
                            hit_corner = ((merged_corner['角台平均'] >= 500) & ((merged_corner['角台平均'] - merged_corner['店舗平均差枚']) >= 500)).sum()
                            pattern_str += f"・角台の優遇 発生率: {(hit_corner / total_days * 100):.1f}%\n"
                            
                        context_data += pattern_str
            
            # --- 1.9. 回収日に甘い機種ランキング ---
            df_scores = backend.load_daily_shop_scores()
            if not df_scores.empty and '予測対象日' in df_scores.columns and not shop_raw.empty:
                df_scores['対象日付'] = pd.to_datetime(df_scores['予測対象日'], errors='coerce').dt.normalize()
                score_shop_col = '店名' if '店名' in df_scores.columns else ('店舗名' if '店舗名' in df_scores.columns else None)
                
                if score_shop_col:
                    shop_scores = df_scores[df_scores[score_shop_col] == selected_shop]
                    cold_dates = shop_scores[shop_scores['店舗平均期待度'] < 0.10]['対象日付'].tolist()
                    
                    if cold_dates:
                        cold_day_data = shop_raw[shop_raw['対象日付'].dt.normalize().isin(cold_dates)].copy()
                        
                        if not cold_day_data.empty:
                            mac_df = cold_day_data
                            mac_df['REG確率_raw'] = mac_df['REG'] / mac_df['累計ゲーム'].replace(0, np.nan)
                            mac_df['valid_play'] = get_valid_play_mask(mac_df['累計ゲーム'], mac_df['差枚'])
                            
                            specs = backend.get_machine_specs()
                            mac_df['合算確率'] = (mac_df['BIG'] + mac_df['REG']) / mac_df['累計ゲーム'].replace(0, np.nan)
                            spec_reg = mac_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
                            spec_tot = mac_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
                            spec_reg3 = mac_df['機種名'].apply(lambda x: 1.0 / specs[backend.get_matched_spec_key(x, specs)].get('設定3', {"REG": 300.0})["REG"])
                            
                            mac_df['高設定挙動'] = ((mac_df['累計ゲーム'] >= 3000) & ((mac_df['REG確率_raw'] >= spec_reg) | ((mac_df['合算確率'] >= spec_tot) & (mac_df['REG確率_raw'] >= spec_reg3)))).astype(int)
                            mac_df['高設定率'] = np.where(mac_df['valid_play'], mac_df['高設定挙動'], np.nan) * 100
                            
                            mac_df['valid_差枚'] = np.where(mac_df['valid_play'], mac_df['差枚'], np.nan)
                            mac_df['REG確率_val'] = np.where(mac_df['累計ゲーム'] > 0, mac_df['REG'] / mac_df['累計ゲーム'], 0)
                            mac_df['valid_REG確率'] = np.where(mac_df['valid_play'], mac_df['REG確率_val'], np.nan)

                            cold_mac_stats = mac_df.groupby('機種名').agg(
                                平均差枚=('valid_差枚', 'mean'), 高設定率=('高設定率', 'mean'), 平均REG確率=('valid_REG確率', 'mean'), サンプル数=('台番号', 'count')
                            ).reset_index().sort_values('平均差枚', ascending=False)
                            
                            if not cold_mac_stats.empty:
                                cold_mac_stats['REG確率'] = cold_mac_stats['平均REG確率'].apply(lambda x: f"1/{int(1/x)}" if pd.notna(x) and x > 0 else "-")
                                cold_mac_stats['平均差枚'] = cold_mac_stats['平均差枚'].apply(lambda x: f"{int(x):+d}" if pd.notna(x) else "-")
                                cold_mac_stats['高設定率'] = cold_mac_stats['高設定率'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
                                
                                context_data += f"\n【{selected_shop} の「回収日」に甘い機種ランキング (AIが回収日と予測した日の実績)】\n"
                                context_data += cold_mac_stats[['機種名', '平均差枚', '高設定率', 'REG確率', 'サンプル数']].to_markdown(index=False) + "\n"

            if df_events is not None and not df_events.empty:
                shop_events = df_events[(df_events['店名'] == selected_shop) & (df_events['イベント日付'].dt.strftime('%Y-%m-%d') == target_date_str)]
                if not shop_events.empty:
                    context_data += f"\n【{selected_shop} の {target_date_str} のイベント情報】\n"
                    for _, ev in shop_events.iterrows():
                        ev_name = ev.get('イベント名', '')
                        ev_rank = ev.get('イベントランク', '不明')
                        ev_type = ev.get('イベント種別', '全体')
                        ev_target = ev.get('対象機種', '指定なし')
                        
                        # --- 過去の同名イベントの実績を集計 ---
                        past_ev_stats_str = "過去実績なし"
                        if not df_raw.empty:
                            past_ev_dates = df_events[(df_events['店名'] == selected_shop) & (df_events['イベント名'] == ev_name) & (df_events['イベント日付'].dt.strftime('%Y-%m-%d') != target_date_str)]['イベント日付'].dt.strftime('%Y-%m-%d').tolist()
                            if past_ev_dates:
                                shop_raw_temp = df_raw[df_raw[shop_col] == selected_shop].copy()
                                shop_raw_temp['対象日付_str'] = pd.to_datetime(shop_raw_temp['対象日付'], errors='coerce').dt.strftime('%Y-%m-%d')
                                past_ev_raw = shop_raw_temp[shop_raw_temp['対象日付_str'].isin(past_ev_dates)]
                                if not past_ev_raw.empty:
                                    past_ev_avg_diff = past_ev_raw.groupby('対象日付_str')['差枚'].mean().mean()
                                    past_ev_stats_str = f"過去実績: 店舗平均 {int(past_ev_avg_diff):+d} 枚"
                                    
                        context_data += f"・{ev_name} (ランク: {ev_rank}, 種別: {ev_type}, 対象: {ev_target}) ｜ {past_ev_stats_str}\n"

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
                    context_data += f"\n【{selected_shop} の設定投入のクセ (AIが重視している特徴量上位10件)】\n"
                    context_data += "※ 相関がプラスなら「その値が大きいほど高設定になりやすい」、マイナスなら「値が小さいほど高設定になりやすい」ことを示します。\n"
                    count = 0
                    for _, row in imp_shop.iterrows():
                        f_key = row['feature']
                        if f_key in FEATURE_NAME_MAP:
                            f_name = FEATURE_NAME_MAP[f_key]
                            importance = row.get('importance', 0)
                            correlation = row.get('correlation', 0)
                            corr_str = f"プラス相関 (+{correlation:.2f})" if correlation >= 0 else f"マイナス相関 ({correlation:.2f})"
                            context_data += f"・{f_name} : 重要度 {importance:.0f} / {corr_str}\n"
                            count += 1
                            if count >= 10: break

            # --- 2.5. 現在のAIハイパーパラメータ設定 ---
            if shop_hyperparams:
                current_hp = shop_hyperparams.get(selected_shop, shop_hyperparams.get("デフォルト", {}))
                context_data += f"\n【{selected_shop} の現在のAIモデル設定 (パラメータ)】\n"
                context_data += f"・学習期間: 直近 {current_hp.get('train_months', 3)} ヶ月\n"
                context_data += f"・学習回数 (n_estimators): {current_hp.get('n_estimators', 300)}\n"
                context_data += f"・学習率 (learning_rate): {current_hp.get('learning_rate', 0.03)}\n"
                context_data += f"・葉の数 (num_leaves): {current_hp.get('num_leaves', 15)}\n"
                context_data += f"・深さ制限 (max_depth): {current_hp.get('max_depth', 4)}\n"
                context_data += f"・最小データ数 (min_child_samples): {current_hp.get('min_child_samples', 50)}\n"
                context_data += f"・L1正則化 (reg_alpha): {current_hp.get('reg_alpha', 0.0)}\n"
                context_data += f"・L2正則化 (reg_lambda): {current_hp.get('reg_lambda', 0.0)}\n"

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
                    # スプレッドシートから正確な店舗全体の平均期待度を取得
                    df_daily_scores = backend.load_daily_shop_scores()
                    avg_pred_diff = None
                    shop_machine_count = shop_pred['台番号'].nunique()
                    if not df_daily_scores.empty and '予測対象日' in df_daily_scores.columns:
                        try:
                            target_dt_str = pd.to_datetime(target_date_str).strftime('%Y-%m-%d')
                            temp_scores = df_daily_scores.copy()
                            temp_scores['予測対象日_str'] = pd.to_datetime(temp_scores['予測対象日'], errors='coerce').dt.strftime('%Y-%m-%d')
                            shop_daily = temp_scores[(temp_scores['店名'] == selected_shop) & (temp_scores['予測対象日_str'] == target_dt_str)]
                            if not shop_daily.empty and '予測平均差枚' in shop_daily.columns:
                                avg_pred_diff = shop_daily['予測平均差枚'].dropna().iloc[-1]
                                if '店舗台数' in shop_daily.columns and pd.notna(shop_daily['店舗台数'].iloc[-1]):
                                    shop_machine_count = shop_daily['店舗台数'].dropna().iloc[-1]
                        except Exception: pass
                            
                    if avg_pred_diff is None or pd.isna(avg_pred_diff):
                        avg_pred_diff = shop_pred['予測差枚数'].mean() if '予測差枚数' in shop_pred.columns else np.nan

                    context_data += f"\n【{selected_shop} の {target_date_str} の店舗全体AI期待度 (還元日/回収日の目安)】\n"
                    if pd.notna(avg_pred_diff):
                        eval_str = backend.classify_shop_eval(avg_pred_diff, shop_machine_count, is_prediction=True)
                        context_data += f"予測平均差枚: {int(avg_pred_diff):+d}枚 / AI評価: {eval_str}\n"
                    else:
                        context_data += "AI店舗評価: 不明\n"

                    context_data += f"\n【{selected_shop} の {target_date_str} の予測データ (期待度上位30台)】\n"
                    cols = ['台番号', '機種名', 'prediction_score', '根拠']
                    available_cols = [c for c in cols if c in shop_pred.columns]
                    
                    display_df = shop_pred[available_cols].head(30).copy()
                    if 'prediction_score' in display_df.columns:
                        display_df['prediction_score'] = display_df['prediction_score'].apply(lambda x: f"{int(x*100)}%")
                    
                    context_data += display_df.to_markdown(index=False) + "\n"

                    # --- 4.2. 島（列）ごとの期待度 ---
                    if 'island_id' in shop_pred.columns:
                        island_pred = shop_pred[shop_pred['island_id'] != "Unknown"].copy()
                        if not island_pred.empty:
                            island_pred['島名'] = island_pred['island_id'].apply(lambda x: str(x).split('_', 1)[1] if '_' in str(x) else str(x))
                            isl_stats = island_pred.groupby('島名')['prediction_score'].mean().sort_values(ascending=False)
                            context_data += f"\n【{selected_shop} の {target_date_str} の島(列)別 平均期待度 (どの島が熱いか)】\n"
                            for i_name, i_score in isl_stats.items():
                                eval_str = "還元島" if i_score >= 0.20 else "通常" if i_score >= 0.10 else "回収島"
                                context_data += f"・{i_name}: 平均期待度 {i_score*100:.1f}% ({eval_str})\n"

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

                    # --- 4.6. 全台の直近3日間の履歴データ（AIによる自由な条件検索・抽出用） ---
                    if not df_raw.empty:
                        shop_raw_all = df_raw[df_raw[shop_col] == selected_shop].copy()
                        if '対象日付' in shop_raw_all.columns:
                            shop_raw_all['対象日付'] = pd.to_datetime(shop_raw_all['対象日付'])
                            base_date = pd.to_datetime(target_date_val) - pd.Timedelta(days=1)
                            cutoff_date = base_date - pd.Timedelta(days=3)
                            hist_all_df = shop_raw_all[(shop_raw_all['対象日付'] > cutoff_date) & (shop_raw_all['対象日付'] <= base_date)].copy()
                            
                            if not hist_all_df.empty:
                                context_data += f"\n【{selected_shop} の全台の直近3日間の個別データ (条件検索・抽出用)】\n"
                                context_data += "※お客様から「特定の条件（REG確率、差枚、連敗など）を満たす台を探して」と依頼された場合は、このデータから条件に合致する台を探して回答してください。確率は(総回転数÷REG回数)等で計算してください。\n"
                                
                                hist_all_df['date_str'] = hist_all_df['対象日付'].dt.strftime('%m/%d')
                                hist_all_df['台番号'] = hist_all_df['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)
                                
                                def make_hist_line(r):
                                    g = int(r.get('累計ゲーム', 0))
                                    if g == 0: return "" # 文字数節約のため稼働0Gの台は省略
                                    diff = int(r.get('差枚', 0))
                                    b = int(r.get('BIG', 0))
                                    reg = int(r.get('REG', 0))
                                    mac = str(r.get('機種名', '')).replace('ジャグラー', 'J') # 文字数節約
                                    return f"[{r['date_str']}]#{r['台番号']}({mac}) {g}G B{b} R{reg} 差{diff:+d}"
                                    
                                hist_lines = hist_all_df.apply(make_hist_line, axis=1).tolist()
                                hist_lines = [line for line in hist_lines if line] # 空行を除外
                                
                                # AIが読み込みやすいよう、適度な件数ごとにパイプ(|)区切りで1行にまとめる
                                chunked_lines = [" | ".join(hist_lines[i:i+100]) for i in range(0, len(hist_lines), 100)]
                                context_data += "\n".join(chunked_lines) + "\n"

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
                            
                            # 店舗の平均期待度を取得して結合
                            df_daily_scores = backend.load_daily_shop_scores()
                            if not df_daily_scores.empty and '予測対象日' in df_daily_scores.columns:
                                df_daily_scores['予測対象日_merge'] = pd.to_datetime(df_daily_scores['予測対象日'], errors='coerce')
                                daily_scores_shop = df_daily_scores[df_daily_scores['店名'] == selected_shop].copy()
                                daily_scores_shop = daily_scores_shop.drop_duplicates(subset=['予測対象日_merge'], keep='last')
                                merged_acc = pd.merge(merged_acc, daily_scores_shop[['予測対象日_merge', '店舗平均期待度']], on='予測対象日_merge', how='left')
                            else:
                                merged_acc['店舗平均期待度'] = np.nan
                                
                            def classify_day(score):
                                if pd.isna(score): return "通常営業"
                                if score >= 0.20: return "還元日"
                                elif score < 0.10: return "回収日"
                                else: return "通常営業"
                                
                            merged_acc['営業予測'] = merged_acc['店舗平均期待度'].apply(classify_day)
                            
                            # 店舗全体の平均差枚を日別に計算して結合
                            daily_shop_diff = df_raw_temp.groupby('対象日付')['差枚'].mean().reset_index()
                            daily_shop_diff = daily_shop_diff.rename(columns={'対象日付': '予測対象日_merge', '差枚': '店舗全体平均差枚'})
                            merged_acc = pd.merge(merged_acc, daily_shop_diff, on='予測対象日_merge', how='left')

                            day_summary = merged_acc[['予測対象日_merge', '営業予測', '店舗全体平均差枚']].drop_duplicates()
                            day_stats = day_summary.groupby('営業予測').agg(
                                発生日数=('予測対象日_merge', 'count'),
                                店舗平均差枚=('店舗全体平均差枚', 'mean')
                            ).reset_index()
                            day_stats_dict = day_stats.set_index('営業予測').to_dict('index')
                            
                            merged_acc['daily_rank'] = merged_acc.groupby('予測対象日_merge')['prediction_score'].rank(method='first', ascending=False)
                            top_k = max(3, min(10, int(df_raw_temp['台番号'].nunique() * 0.10)))
                            
                            high_expect_df = merged_acc[merged_acc['daily_rank'] <= top_k].copy()
                            
                            if not high_expect_df.empty:
                                act_g = pd.to_numeric(high_expect_df['累計ゲーム'], errors='coerce').fillna(0)
                                act_diff = pd.to_numeric(high_expect_df['差枚'], errors='coerce').fillna(0)
                                act_b = pd.to_numeric(high_expect_df['BIG'], errors='coerce').fillna(0)
                                act_r = pd.to_numeric(high_expect_df['REG'], errors='coerce').fillna(0)
                                
                                high_expect_df['valid_play'] = (act_g >= 3000) | ((act_g < 3000) & ((act_diff <= -750) | (act_diff >= 750)))
                                high_expect_df['valid_win'] = high_expect_df['valid_play'] & (act_diff > 0)
                                
                                specs = backend.get_machine_specs()
                                spec_reg_val = high_expect_df['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"REG": 260.0})["REG"])
                                spec_tot_val = high_expect_df['機種名'].apply(lambda x: specs[backend.get_matched_spec_key(x, specs)].get('設定5', {"合算": 128.0})["合算"])
                                
                                reg_prob_den = np.where(act_r > 0, act_g / act_r, 0)
                                tot_prob_den = np.where((act_b + act_r) > 0, act_g / (act_b + act_r), 0)
                                
                                high_expect_df['valid_high_play'] = (act_g >= 3000)
                                high_expect_df['is_high_setting'] = (
                                    (((reg_prob_den > 0) & (reg_prob_den <= spec_reg_val)) | 
                                     ((tot_prob_den > 0) & (tot_prob_den <= spec_tot_val)))
                                ).astype(int)
                                high_expect_df['valid_high'] = high_expect_df['valid_high_play'] & (high_expect_df['is_high_setting'] == 1)

                                valid_count = high_expect_df['valid_play'].sum()
                                win_count = high_expect_df['valid_win'].sum()
                                win_rate = (win_count / valid_count * 100) if valid_count > 0 else 0
                                
                                context_data += f"\n【{selected_shop} のAI予測実績と還元/回収日の傾向 (直近1ヶ月)】\n"
                                context_data += f"・全体: 推奨台勝率 {win_rate:.1f}% (有効稼働 {int(valid_count)}台中 {int(win_count)}台)\n"
                                
                                for day_type in ["還元日", "通常営業", "回収日"]:
                                    type_df = high_expect_df[high_expect_df['営業予測'] == day_type]
                                    t_val = type_df['valid_play'].sum()
                                    t_win = type_df['valid_win'].sum()
                                    t_high_val = type_df['valid_high_play'].sum()
                                    t_high = type_df['valid_high'].sum()
                                    
                                    d_stats = day_stats_dict.get(day_type, {})
                                    d_count = d_stats.get('発生日数', 0)
                                    d_diff = d_stats.get('店舗平均差枚', np.nan)
                                    
                                    if d_count > 0:
                                        diff_str = f"店舗全体平均 {int(d_diff):+d}枚" if pd.notna(d_diff) else "店舗差枚不明"
                                        win_str = f"推奨台勝率 {(t_win/t_val*100):.1f}%" if t_val > 0 else "推奨台勝率 -%"
                                        high_str = f"推奨台高設定率 {(t_high/t_high_val*100):.1f}%" if t_high_val > 0 else "推奨台高設定率 -%"
                                        
                                        context_data += f"・{day_type}予測 ({int(d_count)}日発生): {diff_str} / {win_str}, {high_str}\n"
                                
                                if valid_count < 30:
                                    context_data += "AI自己評価: まだ直近1ヶ月の検証台数（有効稼働）が少なく、たまたまのヒキでブレている可能性が高い状態です。設定は変更せずにもう少し様子を見てください。\n"
                                else:
                                    hot_df = high_expect_df[high_expect_df['営業予測'] == '還元日']
                                    hot_val = hot_df['valid_play'].sum()
                                    hot_win = hot_df['valid_win'].sum()
                                    hot_rate = (hot_win / hot_val * 100) if hot_val > 0 else win_rate
                                    
                                    if hot_rate >= 50:
                                        context_data += "AI自己評価: 還元日の勝率は良好で、AIは店の傾向を正しく読めています。回収日のフェイクに騙されて全体勝率が下がっているだけなので、還元日のみに絞って稼働すれば勝てます！設定の変更は不要です。\n"
                                    elif win_rate >= 50:
                                        context_data += "AI自己評価: 予測が店舗の傾向とよく噛み合っており、非常に信頼できる状態です。\n"
                                    elif win_rate >= 35:
                                        context_data += "AI自己評価: 予測は標準的な精度です。店癖などの他要素も加味して狙い台を絞るのがおすすめです。\n"
                                    else:
                                        context_data += "AI自己評価: 還元日でも勝率が低迷しており、予測がフェイク設定等に騙されています。学習期間を短くするなど設定のチューニングをおすすめします。\n"
                                        
                            # --- 5.5. AIの弱点分析（期待外れ台と逃したお宝台） ---
                            bad_pred = merged_acc[(merged_acc['prediction_score'] >= 0.65) & (merged_acc['差枚'] <= -1000)].copy()
                            missed_pred = merged_acc[(merged_acc['prediction_score'] < 0.20) & (merged_acc['差枚'] >= 2000)].copy()

                            if not bad_pred.empty or not missed_pred.empty:
                                context_data += f"\n【{selected_shop} のAI予測エラー分析 (直近1ヶ月)】\n"
                                if not bad_pred.empty:
                                    context_data += "・期待外れ台 (高評価だったが大負けした台):\n"
                                    for _, r in bad_pred.sort_values('差枚').head(5).iterrows():
                                        context_data += f"  - {r['予測対象日_merge'].strftime('%m/%d')} {r['機種名']} (台番号:{r['台番号']}) 期待度:{r['prediction_score']*100:.0f}% -> 結果:{int(r['差枚'])}枚 (総回転:{int(r['累計ゲーム'])}G BIG:{int(r['BIG'])} REG:{int(r['REG'])})\n"
                                if not missed_pred.empty:
                                    context_data += "・逃したお宝台 (低評価だったが大勝ちした台):\n"
                                    for _, r in missed_pred.sort_values('差枚', ascending=False).head(5).iterrows():
                                        context_data += f"  - {r['予測対象日_merge'].strftime('%m/%d')} {r['機種名']} (台番号:{r['台番号']}) 期待度:{r['prediction_score']*100:.0f}% -> 結果:{int(r['差枚'])}枚 (総回転:{int(r['累計ゲーム'])}G BIG:{int(r['BIG'])} REG:{int(r['REG'])})\n"
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
                    
        else:
            # 店舗未選択時（全体的な相談・店舗選び）
            context_data += "\n【本日の各店舗の予測サマリー (店舗選びの参考にしてください)】\n"
            if not df_predict.empty and shop_col in df_predict.columns and '予測差枚数' in df_predict.columns and 'prediction_score' in df_predict.columns:
                shop_summary = df_predict.groupby(shop_col).agg(
                    平均期待度=('prediction_score', 'mean'),
                    予測差枚数=('予測差枚数', 'mean')
                ).reset_index()
                
                # 過去の実績から「店舗の差枚がプラスだった日（還元日）」の平均稼働を計算して信頼度の担保にする
                if not df_raw.empty and shop_col in df_raw.columns:
                    shop_daily = df_raw.groupby([shop_col, '対象日付']).agg(
                        daily_diff=('差枚', 'mean'),
                        daily_games=('累計ゲーム', 'mean')
                    ).reset_index()
                    hot_days = shop_daily[shop_daily['daily_diff'] > 0]
                    if not hot_days.empty:
                        shop_hot_g = hot_days.groupby(shop_col)['daily_games'].mean().reset_index().rename(columns={'daily_games': '還元日平均稼働'})
                        shop_summary = pd.merge(shop_summary, shop_hot_g, on=shop_col, how='left')
                
                for _, r in shop_summary.iterrows():
                    eval_str = backend.classify_shop_eval(r.get('予測差枚数'), df_predict[df_predict[shop_col]==r[shop_col]]['台番号'].nunique(), is_prediction=True)
                    avg_g_str = f" / 過去還元日の平均稼働: {int(r['還元日平均稼働'])}G" if '還元日平均稼働' in shop_summary.columns and pd.notna(r['還元日平均稼働']) else ""
                    context_data += f"・{r[shop_col]}: 予測店舗平均 {int(r['予測差枚数']):+d}枚 / AI全体期待度 {r['平均期待度']*100:.1f}% ({eval_str}){avg_g_str}\n"
                    
                best_shop = shop_summary.loc[shop_summary['予測差枚数'].idxmax()]
                if best_shop['予測差枚数'] >= 0:
                    context_data += f"\n👉 AIのシステム上の本日のおすすめ店舗は、最も予測差枚が甘い「{best_shop[shop_col]}」です。\n"
                    
                # 本日の属性（曜日、末尾）に関する各店舗の過去実績を追加
                if not df_raw.empty and shop_col in df_raw.columns:
                    target_dt = pd.to_datetime(target_date_str)
                    target_wd = target_dt.dayofweek
                    target_digit = target_dt.day % 10
                    wd_str = ['月', '火', '水', '木', '金', '土', '日'][target_wd]
                    
                    df_raw_temp = df_raw.copy()
                    df_raw_temp['対象日付'] = pd.to_datetime(df_raw_temp['対象日付'])
                    df_raw_temp['曜日'] = df_raw_temp['対象日付'].dt.dayofweek
                    df_raw_temp['末尾'] = df_raw_temp['対象日付'].dt.day % 10
                    
                    shop_daily = df_raw_temp.groupby([shop_col, '対象日付', '曜日', '末尾']).agg(
                        店舗平均差枚=('差枚', 'mean'),
                        店舗平均稼働=('累計ゲーム', 'mean')
                    ).reset_index()
                    
                    wd_stats = shop_daily[shop_daily['曜日'] == target_wd].groupby(shop_col).agg(平均差枚=('店舗平均差枚', 'mean'), 平均稼働=('店舗平均稼働', 'mean')).reset_index()
                    digit_stats = shop_daily[shop_daily['末尾'] == target_digit].groupby(shop_col).agg(平均差枚=('店舗平均差枚', 'mean'), 平均稼働=('店舗平均稼働', 'mean')).reset_index()
                    
                    context_data += f"\n【本日の属性 ({target_dt.strftime('%m/%d')} {wd_str}曜 / {target_digit}のつく日) に対する各店舗の過去実績】\n"
                    for shop in shops:
                        if shop == "店舗を選択してください": continue
                        wd_row = wd_stats[wd_stats[shop_col] == shop]
                        digit_row = digit_stats[digit_stats[shop_col] == shop]
                        wd_text = f"{wd_str}曜の過去平均: {int(wd_row['平均差枚'].iloc[0]):+d}枚 (稼働 {int(wd_row['平均稼働'].iloc[0])}G)" if not wd_row.empty else f"{wd_str}曜の過去実績なし"
                        digit_text = f"{target_digit}のつく日の過去平均: {int(digit_row['平均差枚'].iloc[0]):+d}枚 (稼働 {int(digit_row['平均稼働'].iloc[0])}G)" if not digit_row.empty else f"{target_digit}のつく日の過去実績なし"
                        context_data += f"・{shop}: {wd_text} / {digit_text}\n"
                
                # --- 全店舗の中から期待度上位の台（おすすめ台）を追加 ---
                top_all = df_predict.sort_values('prediction_score', ascending=False).head(10)
                if not top_all.empty:
                    context_data += f"\n【全店舗の激アツ・おすすめ台 (期待度上位10台)】\n"
                    cols = [shop_col, '台番号', '機種名', 'prediction_score', '根拠']
                    available_cols = [c for c in cols if c in top_all.columns]
                    
                    display_df_all = top_all[available_cols].copy()
                    if 'prediction_score' in display_df_all.columns:
                        display_df_all['prediction_score'] = display_df_all['prediction_score'].apply(lambda x: f"{int(x*100)}%")
                    
                    context_data += display_df_all.to_markdown(index=False) + "\n"
            else:
                context_data += "本日の予測データがありません。\n"

        # --- 3日間のスケジュール用データを追加 ---
        context_data += f"\n【向こう3日間のスケジュール検討用データ ({target_date_str} から3日間)】\n"
        context_data += "以下のデータから、向こう3日間の各日の「最も期待値が高いおすすめ店舗」のスケジュールを提案できます。\n"
        
        target_dt = pd.to_datetime(target_date_str)
        weekdays_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
        
        shop_stats_by_wd = {}
        shop_stats_by_digit = {}
        if not df_raw.empty and shop_col in df_raw.columns:
            df_raw_temp = df_raw.copy()
            df_raw_temp['対象日付'] = pd.to_datetime(df_raw_temp['対象日付'])
            df_raw_temp['曜日'] = df_raw_temp['対象日付'].dt.dayofweek
            df_raw_temp['末尾'] = df_raw_temp['対象日付'].dt.day % 10
            
            shop_daily = df_raw_temp.groupby([shop_col, '対象日付', '曜日', '末尾']).agg(
                店舗平均差枚=('差枚', 'mean')
            ).reset_index()
            
            for shop in shops:
                if shop == "店舗を選択してください": continue
                shop_data = shop_daily[shop_daily[shop_col] == shop]
                if not shop_data.empty:
                    shop_stats_by_wd[shop] = shop_data.groupby('曜日')['店舗平均差枚'].mean().to_dict()
                    shop_stats_by_digit[shop] = shop_data.groupby('末尾')['店舗平均差枚'].mean().to_dict()
        
        for i in range(3):
            curr_dt = target_dt + pd.Timedelta(days=i)
            curr_wd = curr_dt.dayofweek
            curr_digit = curr_dt.day % 10
            curr_str = curr_dt.strftime('%m/%d')
            
            context_data += f"\n■ {curr_str} ({weekdays_map[curr_wd]}曜 / {curr_digit}のつく日)\n"
            
            # イベント情報
            if df_events is not None and not df_events.empty:
                day_events = df_events[df_events['イベント日付'].dt.date == curr_dt.date()]
                if not day_events.empty:
                    context_data += "  [イベント]\n"
                    for _, ev in day_events.iterrows():
                        ev_shop = ev.get('店名', '')
                        ev_name = ev.get('イベント名', '')
                        ev_rank = ev.get('イベントランク', '不明')
                        context_data += f"    - {ev_shop}: {ev_name} (ランク: {ev_rank})\n"
            
            # 過去実績に基づく有力店舗
            day_shop_scores = {}
            for shop in shops:
                if shop == "店舗を選択してください": continue
                wd_score = shop_stats_by_wd.get(shop, {}).get(curr_wd, np.nan)
                digit_score = shop_stats_by_digit.get(shop, {}).get(curr_digit, np.nan)
                
                scores = [s for s in [wd_score, digit_score] if pd.notna(s)]
                if scores:
                    day_shop_scores[shop] = max(scores) # 曜日と特定日の強い方を採用
            
            if day_shop_scores:
                sorted_shops = sorted(day_shop_scores.items(), key=lambda x: x[1], reverse=True)
                top_shops = [s for s in sorted_shops if s[1] > 0][:3] # プラスの店舗から上位3件
                if top_shops:
                    context_data += "  [過去実績に基づく有力店舗]\n"
                    for s_name, s_score in top_shops:
                        context_data += f"    - {s_name}: 期待差枚 +{int(s_score)}枚\n"
                else:
                    context_data += "  [過去実績に基づく有力店舗] 該当なし (全店舗マイナス傾向)\n"

        # --- マイ収支データをAIに読み込ませる ---
        df_balance = backend.load_my_balance()
        if not df_balance.empty:
            if '収支' in df_balance.columns:
                df_balance['収支'] = pd.to_numeric(df_balance['収支'], errors='coerce').fillna(0).astype(int)
                
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

            # --- 機種別の通算ボーナス成績をメモ欄から集計 ---
            if 'メモ' in df_balance.columns:
                import re
                temp_bal = df_balance.copy()
                
                def extract_my_data(x, kind):
                    m_new = re.search(r'自分稼働:(\d+)G BIG:(\d+) REG:(\d+)', x)
                    if m_new:
                        if kind == 'G': return int(m_new.group(1))
                        if kind == 'B': return int(m_new.group(2))
                        if kind == 'R': return int(m_new.group(3))
                    m_old = re.search(r'総回転:(\d+)G BIG:(\d+) REG:(\d+)', x)
                    if m_old:
                        if kind == 'G': return int(m_old.group(1))
                        if kind == 'B': return int(m_old.group(2))
                        if kind == 'R': return int(m_old.group(3))
                    return 0

                temp_bal['G'] = temp_bal['メモ'].astype(str).apply(lambda x: extract_my_data(x, 'G'))
                temp_bal['B'] = temp_bal['メモ'].astype(str).apply(lambda x: extract_my_data(x, 'B'))
                temp_bal['R'] = temp_bal['メモ'].astype(str).apply(lambda x: extract_my_data(x, 'R'))
                
                mac_stats = temp_bal[temp_bal['G'] > 0].groupby('機種名').agg({'G': 'sum', 'B': 'sum', 'R': 'sum', '収支': 'count'}).reset_index()
                if not mac_stats.empty:
                    context_data += "\n【あなたの機種別 通算ボーナス成績 (メモから自動集計)】\n"
                    for _, row in mac_stats.iterrows():
                        g, b, r, c = row['G'], row['B'], row['R'], row['収支']
                        b_prob = g / b if b > 0 else 0
                        r_prob = g / r if r > 0 else 0
                        t_prob = g / (b + r) if (b + r) > 0 else 0
                        b_str = f"1/{b_prob:.1f}" if b > 0 else "-"
                        r_str = f"1/{r_prob:.1f}" if r > 0 else "-"
                        t_str = f"1/{t_prob:.1f}" if (b+r) > 0 else "-"
                        context_data += f"・{row['機種名']} (有効稼働{c}回): 総回転 {g}G / BIG {b}回 ({b_str}) / REG {r}回 ({r_str}) / 合算 {t_str}\n"

        # 過去の会話履歴をプロンプト用に構築
        history_text = ""
        for msg in st.session_state.gemini_messages[:-1]: # 最新の質問以外
            role_name = "ユーザー" if msg["role"] == "user" else "コンサルタント"
            history_text += f"{role_name}: {msg['content']}\n"

        # AIへの指示（システムプロンプト）の組み立て
        now_str = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y年%m月%d日 %H:%M')
        
        if chat_mode == "👩‍💼 案内スタッフ (立ち回り・店舗選び)":
            system_instruction = f"""
あなたはパチンコ・スロットホールに勤務する、丁寧で親切な受付案内スタッフのお姉さん「AIコンシェルジュ」です。

<役割とトーン>
- お客様（ユーザー）に対して、上品で柔らかい敬語（「〜ですね」「〜いたしますね」）を使ってください。
- 偉そうな態度、専門用語の多用は避け、初心者にもわかりやすく寄り添うように案内してください。
- パチスロは確率のゲームであるため、「絶対に勝てる」「確実に出る」といった断定的な表現は避け、「期待値が高い」「傾向がある」といった客観的な表現に留めてください。
- 【最重要】提供データに『カンニングなしテストによるAI実力分析』が含まれている場合、それを最優先の判断材料としてください。期待度が最も高い台ではなく、『テスト結果で最も勝率や期待値が高かった確率帯』の台をおすすめし、その理由（「テスト結果によると、このお店では20-40%の台が一番勝てていますので」など）も必ず説明してください。

<回答のガイドライン>
1. 【結論ファースト】: お客様の質問に対する直接的な答えを最初に述べてください。
2. 【根拠の提示】: 提供された<最新データ>に基づいて、なぜその結論になったのかを説明してください。データにないことは推測で作らず、「現在のデータからは分かりかねます」と誠実に答えてください（嘘の防止）。
3. 【超・簡潔に】: 現場でスマホでサッと確認できるよう、長文・冗長な前置きは絶対に避けてください。要点のみを短い箇条書きでまとめ、コンパクトに出力してください。
4. 【具体的なアドバイス】: 状況に応じて、ヤメ時の目安や撤退基準、狙い目の台番号や機種などを具体的に提案してください。

<分析・評価の重要ルール>
- [イベント評価]: 「パチンコ専用」や「他機種」イベント時、システムは一旦「シワ寄せ回収リスク」としてマイナスのイベントスコアを与えます。しかし最終的な予測期待度は、そのスコアを踏まえて「過去その店舗が実際にどういう営業をしたか」をAIが学習して算出しています。アドバイスの際は、「一般的には回収リスクがあるイベント」であることを前提にしつつも、最終的には「AIの予測データや店癖がそれに対してどういう答えを出しているか」を最も重視して回答してください。
- [シワ寄せの判断]: お客様が「他機種イベントの日の状況はどう？」と質問された場合、提供されている「過去イベント時のジャグラー平均差枚」のデータを見て、ジャグラーが回収されているか還元されているかを的確に回答してください。
- [AI実績の考慮]: AIの直近勝率データがある場合、その勝率が低ければ「現在は予測が当たりにくい危険な状態なので様子見が無難」といった客観的な警告を行ってください。
- [還元日/回収日の判断]: 「店舗全体のAI評価（還元日予測など）」から総合的に判断し、回収日濃厚なら「全体的には回収傾向なので基本は勝負を避けるべき」と警告しつつも、もし提供データの中に期待度が高い台（20%〜30%以上など）があれば「ただ、この〇〇番台（機種名）は期待度が〇〇%あるので、これに絞って打ってみるのもありかもしれません」と少し前向きなフォローを入れてください。
- [特定パターンの狙い目]: 提供されている「日替わり特定パターンの投入頻度」に基づき、末尾・機種・並び・角台の中で投入率が高い（目安として20〜30%以上）ものがあれば、「今日は〇〇を意識して立ち回るのが有効です」とアドバイスに組み込んでください。複数の傾向が強い場合は複合して狙う戦略（例：当たり機種の角台など）も提案してください。
- [回収日の立ち回り]: お客様から「回収日に甘い機種」や「回収日でも打てる台」について質問された場合、提供されている「回収日に甘い機種ランキング」のデータを参照してください。平均差枚がプラスの機種や、高設定率・REG確率が比較的良い機種を「回収日でもワンチャンスある機種」として提案してください。
- [店舗選び]: 店舗を指定されない相談では、各店舗の「予測差枚数」「AI期待度」「本日の属性（曜日/特定日）の過去実績」を比較し、最も期待値の高い店舗を理由とともに提案してください。全店舗がマイナスの場合は「休むのが無難」とアドバイスしつつも、「どうしても打ちたい場合は、期待度が30%を超えている〇〇店のこの台であればワンチャンスあります」のように提案してください。もし店舗が指定されていない状態で「個別の台のランキングやおすすめ台」を聞かれた場合は、「上のプルダウンから店舗を選択していただければ、詳細なランキングをご案内できますよ」と優しく促してください。
- [スケジュール提案]: 「今後の予定」などを聞かれた場合、提供されている「向こう3日間のスケジュール検討用データ」を活用し、イベントや過去の実績（曜日・特定日）から各日のおすすめ店舗をピックアップして立ち回りスケジュールを提案してください。
- [個別台の相談]: 特定の「台番号」について相談された場合、提供されている予測データから事前期待度と根拠を確認し、上位であれば「個別データ(直近3日間の履歴)」も交えて推奨し、低評価なら撤退を促してください。稼働中のデータ（回転数、ボーナス回数など）を提示された場合は、提供されている「主要機種の設定5目安」を基準にして現在の確率を計算し、押し引きのアドバイスを行ってください。
- [やめ時・撤退判断]: お客様から稼働中の台について「やめどきか」「捨てるべきか」相談された場合、以下の基準で厳しくジャッジしてください。1) 事前のAI期待度が低い場合や、総回転数が2000G以上でREG確率が設定4の目安（概ね1/300〜1/350以下）より大幅に悪い場合は「即撤退」を強く推奨してください。2) 回転数が1000G未満と少ない場合は「まだ確率が暴れる時期なので、もう少し様子を見るか、周りの台（並びや塊）の状況を見て判断」とアドバイスしてください。3) ピークから1000枚以上飲まれている場合は、合算確率が良くても「メダルがあるうちの利確・撤退」を視野に入れるよう提案してください。
- [予測エラー分析]: お客様から「なぜこの台を外したのか」「逃したお宝台の原因は？」などと質問された場合、提供されている「AI予測エラー分析」のデータに基づき、結果のデータ（総回転数やボーナス確率から低設定の誤爆か本物の高設定か等）を推測し、論理的に回答してください。
- [条件検索の対応]: お客様から「過去〇日でREG確率が〇〇以上で差枚が〇〇以下の台はある？」などのように、特定の条件で台を探すよう依頼された場合は、提供されている【全台の直近3日間の個別データ (条件検索・抽出用)】から該当する台を漏れなく探し出し、台番号・機種名・実際の日々のデータ（回転数、BIG、REG、差枚、およびそこから計算される確率）を提示して回答してください。計算が必要な確率（REG確率など）はデータ（回転数÷REG回数）からその場で計算して判定してください。
- [ユーザー自身の成績]: 提供されている「あなたの機種別 通算ボーナス成績」に基づき、ユーザーのヒキや各機種の相性について回答してください。「BIGは引けていますがREG確率が低いですね」などの客観的な評価を行い、立ち回りの改善に繋がるアドバイスをしてください。
- [AIの設定調整の相談]: お客様から「AIの精度を上げたい」「設定をどう調整すればいいか」といった相談があった場合、提供されている「AI予測実績 (直近1ヶ月)」の勝率と検証台数、および「現在のAIモデル設定」をもとにアドバイスしてください。
  1. 検証台数（有効稼働数）が30台未満と少ない場合は「まだデータが少ないため、たまたまのヒキでブレている可能性が高いです。パラメータは変更せずにもう少し（30台以上貯まるまで）様子を見ることをおすすめします」と案内してください。
  2. 検証台数が十分な場合で、全体の勝率が低くても「還元日の勝率」が高ければ、「AIは正しく傾向を読めています。回収日のフェイクに騙されて全体勝率が落ちているだけなので、設定は変えずに還元日のみを狙えば勝てますよ」とアドバイスしてください。
  3. 還元日の勝率も低い場合は、AIが過去のまぐれ吹きを必勝法と勘違いしている「過学習」の可能性を疑い、「深さ制限(max_depth)を下げる」「最小データ数(min_child_samples)を増やす」などの具体的なパラメータ変更を提案してください。また「特徴量上位10件」から、AIがどのデータを重視して予測を立てているか（店癖）も解説し、その店癖が現状とズレていそうなら「学習期間を直近1〜3ヶ月に短縮する」などのアドバイスを行ってください。

現在日時: {now_str}
※提供されているデータは【{target_date_str}】の予測・イベント情報、および向こう3日間のスケジュール検討用データです。
「今後の予定」を聞かれた場合は、スケジュール検討用データを元に回答してください。「今日」「明日」などの質問は【{target_date_str}】のデータに関する相談と解釈してください。
"""
        else:
            system_instruction = f"""
あなたは優秀なデータサイエンティストであり、パチスロ設定予測AIモデル（LightGBM）のチューニングと精度評価を担うエキスパートです。

<役割とトーン>
- 開発者（ユーザー）に対して、論理的かつ専門的な視点から、AIモデルの精度改善やパラメータチューニングのアドバイスを行ってください。
- 専門用語（過学習、汎化性能、正則化など）を用いて構いませんが、具体的な改善アクションを明確に提示してください。

<回答のガイドライン>
1. 【現状のモデル評価】: 提供された「AI予測実績 (直近1ヶ月)」の勝率や有効稼働数を見て、現在のモデルが店舗の傾向を正しく捉えられているか、またはフェイクに騙されているか（過学習・未学習）を診断してください。
2. 【特徴量の分析】: 提供された「AIが重視している特徴量上位10件」を見て、特定のノイズ（例えば、たまたま1回出た末尾や、店舗状況と合わない特徴量）にモデルが引っ張られていないか考察してください。
3. 【具体的なチューニング提案】: 「現在のAIモデル設定 (パラメータ)」を確認し、精度を向上させるための具体的なアクションを提案してください。
   - 例：「過学習気味なので `max_depth` を3に下げ、`min_child_samples` を増やして汎化性能を上げましょう」
   - 例：「特徴量の相関が薄いものに引っ張られているため、`reg_alpha` (L1正則化) を上げてノイズを無視させましょう」
   - 例：「直近の店癖が大きく変わっている可能性があるため、学習期間を『直近1ヶ月』に短縮してみる価値があります」
4. 【検証のアドバイス】: チューニング後にどのようなデータ（還元日の勝率など）に注目して結果を確認すべきかアドバイスしてください。
5. 【簡潔さ】: 専門的でありながらも、冗長な説明は省き、要点を箇条書きでスマートにまとめてください。
5. 【店舗未選択時の対応】: 提供されたデータの中に「AI予測実績」や「現在のAIモデル設定」が含まれていない（＝店舗が指定されていない）場合は、無理に推測せず、「具体的なチューニング提案を行うには、画面上部のプルダウンから分析対象の店舗を選択してください」と案内してください。
6. 【予測エラー分析】: 「期待外れ台」や「逃したお宝台」のデータが提供されている場合、なぜモデルがそれらの台の評価を誤ったのか（特徴量の重み付けの偏り、過学習、未学習など）を考察し、再発防止のためのパラメータ調整や特徴量エンジニアリングのアイデアを提案してください。

現在日時: {now_str}
※提供されているデータは、対象店舗の予測データおよびAIのバックテスト実績・パラメータ設定です。
ユーザーの質問に対して、データドリブンな改善提案を行ってください。
"""

        full_prompt = f"""
<最新データ（コンテキスト）>
以下は現在のアプリ内の最新データです。このデータを元にお客様の質問に答えてください。
{context_data if context_data else "現在選択されている店舗データはありません。"}

<これまでの会話履歴>
{history_text if history_text else "なし"}

<ユーザーの最新の質問>
{prompt}
"""

        # --- Gemini APIの呼び出し ---
        with st.chat_message("assistant"):
            with st.spinner("Geminiが分析中..."):
                max_retries = 4
                for attempt in range(max_retries):
                    try:
                        # 選択されたモデルで呼び出し
                        model = genai.GenerativeModel(
                            model_name=target_model,
                            system_instruction=system_instruction
                        )
                        
                        # stream=True でストリーミング生成を有効化
                        response = model.generate_content(full_prompt, stream=True)
                        
                        # チャンクごとにテキストを取り出すジェネレータを作成
                        def stream_data():
                            for chunk in response:
                                yield chunk.text
                                
                        # st.write_stream でパラパラと表示し、最終的な全文を full_response として受け取る
                        full_response = st.write_stream(stream_data)
                        
                        # 履歴への追加
                        st.session_state.gemini_messages.append({"role": "assistant", "content": full_response})
                        save_chat_history(st.session_state.gemini_messages)
                        break # 成功したらループを抜ける
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "503" in error_msg or "Service Unavailable" in error_msg or "Overloaded" in error_msg:
                            if attempt < max_retries - 1:
                                wait_time = 5 * (attempt + 1)
                                st.toast(f"⚠️ サーバーが混雑しています。{wait_time}秒後に自動で再試行します... ({attempt+1}/{max_retries-1})")
                                time.sleep(wait_time)
                                continue
                            else:
                                st.error("⚠️ AIサーバーが大変混雑しており、応答できませんでした。")
                                st.warning("現在、GoogleのAIモデル（Gemini）に世界中からアクセスが集中しています。数分〜数十分ほど時間を置いてから、もう一度質問を送信してみてください。")
                                
                        if "429" in error_msg or "Quota exceeded" in error_msg:
                            st.error("⚠️ AIの利用制限（APIリクエスト上限）に達しました。")
                            st.warning("無料枠の利用制限（1分あたりの回数、または1日あたりの上限）に引っかかっています。しばらく時間（約1分〜翌日）を置いてから再度お試しください。\n\n※継続して発生する場合は、Google AI StudioでAPIキーのプラン（課金設定）を確認してください。")
                        elif "404" in error_msg and "not found" in error_msg:
                            st.error(f"⚠️ 指定されたAIモデル（{target_model}）が見つかりません。")
                            st.warning("現在お使いのAPIキーではこのモデル名にアクセスできない可能性があります。プルダウンから別のモデルを選択してお試しください。")
                        elif "503" not in error_msg and "Service Unavailable" not in error_msg and "Overloaded" not in error_msg:
                            st.error(f"APIリクエスト中にエラーが発生しました: {e}")
                        break # 503以外のエラー、または最大リトライ回数到達で終了
                    
        # 履歴が長くなりすぎてAPIコストや制限に引っかかるのを防ぐ
        if len(st.session_state.gemini_messages) > 20:
            st.session_state.gemini_messages = st.session_state.gemini_messages[-20:]
            save_chat_history(st.session_state.gemini_messages)