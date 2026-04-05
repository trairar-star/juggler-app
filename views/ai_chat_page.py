import streamlit as st
import pandas as pd
import backend

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

def render_ai_chat_page(df_predict, df_raw, shop_col):
    st.header("🤖 Gemini スロットコンサルタント")
    st.caption("AI（Gemini）にアプリの最新データを読み込ませて、立ち回りの相談や店舗の傾向についてチャットで質問できます。")

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
        if selected_shop != "店舗を選択してください":
            if not df_predict.empty:
                shop_pred = df_predict[df_predict[shop_col] == selected_shop].sort_values('prediction_score', ascending=False)
                top_10 = shop_pred.head(10)
                if not top_10.empty:
                    context_data += f"\n【{selected_shop} の明日の予測データ (期待度上位10台)】\n"
                    cols = ['台番号', '機種名', 'prediction_score', '根拠']
                    available_cols = [c for c in cols if c in top_10.columns]
                    
                    display_df = top_10[available_cols].copy()
                    if 'prediction_score' in display_df.columns:
                        display_df['prediction_score'] = display_df['prediction_score'].apply(lambda x: f"{int(x*100)}%")
                    
                    context_data += display_df.to_markdown(index=False) + "\n"

            if not df_raw.empty:
                shop_raw = df_raw[df_raw[shop_col] == selected_shop].copy()
                if '対象日付' in shop_raw.columns:
                    recent_date = shop_raw['対象日付'].max() - pd.Timedelta(days=7)
                    recent_data = shop_raw[shop_raw['対象日付'] >= recent_date]
                    if not recent_data.empty:
                        avg_diff = recent_data['差枚'].mean()
                        context_data += f"\n【{selected_shop} の直近1週間の店舗平均差枚】\n約 {int(avg_diff):+d} 枚\n"

        # --- マイ収支データをAIに読み込ませる ---
        df_balance = backend.load_my_balance()
        if not df_balance.empty:
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
        full_prompt = f"""
あなたはプロのスロット立ち回りコンサルタントです。
論理的かつ実践的に、友人のように親身になってアドバイスしてください。
回答は長くなりすぎないよう、箇条書きなどを活用して「要点だけを短く簡潔に」まとめてください。

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