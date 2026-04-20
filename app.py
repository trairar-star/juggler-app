import os
import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import altair as alt # type: ignore

# バックエンド処理をインポート
import backend
from views import shop_detail_page
from views import my_balance_page
from views import island_master_page
from views import event_management_page
from views import feature_analysis_page
from views import daily_result_page
from views import calendar_compare_page
from views import island_map_page
from views import verification_page
from views import ranking_comparison_page
from views import realtime_judgement_page
from views import weekly_schedule_page
from utils import get_confidence_indicator

# ---------------------------------------------------------
# ページ設定 (スマホ閲覧を意識して layout="centered" 推奨)
# ---------------------------------------------------------
st.set_page_config(
    page_title="スロット予測ビューアー",
    page_icon="🎰",
    layout="centered"
)

# --- ページ描画関数: イベント管理 ---
def render_event_management_page():
    st.header("📝 イベント管理")
    st.caption("登録済みの店舗イベント一覧です。不要なイベントはここから削除できます。")
    
    df_events = backend.load_shop_events()
    
    if df_events.empty:
        st.info("現在、登録されているイベントはありません。")
        return

    # 削除用の一意キーと表示用日付を作成
    df_events['date_str'] = df_events['イベント日付'].dt.strftime('%Y-%m-%d')
    df_events['uid'] = df_events['店名'] + " | " + df_events['date_str'] + " | " + df_events['イベント名']
    
    # 表示用データ (日付降順)
    df_display = df_events.sort_values(['イベント日付', '店名'], ascending=[False, True])
    
    st.dataframe(
        df_display[['イベント日付', '店名', 'イベント名', 'イベントランク']],
        column_config={
            "イベント日付": st.column_config.DateColumn("日付", format="YYYY-MM-DD"),
            "店名": st.column_config.TextColumn("店舗"),
            "イベント名": st.column_config.TextColumn("イベント"),
            "イベントランク": st.column_config.TextColumn("ランク"),
        },
        width="stretch",
        hide_index=True
    )
    
    st.divider()
    st.subheader("✏️ イベント編集")
    
    edit_target_uid = st.selectbox("編集するイベントを選択", df_display['uid'].unique(), key="edit_target")
    if edit_target_uid:
        target_row = df_display[df_display['uid'] == edit_target_uid].iloc[0]
        
        with st.form("edit_event_form"):
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                edit_shop = st.text_input("店舗名", value=target_row['店名'])
                try:
                    default_date = pd.to_datetime(target_row['イベント日付']).date()
                except:
                    default_date = pd.Timestamp.now(tz='Asia/Tokyo').date()
                edit_date = st.date_input("イベント日付", value=default_date, key="edit_date")
            with e_col2:
                edit_name = st.text_input("イベント名", value=target_row['イベント名'])
                rank_options = ["S", "A", "B", "C"]
                current_rank = target_row.get('イベントランク', 'A')
                idx = rank_options.index(current_rank) if current_rank in rank_options else 1
                edit_rank = st.selectbox("イベントの強さ", rank_options, index=idx)
                
            if st.form_submit_button("更新を保存", type="primary"):
                if backend.update_shop_event(target_row['店名'], default_date, target_row['イベント名'], edit_shop, edit_date, edit_name, edit_rank):
                    st.success("イベントを更新しました！")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("イベントの更新に失敗しました。")

    st.divider()
    st.subheader("🗑 イベント削除")
    
    # 削除フォーム
    with st.form("delete_event_form"):
        target_uid = st.selectbox("削除するイベントを選択", df_display['uid'].unique())
        if st.form_submit_button("削除実行", type="primary"):
            # 選択されたIDから元データを復元
            target_row = df_display[df_display['uid'] == target_uid].iloc[0]
            
            if backend.delete_shop_event(target_row['店名'], target_row['イベント日付'], target_row['イベント名']):
                st.success(f"削除しました: {target_uid}")
                st.cache_data.clear() # キャッシュクリア
                st.rerun()
            else:
                st.error("削除に失敗しました。")

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
def main():
    # --- スマホ向け アプリ全体のUI調整 (CSS) ---
    st.markdown("""
        <style>
            /* 全体的なテキスト(段落・箇条書き)サイズを縮小 */
            .stMarkdown p, .stMarkdown li {
                font-size: 0.9rem !important;
            }
            .stButton button {
                font-size: 0.9rem !important;
            }
            label p {
                font-size: 0.85rem !important;
            }
            /* 見出しのサイズと余白の調整 */
            h1 { font-size: 1.25rem !important; padding: 0.2rem 0 !important; margin-bottom: 0.2rem !important; line-height: 1.3 !important;}
            h2 { font-size: 1.1rem !important; padding: 0.2rem 0 !important; margin-bottom: 0.2rem !important; line-height: 1.3 !important;}
            h3 { font-size: 1.0rem !important; padding: 0.2rem 0 !important; margin-bottom: 0.2rem !important; line-height: 1.3 !important;}
            /* タイトル下のキャプション(説明文)の余白調整 */
            [data-testid="stCaptionContainer"] { margin-bottom: 0.5rem !important; }
            /* データフレーム(表)の文字サイズをさらに小さく */
            [data-testid="stDataFrame"] {
                font-size: 0.8rem !important;
            }
            /* Metric（大きな数値表示）のサイズ調整 */
            [data-testid="stMetricLabel"] p {
                font-size: 0.8rem !important;
            }
            [data-testid="stMetricValue"] div {
                font-size: 1.4rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- パスワード認証（ログイン機能） ---
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        st.title("🔒 ログイン")
        password = st.text_input("パスワードを入力してください", type="password")
        if st.button("ログイン"):
            # Secretsからパスワードを取得（設定されていない場合は '1234' とする）
            try:
                correct_password = st.secrets.get("app_password", "1234")
            except Exception:
                correct_password = "1234"
            if password == correct_password:
                st.session_state["logged_in"] = True
                # ログイン成功時に自動保存フラグを立てる
                st.session_state['login_save_requested'] = True
                st.rerun()
            else:
                st.error("パスワードが違います。")
        return

    # タイトルや設定は共通
    st.title("🎰 スロット予測ビューアー")
    
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "店舗別詳細データ"
        
    if "global_selected_shop" not in st.session_state:
        st.session_state["global_selected_shop"] = "全て"
        
    pages = ["店舗別詳細データ", "🤖 AIチャット相談", "⏱️ リアルタイム設定判別", "📊 予測の実績検証・AI設定", "📅 週間スケジュール予測", "📅 日別 結果＆予測確認", "🗺️ 店舗間ヒートマップ", "🗺️ 島マップ (神視点)", "島マスター管理", "イベント管理", "💰 マイ収支管理"]
    
    # --- ページ切り替えメニュー (サイドバーの一番上) ---
    page = st.sidebar.radio(
        "メニュー", 
        pages,
        index=pages.index(st.session_state["current_page"]) if st.session_state["current_page"] in pages else 0
    )
    
    if page != st.session_state["current_page"]:
        st.session_state["current_page"] = page
        # スマホ幅 (768px以下) の場合のみサイドバーを閉じるJSを発火
        st.components.v1.html(
            """
            <script>
                if (window.parent.innerWidth <= 768) {
                    var btn = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]') || window.parent.document.querySelector('button[kind="header"]');
                    if (btn) { btn.click(); }
                }
            </script>
            """,
            width=0, height=0
        )
        
    st.sidebar.divider()

    # --- 予測対象日の選択 (サイドバー) ---
    predict_target_date = st.sidebar.date_input(
        "📅 予測対象日",
        value=pd.Timestamp.now(tz='Asia/Tokyo').date(),
        help="指定した日付の予測を行います。過去日を指定した場合は、その前日までのデータのみを使用して予測を再現します。"
    )
    st.sidebar.divider()

    # データ更新ボタン (サイドバー)
    if st.sidebar.button("🔄 データ更新 (再読み込み)"):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("⚠️ 全データ強制再読み込み (時間かかります)"):
        backend.clear_local_cache()
        st.cache_data.clear()
        st.rerun()

    # --- 店舗別AIパラメータの初期化 ---
    if "shop_hyperparams" not in st.session_state:
        st.session_state["shop_hyperparams"] = backend.load_shop_ai_settings()

    # データのロード
    with st.spinner("スプレッドシートからデータを読み込み中..."):
        df_raw = backend.load_data()
    
    if df_raw.empty:
        st.warning("データが取得できませんでした。")
        return

    # イベントデータのロード
    df_events = backend.load_shop_events()
    df_island = backend.load_island_master()

    # --- 学習データの統計情報を表示 (追加) ---
    if '対象日付' in df_raw.columns and not df_raw.empty:
        min_date = df_raw['対象日付'].min()
        max_date = df_raw['対象日付'].max()
        total_records = len(df_raw)
        
        min_str = min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else "不明"
        max_str = max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else "不明"
        
        # --- スプレッドシートの容量（セル数）監視と警告 ---
        estimated_cells = total_records * len(df_raw.columns)
        max_cells = 10000000 # Googleスプレッドシートの物理上限 (1000万セル)
        usage_percent = (estimated_cells / max_cells) * 100
        
        capacity_warning = ""
        if usage_percent >= 90:
            capacity_warning = f"\n\n🚨 **容量警告**: スプレッドシートの限界（{usage_percent:.1f}%）に近づいています！バッチ処理等で古いデータを削除してください。"
        elif usage_percent >= 75:
            capacity_warning = f"\n\n⚠️ **容量注意**: スプレッドシートの容量を {usage_percent:.1f}% 使用しています。そろそろ古いデータの整理を検討してください。"

        st.sidebar.info(f"📚 **学習データ統計**\n\n期間: {min_str} 〜 {max_str}\n総数: {total_records:,} 件{capacity_warning}")

    # 分析実行
    with st.spinner("AIがデータを分析し、予測を生成しています..."):
        # キャッシュキーとしてデータの長さを利用（簡易的）
        df, df_verify, df_importance = backend.run_analysis(df_raw, _df_events=df_events, _df_island=df_island, shop_hyperparams=st.session_state["shop_hyperparams"], target_date=predict_target_date)
    
    if df.empty:
        st.warning(f"指定された予測対象日（{predict_target_date.strftime('%Y-%m-%d')}）以前の分析可能なデータがありません。")
        return

    # 日付表示
    if '対象日付' in df.columns:
        target_date = df['対象日付'].max()
        st.caption(f"📈 過去データ集計期間: 〜 {target_date.strftime('%Y-%m-%d')} (これまでの全履歴の傾向から {predict_target_date.strftime('%Y-%m-%d')} の結果を予測しています)")
    
    # 必要なカラムの補完
    for col in ['おすすめ度', 'prediction_score', '予測差枚数', '根拠']:
        if col not in df.columns:
            df[col] = 0 if col == '予測差枚数' else '-'
            
    # カラム名判定
    shop_col = '店名' if '店名' in df.columns else '店舗名'

    # --- ログイン時の自動保存処理 ---
    if st.session_state.get('login_save_requested'):
        with st.spinner("本日の予測結果を自動保存中..."):
            save_success = backend.save_prediction_log(df)
        st.session_state['login_save_requested'] = False # 再実行時に保存が走らないようにフラグを消す
        if save_success:
            st.toast("✅ 本日の予測結果を自動保存しました！")

    with st.spinner(f"⏳ 「{page}」の画面を構築しています... しばらくお待ちください。"):
        if page == "⏱️ リアルタイム設定判別":
            df_pred_log = backend.load_prediction_log()
            realtime_judgement_page.render_realtime_judgement_page(df_pred_log)
        elif page == "🤖 AIチャット相談":
            from views import ai_chat_page
            ai_chat_page.render_ai_chat_page(df, df_raw, shop_col, df_events=df_events, df_importance=df_importance, shop_hyperparams=st.session_state["shop_hyperparams"])
        elif page == "📊 予測の実績検証・AI設定":
            df_pred_log = backend.load_prediction_log()
            verification_page.render_verification_page(df_pred_log, df_verify, df, df_raw)
        elif page == "📅 日別 結果＆予測確認":
            daily_result_page.render_daily_result_page(df_raw, df_events, df_island, st.session_state["shop_hyperparams"])
        elif page == "📅 週間スケジュール予測":
            weekly_schedule_page.render_weekly_schedule_page(df_raw, df_events, df_island, st.session_state["shop_hyperparams"])
        elif page == "🗺️ 店舗間ヒートマップ":
            calendar_compare_page.render_calendar_compare_page(df_raw, df, predict_target_date)
        elif page == "🗺️ 島マップ (神視点)":
            df_pred_log = backend.load_prediction_log()
            island_map_page.render_island_map_page(df_raw, df_pred_log, df_island)
        elif page == "島マスター管理":
            island_master_page.render_island_master_page(df_raw)
        elif page == "イベント管理":
            event_management_page.render_event_management_page(df_raw)
        elif page == "💰 マイ収支管理":
            my_balance_page.render_my_balance_page(df_raw)
        else:
            df_pred_log = backend.load_prediction_log()
            shop_detail_page.render_shop_detail_page(df, df_raw, shop_col, df_events, df_verify, df_pred_log, df_importance)

if __name__ == "__main__":
    main()