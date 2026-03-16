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
from views import verification_page
from views import ranking_comparison_page
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
        use_container_width=True,
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
        
    pages = ["店舗別詳細データ", "AI傾向分析 (勝利の法則)", "精度検証 (答え合わせ)", "🏆 予測 vs 実際 ランキング", "島マスター管理", "イベント管理", "💰 マイ収支管理"]
    
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
        
    # 予測保存ボタン (サイドバー)
    if st.sidebar.button("💾 予測結果をログ保存"):
        # このボタンを押すと、後述の処理で計算された df を保存します
        st.session_state['save_requested'] = True

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
        
        st.sidebar.info(f"📚 **学習データ統計**\n\n期間: {min_str} 〜 {max_str}\n総数: {total_records:,} 件")

    # --- 店舗イベント登録 (サイドバー) ---
    with st.sidebar.expander("📅 店舗イベント登録", expanded=False):
        st.caption("店舗独自のイベント（取材、特定日など）を登録すると、AIがその傾向を学習します。")
        
        # 店舗リスト取得
        shop_col = '店名' if '店名' in df_raw.columns else '店舗名'
        if shop_col in df_raw.columns:
            unique_shops = df_raw[shop_col].unique()
            
            with st.form("event_reg_form", clear_on_submit=True):
                reg_shop = st.selectbox("店舗", unique_shops)
                reg_date = st.date_input("日付", pd.Timestamp.now(tz='Asia/Tokyo').date())
                reg_name = st.text_input("イベント名 (例: ○○取材, リニューアル)")
                reg_rank = st.selectbox("イベントの強さ (期待度)", ["S", "A", "B", "C"], index=1, help="S:激アツ, A:強い, B:普通, C:弱め")
                submitted = st.form_submit_button("イベントを登録")
                
                if submitted:
                    if backend.save_shop_event(reg_shop, reg_date, reg_name, reg_rank):
                        st.success(f"{reg_shop} のイベントを登録しました！")
                        st.cache_data.clear() # データ更新のためキャッシュクリア
                        st.rerun() # 画面リロード

    # --- ハイパーパラメータ調整 (サイドバー) ---
    with st.sidebar.expander("⚙️ AIモデル設定 (調整)", expanded=False):
        hp_train_months = st.slider("学習データ期間 (直近〇ヶ月)", 1, 12, 3, step=1, help="店長スイッチ対策や処理落ちを防ぐため、直近のデータのみで学習させます。")
        hp_n_estimators = st.slider("学習回数 (n_estimators)", 50, 1000, 300, step=50, help="値を大きくすると学習量が増えますが、時間がかかり過学習のリスクもあります。")
        hp_learning_rate = st.slider("学習率 (learning_rate)", 0.01, 0.3, 0.03, step=0.01, help="値を小さくすると丁寧に学習しますが、回数を増やす必要があります。")
        hp_num_leaves = st.slider("葉の数 (num_leaves)", 10, 127, 15, step=1, help="モデルの複雑さ。スロットのようなノイズが多いデータは小さめ(15〜20)がおすすめです。")
        hp_max_depth = st.slider("深さ制限 (max_depth)", -1, 15, 4, step=1, help="木の深さの上限。ノイズ対策として3〜7程度に制限するのがおすすめです。-1は無制限。")
        hp_min_child_samples = st.slider("最小データ数 (min_child_samples)", 10, 200, 50, step=10, help="1つの条件を法則と認めるために必要な最低データ数。過学習を防ぐため、スロットでは50前後を推奨します。")
        
        hyperparams = {
            'train_months': hp_train_months,
            'n_estimators': hp_n_estimators,
            'learning_rate': hp_learning_rate,
            'num_leaves': hp_num_leaves,
            'max_depth': hp_max_depth,
            'min_child_samples': hp_min_child_samples
        }

    # 分析実行
    with st.spinner("AIがデータを分析し、予測を生成しています..."):
        # キャッシュキーとしてデータの長さを利用（簡易的）
        df, df_verify, df_importance = backend.run_analysis(df_raw, df_events, df_island, hyperparams, target_date=predict_target_date)
    
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

    # --- 保存処理の実行 (キャッシュ外へ移動) ---
    if st.session_state.get('save_requested'):
        backend.save_prediction_log(df)
        st.session_state['save_requested'] = False

    with st.spinner(f"⏳ 「{page}」の画面を構築しています... しばらくお待ちください。"):
        if page == "精度検証 (答え合わせ)":
            df_pred_log = backend.load_prediction_log()
            verification_page.render_verification_page(df_pred_log, df_verify, df, df_raw)
        elif page == "🏆 予測 vs 実際 ランキング":
            df_pred_log = backend.load_prediction_log()
            ranking_comparison_page.render_ranking_comparison_page(df_pred_log, df_verify, df, df_raw)
        elif page == "AI傾向分析 (勝利の法則)":
            feature_analysis_page.render_feature_analysis_page(df_verify, df_importance, df_events)
        elif page == "島マスター管理":
            island_master_page.render_island_master_page(df_raw)
        elif page == "イベント管理":
            event_management_page.render_event_management_page()
        elif page == "💰 マイ収支管理":
            my_balance_page.render_my_balance_page(df_raw)
        else:
            shop_detail_page.render_shop_detail_page(df, df_raw, shop_col, df_events, df_verify)

if __name__ == "__main__":
    main()