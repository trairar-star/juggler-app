import streamlit as st
import pandas as pd
import backend
import datetime
import os
import time

st.title("バッチ分析実行中...")
st.text("この画面はGitHub Actions等での自動実行用です。")

def main():
    print("=== バッチ分析開始 ===")
    
    print("1. データの読み込み中...")
    df_raw = backend.load_data()
    df_events = backend.load_shop_events()
    df_island = backend.load_island_master()
    shop_hyperparams = backend.load_shop_ai_settings()
    
    if df_raw.empty:
        print("データが空です。処理を終了します。")
        os._exit(1)

    # JSTで現在時刻を取得し、明日の日付をターゲットに設定
    jst = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(jst)
    target_date = (now + datetime.timedelta(days=1)).date()
    
    print(f"2. {target_date} の予測を実行中...")
    df_pred, df_train, df_importance = backend.run_analysis(
        df_raw, 
        _df_events=df_events, 
        _df_island=df_island, 
        shop_hyperparams=shop_hyperparams, 
        target_date=target_date
    )
    
    if not df_pred.empty:
        print(f"3. 予測結果をスプレッドシートに保存中... (対象台数: {len(df_pred)})")
        success = backend.save_prediction_log(df_pred)
        print("保存完了しました！" if success else "保存に失敗しました。")
    else:
        print("保存する予測データがありませんでした。")

    print("=== バッチ分析終了 ===")
    
    # スクリプトを強制終了してプロセスを正常に終わらせる
    os._exit(0)

if __name__ == "__main__":
    main()