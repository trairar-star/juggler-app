import os
# PyTorchやLightGBMがスレッドを作りすぎてフリーズするのを防ぐ設定
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import pandas as pd
import datetime
import os
import time
import sys

# Streamlitのアプリ画面外で実行した際のエラーを防ぐための「おまじない」
import streamlit as st
st.warning = lambda *args, **kwargs: print("\n[WARNING]", *args, flush=True)
st.error = lambda *args, **kwargs: print("\n[ERROR]", *args, flush=True)
st.success = lambda *args, **kwargs: print("\n[SUCCESS]", *args, flush=True)
st.info = lambda *args, **kwargs: print("\n[INFO]", *args, flush=True)
st.caption = lambda *args, **kwargs: print("\n[CAPTION]", *args, flush=True)

import backend

def main():
    print("=== バッチ分析開始 ===")
    sys.stdout.flush() # GitHub Actionsの画面にログをリアルタイムで出力させる
    
    print("1. データの読み込み中...")
    sys.stdout.flush()
    df_raw = backend.load_data()
    df_events = backend.load_shop_events()
    df_island = backend.load_island_master()
    shop_hyperparams = backend.load_shop_ai_settings()
    
    if df_raw.empty:
        print("❌ データが空です。読み込みエラーが発生している可能性があります。処理を終了します。", flush=True)
        sys.exit(1)

    # JSTで現在時刻を取得し、実行日（今日）をターゲットに設定
    jst = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(jst)
    target_date = now.date()
    
    print("🔧 バッチ実行環境用のパラメータ調整（LSTMの過負荷・フリーズ防止）を適用します...")
    for shop in shop_hyperparams.keys():
        if shop_hyperparams[shop].get('lstm_epochs', 20) > 5:
            shop_hyperparams[shop]['lstm_epochs'] = 5
            
    print("🧹 既存の予測キャッシュをクリアしています（強制再計算）...")
    sys.stdout.flush()
    backend.clear_spreadsheet_cache()

    print(f"2. {target_date} の予測を実行中... (店舗数が多い場合、数十分かかる場合があります)")
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
    try:
        import torch
        torch.set_num_threads(2) # PyTorchのスレッド競合(フリーズ)を物理的にブロック
    except ImportError:
        pass
    main()