import os
import pandas as pd
import backend

def main():
    print("🤖 事前AI分析（キャッシュ生成）を開始します...")
    
    print("データの読み込み中...")
    df_raw = backend.load_data()
    if df_raw.empty:
        print("❌ データがありません。")
        return
        
    df_events = backend.load_shop_events()
    df_island = backend.load_island_master()
    shop_hyperparams = backend.load_shop_ai_settings()
    
    # データにある最新日の翌日（明日）の予測をターゲットに設定
    if '対象日付' in df_raw.columns:
        latest_date = pd.to_datetime(df_raw['対象日付']).max().date()
        target_date = latest_date + pd.Timedelta(days=1)
    else:
        target_date = (pd.Timestamp.now(tz='Asia/Tokyo') + pd.Timedelta(days=1)).date()
        
    print(f"📅 ターゲット日付: {target_date} (明日の予測) でAI分析を実行します...")
    print("⏳ 特徴量の生成と予測を実行しています。少々お待ちください...")
    
    # 実行してキャッシュを生成
    try:
        df, df_verify, df_importance = backend.run_analysis(
            df_raw, 
            _df_events=df_events, 
            _df_island=df_island, 
            shop_hyperparams=shop_hyperparams, 
            target_date=target_date
        )
        if not df.empty:
            print("✅ 事前分析が完了し、キャッシュファイルが保存されました！")
            print("✨ アプリ（Web画面）を開くと、このキャッシュが読み込まれ瞬時に表示されます。")
    except Exception as e:
        print(f"❌ 分析中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()