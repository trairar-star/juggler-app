import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import optuna
import backend
from lstm_feature_extractor import JugglerTimeSeriesDataset, JugglerLSTM

def objective(trial, X, y, input_size, device):
    # 探索するパラメータの範囲
    hidden_size = trial.suggest_categorical('hidden_size', [4, 8, 16, 32, 64, 128, 256, 512])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 10, 50, step=10)
    num_layers = 2
    
    # データの分割 (80%を学習、20%を検証テストに使用)
    dataset = JugglerTimeSeriesDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = JugglerLSTM(input_size, hidden_size, num_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 学習ループ
    for epoch in range(epochs):
        model.train()
        for seqs, lbls in train_loader:
            seqs, lbls = seqs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            
    # 検証ループ (未学習のデータでどれだけ正確に波を読めたか)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seqs, lbls in val_loader:
            seqs, lbls = seqs.to(device), lbls.to(device)
            outputs = model(seqs)
            loss = criterion(outputs, lbls)
            val_loss += loss.item()
            
    return val_loss / len(val_loader)

def main():
    print("🤖 LSTM(波読み)パラメータ 自動チューニングツールを開始します...")
    print("スプレッドシートから稼働データを読み込んでいます...")
    
    df_raw = backend.load_data()
    if df_raw.empty:
        print("❌ エラー: データがありません。")
        return
        
    print("特徴量を生成中... (少々お待ちください)")
    df, _ = backend._generate_features(df_raw, None, None, None, None)
    
    time_series_features = ['差枚', '累計ゲーム', 'REG', 'BIG']
    df_scaled = df.copy()
    for col in time_series_features:
        if col in df_scaled.columns:
            df_scaled[col] = df_scaled[col].fillna(0)
            max_val = df_scaled[col].abs().max()
            if max_val > 0:
                df_scaled[f'{col}_scaled'] = df_scaled[col] / max_val
            else:
                df_scaled[f'{col}_scaled'] = 0.0
    
    scaled_features = [f'{col}_scaled' for col in time_series_features]
    input_size = len(scaled_features)
    
    shop_col = '店名' if '店名' in df_scaled.columns else '店舗名'
    all_shops = df_scaled[shop_col].dropna().unique().tolist()
    shop_hyperparams = backend.load_shop_ai_settings()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用デバイス: {device}")
    
    # 各店舗の「デフォルト(全体)」モデルとしてチューニングを実行
    # ※今回は全店舗のデータをまとめて一番強力な波読み汎用パラメータを探します
    print("\n--- 汎用LSTMモデルのチューニングを開始 ---")
    sequences, targets = [], []
    grouped = df_scaled.groupby([shop_col, '台番号'])
    
    for _, group in grouped:
        group_vals = group[scaled_features].values
        group_targets = group['target'].values if 'target' in group.columns else np.zeros(len(group))
        for i in range(len(group)):
            if pd.isna(group_targets[i]): continue
            if i < 6: continue # 最低7日分の履歴があるデータのみ抽出
            seq = group_vals[i - 6 : i + 1]
            sequences.append(seq)
            targets.append(group_targets[i])
                
    X = np.array(sequences)
    y = np.array(targets)
    print(f"抽出された波のパターン数: {len(X)} 件")
    
    study = optuna.create_study(direction='minimize')
    # 探索回数 (時間がある時は20や30に増やしてください)
    study.optimize(lambda t: objective(t, X, y, input_size, device), n_trials=10)
    
    print(f"\n🎉 チューニング完了！ 最適なパラメータ: {study.best_params}")
    print("スプレッドシートに設定を自動保存しています...")
    
    for shop in shop_hyperparams.keys():
        shop_hyperparams[shop]['lstm_hidden_size'] = study.best_params['hidden_size']
        shop_hyperparams[shop]['lstm_lr'] = study.best_params['lr']
        shop_hyperparams[shop]['lstm_epochs'] = study.best_params['epochs']
        
    backend.save_shop_ai_settings(shop_hyperparams)
    print("✅ スプレッドシートへの自動保存が完了しました！アプリの画面をリロードすると即座に反映されます。")

if __name__ == '__main__':
    main()