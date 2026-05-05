import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import streamlit as st

# --- 1. PyTorch用 データセットクラス ---
class JugglerTimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# --- 2. LSTM モデル定義 ---
class JugglerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(JugglerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM層 (batch_first=True で (Batch, Seq, Feature) の入力を受け付ける)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 最終出力を確率(0〜1)に変換する層
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 初期隠れ状態とセル状態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTMの順伝播
        out, _ = self.lstm(x, (h0, c0))
        
        # 最後のタイムステップ (7日目) の出力のみを使用
        out = out[:, -1, :] 
        
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out.squeeze()

# --- 3. データ前処理 & 学習・推論関数 ---
def add_lstm_features(df, shop_col='店名', seq_length=7, hidden_size=64, lr=0.001, epochs=20):
    """
    DataFrameから過去7日間のシーケンスを作成し、LSTMで学習・推論を行って
    新しい特徴量 'lstm_pred_score' を付与して返す
    """
    # ★調整ポイント1: LSTMに入力する特徴量
    # 'REG確率' や 'is_win'(前日の高設定挙動フラグ) がdfに存在すれば追加するのも有効です
    time_series_features = ['差枚', '累計ゲーム', 'REG', 'BIG'] 
    
    # 欠損値や無限大の処理と正規化 (簡易的なMinMaxスケーリング)
    df_scaled = df.copy()
    df_scaled['_orig_index'] = df_scaled.index  # ★元のインデックスを記憶しておく
    
    for col in time_series_features:
        if col in df_scaled.columns:
            df_scaled[col] = df_scaled[col].fillna(0)
            max_val = df_scaled[col].abs().max()
            if max_val > 0:
                df_scaled[f'{col}_scaled'] = df_scaled[col] / max_val
            else:
                df_scaled[f'{col}_scaled'] = 0.0
                
    scaled_features = [f'{col}_scaled' for col in time_series_features]
    
    # 台ごとの時系列データ構築
    # 処理を早くするために、ソートしてグループ化
    df_scaled = df_scaled.sort_values([shop_col, '台番号', '対象日付']).reset_index(drop=True)
    
    sequences = []
    targets = []
    valid_indices = []
    
    # groupbyで台ごとの履歴を取得
    grouped = df_scaled.groupby([shop_col, '台番号'])
    
    for _, group in grouped:
        group_vals = group[scaled_features].values
        # ターゲットは backend.py で定義された 'target' カラムを使用
        group_targets = group['target'].values if 'target' in group.columns else np.zeros(len(group))
        group_indices = group['_orig_index'].values  # ★記憶しておいたインデックスを使う
        
        # シーケンスの作成 (スライディングウィンドウ)
        for i in range(len(group)):
            if i < seq_length - 1:
                # 履歴が足りない場合は0埋めパディング
                pad_len = seq_length - 1 - i
                pad = np.zeros((pad_len, len(scaled_features)))
                seq = np.vstack([pad, group_vals[:i+1]])
            else:
                seq = group_vals[i - seq_length + 1 : i + 1]
                
            sequences.append(seq)
            targets.append(group_targets[i])
            valid_indices.append(group_indices[i])
            
    X = np.array(sequences)
    y = np.array(targets)
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルの初期化
    input_size = len(scaled_features)
    num_layers = 2
    model = JugglerLSTM(input_size, hidden_size, num_layers).to(device)
    
    # --- 学習フェーズ ---
    # 推論対象（ターゲットが未定 = 最新の日付）以外のデータで学習
    if 'target' in df.columns:
        train_mask = ~df_scaled['target'].isna()
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        if len(X_train) > 100:
            dataset = JugglerTimeSeriesDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            criterion = nn.BCELoss() # 2値分類の損失関数
            optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
            
            model.train()
            for epoch in range(epochs):
                for seqs, lbls in dataloader:
                    seqs, lbls = seqs.to(device), lbls.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(seqs)
                    loss = criterion(outputs, lbls)
                    loss.backward()
                    optimizer.step()
    
    # --- 推論フェーズ ---
    # 全データに対してLSTMの予測スコアを計算（これをLightGBMに渡す）
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        predictions = model(X_tensor).cpu().numpy()
        
    # 元のDataFrameに特徴量として結合
    df.loc[valid_indices, 'lstm_wave_score'] = predictions
    
    return df