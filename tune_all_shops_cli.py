import os
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb # type: ignore
import backend
from config import BASE_FEATURES
from shop_trends import diagnose_allocation_types
from config import MACHINE_SPECS

def main():
    print("🤖 全店舗のLightGBMパラメータ一括自動チューニングを開始します...")
    
    print("データの読み込みと特徴量生成中...")
    df_raw = backend.load_data()
    if df_raw.empty:
        print("❌ データがありません。")
        return
        
    df_events = backend.load_shop_events()
    df_island = backend.load_island_master()
    shop_hyperparams = backend.load_shop_ai_settings()
    
    df, features = backend._generate_features(df_raw, df_events, df_island, None, None)
    train_df = df.dropna(subset=['next_diff']).copy()
    train_df['valid_play_mask'] = backend.get_valid_play_mask(train_df['next_累計ゲーム'], train_df['next_diff'])
    
    shop_col = '店名' if '店名' in train_df.columns else ('店舗名' if '店舗名' in train_df.columns else None)
    if not shop_col:
        print("❌ 店舗データがありません。")
        return
        
    all_shops = train_df[shop_col].dropna().unique().tolist()
    if not all_shops:
        print("❌ チューニング対象の店舗がありません。")
        return

    alloc_types = diagnose_allocation_types(train_df, shop_col, MACHINE_SPECS)
    
    actual_features = [f for f in BASE_FEATURES if f in train_df.columns]
    cat_features = [f for f in ['machine_code', 'shop_code', 'event_code', 'target_weekday', 'target_date_end_digit'] if f in actual_features]
    
    keep_allowed_features = [
        '累計ゲーム', 'REG確率', 'BIG確率', '差枚', 'reg_ratio',
        'prev_bonus_balance', 'prev_unlucky_gap',
        'is_prev_high_reg', 'is_high_reg_plus_diff', 'is_low_reg_plus_diff',
        'prev2_差枚', 'prev3_差枚', 'prev2_REG確率', 'prev3_REG確率', 'prev2_累計ゲーム', 'prev3_累計ゲーム',
        'mean_3days_diff', 'mean_3days_reg_prob', 'mean_3days_games',
        'mean_7days_diff', 'mean_7days_reg_prob', 'mean_7days_games',
        '連続マイナス日数', '連続プラス日数', 'cons_high_reg_days',
        'island_high_setting_ratio', 'reg_diff_interaction', 'big_reg_ratio_gap', 
        'reg_efficiency_penalty', 'machine_3days_avg_diff', 
        'machine_3days_high_setting_ratio', 'machine_prev_avg_games',
        'days_since_last_high', 'rotation_priority_rank', 'island_unexplored_flag',
        'prev_shop_fake_rate', 'is_sandwich_target', 'relative_abandon_score',
        'island_win_rate', 'island_fake_ratio', 'past_island_fake_rate',
        'is_corner_showpiece', 'heavy_play_fake_penalty', 'post_ev_sueoki_trust'
    ]
    keep_features = [f for f in actual_features if f in keep_allowed_features]
    change_features = actual_features.copy()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    total_shops = len(all_shops)
    for idx, shop_name in enumerate(all_shops):
        print(f"\n[{idx+1}/{total_shops}] {shop_name} の最適化を実行中...")
        is_point = alloc_types.get(shop_name, {}).get("is_point", False)
        
        shop_df = train_df[train_df[shop_col] == shop_name].copy()
        if len(shop_df) < 150:
            print(f"  -> データ不足 ({len(shop_df)}件) のためスキップします。")
            continue
            
        shop_df['対象日付'] = pd.to_datetime(shop_df['対象日付'])
        shop_df = shop_df.sort_values('対象日付')
        max_date = shop_df['対象日付'].max()
        
        # 時系列交差検証 (Time Series Split) 用データチェック (3fold x 14days = 42days)
        cutoff_check = max_date - pd.Timedelta(days=42)
        if len(shop_df[shop_df['対象日付'] <= cutoff_check]) < 30:
            print(f"  -> 時系列交差検証に必要な過去データ不足のためスキップします。")
            continue
        
        best_c_params = None
        best_k_params = None
        
        # --- 据え置き前提NOの日をテストから除外するための処理 ---
        sueoki_no_dates = set()
        df_raw_shop = df_raw[df_raw[shop_col] == shop_name].copy()
        for d in shop_df['対象日付'].dt.date.unique():
            tgt_date = pd.to_datetime(d) + pd.Timedelta(days=1)
            premise, _ = backend.evaluate_sueoki_premise(df_raw_shop, tgt_date, df_events)
            if premise == "NO":
                sueoki_no_dates.add(pd.to_datetime(d))
        
        current_hp = shop_hyperparams.get(shop_name, shop_hyperparams.get("デフォルト", {}))
        
        for mode in ['change', 'keep']:
            target_val = 0 if mode == 'change' else 1
            mode_full_data = shop_df[shop_df['is_prev_high_reg'] == target_val].copy()
                
            current_features = change_features if mode == 'change' else keep_features
            if is_point:
                ignore_features = ['is_neighbor_high_reg', 'neighbor_reg_reliability_score', 'neighbor_high_setting_count', 'past_island_reg_prob', 'is_main_island', 'is_wall_island']
                current_features = [f for f in current_features if f not in ignore_features]
                
            current_cat_features = [f for f in cat_features if f in current_features]
            
            def objective_mode(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 80, step=10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0)
                }
                max_leaves = min(127, (2 ** params['max_depth']) - 1)
                params['num_leaves'] = trial.suggest_int('num_leaves', min(7, max_leaves), max_leaves)
                
                n_splits = 3
                test_days = 14
                fold_scores = []
                
                for fold in range(n_splits):
                    fold_cutoff_end = max_date - pd.Timedelta(days=fold * test_days)
                    fold_cutoff_start = fold_cutoff_end - pd.Timedelta(days=test_days)
                    
                    fold_train = mode_full_data[mode_full_data['対象日付'] <= fold_cutoff_start].copy()
                    fold_test = mode_full_data[(mode_full_data['対象日付'] > fold_cutoff_start) & (mode_full_data['対象日付'] <= fold_cutoff_end)].copy()
                    
                    if mode == 'keep':
                        sueoki_no_obj_dates = [d - pd.Timedelta(days=1) for d in sueoki_no_dates]
                        fold_test = fold_test[~fold_test['対象日付'].isin(sueoki_no_obj_dates)].copy()
                        
                    if len(fold_train) < 30 or len(fold_test) < 5:
                        continue
                        
                    X_fold_train, y_fold_train = fold_train[current_features], fold_train['target']
                    X_fold_test, y_fold_test = fold_test[current_features], fold_test['target']
                    
                    fold_max_date = fold_train['対象日付'].max()
                    days_diff = (fold_max_date - fold_train['対象日付']).dt.days
                    sample_weights = 0.995 ** days_diff
                    
                    try:
                        reg_model = lgb.LGBMRegressor(random_state=42, verbose=-1, **params, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
                        reg_model.fit(X_fold_train, fold_train['next_diff'], sample_weight=sample_weights, categorical_feature=current_cat_features)
                        X_train_st = X_fold_train.copy()
                        X_train_st['predicted_diff'] = reg_model.predict(X_fold_train)
                        
                        model = lgb.LGBMClassifier(objective='binary', random_state=42, verbose=-1, **params, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
                        model.fit(X_train_st, y_fold_train, sample_weight=sample_weights, categorical_feature=current_cat_features)
                        
                        X_test_st = X_fold_test.copy()
                        X_test_st['predicted_diff'] = reg_model.predict(X_fold_test)
                        preds = model.predict_proba(X_test_st)[:, 1]
                        
                        if len(np.unique(preds)) <= 1: continue
                        
                        valid_play = backend.get_valid_play_mask(fold_test['next_累計ゲーム'], fold_test['next_diff'])
                        valid_win = valid_play & (pd.to_numeric(fold_test['next_diff'], errors='coerce').fillna(0) > 0)
                        threshold = pd.Series(preds).quantile(0.85)
                        target_idx = np.where(preds >= threshold)[0]
                        
                        if len(target_idx) == 0: continue
                        valid_target = valid_play.iloc[target_idx]
                        if valid_target.sum() == 0: continue
                        
                        win_rate = valid_target['valid_win'].mean()
                        avg_diff = fold_test['next_diff'].iloc[target_idx][valid_target].mean()
                        
                        base_win_rate = valid_win.sum() / valid_play.sum() if valid_play.sum() > 0 else 0
                        win_lift = max(0, win_rate - base_win_rate)
                        
                        score = (win_rate * 100) + (win_lift * 200) + (avg_diff / 10)
                        fold_scores.append(score)
                    except Exception:
                        continue
                        
                if not fold_scores: return -1.0
                return sum(fold_scores) / len(fold_scores)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective_mode, n_trials=10) # ★精度を上げるならここを 20 や 30 に増やす
            
            if mode == 'change': 
                best_c_params = study.best_params
                if 'num_leaves' not in best_c_params: best_c_params['num_leaves'] = min(127, (2 ** best_c_params['max_depth']) - 1)
                print(f"  [変更] 最適パラメータ: {best_c_params}")
            else: 
                best_k_params = study.best_params
                if 'num_leaves' not in best_k_params: best_k_params['num_leaves'] = min(127, (2 ** best_k_params['max_depth']) - 1)
                print(f"  [据え] 最適パラメータ: {best_k_params}")
            
        if best_c_params is None: best_c_params = current_hp
        if best_k_params is None: best_k_params = current_hp
        
        shop_hyperparams[shop_name] = {
            'train_months': current_hp.get('train_months', 3), 
            'n_estimators': best_c_params.get('n_estimators', current_hp.get('n_estimators')),
            'learning_rate': best_c_params.get('learning_rate', current_hp.get('learning_rate')),
            'num_leaves': best_c_params.get('num_leaves', current_hp.get('num_leaves')),
            'max_depth': best_c_params.get('max_depth', current_hp.get('max_depth')),
            'min_child_samples': best_c_params.get('min_child_samples', current_hp.get('min_child_samples')),
            'reg_alpha': best_c_params.get('reg_alpha', current_hp.get('reg_alpha', 0.0)),
            'reg_lambda': best_c_params.get('reg_lambda', current_hp.get('reg_lambda', 0.0)),
            'k_train_months': current_hp.get('k_train_months', 6),
            'k_n_estimators': best_k_params.get('n_estimators', current_hp.get('k_n_estimators')),
            'k_learning_rate': best_k_params.get('learning_rate', current_hp.get('k_learning_rate')),
            'k_num_leaves': best_k_params.get('num_leaves', current_hp.get('k_num_leaves')),
            'k_max_depth': best_k_params.get('max_depth', current_hp.get('k_max_depth')),
            'k_min_child_samples': best_k_params.get('min_child_samples', current_hp.get('k_min_child_samples')),
            'k_reg_alpha': best_k_params.get('reg_alpha', current_hp.get('k_reg_alpha', 0.0)),
            'k_reg_lambda': best_k_params.get('reg_lambda', current_hp.get('k_reg_lambda', 0.0)),
            'lstm_hidden_size': current_hp.get('lstm_hidden_size', 64),
            'lstm_lr': current_hp.get('lstm_lr', 0.001),
            'lstm_epochs': current_hp.get('lstm_epochs', 20)
        }
        
    print("\n💾 最適化されたパラメータをスプレッドシートに保存しています...")
    backend.save_shop_ai_settings(shop_hyperparams)
    print("✅ 完了しました！アプリをリロードすると全店舗の最新設定が反映されます。")

if __name__ == "__main__":
    main()