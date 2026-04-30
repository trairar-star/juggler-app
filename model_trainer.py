import pandas as pd
import numpy as np
import lightgbm as lgb # type: ignore
from shop_trends import diagnose_allocation_types
from config import MACHINE_SPECS

def train_models(train_df, predict_df, features, shop_hyperparams):
    """
    LightGBMを用いて設定判別モデルを学習し、予測結果と特徴量重要度を返す
    """
    shop_col = '店名' if '店名' in train_df.columns else ('店舗名' if '店舗名' in train_df.columns else None)
    default_hp = shop_hyperparams.get("デフォルト", {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50, 'reg_alpha': 0.0, 'reg_lambda': 0.0})
    default_t_m = default_hp.get('train_months', 3)

    # 配分型の事前診断を取得
    alloc_types = {}
    if shop_col and not train_df.empty:
        alloc_types = diagnose_allocation_types(train_df, shop_col, MACHINE_SPECS)

    # 据え置き予想モデル思想「昨日の続きを打てるか」のみを評価し、深い推測(曜日、イベント等)を排除する
    keep_allowed_features = [
        '累計ゲーム', 'REG確率', 'BIG確率', '差枚', 'reg_ratio',
        'prev_bonus_balance', 'prev_unlucky_gap',
        'is_prev_high_reg', 'is_high_reg_plus_diff', 'is_low_reg_plus_diff',
        'prev2_差枚', 'prev3_差枚', 'prev2_REG確率', 'prev3_REG確率', 'prev2_累計ゲーム', 'prev3_累計ゲーム',
        'mean_3days_diff', 'mean_3days_reg_prob', 'mean_3days_games',
        'mean_7days_diff', 'mean_7days_reg_prob', 'mean_7days_games',
        '連続マイナス日数', '連続プラス日数', 'cons_high_reg_days'
    ]
    keep_features = [f for f in features if f in keep_allowed_features]
    change_features = features.copy()

    # カテゴリ変数として扱う特徴量のリストを定義
    cat_features = [f for f in ['machine_code', 'shop_code', 'event_code', 'target_weekday', 'target_date_end_digit'] if f in features]

    # 予測結果格納用の列を初期化
    if not predict_df.empty:
        predict_df['予測差枚数'] = np.nan
        predict_df['prediction_score'] = np.nan
        predict_df['sueoki_score'] = np.nan
        predict_df['ai_version'] = ""
    if not train_df.empty:
        train_df['予測差枚数'] = np.nan
        train_df['prediction_score'] = np.nan
        train_df['sueoki_score'] = np.nan
    
    feature_importances_list = []

    # 相関計算用ヘルパー関数
    def get_correlations(df_sub, feature_list):
        corrs = []
        for f in feature_list:
            if f in df_sub.columns and pd.api.types.is_numeric_dtype(df_sub[f]):
                c = df_sub[f].corr(df_sub['target'])
                corrs.append(c if not pd.isna(c) else 0.0)
            else:
                corrs.append(0.0)
        return corrs

    # 変更予測(上げ狙い: is_prev_high_reg=0) と 据え置き予測(据え狙い: is_prev_high_reg=1) でループ
    for mode in ['change', 'keep']:
        p_prefix = "" if mode == 'change' else "k_"
        t_m_key = 'train_months' if mode == 'change' else 'k_train_months'
        default_t_m = default_hp.get(t_m_key, default_hp.get('train_months', 6 if mode == 'keep' else 3))
        target_val = 0 if mode == 'change' else 1
        mode_label = "変更予測" if mode == 'change' else "据え置き予測"
        version_prefix = "v3.0(変更)" if mode == 'change' else "v3.0(据え)"
        target_col = 'prediction_score' if mode == 'change' else 'sueoki_score'
        
        train_mode_mask = train_df['is_prev_high_reg'] == target_val
        t_df = train_df[train_mode_mask].copy()
        
        pred_mode_mask = predict_df['is_prev_high_reg'] == target_val if not predict_df.empty else pd.Series(dtype=bool)
        p_df = predict_df[pred_mode_mask].copy() if not predict_df.empty else pd.DataFrame()
        
        if t_df.empty:
            continue
            
        current_features = change_features if mode == 'change' else keep_features
        current_cat_features = [f for f in cat_features if f in current_features]

        # 共通モデル用にデフォルトの学習期間で絞り込んだデータを作成
        if '対象日付' in t_df.columns and not t_df.empty:
            max_d = t_df['対象日付'].max()
            cutoff = max_d - pd.DateOffset(months=default_t_m)
            train_df_common = t_df[t_df['対象日付'] >= cutoff].copy()
        else:
            train_df_common = t_df.copy()
            
        if train_df_common.empty:
            continue

        X = train_df_common[current_features]
        y = train_df_common['target']
        
        sample_weights = None
        if '対象日付' in train_df_common.columns:
            max_date = train_df_common['対象日付'].max()
            days_diff = (max_date - train_df_common['対象日付']).dt.days
            sample_weights = 0.985 ** days_diff


        n_est = default_hp.get(f'{p_prefix}n_estimators', default_hp.get('n_estimators', 300))
        lr = default_hp.get(f'{p_prefix}learning_rate', default_hp.get('learning_rate', 0.03))
        nl = default_hp.get(f'{p_prefix}num_leaves', default_hp.get('num_leaves', 15))
        md = default_hp.get(f'{p_prefix}max_depth', default_hp.get('max_depth', 4))
        mcs = default_hp.get(f'{p_prefix}min_child_samples', default_hp.get('min_child_samples', 50))
        r_alpha = default_hp.get(f'{p_prefix}reg_alpha', default_hp.get('reg_alpha', 0.0))
        r_lambda = default_hp.get(f'{p_prefix}reg_lambda', default_hp.get('reg_lambda', 0.0))

        # --- 全店舗共通モデルの学習と推論 ---
        y_reg_common = train_df_common['next_diff'].clip(lower=-3000, upper=4000)
        reg_model = lgb.LGBMRegressor(
            objective='huber',
            random_state=42, verbose=-1, 
            n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
            reg_alpha=r_alpha, reg_lambda=r_lambda,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8
        )
        reg_model.fit(X, y_reg_common, sample_weight=sample_weights, categorical_feature=current_cat_features)
        
        # 【スタッキング】回帰モデルの予測差枚数を特徴量に追加
        X_stacked = X.copy()
        X_stacked['predicted_diff'] = reg_model.predict(X)
        stacked_features = current_features + ['predicted_diff']

        model = lgb.LGBMClassifier(
            objective='binary', random_state=42, verbose=-1, 
            n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
            reg_alpha=r_alpha, reg_lambda=r_lambda,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02
        )
        
        # 分類モデルは放置台を除外したデータ(valid_play_mask=True)のみで学習させる
        train_df_common_cls = train_df_common[train_df_common['valid_play_mask']]
        if len(train_df_common_cls) > 0 and train_df_common_cls['target'].sum() > 0:
            X_cls = X_stacked.loc[train_df_common_cls.index]
            y_cls = y.loc[train_df_common_cls.index]
            sw_cls = sample_weights.loc[train_df_common_cls.index] if sample_weights is not None else None
            model.fit(X_cls, y_cls, sample_weight=sw_cls, categorical_feature=current_cat_features)
            
            if not p_df.empty:
                p_df['予測差枚数'] = reg_model.predict(p_df[current_features]).astype(int)
                X_pred_stacked = p_df[current_features].copy()
                X_pred_stacked['predicted_diff'] = p_df['予測差枚数']
                p_df[target_col] = model.predict_proba(X_pred_stacked)[:, 1]
                p_df['ai_version'] = f"{version_prefix}(共通+ST)"
            if not t_df.empty:
                t_df['予測差枚数'] = reg_model.predict(t_df[current_features]).astype(int)
                X_train_stacked = t_df[current_features].copy()
                X_train_stacked['predicted_diff'] = t_df['予測差枚数']
                t_df[target_col] = model.predict_proba(X_train_stacked)[:, 1]

            corrs_all = get_correlations(train_df_common.assign(predicted_diff=X_stacked['predicted_diff']), stacked_features)
            feature_importances_list.append(pd.DataFrame({
                'shop_name': f"全店舗({mode_label})",
                'category': '全体',
                'feature': stacked_features,
                'importance': model.feature_importances_,
                'correlation': corrs_all
            }))
        
        # --- 店舗個別モデルの学習と推論の上書き ---
        if shop_col:
            for shop in t_df[shop_col].unique():
                # 単体型（点配分）の店舗の場合、並びや島に関する特徴量をAIに無視させる
                shop_features = current_features.copy()
                if alloc_types.get(shop, {}).get("is_point", False):
                    ignore_features = [
                        'is_neighbor_high_reg', 'neighbor_reg_reliability_score', 'neighbor_high_setting_count',
                        'past_island_reg_prob', 'is_main_island', 'is_wall_island'
                    ]
                    shop_features = [f for f in shop_features if f not in ignore_features]
                shop_cat_features = [f for f in cat_features if f in shop_features]

                shop_hp = shop_hyperparams.get(shop, default_hp)
                t_m = shop_hp.get(t_m_key, default_t_m)
                s_n_est = shop_hp.get(f'{p_prefix}n_estimators', n_est)
                s_lr = shop_hp.get(f'{p_prefix}learning_rate', lr)
                s_nl = shop_hp.get(f'{p_prefix}num_leaves', nl)
                s_md = shop_hp.get(f'{p_prefix}max_depth', md)
                s_mcs = shop_hp.get(f'{p_prefix}min_child_samples', mcs)
                s_ra = shop_hp.get(f'{p_prefix}reg_alpha', r_alpha)
                s_rl = shop_hp.get(f'{p_prefix}reg_lambda', r_lambda)
                
                shop_df_full = t_df[t_df[shop_col] == shop]
                if not shop_df_full.empty and '対象日付' in shop_df_full.columns:
                    s_max_d = shop_df_full['対象日付'].max()
                    s_cutoff = s_max_d - pd.DateOffset(months=t_m)
                    shop_train = shop_df_full[shop_df_full['対象日付'] >= s_cutoff].copy()
                else:
                    shop_train = shop_df_full.copy()
                    
                shop_train_cls = shop_train[shop_train['valid_play_mask']]
                y_shop_check = shop_train_cls['target']
                
                # サンプル数条件をやや緩和 (分割によりデータ数が半減するため)
                if len(shop_train_cls) >= 100 and y_shop_check.sum() >= 3:
                    X_shop = shop_train[shop_features]
                    sw_shop = None
                    if '対象日付' in shop_train.columns:
                        s_max_date = shop_train['対象日付'].max()
                        s_days_diff = (s_max_date - shop_train['対象日付']).dt.days
                        sw_shop = 0.985 ** s_days_diff
                    
                    shop_model = lgb.LGBMClassifier(
                        objective='binary', random_state=42, verbose=-1, 
                        n_estimators=s_n_est, learning_rate=s_lr, num_leaves=s_nl, max_depth=s_md, min_child_samples=s_mcs,
                        reg_alpha=s_ra, reg_lambda=s_rl,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02
                    )
                    shop_reg = lgb.LGBMRegressor(
                        objective='huber',
                        random_state=42, verbose=-1, 
                        n_estimators=s_n_est, learning_rate=s_lr, num_leaves=s_nl, max_depth=s_md, min_child_samples=s_mcs,
                        reg_alpha=s_ra, reg_lambda=s_rl,
                        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02
                    )
                    
                    try:
                        y_shop_reg = shop_train['next_diff'].clip(lower=-3000, upper=4000)
                        shop_reg.fit(X_shop, y_shop_reg, sample_weight=sw_shop, categorical_feature=shop_cat_features)
                        
                        X_shop_stacked = X_shop.copy()
                        X_shop_stacked['predicted_diff'] = shop_reg.predict(X_shop)
                        
                        X_shop_cls = X_shop_stacked.loc[shop_train_cls.index]
                        y_shop_cls = shop_train_cls['target']
                        sw_shop_cls = sw_shop.loc[shop_train_cls.index] if sw_shop is not None else None
                        shop_model.fit(X_shop_cls, y_shop_cls, sample_weight=sw_shop_cls, categorical_feature=shop_cat_features)
                        shop_stacked_features = shop_features + ['predicted_diff']
                        corrs_shop = get_correlations(shop_train.assign(predicted_diff=X_shop_stacked['predicted_diff']), shop_stacked_features)
                        feature_importances_list.append(pd.DataFrame({
                            'shop_name': f"{shop}({mode_label})",
                            'category': '店舗',
                            'feature': shop_stacked_features,
                            'importance': shop_model.feature_importances_,
                            'correlation': corrs_shop
                        }))
                        
                        if not p_df.empty:
                            shop_pred_idx = p_df[p_df[shop_col] == shop].index
                            if len(shop_pred_idx) > 0:
                                p_df.loc[shop_pred_idx, '予測差枚数'] = shop_reg.predict(p_df.loc[shop_pred_idx, shop_features]).astype(int)
                                X_shop_pred_stacked = p_df.loc[shop_pred_idx, shop_features].copy()
                                X_shop_pred_stacked['predicted_diff'] = p_df.loc[shop_pred_idx, '予測差枚数']
                                p_df.loc[shop_pred_idx, target_col] = shop_model.predict_proba(X_shop_pred_stacked)[:, 1]
                                p_df.loc[shop_pred_idx, 'ai_version'] = f"{version_prefix}(m{t_m}_n{s_n_est})"
                        if not t_df.empty:
                            shop_train_idx = t_df[t_df[shop_col] == shop].index
                            if len(shop_train_idx) > 0:
                                t_df.loc[shop_train_idx, '予測差枚数'] = shop_reg.predict(t_df.loc[shop_train_idx, shop_features]).astype(int)
                                X_shop_train_stacked = t_df.loc[shop_train_idx, shop_features].copy()
                                X_shop_train_stacked['predicted_diff'] = t_df.loc[shop_train_idx, '予測差枚数']
                                t_df.loc[shop_train_idx, target_col] = shop_model.predict_proba(X_shop_train_stacked)[:, 1]
                    except: pass

        # ループの最後で元のDataFrameに結果をマージ
        if not p_df.empty:
            predict_df.loc[p_df.index, '予測差枚数'] = p_df['予測差枚数']
            predict_df.loc[p_df.index, target_col] = p_df[target_col]
            predict_df.loc[p_df.index, 'ai_version'] = p_df['ai_version']
            
        if not t_df.empty:
            train_df.loc[t_df.index, '予測差枚数'] = t_df['予測差枚数']
            train_df.loc[t_df.index, target_col] = t_df[target_col]

    feature_importances = pd.concat(feature_importances_list, ignore_index=True) if feature_importances_list else pd.DataFrame()
    
    # 欠損値(NaN)のままの行を埋める(万が一予測できなかった台用)
    if not predict_df.empty:
        predict_df['予測差枚数'] = predict_df['予測差枚数'].fillna(0).astype(int)
        predict_df['prediction_score'] = predict_df['prediction_score'].fillna(0.0)
        predict_df['sueoki_score'] = predict_df['sueoki_score'].fillna(0.0)
    if not train_df.empty:
        train_df['予測差枚数'] = train_df['予測差枚数'].fillna(0).astype(int)
        train_df['prediction_score'] = train_df['prediction_score'].fillna(0.0)
        train_df['sueoki_score'] = train_df['sueoki_score'].fillna(0.0)
        
    return predict_df, train_df, feature_importances