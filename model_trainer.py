import pandas as pd
import lightgbm as lgb # type: ignore

def train_models(train_df, predict_df, features, shop_hyperparams):
    """
    LightGBMを用いて設定判別モデルを学習し、予測結果と特徴量重要度を返す
    """
    shop_col = '店名' if '店名' in train_df.columns else ('店舗名' if '店舗名' in train_df.columns else None)
    default_hp = shop_hyperparams.get("デフォルト", {'n_estimators': 300, 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 50, 'reg_alpha': 0.0, 'reg_lambda': 0.0})
    
    # 共通モデル用にデフォルトの学習期間で絞り込んだデータを作成
    default_t_m = default_hp.get('train_months', 3)
    if '対象日付' in train_df.columns and not train_df.empty:
        max_d = train_df['対象日付'].max()
        cutoff = max_d - pd.DateOffset(months=default_t_m)
        train_df_common = train_df[train_df['対象日付'] >= cutoff].copy()
    else:
        train_df_common = train_df.copy()

    X = train_df_common[features]
    y = train_df_common['target']
    
    sample_weights = None
    if '対象日付' in train_df_common.columns:
        max_date = train_df_common['対象日付'].max()
        days_diff = (max_date - train_df_common['対象日付']).dt.days
        sample_weights = 0.985 ** days_diff # 0.995から0.985に変更し、より直近の傾向を強く重視する

    n_est = default_hp.get('n_estimators', 300)
    lr = default_hp.get('learning_rate', 0.03)
    nl = default_hp.get('num_leaves', 15)
    md = default_hp.get('max_depth', 4)
    mcs = default_hp.get('min_child_samples', 50)
    r_alpha = default_hp.get('reg_alpha', 0.0)
    r_lambda = default_hp.get('reg_lambda', 0.0)

    # カテゴリ変数として扱う特徴量のリストを定義
    cat_features = [f for f in ['machine_code', 'shop_code', 'event_code', 'target_weekday', 'target_date_end_digit'] if f in features]

    # --- 全店舗共通モデルの学習と推論 ---
    y_reg_common = train_df_common['next_diff'].clip(lower=-3000, upper=4000)
    reg_model = lgb.LGBMRegressor(
        objective='huber',
        random_state=42, verbose=-1, 
        n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
        reg_alpha=r_alpha, reg_lambda=r_lambda,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8
    )
    reg_model.fit(X, y_reg_common, sample_weight=sample_weights, categorical_feature=cat_features)
    
    # 【スタッキング】回帰モデルの予測差枚数を特徴量に追加
    X_stacked = X.copy()
    X_stacked['predicted_diff'] = reg_model.predict(X)
    stacked_features = features + ['predicted_diff']

    model = lgb.LGBMClassifier(
        objective='binary', random_state=42, verbose=-1, 
        n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
        reg_alpha=r_alpha, reg_lambda=r_lambda,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02
    )
    
    # 分類モデルは放置台を除外したデータ(valid_play_mask=True)のみで学習させる
    train_df_common_cls = train_df_common[train_df_common['valid_play_mask']]
    X_cls = X_stacked.loc[train_df_common_cls.index]
    y_cls = y.loc[train_df_common_cls.index]
    sw_cls = sample_weights.loc[train_df_common_cls.index] if sample_weights is not None else None
    model.fit(X_cls, y_cls, sample_weight=sw_cls, categorical_feature=cat_features)
    
    if not predict_df.empty:
        predict_df['予測差枚数'] = reg_model.predict(predict_df[features]).astype(int)
        X_pred_stacked = predict_df[features].copy()
        X_pred_stacked['predicted_diff'] = predict_df['予測差枚数']
        predict_df['prediction_score'] = model.predict_proba(X_pred_stacked)[:, 1]
        predict_df['ai_version'] = "v2.3(共通+ST)"
    if not train_df.empty:
        train_df['予測差枚数'] = reg_model.predict(train_df[features]).astype(int)
        X_train_stacked = train_df[features].copy()
        X_train_stacked['predicted_diff'] = train_df['予測差枚数']
        train_df['prediction_score'] = model.predict_proba(X_train_stacked)[:, 1]
    
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

    corrs_all = get_correlations(train_df_common.assign(predicted_diff=X_stacked['predicted_diff']), stacked_features)
    feature_importances_list = []
    feature_importances_list.append(pd.DataFrame({
        'shop_name': '全店舗',
        'category': '全体',
        'feature': stacked_features,
        'importance': model.feature_importances_,
        'correlation': corrs_all
    }))
    
    # --- 店舗個別モデルの学習と推論の上書き ---
    if shop_col:
        for shop in train_df[shop_col].unique():
            shop_hp = shop_hyperparams.get(shop, default_hp)
            t_m = shop_hp.get('train_months', default_t_m)
            s_n_est = shop_hp.get('n_estimators', 300)
            s_lr = shop_hp.get('learning_rate', 0.03)
            s_nl = shop_hp.get('num_leaves', 15)
            s_md = shop_hp.get('max_depth', 4)
            s_mcs = shop_hp.get('min_child_samples', 50)
            s_ra = shop_hp.get('reg_alpha', 0.0)
            s_rl = shop_hp.get('reg_lambda', 0.0)
            
            shop_df_full = train_df[train_df[shop_col] == shop]
            if not shop_df_full.empty and '対象日付' in shop_df_full.columns:
                s_max_d = shop_df_full['対象日付'].max()
                s_cutoff = s_max_d - pd.DateOffset(months=t_m)
                shop_train = shop_df_full[shop_df_full['対象日付'] >= s_cutoff].copy()
            else:
                shop_train = shop_df_full.copy()
                
            shop_train_cls = shop_train[shop_train['valid_play_mask']]
            y_shop_check = shop_train_cls['target']
            
            # ノイズ過学習防止と、正例(当たり台)が少なすぎて確率が0%に張り付くバグを防ぐため、
            # 最低サンプル数150件 ＋ 正例が5件以上ある場合のみ専用モデルを構築する
            if len(shop_train_cls) >= 150 and y_shop_check.sum() >= 5:
                X_shop = shop_train[features]
                sw_shop = None
                if '対象日付' in shop_train.columns:
                    s_max_date = shop_train['対象日付'].max()
                    s_days_diff = (s_max_date - shop_train['対象日付']).dt.days
                    sw_shop = 0.985 ** s_days_diff # 店舗個別モデルも同様に直近重視へ
                
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
                    shop_reg.fit(X_shop, y_shop_reg, sample_weight=sw_shop, categorical_feature=cat_features)
                    
                    X_shop_stacked = X_shop.copy()
                    X_shop_stacked['predicted_diff'] = shop_reg.predict(X_shop)
                    
                    X_shop_cls = X_shop_stacked.loc[shop_train_cls.index]
                    y_shop_cls = shop_train_cls['target']
                    sw_shop_cls = sw_shop.loc[shop_train_cls.index] if sw_shop is not None else None
                    shop_model.fit(X_shop_cls, y_shop_cls, sample_weight=sw_shop_cls, categorical_feature=cat_features)
                    corrs_shop = get_correlations(shop_train.assign(predicted_diff=X_shop_stacked['predicted_diff']), stacked_features)
                    feature_importances_list.append(pd.DataFrame({
                        'shop_name': shop,
                        'category': '店舗',
                        'feature': stacked_features,
                        'importance': shop_model.feature_importances_,
                        'correlation': corrs_shop
                    }))
                    
                    # その店舗の推論結果を専用モデルで上書きする
                    if not predict_df.empty:
                        shop_pred_idx = predict_df[predict_df[shop_col] == shop].index
                        if len(shop_pred_idx) > 0:
                            predict_df.loc[shop_pred_idx, '予測差枚数'] = shop_reg.predict(predict_df.loc[shop_pred_idx, features]).astype(int)
                            X_shop_pred_stacked = predict_df.loc[shop_pred_idx, features].copy()
                            X_shop_pred_stacked['predicted_diff'] = predict_df.loc[shop_pred_idx, '予測差枚数']
                            predict_df.loc[shop_pred_idx, 'prediction_score'] = shop_model.predict_proba(X_shop_pred_stacked)[:, 1]
                            predict_df.loc[shop_pred_idx, 'ai_version'] = f"v2.4(m{t_m}_n{s_n_est}_d{s_md}_ra{s_ra})"
                    if not train_df.empty:
                        shop_train_idx = train_df[train_df[shop_col] == shop].index
                        if len(shop_train_idx) > 0:
                            train_df.loc[shop_train_idx, '予測差枚数'] = shop_reg.predict(train_df.loc[shop_train_idx, features]).astype(int)
                            X_shop_train_stacked = train_df.loc[shop_train_idx, features].copy()
                            X_shop_train_stacked['predicted_diff'] = train_df.loc[shop_train_idx, '予測差枚数']
                            train_df.loc[shop_train_idx, 'prediction_score'] = shop_model.predict_proba(X_shop_train_stacked)[:, 1]
                except: pass
                    
    # --- 曜日別モデルの学習 ---
    weekdays_map = {0: '月曜', 1: '火曜', 2: '水曜', 3: '木曜', 4: '金曜', 5: '土曜', 6: '日曜'}
    if 'target_weekday' in train_df_common.columns:
        for wd in sorted(train_df_common['target_weekday'].unique()):
            wd_train = train_df_common[train_df_common['target_weekday'] == wd]
            wd_train_cls = wd_train[wd_train['valid_play_mask']]
            y_wd_check = wd_train_cls['target']
            if len(wd_train_cls) >= 150 and y_wd_check.sum() >= 5:
                X_wd = wd_train[features]
                sw_wd = sample_weights.loc[wd_train.index] if sample_weights is not None and wd_train.index.isin(sample_weights.index).all() else None
                
                wd_reg = lgb.LGBMRegressor(
                    objective='huber',
                    random_state=42, verbose=-1, 
                    n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
                    reg_alpha=r_alpha, reg_lambda=r_lambda,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02
                )
                wd_model = lgb.LGBMClassifier(
                    objective='binary', random_state=42, verbose=-1, 
                    n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
                    reg_alpha=r_alpha, reg_lambda=r_lambda,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02
                )
                try:
                    y_wd_reg = wd_train['next_diff'].clip(lower=-3000, upper=4000)
                    wd_reg.fit(X_wd, y_wd_reg, sample_weight=sw_wd, categorical_feature=cat_features)
                    X_wd_stacked = X_wd.copy()
                    X_wd_stacked['predicted_diff'] = wd_reg.predict(X_wd)
                    
                    X_wd_cls = X_wd_stacked.loc[wd_train_cls.index]
                    y_wd_cls = wd_train_cls['target']
                    sw_wd_cls = sample_weights.loc[wd_train_cls.index] if sample_weights is not None and wd_train_cls.index.isin(sample_weights.index).all() else None
                    wd_model.fit(X_wd_cls, y_wd_cls, sample_weight=sw_wd_cls, categorical_feature=cat_features)
                    corrs_wd = get_correlations(wd_train.assign(predicted_diff=X_wd_stacked['predicted_diff']), stacked_features)
                    feature_importances_list.append(pd.DataFrame({
                        'shop_name': weekdays_map.get(wd, f"曜日{wd}"),
                        'category': '曜日',
                        'feature': stacked_features,
                        'importance': wd_model.feature_importances_,
                        'correlation': corrs_wd
                    }))
                except: pass
                
    # --- イベント有無別モデルの学習 ---
    if 'イベント名' in train_df_common.columns:
        train_df_ev = train_df_common.copy()
        train_df_ev['is_event'] = train_df_ev['イベント名'].apply(lambda x: '通常日' if x == '通常' else 'イベント日')
        for ev_type in ['通常日', 'イベント日']:
            ev_train = train_df_ev[train_df_ev['is_event'] == ev_type]
            ev_train_cls = ev_train[ev_train['valid_play_mask']]
            y_ev_check = ev_train_cls['target']
            if len(ev_train_cls) >= 150 and y_ev_check.sum() >= 5:
                X_ev = ev_train[features]
                sw_ev = sample_weights.loc[ev_train.index] if sample_weights is not None and ev_train.index.isin(sample_weights.index).all() else None
                
                ev_reg = lgb.LGBMRegressor(
                    objective='huber',
                    random_state=42, verbose=-1, 
                    n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
                    reg_alpha=r_alpha, reg_lambda=r_lambda,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02
                )
                ev_model = lgb.LGBMClassifier(
                    objective='binary', random_state=42, verbose=-1, 
                    n_estimators=n_est, learning_rate=lr, num_leaves=nl, max_depth=md, min_child_samples=mcs,
                    reg_alpha=r_alpha, reg_lambda=r_lambda,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, min_split_gain=0.02
                )
                try:
                    y_ev_reg = ev_train['next_diff'].clip(lower=-3000, upper=4000)
                    ev_reg.fit(X_ev, y_ev_reg, sample_weight=sw_ev, categorical_feature=cat_features)
                    X_ev_stacked = X_ev.copy()
                    X_ev_stacked['predicted_diff'] = ev_reg.predict(X_ev)
                    
                    X_ev_cls = X_ev_stacked.loc[ev_train_cls.index]
                    y_ev_cls = ev_train_cls['target']
                    sw_ev_cls = sample_weights.loc[ev_train_cls.index] if sample_weights is not None and ev_train_cls.index.isin(sample_weights.index).all() else None
                    ev_model.fit(X_ev_cls, y_ev_cls, sample_weight=sw_ev_cls, categorical_feature=cat_features)
                    corrs_ev = get_correlations(ev_train.assign(predicted_diff=X_ev_stacked['predicted_diff']), stacked_features)
                    feature_importances_list.append(pd.DataFrame({
                        'shop_name': ev_type,
                        'category': 'イベント',
                        'feature': stacked_features,
                        'importance': ev_model.feature_importances_,
                        'correlation': corrs_ev
                    }))
                except: pass

    feature_importances = pd.concat(feature_importances_list, ignore_index=True) if feature_importances_list else pd.DataFrame()
    
    return predict_df, train_df, feature_importances