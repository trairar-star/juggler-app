import pandas as pd
import numpy as np

def get_confidence_indicator(sample_size, thresholds=(10, 30)):
    """サンプル数に基づいて信頼度を示す文字列を返す"""
    if sample_size is None or pd.isna(sample_size):
        return "N/A"
    try:
        size = int(sample_size)
        if size < thresholds[0]:
            return "🔻低"
        elif size < thresholds[1]:
            return "🔸中"
        else:
            return "🔼高"
    except (ValueError, TypeError):
        return "N/A"

def get_valid_play_mask(games_series, diff_series, min_games=3000, min_diff=750):
    """
    有効稼働の判定マスク（Pandas Seriesのブール値）を返す。
    指定ゲーム数以上、または指定ゲーム数未満でも差枚の絶対値が指定枚数以上の場合にTrueとなる。
    """
    g = pd.to_numeric(games_series, errors='coerce').fillna(0)
    diff = pd.to_numeric(diff_series, errors='coerce').fillna(0)
    return (g >= min_games) | ((g < min_games) & ((diff <= -min_diff) | (diff >= min_diff)))

def get_matched_spec_key(machine_name, specs):
    """機種名から最も一致するスペックキーを探す。見つからなければデフォルトを返す"""
    if not isinstance(machine_name, str) or not machine_name:
        return "ジャグラー（デフォルト）"
    if machine_name in specs:
        return machine_name
    for spec_key in specs.keys():
        if spec_key == "ジャグラー（デフォルト）": continue
        chk_word = spec_key.split('ジャグラー')[0] if 'ジャグラー' in spec_key else spec_key
        if not chk_word: chk_word = "ガールズ" if "ガールズ" in spec_key else spec_key
        if chk_word and chk_word in machine_name:
            return spec_key
    return "ジャグラー（デフォルト）"

def classify_shop_eval(avg_diff, machine_count=None, is_prediction=False):
    """
    平均差枚から、店舗の営業状態（還元/通常/回収）を判定する共通関数。
    is_prediction=True の場合は「予測」のテキストを付加する。
    """
    if pd.isna(avg_diff):
        return "⚖️ 通常営業予測" if is_prediction else "⚖️ 通常営業"
        
    suffix = "予測" if is_prediction else ""
    if avg_diff >= 100: return f"🔥 還元日{suffix}"
    elif avg_diff <= -100: return f"🥶 回収日{suffix}"
    else: return f"⚖️ 通常営業{suffix}"

def calculate_high_setting_mask(df, specs, g_col='累計ゲーム', b_col='BIG', r_col='REG', mac_col='機種名', include_bb_filter=False):
    """
    データフレームから高設定挙動（設定5基準）のマスク(Boolean Series)を計算して返す。
    Zスコア（設定1否定度）による補正も含む。
    """
    if df.empty:
        return pd.Series(dtype=bool)

    g = pd.to_numeric(df[g_col], errors='coerce').fillna(0)
    b = pd.to_numeric(df[b_col], errors='coerce').fillna(0)
    r = pd.to_numeric(df[r_col], errors='coerce').fillna(0)

    mac_series = df[mac_col]
    unique_machines = mac_series.dropna().unique()
    
    reg5_map = {}
    tot5_map = {}
    reg3_map = {}
    reg1_map = {}
    b6_map = {}

    for m in unique_machines:
        k = get_matched_spec_key(m, specs)
        sp = specs.get(k, {})
        reg5_map[m] = sp.get('設定5', {}).get("REG", 260.0)
        tot5_map[m] = sp.get('設定5', {}).get("合算", 128.0)
        reg3_map[m] = sp.get('設定3', {}).get("REG", 300.0)
        reg1_map[m] = sp.get('設定1', {}).get("REG", 400.0)
        b6_map[m] = sp.get('設定6', {}).get("BIG", 260.0)

    spec_reg5_den = mac_series.map(reg5_map).fillna(260.0)
    spec_tot5_den = mac_series.map(tot5_map).fillna(128.0)
    spec_reg3_den = mac_series.map(reg3_map).fillna(300.0)
    spec_reg1_den = mac_series.map(reg1_map).fillna(400.0)

    reg_prob_den = np.where(r > 0, g / r, 9999)
    tot_prob_den = np.where((b + r) > 0, g / (b + r), 9999)

    p_reg1 = 1.0 / spec_reg1_den
    exp_r1 = g * p_reg1
    std_r1 = np.sqrt(g * p_reg1 * (1.0 - p_reg1))
    z_score = np.where(std_r1 > 0, (r - exp_r1) / std_r1, 0)

    cond_reg5 = (reg_prob_den > 0) & (reg_prob_den <= spec_reg5_den)
    cond_tot5_reg3 = (tot_prob_den > 0) & (tot_prob_den <= spec_tot5_den) & (reg_prob_den > 0) & (reg_prob_den <= spec_reg3_den)
    cond_zscore = z_score >= 1.64

    is_high = cond_reg5 | cond_tot5_reg3 | cond_zscore

    if include_bb_filter:
        spec_b6_den = mac_series.map(b6_map).fillna(260.0)
        big_prob_den = np.where(b > 0, g / b, 9999)
        is_high = is_high & (big_prob_den <= spec_b6_den + 100)

    return is_high