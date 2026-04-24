import pandas as pd

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