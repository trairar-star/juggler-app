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