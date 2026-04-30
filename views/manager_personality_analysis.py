import pandas as pd
import numpy as np
import streamlit as st # type: ignore

def render_manager_personality_analysis(selected_shop, top_trends_df, analysis_df, df_raw_shop):
    with st.expander(f"👨‍💼 {selected_shop} 店長 性格診断 (設定投入のクセ)", expanded=True):
        st.caption("AIが発見した店癖や過去のデータから、店長がどのような台に設定を入れやすいかを性格診断風に分析します。")
        
        personality_traits = []
        
        # --- 0. 店舗の「配分型」完全マップ診断 ---
        if analysis_df is not None and not analysis_df.empty:
            import backend
            from shop_trends import diagnose_allocation_types
            specs = backend.get_machine_specs()
            shop_col = '店名' if '店名' in analysis_df.columns else '店舗名'
            alloc_types = diagnose_allocation_types(analysis_df, shop_col, specs)
            shop_alloc = alloc_types.get(selected_shop, {})
            
            if shop_alloc.get("messages"):
                st.markdown("##### 🏢 店舗の「配分型」完全マップ診断")
                st.info("💡 **AIの結論**: この店舗の配分思想のベースです。AIはこの思想を前提に予測ロジックを自動調整しています。\n\n" + "\n\n".join(shop_alloc["messages"]))
                st.divider()

        # --- 1. AIが発見した店癖に基づく性格診断 (リファクタリング) ---
        st.markdown("##### 💡 AIの分析による店癖")
        st.caption("スマートな実装への改善提案：診断ロジックをデータ構造として定義し、ループ処理で簡潔に記述しました。")
        if top_trends_df is not None and not top_trends_df.empty:
            hot_conditions = top_trends_df['条件'].tolist()
            
            trend_definitions = {
                "連続マイナス": "😇 **お詫び還元タイプ**: 負けが続いている台に救いの手を差し伸べる（上げリセットする）優しさを持っています。",
                "連続プラス": "👑 **イケイケ据え置きタイプ**: 調子が良い台はそのまま据え置く自信家です。",
                "角台": "✨ **見せびらかし屋タイプ**: 角台を出してアピールするのが好きなようです。",
                "並び": "🤝 **仕掛け人タイプ**: 並びや塊で出玉を演出し、ホールの盛り上がりを画策するのが好きなようです。",
                "隣": "🤝 **仕掛け人タイプ**: 並びや塊で出玉を演出し、ホールの盛り上がりを画策するのが好きなようです。",
                "不発": "🧐 **テクニカルタイプ**: REGは引けているのに出玉が伴わなかった不発台を、翌日しっかり上げてくる几帳面さがあります。",
            }
            
            # setを使って重複する性格診断を避ける
            found_traits = set()
            for keyword, trait in trend_definitions.items():
                if any(keyword in c for c in hot_conditions):
                    found_traits.add(trait)
            personality_traits.extend(list(found_traits))

        # --- 2. 据え置き vs 上げリセット の傾向 ---
        if analysis_df is not None and not analysis_df.empty and '連続マイナス日数' in analysis_df.columns and '連続プラス日数' in analysis_df.columns:
            sum_df = analysis_df.copy()
            sum_df['valid_high_play'] = sum_df['next_累計ゲーム'] >= 3000
            sum_df['target_rate'] = np.where(sum_df['valid_high_play'], sum_df['target'], np.nan) * 100
            
            minus_df = sum_df[sum_df['連続マイナス日数'] >= 2]
            plus_df = sum_df[sum_df['連続プラス日数'] >= 2]
            
            age_rate = minus_df['target_rate'].mean() if not minus_df.empty else 0
            sue_rate = plus_df['target_rate'].mean() if not plus_df.empty else 0
            
            age_rate = age_rate if pd.notna(age_rate) else 0
            sue_rate = sue_rate if pd.notna(sue_rate) else 0

            if age_rate > sue_rate and age_rate > 10:
                 personality_traits.append("📈 **凹み台救済派**: 連日勝っている台よりも、連日凹んでいる台を救済する（上げる）傾向が強いです。")
            elif sue_rate > age_rate and sue_rate > 10:
                 personality_traits.append("🔁 **好調台キープ派**: 凹み台よりも、連日勝っている台をそのまま使う（据え置く）傾向が強いです。")

        # --- 3. 並び・塊の投入頻度 ---
        if not df_raw_shop.empty and '対象日付' in df_raw_shop.columns and '台番号' in df_raw_shop.columns and '差枚' in df_raw_shop.columns:
            temp_df_n = df_raw_shop[['対象日付', '台番号', '差枚']].copy()
            temp_df_n['台番号'] = pd.to_numeric(temp_df_n['台番号'], errors='coerce')
            temp_df_n = temp_df_n.dropna(subset=['台番号']).sort_values(['対象日付', '台番号'])
            narabi_hit_days = 0
            narabi_total_days = temp_df_n['対象日付'].nunique()
            if narabi_total_days > 0:
                for date, group in temp_df_n.groupby('対象日付'):
                    group = group.sort_values('台番号')
                    group['is_hot'] = group['差枚'] >= 1000
                    group['block'] = (group['is_hot'] != group['is_hot'].shift()).cumsum()
                    hot_blocks = group[group['is_hot']].groupby('block').size()
                    if not hot_blocks.empty and hot_blocks.max() >= 3:
                        narabi_hit_days += 1
                narabi_hit_rate = (narabi_hit_days / narabi_total_days * 100)
                if narabi_hit_rate >= 30:
                    personality_traits.append("⛓️ **並び好き**: 3割以上の営業日で3台以上の並びを作っており、塊で設定を入れるのが好きなようです。")
        
        # --- 4. 月末のノルマ達成状況による設定の入れ方 ---
        if analysis_df is not None and not analysis_df.empty and 'shop_monthly_cumulative_diff' in analysis_df.columns:
            # 後の処理に影響を与えないように、この分析専用のDataFrameをコピーして使う
            df_budget_analysis = analysis_df.copy()
            # 月末ノルマ分析に必要なカラムを計算
            df_budget_analysis['valid_play_next'] = (pd.to_numeric(df_budget_analysis['next_累計ゲーム'], errors='coerce').fillna(0) >= 3000) | \
                                       ((pd.to_numeric(df_budget_analysis['next_累計ゲーム'], errors='coerce').fillna(0) < 3000) & \
                                        ((pd.to_numeric(df_budget_analysis['next_diff'], errors='coerce').fillna(0) <= -750) | (pd.to_numeric(df_budget_analysis['next_diff'], errors='coerce').fillna(0) >= 750)))
            df_budget_analysis['valid_next_diff'] = np.where(df_budget_analysis['valid_play_next'], df_budget_analysis['next_diff'], np.nan)

            month_end_df = df_budget_analysis[df_budget_analysis['対象日付'].dt.day >= 25].copy()
            if not month_end_df.empty:
                # 差枚は客側。客がマイナス＝店が黒字
                month_end_df['is_shop_winning'] = month_end_df['shop_monthly_cumulative_diff'] <= 0
                
                budget_stats = month_end_df.groupby('is_shop_winning').agg(
                    avg_next_diff=('valid_next_diff', 'mean'),
                    sample_days=('対象日付', 'nunique')
                ).reset_index()
                
                shop_win_stats = budget_stats[budget_stats['is_shop_winning'] == True]
                shop_lose_stats = budget_stats[budget_stats['is_shop_winning'] == False]
                
                if not shop_win_stats.empty and not shop_lose_stats.empty:
                    shop_win_diff = shop_win_stats['avg_next_diff'].iloc[0]
                    shop_lose_diff = shop_lose_stats['avg_next_diff'].iloc[0]
                    
                    if pd.notna(shop_win_diff) and pd.notna(shop_lose_diff):
                        if shop_win_diff < -50 and shop_lose_diff > 50:
                            personality_traits.append("💼 **ノルマ意識型マネージャー**: 月末に店の収支が黒字だと回収に走り、赤字だとお詫び還元する、典型的なサラリーマン店長です。")
                        elif shop_win_diff > 50 and shop_lose_diff < -50:
                            personality_traits.append("🃏 **あまのじゃく型ギャンブラー**: 月末に店の収支が黒字だとさらに放出し、赤字だと徹底的に回収する、予測不能なギャンブラー気質の店長です。")

        if personality_traits:
            st.info("\n\n".join([f"- {trait}" for trait in set(personality_traits)]))
        else:
            st.warning("この店長の性格を特定できるほどの明確なデータ傾向が見つかりませんでした。")