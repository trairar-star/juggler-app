import pandas as pd
import numpy as np
import streamlit as st # type: ignore
import backend

def render_summary_map_page(df_raw, df_island):
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.header("🗺️ 機種・島サマリーマップ")
        st.caption("各店舗の「機種別」および「島（列）別」の成績を日別で一覧表示します。")
    with col_h2:
        st.components.v1.html("""
            <button id="fs-btn" onclick="toggleFullscreen()" style="width: 100%; height: 36px; border-radius: 6px; background-color: #42A5F5; color: white; border: none; cursor: pointer; font-weight: bold; font-family: sans-serif; font-size: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                🖥️ 領域を最大化
            </button>
            <script>
            function toggleFullscreen() {
                const doc = window.parent.document;
                const mainBlock = doc.querySelector('[data-testid="stMainBlockContainer"]');
                const btn = document.getElementById('fs-btn');
                
                if (!doc.fullscreenElement) {
                    if (mainBlock) {
                        mainBlock.requestFullscreen().then(() => {
                            mainBlock.style.backgroundColor = window.getComputedStyle(doc.body).backgroundColor || '#ffffff';
                            mainBlock.style.maxWidth = '100%';
                            mainBlock.style.padding = '1rem';
                            mainBlock.style.overflowY = 'auto';
                            btn.innerHTML = '↙️ 元に戻す';
                        }).catch(err => { console.log('Error:', err); });
                    }
                } else { 
                    doc.exitFullscreen(); 
                }
            }
            
            window.parent.document.addEventListener('fullscreenchange', () => {
                const doc = window.parent.document;
                const mainBlock = doc.querySelector('[data-testid="stMainBlockContainer"]');
                const btn = document.getElementById('fs-btn');
                if (!doc.fullscreenElement) {
                    if (mainBlock) { mainBlock.style.backgroundColor = ''; mainBlock.style.maxWidth = ''; mainBlock.style.padding = ''; }
                    if (btn) btn.innerHTML = '🖥️ 領域を最大化';
                }
            });
            </script>
        """, height=50)

    if df_raw.empty:
        st.warning("データがありません。")
        st.stop()

    date_col = '対象日付'
    temp_df = df_raw.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
    available_dates = sorted(temp_df[date_col].dropna().dt.date.unique(), reverse=True)
    available_months = sorted(list(set([d.strftime('%Y-%m') for d in available_dates])), reverse=True)

    if not available_dates:
        st.warning("有効な日付データがありません。")
        st.stop()

    shop_col = '店名' if '店名' in temp_df.columns else ('店舗名' if '店舗名' in temp_df.columns else None)
    shops = ["店舗を選択してください"] + sorted(list(temp_df[shop_col].dropna().unique()))
    
    default_index = 0
    saved_shop = st.session_state.get("global_selected_shop", "店舗を選択してください")
    if saved_shop in shops:
        default_index = shops.index(saved_shop)

    with st.sidebar:
        st.markdown("---")
        st.subheader("🛠️ 表示設定")
        selected_shop = st.selectbox("🏬 店舗を選択", shops, index=default_index, key="summary_map_shop")
        
        if selected_shop != "店舗を選択してください":
            st.session_state["global_selected_shop"] = selected_shop

        st.markdown("---")
        st.subheader("📅 データ表 設定")
        table_metric = st.radio("📊 色分けの基準", ["平均差枚", "REG確率", "合計差枚"], horizontal=True)
        table_period = st.selectbox("表示期間", ["直近30日", "直近14日"] + available_months, key="summary_map_period")

    if selected_shop == "店舗を選択してください":
        st.info("👆 サイドバーから店舗を選択すると、機種・島別のサマリー表が表示されます。")
        st.stop()

    st.divider()

    df_shop = df_raw[df_raw[shop_col] == selected_shop].copy()
    df_shop['対象日付'] = pd.to_datetime(df_shop['対象日付'], errors='coerce')
    df_shop = df_shop.dropna(subset=['対象日付', '台番号'])
    df_shop['台番号'] = df_shop['台番号'].astype(str).str.replace(r'\.0$', '', regex=True)

    if table_period == "直近30日":
        max_date = df_shop['対象日付'].max()
        cutoff_date = max_date - pd.Timedelta(days=30)
        df_month = df_shop[df_shop['対象日付'] > cutoff_date].copy()
    elif table_period == "直近14日":
        max_date = df_shop['対象日付'].max()
        cutoff_date = max_date - pd.Timedelta(days=14)
        df_month = df_shop[df_shop['対象日付'] > cutoff_date].copy()
    else:
        df_month = df_shop[df_shop['対象日付'].dt.strftime('%Y-%m') == table_period].copy()

    if df_month.empty:
        st.warning("表示するデータがありません。")
        st.stop()

    weekdays_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
    df_month['day_str'] = df_month['対象日付'].dt.strftime('%m/%d') + "(" + df_month['対象日付'].dt.dayofweek.map(weekdays_map) + ")"
    
    df_month['g'] = pd.to_numeric(df_month['累計ゲーム'], errors='coerce').fillna(0)
    df_month['b'] = pd.to_numeric(df_month['BIG'], errors='coerce').fillna(0)
    df_month['r'] = pd.to_numeric(df_month['REG'], errors='coerce').fillna(0)
    df_month['diff'] = pd.to_numeric(df_month['差枚'], errors='coerce').fillna(0)

    # 機種名の統一
    specs = backend.get_machine_specs()
    df_month['機種名_統一'] = df_month['機種名'].apply(lambda x: backend.get_matched_spec_key(x, specs) if pd.notna(x) else '不明')
    
    # 同一日・同台番号の重複は最新のデータを優先する
    df_month = df_month.sort_values('対象日付', ascending=False)
    df_month = df_month.drop_duplicates(subset=['台番号', 'day_str'], keep='first')

    # 島情報の付与
    df_month = backend._apply_island_features(df_month, df_island, shop_col)

    tab_mac, tab_isl = st.tabs(["🎰 機種別 サマリー", "🏝️ 島(列)別 サマリー"])

    # 共通のCSSとJS
    custom_css_js = """
    <style>
        .scroll-container { position: relative; overflow: auto; max-height: 85vh; width: 100%; border: 1px solid #ccc; }
        .scroll-container table { border-collapse: separate; border-spacing: 0; min-width: 100%; font-size: 11px; font-family: sans-serif; text-align: center; user-select: none; -webkit-user-select: none; }
        .scroll-container th, .scroll-container td { border-right: 1px solid #ccc; border-bottom: 1px solid #ccc; padding: 4px; white-space: nowrap; }
        .scroll-container thead th { position: sticky; top: 0; background-color: #eeeeee; z-index: 10; border-top: 1px solid #ccc; }
        .scroll-container tr th:first-child, .scroll-container tr td:first-child { border-left: 1px solid #ccc; position: sticky; left: 0; background-color: #f9f9f9; z-index: 5; }
        .scroll-container thead th:first-child { z-index: 15; background-color: #eeeeee; }
        .selected-cell { box-shadow: inset 0 0 0 3px #E91E63 !important; }
        #calc-bar { position: sticky; bottom: 0; left: 0; width: 100%; background-color: rgba(30, 30, 30, 0.95); color: white; padding: 8px 15px; font-size: 14px; line-height: 1.4; font-weight: bold; z-index: 10; display: flex; visibility: hidden; opacity: 0; pointer-events: none; box-shadow: 0 -2px 8px rgba(0,0,0,0.4); border-radius: 4px 4px 0 0; justify-content: space-between; align-items: center; box-sizing: border-box; }
        #calc-bar.show { visibility: visible; opacity: 1; pointer-events: auto; }
        #calc-content { text-align: left; flex-grow: 1; font-size: 13px; line-height: 1.4; }
        #clear-btn { background-color: #F44336; color: white; border: none; border-radius: 4px; padding: 8px 12px; cursor: pointer; font-weight: bold; font-size: 12px; margin-left: 10px; flex-shrink: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
        #clear-btn:active { background-color: #D32F2F; }
        .cell-val { line-height: 1.3; padding: 2px; }
        @media (max-width: 768px) {
            .scroll-container table { font-size: 9px; }
            .scroll-container th, .scroll-container td { padding: 2px; }
            .cell-val { line-height: 1.1; padding: 1px; }
            #calc-bar { padding: 6px 8px; }
            #calc-content { font-size: 10.5px; line-height: 1.3; }
            #clear-btn { padding: 6px 8px; font-size: 11px; }
        }
    </style>
    <script>
        function setupSelection(containerId) {
            const container = document.getElementById(containerId);
            if (!container) return;
            const table = container.querySelector('table');
            if (!table) return;
            let isMouseDown = false;
            let selectionMode = true;
            
            const calcBar = container.querySelector('.calc-bar');
            const calcContent = container.querySelector('.calc-content');
            const clearBtn = container.querySelector('.clear-btn');
            
            clearBtn.addEventListener('click', () => {
                table.querySelectorAll('.selected-cell').forEach(c => c.classList.remove('selected-cell'));
                updateCalc();
            });

            function updateCalc() {
                let totalG = 0; let totalB = 0; let totalR = 0; let totalDiff = 0; let count = 0; let totalMachines = 0;
                const selected = table.querySelectorAll('.selected-cell');
                selected.forEach(td => {
                    const cellVal = td.querySelector('.cell-val');
                    if(cellVal) {
                        totalG += parseInt(cellVal.getAttribute('data-g') || '0', 10);
                        totalB += parseInt(cellVal.getAttribute('data-b') || '0', 10);
                        totalR += parseInt(cellVal.getAttribute('data-r') || '0', 10);
                        totalDiff += parseInt(cellVal.getAttribute('data-diff') || '0', 10);
                        totalMachines += parseInt(cellVal.getAttribute('data-c') || '0', 10);
                        count++;
                    }
                });

                if (count > 0) {
                    let regProbStr = totalR > 0 ? "1/" + Math.floor(totalG / totalR) : "-";
                    let bigProbStr = totalB > 0 ? "1/" + Math.floor(totalG / totalB) : "-";
                    let totProbStr = (totalB + totalR) > 0 ? "1/" + Math.floor(totalG / (totalB + totalR)) : "-";
                    let diffSign = totalDiff > 0 ? "+" : "";
                    let avgDiff = totalMachines > 0 ? Math.floor(totalDiff / totalMachines) : 0;
                    let avgDiffSign = avgDiff > 0 ? "+" : "";
                    calcContent.innerHTML = `🎰 [選択: ${count}セル / 計${totalMachines}台] ｜ 総回転: ${totalG}G ｜ 差枚: <span style="color:${totalDiff>0?'#FFCA28':'#81D4FA'}">${diffSign}${totalDiff}枚 (台平均 ${avgDiffSign}${avgDiff}枚)</span><br>BIG: ${totalB}回 (<span style="color:#FFCA28">${bigProbStr}</span>) ｜ REG: ${totalR}回 (<span style="color:#FFCA28">${regProbStr}</span>) ｜ 合算: <span style="color:#FFCA28">${totProbStr}</span>`;
                    calcBar.classList.add('show');
                } else {
                    calcBar.classList.remove('show');
                }
            }

            table.addEventListener('mousedown', function(e) {
                let td = e.target.closest('td');
                if (!td || td.cellIndex < 1) return; // 1列目(インデックス0)はヘッダ扱い
                isMouseDown = true;
                selectionMode = !td.classList.contains('selected-cell');
                if (selectionMode) { td.classList.add('selected-cell'); } else { td.classList.remove('selected-cell'); }
                updateCalc();
            });

            table.addEventListener('mouseover', function(e) {
                if (!isMouseDown) return;
                let td = e.target.closest('td');
                if (!td || td.cellIndex < 1) return;
                if (selectionMode) { td.classList.add('selected-cell'); } else { td.classList.remove('selected-cell'); }
                updateCalc();
            });

            document.addEventListener('mouseup', function(e) { isMouseDown = false; });
            
            table.addEventListener('touchstart', function(e) { isMouseDown = true; }, { passive: true });
            table.addEventListener('touchmove', function(e) {
                if (!isMouseDown) return;
                let touch = e.touches[0];
                let element = document.elementFromPoint(touch.clientX, touch.clientY);
                if (!element) return;
                let td = element.closest('td');
                if (!td || td.cellIndex < 1) return;
                if (selectionMode) { td.classList.add('selected-cell'); } else { td.classList.remove('selected-cell'); }
                updateCalc();
            }, { passive: true });
            document.addEventListener('touchend', function(e) { isMouseDown = false; });
        }
    </script>
    """

    def generate_html_table(pivot_df, index_name, container_id):
        date_cols = sorted([c for c in pivot_df.columns])
        
        def style_summary_table(row):
            styles = [''] * len(row)
            for i, col in enumerate(row.index):
                val = row[col]
                if not isinstance(val, tuple): continue
                g, b, r, diff, count = val
                if count == 0 or g == 0: continue
                
                bg_color = ""
                text_color = ""
                avg_diff = diff / count
                
                if table_metric == "平均差枚":
                    if avg_diff >= 500: bg_color = "#FFCDD2"
                    elif avg_diff >= 200: bg_color = "#FFE082"
                    elif avg_diff > 0: bg_color = "#FFF59D"
                    elif avg_diff <= -500: bg_color = "#E3F2FD"
                    
                    if avg_diff > 200: text_color = "#D32F2F"
                    elif avg_diff > 0: text_color = "#EF6C00"
                    elif avg_diff < 0: text_color = "#1565C0"
                elif table_metric == "合計差枚":
                    if diff >= 3000: bg_color = "#FFCDD2"
                    elif diff >= 1000: bg_color = "#FFE082"
                    elif diff > 0: bg_color = "#FFF59D"
                    elif diff <= -3000: bg_color = "#E3F2FD"
                    
                    if diff > 1000: text_color = "#D32F2F"
                    elif diff > 0: text_color = "#EF6C00"
                    elif diff < 0: text_color = "#1565C0"
                elif table_metric == "REG確率":
                    prob = g / r if r > 0 else 9999
                    if prob <= 260: bg_color = "#FFCDD2"
                    elif prob <= 300: bg_color = "#FFE082"
                    elif prob <= 340: bg_color = "#FFF59D"
                    
                style_str = "vertical-align: middle; "
                if bg_color: style_str += f"background-color: {bg_color}; "
                if text_color: 
                    style_str += f"color: {text_color}; "
                    if (table_metric == "平均差枚" and avg_diff > 200) or (table_metric == "合計差枚" and diff > 1000):
                        style_str += "font-weight: bold; "
                styles[i] = style_str
            return styles

        def fmt_cell(val):
            if isinstance(val, tuple):
                g, b, r, diff, count = val
                if count == 0 or g == 0: return "-"
                
                avg_diff = int(diff / count)
                prob_str = f"1/{int(g/r)}" if r > 0 else "-"
                
                return f"<div class='cell-val' data-g='{int(g)}' data-b='{int(b)}' data-r='{int(r)}' data-diff='{int(diff)}' data-c='{int(count)}'>{int(count)}台<br>{int(g)}G<br>{int(r)}R ({prob_str})<br>{int(diff):+d}枚 (台{avg_diff:+d})</div>"
            return "-"

        format_dict = {c: fmt_cell for c in date_cols}
        styled_df = pivot_df.style.apply(style_summary_table, axis=1).format(format_dict, na_rep="-")
        
        html_table = styled_df.to_html(escape=False)
        
        return f"""
        <div class='scroll-container' id='{container_id}'>
            {html_table}
            <div class="calc-bar" id="calc-bar-{container_id}">
                <div class="calc-content"></div>
                <button class="clear-btn">✖ クリア</button>
            </div>
        </div>
        <script>
            setTimeout(() => setupSelection('{container_id}'), 100);
        </script>
        """

    # --- 🎰 機種別 サマリー ---
    with tab_mac:
        mac_daily = df_month.groupby(['機種名_統一', 'day_str']).agg(
            g=('g', 'sum'), b=('b', 'sum'), r=('r', 'sum'), diff=('diff', 'sum'), count=('台番号', 'nunique')
        ).reset_index()
        
        mac_order = mac_daily.groupby('機種名_統一')['g'].sum().sort_values(ascending=False).index.tolist()
        
        records = []
        for (m, d), group in mac_daily.groupby(['機種名_統一', 'day_str']):
            records.append({'機種名': m, 'day_str': d, 'val': (group['g'].sum(), group['b'].sum(), group['r'].sum(), group['diff'].sum(), group['count'].sum())})
            
        if records:
            pivot_mac = pd.DataFrame(records).pivot(index='機種名', columns='day_str', values='val')
            date_cols = sorted(pivot_mac.columns)
            pivot_mac = pivot_mac.reindex(mac_order)
            pivot_mac = pivot_mac[date_cols]
            
            st.markdown(f"**(色分けの目安 - {table_metric})**")
            if table_metric == "平均差枚": st.markdown("🟥: 台平均+500枚以上 / 🟧: 台平均+200枚以上 / 🟨: プラス / 🟦: 台平均-500枚以下")
            elif table_metric == "合計差枚": st.markdown("🟥: 機種合計+3000枚以上 / 🟧: 機種合計+1000枚以上 / 🟨: プラス / 🟦: 機種合計-3000枚以下")
            else: st.markdown("🟥: 1/260以下 / 🟧: 1/300以下 / 🟨: 1/340以下")
            
            st.info("💡 **便利機能**: 表のセルをマウスでドラッグして複数選択すると、選択した日・機種の **合計データ** が下部に自動計算されます！")
            
            html_content_mac = custom_css_js + generate_html_table(pivot_mac, '機種名', 'mac-scroll')
            st.components.v1.html(html_content_mac, height=600, scrolling=False)
        else:
            st.info("機種別データがありません。")

    # --- 🏝️ 島(列)別 サマリー ---
    with tab_isl:
        if 'island_id' in df_month.columns:
            isl_df = df_month[df_month['island_id'] != "Unknown"].copy()
            if not isl_df.empty:
                isl_df['島名'] = isl_df['island_id'].apply(lambda x: str(x).split('_', 1)[1] if '_' in str(x) else str(x))
                
                isl_daily = isl_df.groupby(['島名', 'day_str']).agg(
                    g=('g', 'sum'), b=('b', 'sum'), r=('r', 'sum'), diff=('diff', 'sum'), count=('台番号', 'nunique')
                ).reset_index()
                
                isl_order = isl_daily.groupby('島名')['count'].max().sort_values(ascending=False).index.tolist()
                
                records = []
                for (m, d), group in isl_daily.groupby(['島名', 'day_str']):
                    records.append({'島名': m, 'day_str': d, 'val': (group['g'].sum(), group['b'].sum(), group['r'].sum(), group['diff'].sum(), group['count'].sum())})
                    
                if records:
                    pivot_isl = pd.DataFrame(records).pivot(index='島名', columns='day_str', values='val')
                    date_cols = sorted(pivot_isl.columns)
                    pivot_isl = pivot_isl.reindex(isl_order)
                    pivot_isl = pivot_isl[date_cols]
                    
                    st.markdown(f"**(色分けの目安 - {table_metric})**")
                    if table_metric == "平均差枚": st.markdown("🟥: 台平均+500枚以上 / 🟧: 台平均+200枚以上 / 🟨: プラス / 🟦: 台平均-500枚以下")
                    elif table_metric == "合計差枚": st.markdown("🟥: 島合計+3000枚以上 / 🟧: 島合計+1000枚以上 / 🟨: プラス / 🟦: 島合計-3000枚以下")
                    else: st.markdown("🟥: 1/260以下 / 🟧: 1/300以下 / 🟨: 1/340以下")
                    
                    st.info("💡 **便利機能**: 表のセルをマウスでドラッグして複数選択すると、選択した日・島の **合計データ** が下部に自動計算されます！")
                    
                    html_content_isl = custom_css_js + generate_html_table(pivot_isl, '島名', 'isl-scroll')
                    st.components.v1.html(html_content_isl, height=600, scrolling=False)
                else:
                    st.info("島別データがありません。")
            else:
                st.info("島マスターに登録された台の稼働データがありません。サイドバーの「⚙️ 島マスター管理」から島を登録してください。")
        else:
            st.info("島データがありません。")