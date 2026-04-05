import pandas as pd
import streamlit as st # type: ignore
import backend

def render_island_master_page(df_raw):
    st.header("🗺️ 島マスター管理 (角・並び設定)")
    st.caption("台が属する「島（列）」を登録することで、AIが正確な『角台』や『通路を跨がない隣台（並び）』を認識できるようになり、精度が劇的に向上します。")
    
    df_island = backend.load_island_master()
    
    with st.expander("📝 新しい島（列）を登録", expanded=True):
        st.info("💡 **登録のコツ**: 店舗を選ぶと、その店舗の全台番号がリスト化され、ポチポチ選んで登録できます！ハイフンを使った手入力（例: `501-510`）との併用も可能です。")
        
        shops = ["店舗を選択してください"]
        shop_col = '店名'
        if not df_raw.empty:
            shop_col = '店名' if '店名' in df_raw.columns else '店舗名'
            if shop_col in df_raw.columns:
                shops.extend(list(df_raw[shop_col].unique()))
                
        # 店舗選択をフォーム外に出すことで、選択に連動して台番号リストを動的に更新する
        input_shop = st.selectbox("店舗名", shops, key="island_reg_shop")
        
        machine_list = []
        registered_machines = set()
        
        if input_shop and input_shop != "店舗を選択してください":
            if not df_island.empty:
                shop_islands = df_island[df_island['店名'] == input_shop]
                for _, i_row in shop_islands.iterrows():
                    rule = str(i_row.get('台番号ルール', ''))
                    if rule and rule.strip() != '' and rule != 'nan':
                        for part in rule.split(','):
                            part = part.strip()
                            if not part: continue
                            if '-' in part:
                                try:
                                    s_str, e_str = part.split('-', 1)
                                    registered_machines.update([str(m) for m in range(int(s_str), int(e_str) + 1)])
                                except: pass
                            else:
                                registered_machines.add(str(part))
                    else:
                        try:
                            s = int(i_row.get('開始台番号', 0))
                            e = int(i_row.get('終了台番号', 0))
                            if s > 0 and e >= s:
                                registered_machines.update([str(m) for m in range(s, e + 1)])
                        except: pass

        if input_shop and input_shop != "店舗を選択してください" and not df_raw.empty:
            raw_machines = df_raw[df_raw[shop_col] == input_shop]['台番号'].astype(str).str.replace(r'\.0$', '', regex=True).unique()
            available_machines = [m for m in raw_machines if m not in registered_machines]
            try:
                machine_list = sorted(available_machines, key=lambda x: int(x))
            except:
                machine_list = sorted(available_machines)

        input_island = st.text_input("島名 (例: マイジャグA列)", placeholder="マイジャグA列", key="island_reg_name")
        
        st.markdown("**対象台番号の指定** (両方組み合わせての指定も可能です)")
        selected_machines = st.multiselect("① リストから選択 (未登録の台から複数選択可)", machine_list, help="クリックしてポチポチ選べます。すでに別の島に登録されている台番号は表示されません。", key="island_reg_machines")
        input_rule_text = st.text_input("② 手入力・範囲指定 (例: 501-510 または 786, 880-885)", placeholder="501-510", key="island_reg_rule")
        
        candidate_machines = set(selected_machines)
        if input_rule_text:
            for part in input_rule_text.split(','):
                part = part.strip()
                if not part: continue
                if '-' in part:
                    try:
                        s_str, e_str = part.split('-', 1)
                        candidate_machines.update([str(m) for m in range(int(s_str), int(e_str) + 1)])
                    except: pass
                else:
                    candidate_machines.add(str(part))
                    
        try:
            candidate_list = sorted(list(candidate_machines), key=lambda x: int(x))
        except:
            candidate_list = sorted(list(candidate_machines))
            
        corner_options = ["指定なし"] + candidate_list if candidate_list else ["対象台番号を指定してください"]
        
        st.markdown("**アピール情報 (任意)**")
        input_main_corner = st.selectbox("目立つ方の角番 (メイン通路側など)", corner_options, help="店長が一番出玉をアピールしたい場所（メイン通路側の角など）があれば指定してください。上の『対象台番号』に入力した台番号から選べます。", key="island_reg_main_corner")
        input_island_type = st.radio("島の目立ち度", ["普通", "メイン通路沿い (目立つ)", "壁側・奥 (目立たない)"], horizontal=True, key="island_reg_island_type")

        if st.button("島を登録", type="primary"):
            if input_shop == "店舗を選択してください":
                st.error("店舗を選択してください。")
            elif not input_island:
                st.error("島名を入力してください。")
            else:
                rules = []
                if selected_machines:
                    rules.append(", ".join(selected_machines))
                if str(input_rule_text).strip():
                    rules.append(str(input_rule_text).strip())
                    
                final_rule = ", ".join(rules)
                
                if not final_rule:
                    st.error("対象台番号を指定してください。")
                else:
                    main_corner_val = "指定なし" if input_main_corner == "対象台番号を指定してください" else input_main_corner
                    if backend.save_island_master(input_shop, input_island, final_rule, main_corner_val, input_island_type):
                        st.success(f"{input_shop}の島マスターを登録しました！\n\n💡 すべての登録が終わったら、サイドバーの「🔄 データ更新 (再読み込み)」を押してAIに反映させてください。")
                        for k in ["island_reg_name", "island_reg_machines", "island_reg_rule", "island_reg_main_corner", "island_reg_island_type"]:
                            if k in st.session_state: del st.session_state[k]
                        backend.load_island_master.clear()
                        st.rerun()

    if not df_island.empty:
        st.subheader("📋 登録済みの島一覧")
        
        def get_display_rule(r):
            rule = r.get('台番号ルール', '')
            if pd.notna(rule) and str(rule).strip() != '':
                return str(rule)
            s = r.get('開始台番号', '')
            e = r.get('終了台番号', '')
            return f"{s}〜{e}"
            
        df_island['対象台番号'] = df_island.apply(get_display_rule, axis=1)
        
        if 'メイン角番' not in df_island.columns: df_island['メイン角番'] = ''
        if '島属性' not in df_island.columns: df_island['島属性'] = '普通'
        
        df_island['uid_label'] = df_island['店名'].astype(str) + " | " + df_island['島名'].astype(str) + " (" + df_island['対象台番号'].astype(str) + ")"
        st.dataframe(df_island[['店名', '島名', '対象台番号', 'メイン角番', '島属性']], width="stretch", hide_index=True)
        
        st.divider()
        st.subheader("✏️ 島情報の編集・削除")
        
        def on_edit_island_change():
            target_uid = st.session_state.edit_island_target
            if target_uid:
                try:
                    t_row = df_island[df_island['登録日時'] == target_uid].iloc[0]
                    st.session_state.ei_shop = t_row['店名']
                    st.session_state.ei_name = t_row['島名']
                    
                    c_rule = str(t_row.get('台番号ルール', ''))
                    if not c_rule.strip():
                        c_rule = f"{t_row.get('開始台番号', '')}-{t_row.get('終了台番号', '')}"
                    st.session_state.ei_rule = c_rule
                    
                    st.session_state.ei_corner = str(t_row.get('メイン角番', ''))
                except:
                    pass

        edit_target = st.selectbox(
            "編集・削除する島を選択", 
            df_island['登録日時'].unique(), 
            format_func=lambda x: df_island[df_island['登録日時']==x].iloc[0]['uid_label'],
            key="edit_island_target",
            on_change=on_edit_island_change
        )
        
        if 'ei_shop' not in st.session_state and not df_island.empty:
            on_edit_island_change()
            
        if edit_target:
            target_row = df_island[df_island['登録日時'] == edit_target].iloc[0]
            
            with st.form("edit_island_form"):
                e_col1, e_col2 = st.columns(2)
                with e_col1:
                    edit_shop = st.text_input("店舗名", key="ei_shop")
                    edit_name = st.text_input("島名", key="ei_name")
                with e_col2:
                    edit_rule = st.text_input("対象台番号 (カンマ区切り、またはハイフンで範囲指定)", key="ei_rule")
                    edit_corner = st.text_input("目立つ方の角番 (指定なしの場合は空欄)", key="ei_corner")
                
                type_options = ["普通", "メイン通路沿い (目立つ)", "壁側・奥 (目立たない)"]
                current_type = str(target_row.get('島属性', '普通'))
                type_idx = type_options.index(current_type) if current_type in type_options else 0
                edit_type = st.radio("島の目立ち度", type_options, index=type_idx, horizontal=True)
                
                if st.form_submit_button("更新を保存", type="primary"):
                    if not edit_shop or not edit_name or not edit_rule:
                        st.error("店舗名、島名、対象台番号は必須です。")
                    else:
                        main_cor = "指定なし" if not edit_corner.strip() else edit_corner.strip()
                        if backend.update_island_master(edit_target, edit_shop, edit_name, edit_rule, main_cor, edit_type):
                            st.success("島情報を更新しました！\n\n💡 すべての編集が終わったら、サイドバーの「🔄 データ更新 (再読み込み)」を押してAIに反映させてください。")
                            backend.load_island_master.clear()
                            st.rerun()
                            
            with st.form("delete_island_form"):
                if st.form_submit_button("この島を削除"):
                    if backend.delete_island_master(edit_target):
                        st.success("削除しました。\n\n💡 すべての削除が終わったら、サイドバーの「🔄 データ更新 (再読み込み)」を押してAIに反映させてください。")
                        backend.load_island_master.clear()
                        st.rerun()