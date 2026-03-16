import pandas as pd
import streamlit as st # type: ignore
import backend

def render_island_master_page(df_raw):
    st.header("🗺️ 島マスター管理 (角・並び設定)")
    st.caption("台が属する「島（列）」を登録することで、AIが正確な『角台』や『通路を跨がない隣台（並び）』を認識できるようになり、精度が劇的に向上します。")
    
    df_island = backend.load_island_master()
    
    with st.expander("📝 新しい島（列）を登録", expanded=True):
        st.info("💡 **登録のコツ**: ハイフン（範囲）とカンマ（区切り）を組み合わせて、複雑な形でも指定できます。\n\n例1: `501-510` (一般的な10台島)\n例2: `785-786, 880-885` (特殊な並びや飛び番)")
        with st.form("island_form", clear_on_submit=True):
            shops = []
            if not df_raw.empty:
                shop_col = '店名' if '店名' in df_raw.columns else '店舗名'
                if shop_col in df_raw.columns:
                    shops = list(df_raw[shop_col].unique())
            input_shop = st.selectbox("店舗名", shops)
            input_island = st.text_input("島名 (例: マイジャグA列)", placeholder="マイジャグA列")
            input_rule = st.text_input("対象台番号 (例: 501-510 または 786, 880-885)", placeholder="501-510, 786, 880")
            
            submitted = st.form_submit_button("島を登録", type="primary")
            if submitted:
                if not input_island:
                    st.error("島名を入力してください。")
                elif not input_rule:
                    st.error("対象台番号を入力してください。")
                else:
                    if backend.save_island_master(input_shop, input_island, input_rule):
                        st.success(f"{input_shop}の島マスターを登録しました！")
                        st.cache_data.clear()
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
        df_island['uid_label'] = df_island['店名'].astype(str) + " | " + df_island['島名'].astype(str) + " (" + df_island['対象台番号'].astype(str) + ")"
        st.dataframe(df_island[['店名', '島名', '対象台番号']], use_container_width=True, hide_index=True)
        
        with st.form("delete_island_form"):
            target = st.selectbox("削除する島を選択", df_island['登録日時'].unique(), format_func=lambda x: df_island[df_island['登録日時']==x].iloc[0]['uid_label'])
            if st.form_submit_button("削除"):
                if backend.delete_island_master(target):
                    st.success("削除しました。")
                    st.cache_data.clear()
                    st.rerun()