import pandas as pd
import streamlit as st # type: ignore
import backend

def render_event_management_page(df_raw):
    st.header("📝 イベント管理")
    st.caption("店舗独自のイベント（取材、特定日など）を登録・編集・削除します。")

    # --- イベント登録フォーム ---
    with st.expander("新しいイベントを登録", expanded=False):
        shop_col = '店名' if '店名' in df_raw.columns else '店舗名'
        if shop_col in df_raw.columns:
            unique_shops = df_raw[shop_col].unique()
            
            with st.form("event_reg_form", clear_on_submit=True):
                reg_shop = st.selectbox("店舗", unique_shops)
                reg_date = st.date_input("日付", pd.Timestamp.now(tz='Asia/Tokyo').date())
                reg_name = st.text_input("イベント名 (例: ○○取材, 周年, リニューアル)", help="※イベント名に『周年』が含まれる場合、登録時の「年」は無視され、毎年自動的にループ適用されます。")
                reg_rank = st.selectbox("イベントの強さ (期待度)", ["SS (周年)", "S", "A", "B", "C"], index=1, help="SS:周年・グランド・リニューアル等, S:激アツ, A:強い(新台6台〜など), B:普通(新台3〜5台), C:弱め(新台1〜2台)")
                
                st.markdown("**対象の絞り込み**")
                reg_type = st.radio("イベント種別", ["全体", "スロット専用", "パチンコ専用", "対象外(無効)"], horizontal=True, help="「パチンコ専用」にすると、AIは『パチンコの特日』という目印をつけて学習し、過去の傾向に基づいてスロットが回収されるか判断します。出玉に一切関係ない場合は「対象外(無効)」にしてください。")
                machine_list = ["指定なし", "ジャグラー全体", "ジャグラー以外 (パチスロ他機種)"]
                if '機種名' in df_raw.columns:
                    machine_list.extend(sorted(list(df_raw['機種名'].dropna().unique())))
                reg_target = st.selectbox("対象機種 (新台入替や特定機種イベントの場合)", machine_list, index=0, help="新台入替や特定機種のイベントの場合、ここに対象機種を指定してください。ジャグラー以外の新台入替なら『ジャグラー以外』を選べます。")
                
                submitted = st.form_submit_button("イベントを登録")
                
                if submitted:
                    t_mac = reg_target.strip() if reg_target.strip() else "指定なし"
                    if backend.save_shop_event(reg_shop, reg_date, reg_name, reg_rank, reg_type, t_mac):
                        st.success(f"{reg_shop} のイベントを登録しました！\n\n💡 **続けて登録できます。**\nすべての登録が終わったら、サイドバーの「🔄 データ更新 (再読み込み)」を押してAIに反映させてください。")
                        backend.load_shop_events.clear()
                        st.rerun()
        else:
            st.warning("店舗データが見つからないため、イベントを登録できません。")
    
    df_events = backend.load_shop_events()
    st.subheader("登録済みイベント一覧")
    
    if df_events.empty:
        st.info("現在、登録されているイベントはありません。")
        return

    # 削除用の一意キーと表示用日付を作成
    df_events['date_str'] = df_events['イベント日付'].dt.strftime('%Y-%m-%d')
    df_events['uid'] = df_events['店名'] + " | " + df_events['date_str'] + " | " + df_events['イベント名']
    
    # 表示用データ (日付降順)
    df_display = df_events.sort_values(['イベント日付', '店名'], ascending=[False, True])
    
    if 'イベント種別' not in df_display.columns: df_display['イベント種別'] = '全体'
    df_display['イベント種別'] = df_display['イベント種別'].replace('スロット/全体', '全体')
    if '対象機種' not in df_display.columns: df_display['対象機種'] = '指定なし'

    st.dataframe(
        df_display[['イベント日付', '店名', 'イベント名', 'イベントランク', 'イベント種別', '対象機種']],
        column_config={
            "イベント日付": st.column_config.DateColumn("日付", format="YYYY-MM-DD"),
            "店名": st.column_config.TextColumn("店舗"),
            "イベント名": st.column_config.TextColumn("イベント"),
            "イベントランク": st.column_config.TextColumn("ランク"),
            "イベント種別": st.column_config.TextColumn("種別"),
            "対象機種": st.column_config.TextColumn("対象機種"),
        },
        width="stretch",
        hide_index=True
    )
    
    st.divider()
    st.subheader("✏️ イベント編集")
    
    edit_target_uid = st.selectbox("編集するイベントを選択", df_display['uid'].unique(), key="edit_target")
    if edit_target_uid:
        target_row = df_display[df_display['uid'] == edit_target_uid].iloc[0]
        
        with st.form("edit_event_form"):
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                edit_shop = st.text_input("店舗名", value=target_row['店名'])
                try:
                    default_date = pd.to_datetime(target_row['イベント日付']).date()
                except:
                    default_date = pd.Timestamp.now(tz='Asia/Tokyo').date()
                edit_date = st.date_input("イベント日付", value=default_date, key="edit_date")
            with e_col2:
                edit_name = st.text_input("イベント名", value=target_row['イベント名'], help="※イベント名に『周年』が含まれる場合、登録時の「年」は無視され、毎年自動的にループ適用されます。")
                rank_options = ["SS (周年)", "S", "A", "B", "C"]
                current_rank = target_row.get('イベントランク', 'A')
                idx = rank_options.index(current_rank) if current_rank in rank_options else 1
                edit_rank = st.selectbox("イベントの強さ", rank_options, index=idx, help="SS:周年・グランド・リニューアル等, S:激アツ, A:強い(新台6台〜など), B:普通(新台3〜5台), C:弱め(新台1〜2台)")
                
            st.markdown("**対象の絞り込み**")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                type_options = ["全体", "スロット専用", "パチンコ専用", "対象外(無効)"]
                current_type = target_row.get('イベント種別', '全体')
                if current_type == 'スロット/全体': current_type = '全体'
                t_idx = type_options.index(current_type) if current_type in type_options else 0
                edit_type = st.radio("イベント種別", type_options, index=t_idx, horizontal=True, help="「パチンコ専用」にすると『パチンコの特日』として学習し、過去の実績からスロットの期待度を判断します。")
            with t_col2:
                current_target = target_row.get('対象機種', '指定なし')
                machine_list = ["指定なし", "ジャグラー全体", "ジャグラー以外 (パチスロ他機種)"]
                if '機種名' in df_raw.columns:
                    machine_list.extend(sorted(list(df_raw['機種名'].dropna().unique())))
                if current_target not in machine_list:
                    machine_list.insert(0, current_target)
                target_idx = machine_list.index(current_target)
                
                edit_target = st.selectbox("対象機種", machine_list, index=target_idx, help="特定機種のイベントの場合、対象外の機種には『別機種の特日』という目印をつけて学習します。")
                
            if st.form_submit_button("更新を保存", type="primary"):
                e_t_mac = edit_target.strip() if edit_target.strip() else "指定なし"
                if backend.update_shop_event(target_row['店名'], default_date, target_row['イベント名'], edit_shop, edit_date, edit_name, edit_rank, edit_type, e_t_mac):
                    st.success("イベントを更新しました！\n\n💡 すべての編集が終わったら、サイドバーの「🔄 データ更新 (再読み込み)」を押してAIに反映させてください。")
                    backend.load_shop_events.clear()
                    st.rerun()
                else:
                    st.error("イベントの更新に失敗しました。")

    st.divider()
    st.subheader("🗑 イベント削除")
    
    # 削除フォーム
    with st.form("delete_event_form"):
        target_uid = st.selectbox("削除するイベントを選択", df_display['uid'].unique())
        if st.form_submit_button("削除実行", type="primary"):
            # 選択されたIDから元データを復元
            target_row = df_display[df_display['uid'] == target_uid].iloc[0]
            
            if backend.delete_shop_event(target_row['店名'], target_row['イベント日付'], target_row['イベント名']):
                st.success(f"削除しました: {target_uid}\n\n💡 すべての削除が終わったら、サイドバーの「🔄 データ更新 (再読み込み)」を押してAIに反映させてください。")
                backend.load_shop_events.clear()
                st.rerun()
            else:
                st.error("削除に失敗しました。")