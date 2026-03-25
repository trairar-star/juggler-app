import backend
import unicodedata
import gspread.utils

def main():
    print("🔍 スプレッドシートの過去データをチェックしています...")
    try:
        # backend.pyの認証情報をそのまま使い回す
        gc = backend._get_gspread_client()
        sh = gc.open_by_key(backend.SPREADSHEET_KEY)
        worksheet = sh.worksheet(backend.SHEET_NAME)
        
        header = worksheet.row_values(1)
        if '機種名' not in header:
            print("❌ エラー: 機種名カラムが見つかりません。")
            return
            
        col_idx = header.index('機種名') + 1
        machine_names = worksheet.col_values(col_idx)
        
        specs = backend.get_machine_specs()
        new_machine_names = []
        update_count = 0
        
        for i, name in enumerate(machine_names):
            if i == 0 or not name:
                new_machine_names.append([name])
                continue
                
            normalized_name = unicodedata.normalize('NFKC', str(name))
            matched_key = backend.get_matched_spec_key(normalized_name, specs)
            
            # 表記ゆれがあり、正規化対象となった場合のみ更新
            if matched_key != "ジャグラー（デフォルト）" and name != matched_key:
                new_machine_names.append([matched_key])
                update_count += 1
            else:
                new_machine_names.append([name])
                
        if update_count > 0:
            print(f"⚠️ {update_count}件の機種名表記ゆれを発見しました。スプレッドシートを一括修正しています...")
            col_letter = gspread.utils.rowcol_to_a1(1, col_idx).replace('1', '')
            range_str = f"{col_letter}1:{col_letter}{len(new_machine_names)}"
            
            worksheet.update(values=new_machine_names, range_name=range_str)
            
            print("🧹 ローカルの古いキャッシュを削除しています...")
            backend.clear_local_cache()
            print("✅ すべての修正が完了しました！")
        else:
            print("✨ 修正が必要なデータはありませんでした。（すべて正式名称です）")

    except Exception as e:
        print(f"❌ 実行中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()