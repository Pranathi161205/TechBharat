import sqlite3, json, os
db = 'catalog/catalog.db'
if not os.path.exists(db):
    print("catalog missing:", db)
    raise SystemExit(1)
conn = sqlite3.connect(db)
for r in conn.execute('SELECT id, original_filename, status, columns_json FROM datasets'):
    id, fname, status, cols = r
    print("ID:", id)
    print(" filename:", fname)
    print(" status:", status)
    try:
        print(" columns_json:", json.loads(cols) if cols else cols)
    except Exception:
        print(" columns_json(raw):", cols)
    print("-"*40)
conn.close()
