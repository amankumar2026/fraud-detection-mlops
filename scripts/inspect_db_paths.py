import sqlite3
import sys

c = sqlite3.connect(sys.argv[1])
tables = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
for t in tables:
    cols = [r[1] for r in c.execute(f"pragma table_info({t})")]
    for col in cols:
        if any(k in col for k in ("location", "uri", "path", "source")):
            vals = c.execute(f"select {col} from {t} where {col} is not null limit 3").fetchall()
            if vals:
                print(t, col, vals)
