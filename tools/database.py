"""
database.py — SQLite query tool scoped to the sandbox directory.

The agent specifies both a database filename (relative to sandbox) and
a SQL query. Only SELECT and read-like queries produce row results;
INSERT/UPDATE/DELETE return affected row counts.
"""

import json
import sqlite3
from pathlib import Path


def make_db_tool(work_dir: Path):
    _root = work_dir.resolve()

    def sql_query(database: str, query: str) -> str:
        # Resolve path inside sandbox
        try:
            db_path = (_root / database).resolve()
            if not str(db_path).startswith(str(_root)):
                return json.dumps({"status": "error",
                                   "message": "Access denied: path escapes sandbox."})
        except Exception as ex:
            return json.dumps({"status": "error", "message": str(ex)})

        if not db_path.exists():
            return json.dumps({"status": "error",
                               "message": f"Database not found: '{database}'."})

        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)

            upper = query.strip().upper()
            if upper.startswith("SELECT") or upper.startswith("PRAGMA") or upper.startswith("WITH"):
                rows = [dict(r) for r in cursor.fetchall()]
                cols = [d[0] for d in cursor.description] if cursor.description else []
                conn.close()
                return json.dumps({
                    "status": "ok",
                    "database": database,
                    "columns": cols,
                    "rows": rows,
                    "count": len(rows),
                })
            else:
                conn.commit()
                affected = cursor.rowcount
                conn.close()
                return json.dumps({
                    "status": "ok",
                    "message": f"Query executed successfully. {affected} row(s) affected.",
                })

        except PermissionError:
            return json.dumps({"status": "error",
                               "message": f"Permission denied: cannot open '{database}'."})
        except sqlite3.OperationalError as ex:
            return json.dumps({"status": "error",
                               "message": f"SQL error: {ex}"})
        except sqlite3.DatabaseError as ex:
            return json.dumps({"status": "error",
                               "message": f"Database error: {ex}"})
        except Exception as ex:
            return json.dumps({"status": "error", "message": str(ex)})

    return sql_query
