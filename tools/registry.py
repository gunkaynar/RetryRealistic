"""
registry.py — Builds the per-run tool registry.

Tools that depend on the sandbox (filesystem, shell, DB) are instantiated
fresh each run so they point to the correct working directory.
"""

from pathlib import Path

from tools.calculator import tool_calculator
from tools.filesystem import make_filesystem_tools
from tools.shell import make_shell_tool
from tools.database import make_db_tool
from tools.http_client import make_http_tool

# Shared JSON-schema fragments
_STR = {"type": "string"}
_DICT = {"type": "object"}


def build_tool_registry(work_dir: Path, mock_server_url: str,
                        allow_permission_repair: bool = False) -> dict:
    """
    Return a dict of {tool_name: {function, description, parameters}}.
    Called once per experiment run.
    """
    file_read, file_write, list_dir, file_exists = make_filesystem_tools(work_dir)
    shell_exec = make_shell_tool(work_dir, allow_permission_repair)
    sql_query = make_db_tool(work_dir)
    http_request = make_http_tool(mock_server_url)

    return {
        "file_read": {
            "function": file_read,
            "description": (
                "Read the contents of a file in the working directory. "
                "Returns the file content as a string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {**_STR, "description": "Relative path to the file."},
                },
                "required": ["filename"],
            },
        },
        "file_write": {
            "function": file_write,
            "description": (
                "Write text content to a file in the working directory. "
                "Creates the file (and any missing parent directories) if it does not exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {**_STR, "description": "Relative path to write to."},
                    "content": {**_STR, "description": "Text content to write."},
                },
                "required": ["filename", "content"],
            },
        },
        "list_dir": {
            "function": list_dir,
            "description": (
                "List the contents of a directory in the working directory. "
                "Returns names, types (file/dir), and sizes of all entries. "
                "Use this to discover what files are available before reading them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {**_STR, "description": "Relative path to list. Defaults to '.' (root of working directory)."},
                },
                "required": [],
            },
        },
        "file_exists": {
            "function": file_exists,
            "description": "Check whether a file or directory exists in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {**_STR, "description": "Relative path to check."},
                },
                "required": ["filename"],
            },
        },
        "shell_exec": {
            "function": shell_exec,
            "description": (
                "Execute a shell command in the working directory. "
                "Returns stdout, stderr, and exit code. "
                "Allowed commands: grep, awk, sed, sort, uniq, wc, head, tail, "
                "cat, find, cut, tr, diff, python3, echo, date, stat. "
                "Commands run with a 30-second timeout."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {**_STR, "description": "Shell command to run."},
                },
                "required": ["command"],
            },
        },
        "sql_query": {
            "function": sql_query,
            "description": (
                "Execute a SQL query against a SQLite database file in the working directory. "
                "For SELECT queries, returns column names and rows. "
                "For INSERT/UPDATE/DELETE, returns affected row count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "database": {**_STR, "description": "Relative path to the .db file."},
                    "query": {**_STR, "description": "SQL query to execute."},
                },
                "required": ["database", "query"],
            },
        },
        "http_request": {
            "function": http_request,
            "description": (
                "Make an HTTP request to an API endpoint. "
                "Returns the status code and response body. "
                "Use this to fetch external data such as exchange rates, prices, or supplier info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {**_STR, "description": "API path, e.g. '/api/rates' or '/api/suppliers'."},
                    "method": {**_STR, "description": "HTTP method: 'GET' (default) or 'POST'."},
                    "params": {
                        "type": "object",
                        "description": "Query parameters as key-value pairs, e.g. {\"ticker\": \"AAPL\"}.",
                    },
                    "body": {
                        "type": "object",
                        "description": "JSON body for POST requests.",
                    },
                },
                "required": ["path"],
            },
        },
        "calculator": {
            "function": tool_calculator,
            "description": (
                "Evaluate a mathematical expression. "
                "Supports arithmetic, sqrt, log, sin, cos, pi, e, round, abs, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {**_STR, "description": "Python math expression, e.g. '28.78 * 0.92'."},
                },
                "required": ["expression"],
            },
        },
    }
