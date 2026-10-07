import json
import math


def tool_calculator(expression: str) -> str:
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "len": len, "pow": pow, "int": int, "float": float,
        "pi": math.pi, "e": math.e, "sqrt": math.sqrt, "log": math.log,
        "log10": math.log10, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "ceil": math.ceil, "floor": math.floor,
    }
    try:
        # Names must live in globals: comprehensions/genexps get their own
        # scope that resolves free names against globals, not outer locals.
        result = eval(expression, {"__builtins__": {}, **allowed_names})
        return json.dumps({"status": "ok", "result": result})
    except Exception as ex:
        return json.dumps({"status": "error", "message": str(ex)})
