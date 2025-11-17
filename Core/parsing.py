# Core/parsing.py
import sympy as sp

x = sp.Symbol('x')

def construir_funcion(expr_str: str):
    """
    Recibe un string como 'sin(x) - x/2' y devuelve:
    - expr: expresión simbólica de sympy
    - f: función numérica f(x) usable con float o numpy
    """
    try:
        expr = sp.sympify(expr_str)
    except Exception as e:
        raise ValueError(f"Expresión inválida: {e}")

    f = sp.lambdify(x, expr, "numpy")
    return expr, f


def construir_derivada(expr_str: str):
    """
    A partir de f(x) en texto, devuelve:
    - deriv_expr: derivada simbólica
    - df: función numérica de la derivada
    """
    expr = sp.sympify(expr_str)
    deriv_expr = sp.diff(expr, x)
    df = sp.lambdify(x, deriv_expr, "numpy")
    return deriv_expr, df
