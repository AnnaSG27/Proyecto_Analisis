# Core/cap1_no_lineales.py
from typing import Callable, Dict, List, Optional

from Core.errores import error_absoluto, error_relativo, error_condicion


# ---------------------------------------------------------------------
# Utilidad para escoger el error según el tipo
# ---------------------------------------------------------------------
def _calcular_error(tipo_error: str,
                    x_new: float,
                    x_old: Optional[float],
                    fx: float) -> Optional[float]:
    if x_old is None:
        return None

    if tipo_error == "absoluto":
        return error_absoluto(x_new, x_old)
    elif tipo_error == "relativo":
        return error_relativo(x_new, x_old)
    else:  # condicion
        return error_condicion(fx)


# ---------------------------------------------------------------------
# BISECCIÓN
# ---------------------------------------------------------------------
def biseccion(f: Callable[[float], float],
              a: float,
              b: float,
              tol: float,
              max_iter: int,
              tipo_error: str = "relativo") -> Dict:
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos.")

    tabla: List[Dict] = []
    x_old: Optional[float] = None
    err: Optional[float] = None

    for n in range(1, max_iter + 1):
        xm = (a + b) / 2.0
        fxm = f(xm)

        err = _calcular_error(tipo_error, xm, x_old, fxm)

        tabla.append({
            "iter": n,
            "a": a,
            "b": b,
            "xm": xm,
            "f(xm)": fxm,
            "error": err,
        })

        if fxm == 0:
            break
        if err is not None and err < tol:
            break

        if fa * fxm < 0:
            b = xm
            fb = fxm
        else:
            a = xm
            fa = fxm

        x_old = xm

    convergio = (err is not None and err < tol) or fxm == 0

    return {
        "raiz": xm,
        "iteraciones": n,
        "tabla": tabla,
        "convergio": convergio,
    }


# ---------------------------------------------------------------------
# REGLA FALSA (False Position)
# ---------------------------------------------------------------------
def regla_falsa(f: Callable[[float], float],
                a: float,
                b: float,
                tol: float,
                max_iter: int,
                tipo_error: str = "relativo") -> Dict:
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos.")

    tabla: List[Dict] = []
    x_old: Optional[float] = None
    err: Optional[float] = None

    for n in range(1, max_iter + 1):
        # fórmula de regla falsa
        xm = b - fb * (b - a) / (fb - fa)
        fxm = f(xm)

        err = _calcular_error(tipo_error, xm, x_old, fxm)

        tabla.append({
            "iter": n,
            "a": a,
            "b": b,
            "xm": xm,
            "f(xm)": fxm,
            "error": err,
        })

        if fxm == 0:
            break
        if err is not None and err < tol:
            break

        if fa * fxm < 0:
            b = xm
            fb = fxm
        else:
            a = xm
            fa = fxm

        x_old = xm

    convergio = (err is not None and err < tol) or fxm == 0

    return {
        "raiz": xm,
        "iteraciones": n,
        "tabla": tabla,
        "convergio": convergio,
    }


# ---------------------------------------------------------------------
# PUNTO FIJO
# f(x) = 0   ->   x = g(x)
# Necesitamos f para el error de condición y g para la iteración.
# ---------------------------------------------------------------------
def punto_fijo(f: Callable[[float], float],
               g: Callable[[float], float],
               x0: float,
               tol: float,
               max_iter: int,
               tipo_error: str = "relativo") -> Dict:
    tabla: List[Dict] = []
    x = x0
    x_old: Optional[float] = None
    err: Optional[float] = None

    for n in range(1, max_iter + 1):
        x_new = g(x)
        fx = f(x_new)

        err = _calcular_error(tipo_error, x_new, x_old, fx)

        tabla.append({
            "iter": n,
            "a": None,
            "b": None,
            "xm": x_new,
            "f(xm)": fx,
            "error": err,
        })

        if fx == 0:
            break
        if err is not None and err < tol:
            break

        x_old = x_new
        x = x_new

    convergio = (err is not None and err < tol) or fx == 0

    return {
        "raiz": x_new,
        "iteraciones": n,
        "tabla": tabla,
        "convergio": convergio,
    }


# ---------------------------------------------------------------------
# NEWTON-RAPHSON
# ---------------------------------------------------------------------
def newton(f: Callable[[float], float],
           df: Callable[[float], float],
           x0: float,
           tol: float,
           max_iter: int,
           tipo_error: str = "relativo") -> Dict:
    tabla: List[Dict] = []
    x = x0
    x_old: Optional[float] = None
    err: Optional[float] = None

    for n in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError(f"Derivada cero en iteración {n}, no se puede continuar.")

        x_new = x - fx / dfx

        err = _calcular_error(tipo_error, x_new, x_old, fx)

        tabla.append({
            "iter": n,
            "a": None,
            "b": None,
            "xm": x_new,
            "f(xm)": f(x_new),
            "error": err,
        })

        if fx == 0:
            break
        if err is not None and err < tol:
            break

        x_old = x_new
        x = x_new

    convergio = (err is not None and err < tol) or fx == 0

    return {
        "raiz": x_new,
        "iteraciones": n,
        "tabla": tabla,
        "convergio": convergio,
    }


# ---------------------------------------------------------------------
# SECANTE
# ---------------------------------------------------------------------
def secante(f: Callable[[float], float],
            x0: float,
            x1: float,
            tol: float,
            max_iter: int,
            tipo_error: str = "relativo") -> Dict:
    tabla: List[Dict] = []
    x_prev = x0
    x = x1
    x_old: Optional[float] = None
    err: Optional[float] = None

    for n in range(1, max_iter + 1):
        fx = f(x)
        fx_prev = f(x_prev)
        denom = fx - fx_prev
        if denom == 0:
            raise ValueError(f"Denominador cero en iteración {n}, no se puede continuar.")

        x_new = x - fx * (x - x_prev) / denom
        fx_new = f(x_new)

        err = _calcular_error(tipo_error, x_new, x_old, fx_new)

        tabla.append({
            "iter": n,
            "a": None,
            "b": None,
            "xm": x_new,
            "f(xm)": fx_new,
            "error": err,
        })

        if fx_new == 0:
            break
        if err is not None and err < tol:
            break

        x_prev, x = x, x_new
        x_old = x_new

    convergio = (err is not None and err < tol) or fx_new == 0

    return {
        "raiz": x_new,
        "iteraciones": n,
        "tabla": tabla,
        "convergio": convergio,
    }


# ---------------------------------------------------------------------
# NEWTON PARA RAÍCES MÚLTIPLES
# x_{n+1} = x_n - m * f(x_n) / f'(x_n)
# m: multiplicidad conocida (por ejemplo 2, 3, ...)
# ---------------------------------------------------------------------
def newton_multiples(f: Callable[[float], float],
                     df: Callable[[float], float],
                     m: int,
                     x0: float,
                     tol: float,
                     max_iter: int,
                     tipo_error: str = "relativo") -> Dict:
    if m <= 1:
        raise ValueError("La multiplicidad m debe ser mayor que 1.")

    tabla: List[Dict] = []
    x = x0
    x_old: Optional[float] = None
    err: Optional[float] = None

    for n in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError(f"Derivada cero en iteración {n}, no se puede continuar.")

        x_new = x - m * fx / dfx
        fx_new = f(x_new)

        err = _calcular_error(tipo_error, x_new, x_old, fx_new)

        tabla.append({
            "iter": n,
            "a": None,
            "b": None,
            "xm": x_new,
            "f(xm)": fx_new,
            "error": err,
        })

        if fx_new == 0:
            break
        if err is not None and err < tol:
            break

        x_old = x_new
        x = x_new

    convergio = (err is not None and err < tol) or fx_new == 0

    return {
        "raiz": x_new,
        "iteraciones": n,
        "tabla": tabla,
        "convergio": convergio,
    }
