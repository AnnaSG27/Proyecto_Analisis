# Core/cap3_interpolacion.py
import numpy as np
from typing import Dict, List


# ============================================================
# POLINOMIOS: VANDERMONDE, NEWTON, LAGRANGE
# ============================================================

def polinomio_vandermonde(x: List[float], y: List[float]) -> Dict:
    """
    Construye el polinomio interpolante usando el sistema de Vandermonde.
    Retorna coeficientes en base estándar (descendentes).
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    V = np.vander(x, N=n, increasing=False)  # columnas: x^{n-1},...,x^0
    coef = np.linalg.solve(V, y)

    return {"coef": coef, "grado": n - 1}


def polinomio_newton(x: List[float], y: List[float]) -> Dict:
    """
    Construye el polinomio de Newton por diferencias divididas
    y lo convierte a coeficientes estándar (descendentes).
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    # Tabla de diferencias divididas
    dd = np.zeros((n, n), dtype=float)
    dd[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (x[i + j] - x[i])

    coef_newton = dd[0, :]  # a0, a1, ..., a_{n-1}

    # Convertir a base estándar usando poly1d
    p = np.poly1d([0.0])
    for k in range(n):
        term = np.poly1d([1.0])
        for j in range(k):
            term *= np.poly1d([1.0, -x[j]])
        p += coef_newton[k] * term

    coef_std = p.coeffs  # descendentes

    return {
        "coef": coef_std,
        "grado": len(coef_std) - 1,
        "coef_newton": coef_newton,
        "nodos": x,
    }


def polinomio_lagrange(x: List[float], y: List[float]) -> Dict:
    """
    Construye el polinomio de Lagrange y devuelve coeficientes estándar.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    p = np.poly1d([0.0])
    for i in range(n):
        Li = np.poly1d([1.0])
        for j in range(n):
            if j == i:
                continue
            Li *= np.poly1d([1.0, -x[j]]) / (x[i] - x[j])
        p += y[i] * Li

    coef = p.coeffs

    return {"coef": coef, "grado": len(coef) - 1}


# ============================================================
# SPLINES: LINEAL Y CÚBICO NATURAL
# ============================================================

def spline_lineal(x: List[float], y: List[float]) -> Dict:
    """
    Spline lineal por tramos: en cada [x_i, x_{i+1}]
    S_i(x) = a_i + b_i (x - x_i)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    if n < 2:
        raise ValueError("Se requieren al menos 2 puntos para spline lineal.")

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("Los x_i deben ser estrictamente crecientes.")

    a = y[:-1]
    b = np.diff(y) / h

    return {"tipo": "lineal", "x": x, "a": a, "b": b}


def spline_cubico_natural(x: List[float], y: List[float]) -> Dict:
    """
    Spline cúbico natural:
    S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3
    con condiciones naturales S''(x_0)=S''(x_n)=0.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    if n < 3:
        raise ValueError("Se requieren al menos 3 puntos para spline cúbico natural.")

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("Los x_i deben ser estrictamente crecientes.")

    # Sistema tridiagonal para segundas derivadas M
    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    M = np.zeros(n)
    for j in range(n - 2, -1, -1):
        M[j] = z[j] - mu[j] * M[j + 1]

    # Coeficientes por tramo
    a = y[:-1].copy()
    b = np.zeros(n - 1)
    c = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for i in range(n - 1):
        hi = h[i]
        a[i] = y[i]
        c[i] = M[i] / 2.0
        d[i] = (M[i + 1] - M[i]) / (6.0 * hi)
        b[i] = (y[i + 1] - y[i]) / hi - (2 * M[i] + M[i + 1]) * hi / 6.0

    return {"tipo": "cubico", "x": x, "a": a, "b": b, "c": c, "d": d}


# ============================================================
# EVALUACIÓN Y TEXTO
# ============================================================

def evaluar_polinomio(coef, x_eval):
    """Evalúa polinomio con coeficientes descendentes en puntos x_eval."""
    return np.polyval(coef, x_eval)


def evaluar_spline_lineal(modelo: Dict, x_eval):
    x = modelo["x"]
    a = modelo["a"]
    b = modelo["b"]

    x_eval = np.array(x_eval, dtype=float)
    y_eval = np.zeros_like(x_eval)

    for idx, xv in enumerate(x_eval):
        if xv <= x[0]:
            i = 0
        elif xv >= x[-1]:
            i = len(x) - 2
        else:
            i = np.searchsorted(x, xv) - 1

        y_eval[idx] = a[i] + b[i] * (xv - x[i])

    return y_eval


def evaluar_spline_cubico(modelo: Dict, x_eval):
    x = modelo["x"]
    a = modelo["a"]
    b = modelo["b"]
    c = modelo["c"]
    d = modelo["d"]

    x_eval = np.array(x_eval, dtype=float)
    y_eval = np.zeros_like(x_eval)

    for idx, xv in enumerate(x_eval):
        if xv <= x[0]:
            i = 0
        elif xv >= x[-1]:
            i = len(x) - 2
        else:
            i = np.searchsorted(x, xv) - 1

        t = xv - x[i]
        y_eval[idx] = a[i] + b[i] * t + c[i] * t**2 + d[i] * t**3

    return y_eval


def polinomio_to_str(coef, variable: str = "x") -> str:
    """
    Convierte coeficientes (descendentes) a texto tipo:
    c0*x**n + c1*x**(n-1) + ...
    """
    coef = np.array(coef, dtype=float)
    n = len(coef)
    terms = []

    for i, c in enumerate(coef):
        if abs(c) < 1e-14:
            continue
        power = n - i - 1
        if power == 0:
            term = f"{c:.6g}"
        elif power == 1:
            term = f"{c:.6g}*{variable}"
        else:
            term = f"{c:.6g}*{variable}**{power}"
        terms.append(term)

    if not terms:
        return "0"

    return " + ".join(terms)


def spline_lineal_to_str(modelo: Dict, variable: str = "x") -> str:
    x = modelo["x"]
    a = modelo["a"]
    b = modelo["b"]
    partes = []
    for i in range(len(a)):
        partes.append(
            f"Para {x[i]:.6g} <= {variable} <= {x[i+1]:.6g}: "
            f"{a[i]:.6g} + {b[i]:.6g}*({variable} - {x[i]:.6g})"
        )
    return "\n".join(partes)


def spline_cubico_to_str(modelo: Dict, variable: str = "x") -> str:
    x = modelo["x"]
    a = modelo["a"]
    b = modelo["b"]
    c = modelo["c"]
    d = modelo["d"]
    partes = []
    for i in range(len(a)):
        partes.append(
            f"Para {x[i]:.6g} <= {variable} <= {x[i+1]:.6g}: "
            f"{a[i]:.6g} + {b[i]:.6g}*({variable}-{x[i]:.6g})"
            f" + {c[i]:.6g}*({variable}-{x[i]:.6g})**2"
            f" + {d[i]:.6g}*({variable}-{x[i]:.6g})**3"
        )
    return "\n".join(partes)
