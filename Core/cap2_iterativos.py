# Core/cap2_iterativos.py
import numpy as np
from typing import Dict, List, Optional


def _validar_matriz_y_vectores(A, b, x0=None):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    n, m = A.shape

    if n != m:
        raise ValueError("La matriz A debe ser cuadrada.")
    if n > 7:
        raise ValueError("El tamaño máximo permitido es 7x7.")
    if b.shape[0] != n:
        raise ValueError("El tamaño de b debe coincidir con el de A.")

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float).reshape(-1)
        if x.shape[0] != n:
            raise ValueError("El tamaño de x0 debe coincidir con el de A.")

    return A, b, x, n


def _radio_espectral(T: np.ndarray) -> float:
    vals = np.linalg.eigvals(T)
    return float(max(abs(vals)))


# ---------------------------------------------------------------------
# JACOBI
# ---------------------------------------------------------------------
def jacobi(A, b, x0=None, tol: float = 1e-6, max_iter: int = 100) -> Dict:
    A, b, x, n = _validar_matriz_y_vectores(A, b, x0)

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    if np.any(np.isclose(np.diag(D), 0.0)):
        raise ValueError("Jacobi: hay ceros en la diagonal de A. No se puede invertir D.")

    D_inv = np.linalg.inv(D)
    T = -D_inv @ (L + U)
    c = D_inv @ b

    rho = _radio_espectral(T)

    tabla: List[Dict] = []
    err: Optional[float] = None

    for k in range(1, max_iter + 1):
        x_new = T @ x + c
        err = float(np.linalg.norm(x_new - x, ord=np.inf))
        tabla.append({
            "iter": k,
            "x": x_new.copy(),
            "error": err,
        })
        x = x_new
        if err < tol:
            break

    convergio = err is not None and err < tol

    return {
        "sol": x_new,
        "iteraciones": k,
        "tabla": tabla,
        "radio_espectral": rho,
        "convergio": convergio,
    }


# ---------------------------------------------------------------------
# GAUSS-SEIDEL
# ---------------------------------------------------------------------
def gauss_seidel(A, b, x0=None, tol: float = 1e-6, max_iter: int = 100) -> Dict:
    A, b, x, n = _validar_matriz_y_vectores(A, b, x0)

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    DL = D + L
    if np.linalg.det(DL) == 0:
        raise ValueError("Gauss-Seidel: la matriz (D+L) es singular.")

    DL_inv = np.linalg.inv(DL)
    T = -DL_inv @ U
    c = DL_inv @ b

    rho = _radio_espectral(T)

    tabla: List[Dict] = []
    err: Optional[float] = None

    for k in range(1, max_iter + 1):
        x_new = T @ x + c
        err = float(np.linalg.norm(x_new - x, ord=np.inf))
        tabla.append({
            "iter": k,
            "x": x_new.copy(),
            "error": err,
        })
        x = x_new
        if err < tol:
            break

    convergio = err is not None and err < tol

    return {
        "sol": x_new,
        "iteraciones": k,
        "tabla": tabla,
        "radio_espectral": rho,
        "convergio": convergio,
    }


# ---------------------------------------------------------------------
# SOR
# ---------------------------------------------------------------------
def sor(A, b, x0=None, omega: float = 1.2, tol: float = 1e-6, max_iter: int = 100) -> Dict:
    if omega <= 0 or omega >= 2:
        raise ValueError("SOR: omega debe estar en (0, 2) para un comportamiento razonable.")

    A, b, x, n = _validar_matriz_y_vectores(A, b, x0)

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    DwL = D + omega * L
    if np.linalg.det(DwL) == 0:
        raise ValueError("SOR: la matriz (D + omega*L) es singular.")

    DwL_inv = np.linalg.inv(DwL)
    T = DwL_inv @ ((1 - omega) * D - omega * U)
    c = omega * (DwL_inv @ b)

    rho = _radio_espectral(T)

    tabla: List[Dict] = []
    err: Optional[float] = None

    for k in range(1, max_iter + 1):
        x_new = T @ x + c
        err = float(np.linalg.norm(x_new - x, ord=np.inf))
        tabla.append({
            "iter": k,
            "x": x_new.copy(),
            "error": err,
        })
        x = x_new
        if err < tol:
            break

    convergio = err is not None and err < tol

    return {
        "sol": x_new,
        "iteraciones": k,
        "tabla": tabla,
        "radio_espectral": rho,
        "convergio": convergio,
    }
