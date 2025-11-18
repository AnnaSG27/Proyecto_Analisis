# **Información Técnica del Proyecto**

-   **Versión de Python**: 3.8

-   **Sistema Operativo**: Windows 11 Pro 25H2

-   **Dependencias**: sympy, numpy, matplotlib, pandas

# Instalación y Configuración 

## Instalación de Dependencias 

    pip install sympy numpy matplotlib pandas

## Ejecución del Proyecto

    python main.py

# Descripción General 

Este proyecto es una aplicación gráfica completa desarrollada en Python
que implementa los principales métodos numéricos organizados en tres
capítulos fundamentales:

-   **Capítulo 1**: Métodos para resolver ecuaciones no lineales

-   **Capítulo 2**: Métodos iterativos para sistemas de ecuaciones
    lineales

-   **Capítulo 3**: Métodos de interpolación polinomial y por splines

# Estructura del Proyecto 

## Directorio Raíz

    ProyectoTOTAL/
    ├── Core/                    # Lógica principal de los métodos
    ├── informes/                # Informes generados automáticamente
    ├── UI/                      # Interfaz de usuario
    ├── main.py                  # Punto de entrada
    └── requirements.txt         # Dependencias

# Archivo Principal: main.py 

``` {.python language="Python" caption="main.py"}
# main.py
import tkinter as tk
from UI.ventana_principal import VentanaPrincipal

def main():
    root = tk.Tk()
    root.title("Proyecto Métodos Numéricos")
    root.geometry("1200x700")
    app = VentanaPrincipal(root)
    root.mainloop()

if __name__ == "__main__":
    main()
```

# Archivo de Dependencias: requirements.txt 

``` {caption="requirements.txt"}
sympy
numpy
matplotlib
pandas
```

# Módulo Core - Métodos Numéricos 

## cap1_no_lineales.py 

``` {.python language="Python"}
# Core/cap1_no_lineales.py
from typing import Callable, Dict, List, Optional
from Core.errores import error_absoluto, error_relativo, error_condicion

def biseccion(f: Callable[[float], float],
              a: float, b: float,
              tol: float, max_iter: int,
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
            "iter": n, "a": a, "b": b, 
            "xm": xm, "f(xm)": fxm, "error": err
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

def newton(f: Callable[[float], float],
           df: Callable[[float], float],
           x0: float, tol: float, max_iter: int,
           tipo_error: str = "relativo") -> Dict:
    
    tabla: List[Dict] = []
    x = x0
    x_old: Optional[float] = None
    err: Optional[float] = None

    for n in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError(f"Derivada cero en iteración {n}")

        x_new = x - fx / dfx
        err = _calcular_error(tipo_error, x_new, x_old, fx)

        tabla.append({
            "iter": n, "xm": x_new, 
            "f(xm)": f(x_new), "error": err
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
```

## cap2_iterativos.py 

``` {.python language="Python"}
# Core/cap2_iterativos.py
import numpy as np
from typing import Dict, List

def jacobi(A, b, x0=None, tol: float = 1e-6, 
           max_iter: int = 100) -> Dict:
    
    A, b, x, n = _validar_matriz_y_vectores(A, b, x0)

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    if np.any(np.isclose(np.diag(D), 0.0)):
        raise ValueError("Jacobi: ceros en la diagonal")

    D_inv = np.linalg.inv(D)
    T = -D_inv @ (L + U)
    c = D_inv @ b

    rho = _radio_espectral(T)

    tabla: List[Dict] = []
    err: Optional[float] = None

    for k in range(1, max_iter + 1):
        x_new = T @ x + c
        err = float(np.linalg.norm(x_new - x, ord=np.inf))
        tabla.append({"iter": k, "x": x_new.copy(), "error": err})
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
```

## cap3_interpolacion.py 

``` {.python language="Python"}
# Core/cap3_interpolacion.py
import numpy as np
from typing import Dict, List

def polinomio_vandermonde(x: List[float], 
                         y: List[float]) -> Dict:
    
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    V = np.vander(x, N=n, increasing=False)
    coef = np.linalg.solve(V, y)

    return {"coef": coef, "grado": n - 1}

def spline_cubico_natural(x: List[float], 
                         y: List[float]) -> Dict:
    
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    if n < 3:
        raise ValueError("Se requieren al menos 3 puntos")

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x_i deben ser crecientes")

    # Sistema para segundas derivadas M
    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = (3/h[i])*(y[i+1]-y[i]) - (3/h[i-1])*(y[i]-y[i-1])

    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    for i in range(1, n - 1):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]

    M = np.zeros(n)
    for j in range(n-2, -1, -1):
        M[j] = z[j] - mu[j] * M[j+1]

    # Coeficientes por tramo
    a = y[:-1].copy()
    b = np.zeros(n-1)
    c = np.zeros(n-1)
    d = np.zeros(n-1)

    for i in range(n-1):
        hi = h[i]
        a[i] = y[i]
        c[i] = M[i] / 2.0
        d[i] = (M[i+1] - M[i]) / (6.0 * hi)
        b[i] = (y[i+1]-y[i])/hi - (2*M[i]+M[i+1])*hi/6.0

    return {"tipo": "cubico", "x": x, 
            "a": a, "b": b, "c": c, "d": d}
```

# Módulo UI - Interfaces Gráficas {

## ventana_principal.py 

``` {.python language="Python"}
# UI/ventana_principal.py
import tkinter as tk
from tkinter import ttk

from UI.capitulo1_view import Capitulo1View
from UI.capitulo2_view import Capitulo2View
from UI.capitulo3_view import Capitulo3View

class VentanaPrincipal:
    def __init__(self, root: tk.Tk):
        self.root = root

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        frame1 = ttk.Frame(notebook)
        frame2 = ttk.Frame(notebook)
        frame3 = ttk.Frame(notebook)

        notebook.add(frame1, text="Capítulo 1: No lineales")
        notebook.add(frame2, text="Capítulo 2: Sistemas iterativos")
        notebook.add(frame3, text="Capítulo 3: Interpolación")

        self.cap1 = Capitulo1View(frame1)
        self.cap2 = Capitulo2View(frame2)
        self.cap3 = Capitulo3View(frame3)
```

## capitulo1_view.py 

``` {.python language="Python"}
# UI/capitulo1_view.py (Extracto)
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from Core.parsing import construir_funcion, construir_derivada
from Core.cap1_no_lineales import (
    biseccion, regla_falsa, punto_fijo, 
    newton, secante, newton_multiples
)

class Capitulo1View:
    def __init__(self, master):
        self.master = master
        self.resultados_cap1 = None

        # Panel de entradas
        frame_inputs = ttk.LabelFrame(master, 
                                    text="Entrada de datos - Capítulo 1")
        frame_inputs.pack(fill="x", padx=10, pady=10)

        # Método
        ttk.Label(frame_inputs, text="Método:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5)
        self.metodo = tk.StringVar(value="Bisección")
        self.combo_metodo = ttk.Combobox(
            frame_inputs, textvariable=self.metodo,
            state="readonly",
            values=["Bisección", "Regla falsa", "Punto fijo", 
                   "Newton", "Secante", "Newton raíces múltiples"],
            width=22)
        self.combo_metodo.grid(row=0, column=1, sticky="w", 
                             padx=5, pady=5)

        # f(x)
        ttk.Label(frame_inputs, text="f(x):").grid(
            row=1, column=0, sticky="w", padx=5, pady=5)
        self.entry_fx = ttk.Entry(frame_inputs, width=40)
        self.entry_fx.grid(row=1, column=1, columnspan=3, 
                         sticky="w", padx=5, pady=5)
        self.entry_fx.insert(0, "x**3 - x - 1")

        # Botones
        btn_ejecutar = ttk.Button(frame_inputs, 
                                text="Ejecutar método", 
                                command=self.ejecutar_metodo)
        btn_ejecutar.grid(row=8, column=0, columnspan=2, pady=10)

        btn_todos = ttk.Button(frame_inputs, 
                             text="Ejecutar todos y comparar", 
                             command=self.ejecutar_todos_y_comparar)
        btn_todos.grid(row=8, column=2, columnspan=2, pady=10)

    def ejecutar_metodo(self):
        # Limpiar tabla
        for item in self.tree.get_children():
            self.tree.delete(item)

        fx_str = self.entry_fx.get().strip()
        if not fx_str:
            messagebox.showerror("Error", "Debe ingresar f(x).")
            return

        try:
            expr_f, f = construir_funcion(fx_str)
        except Exception as e:
            messagebox.showerror("Error en f(x)", str(e))
            return

        # ... resto de la implementacion
```

# Utilidades

## parsing.py 

``` {.python language="Python"}
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
```

## errores.py 

``` {.python language="Python"}
# Core/errores.py

def error_absoluto(x_new, x_old):
    return abs(x_new - x_old)

def error_relativo(x_new, x_old):
    if x_new == 0:
        return float('inf')
    return abs((x_new - x_old) / x_new)

def error_condicion(fx):
    # criterio de parada basado en |f(x)|
    return abs(fx)
```

# Métodos Implementados 


| Capítulo | Método | Descripción | Parámetros Requeridos | Convergencia |
|----------|--------|-------------|----------------------|--------------|
| **1 - Ecuaciones No Lineales** | | | | |
| | **Bisección** | División sucesiva de intervalos | `[a, b]` con `f(a)*f(b) < 0` | Lineal |
| | **Regla Falsa** | Interpolación lineal en intervalo | `[a, b]` con `f(a)*f(b) < 0` | Superlineal |
| | **Punto Fijo** | Iteración `x = g(x)` | `g(x)` y `x₀` | Lineal |
| | **Newton-Raphson** | Uso de derivadas para convergencia rápida | `f(x)`, `x₀` (derivada automática) | Cuadrática |
| | **Secante** | Aproximación de derivada con dos puntos | `x₀`, `x₁` | Superlineal (1.618) |
| | **Newton Múltiples** | Para raíces con multiplicidad conocida | `f(x)`, `x₀`, `m` (multiplicidad) | Lineal |
| **2 - Sistemas Lineales** | | | | |
| | **Jacobi** | Descomposición `T = -D⁻¹(L+U)`, `c = D⁻¹b` | Matriz A, vector b, `x₀` opcional | `ρ(T) < 1` |
| | **Gauss-Seidel** | Mejora de Jacobi: `T = -(D+L)⁻¹U`, `c = (D+L)⁻¹b` | Matriz A, vector b, `x₀` opcional | `ρ(T) < 1` |
| | **SOR** | Relajación sucesiva: `T = (D+ωL)⁻¹[(1-ω)D - ωU]` | Matriz A, vector b, `ω ∈ (0,2)`, `x₀` opcional | `ρ(T) < 1` |
| **3 - Interpolación** | | | | |
| | **Vandermonde** | Sistema lineal directo | Puntos `(xᵢ, yᵢ)` | Exacta en puntos |
| | **Newton** | Diferencias divididas | Puntos `(xᵢ, yᵢ)` | Exacta en puntos |
| | **Lagrange** | Base polinomial de Lagrange | Puntos `(xᵢ, yᵢ)` | Exacta en puntos |
| | **Spline Lineal** | Continuidad `C⁰` (por tramos lineales) | Puntos `(xᵢ, yᵢ)` | Continua |
| | **Spline Cúbico** | Continuidad `C²` (por tramos cúbicos) | Puntos `(xᵢ, yᵢ)` | Suave |

## Características de los Métodos

### Capítulo 1: Ecuaciones No Lineales
- **Criterios de parada**: Error absoluto, relativo o por condición `|f(x)|`
- **Validaciones**: Signos opuestos en intervalo, derivada no nula
- **Máximo de iteraciones**: Configurable por usuario

### Capítulo 2: Sistemas Lineales Iterativos
- **Límite de tamaño**: Matrices hasta 7×7
- **Criterio de parada**: Norma infinito `||x⁽ᵏ⁺¹⁾ - x⁽ᵏ⁾||∞ < tol`
- **Análisis de convergencia**: Radio espectral `ρ(T)`

### Capítulo 3: Interpolación
- **Máximo de puntos**: 8 puntos
- **Validación**: Puntos `xᵢ` estrictamente crecientes
- **Evaluación**: En cualquier punto del dominio

# Requisitos del Sistema 

## Especificaciones Técnicas 

-   **Python**: Versión 3.8 o superior

-   **Sistema Operativo**: Windows 11 Pro 25H2 (compatible con otros
    sistemas)

-   **Memoria RAM**: Mínimo 4GB recomendado

-   **Espacio en disco**: 100MB libres

## Dependencias Requeridas 

    sympy>=1.8    # Procesamiento simbólico
    numpy>=1.20   # Cálculos numéricos
    matplotlib>=3.3  # Visualizaciones
    pandas>=1.3   # Manejo de datos

# Características del Proyecto 

-   Interfaz gráfica intuitiva con Tkinter

-   Validación en tiempo real de entradas

-   Visualización de resultados con Matplotlib

-   Comparación automática entre métodos

-   Generación de informes detallados

-   Procesamiento simbólico con SymPy

-   Soporte para funciones matemáticas complejas

-   Ejecución en Windows 11 Pro 25H2

-   Compatible con Python 3.8+

# Desarrolladores 

-   **Samuel Aguilar Villada** 
-   **Miguel Angel Montoya** 
-   **Anna Sofia Giraldo Carvajal** 

# Notas de Instalación

## Para usuarios de Windows 

1.  Asegurarse de tener Python 3.8 o superior instalado

2.  Abrir PowerShell o Símbolo del sistema como administrador

3.  Ejecutar: `pip install sympy numpy matplotlib pandas`

4.  Navegar al directorio del proyecto

5.  Ejecutar: `python main.py`

## Verificación de instalación

    python --version
    pip list | findstr "sympy numpy matplotlib pandas"
