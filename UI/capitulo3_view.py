# UI/capitulo3_view.py
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from Core.cap3_interpolacion import (
    polinomio_vandermonde,
    polinomio_newton,
    polinomio_lagrange,
    spline_lineal,
    spline_cubico_natural,
    evaluar_polinomio,
    evaluar_spline_lineal,
    evaluar_spline_cubico,
    polinomio_to_str,
    spline_lineal_to_str,
    spline_cubico_to_str,
)
from Core.informes import escribir_informe_cap3


class Capitulo3View:
    def __init__(self, master):
        self.master = master
        self.resultados_cap3: Dict[str, Dict[str, Any]] | None = None

        # ---------------- ENTRADA ----------------
        frame_inputs = ttk.LabelFrame(master, text="Entrada de datos - Capítulo 3 (Interpolación)")
        frame_inputs.pack(fill="x", padx=10, pady=10)

        ttk.Label(frame_inputs, text="Método:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.metodo = tk.StringVar(value="Vandermonde")
        self.combo_metodo = ttk.Combobox(
            frame_inputs,
            textvariable=self.metodo,
            state="readonly",
            values=["Vandermonde", "Newton interpolante", "Lagrange", "Spline lineal", "Spline cúbico"],
            width=25,
        )
        self.combo_metodo.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(
            frame_inputs,
            text=(
                "Ingrese hasta 8 puntos. x_i estrictamente crecientes.\n"
                "x: un valor por línea.  y: un valor por línea en el mismo orden."
            ),
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=5, pady=5)

        frame_textos = ttk.Frame(master)
        frame_textos.pack(fill="x", padx=10, pady=5)

        frame_x = ttk.LabelFrame(frame_textos, text="Valores x_i")
        frame_x.pack(side="left", fill="both", expand=True, padx=5)
        self.text_x = tk.Text(frame_x, width=20, height=10)
        self.text_x.pack(fill="both", expand=True, padx=3, pady=3)

        frame_y = ttk.LabelFrame(frame_textos, text="Valores y_i")
        frame_y.pack(side="left", fill="both", expand=True, padx=5)
        self.text_y = tk.Text(frame_y, width=20, height=10)
        self.text_y.pack(fill="both", expand=True, padx=3, pady=3)

        # Ejemplo sencillo
        ejemplo_x = "0\n1\n2\n3"
        ejemplo_y = "1\n2\n0\n4"
        self.text_x.insert("1.0", ejemplo_x)
        self.text_y.insert("1.0", ejemplo_y)

        # ---------------- BOTONES ----------------
        frame_botones = ttk.Frame(master)
        frame_botones.pack(fill="x", padx=10, pady=5)

        ttk.Button(frame_botones, text="Ejecutar método", command=self.ejecutar_metodo).pack(side="left", padx=5)
        ttk.Button(frame_botones, text="Ejecutar todos y comparar", command=self.ejecutar_todos_y_comparar).pack(
            side="left", padx=5
        )
        ttk.Button(frame_botones, text="Generar informe (todos)", command=self.generar_informe_todos).pack(
            side="left", padx=5
        )

        # ---------------- POLINOMIO TEXTO ----------------
        frame_poly = ttk.LabelFrame(master, text="Polinomio / función interpolante")
        frame_poly.pack(fill="both", expand=False, padx=10, pady=5)

        self.text_poly = tk.Text(frame_poly, height=6)
        self.text_poly.pack(fill="both", expand=True, padx=3, pady=3)
        self.text_poly.configure(state="disabled")

        # ---------------- GRÁFICA ----------------
        frame_graf = ttk.LabelFrame(master, text="Gráfica interpolante")
        frame_graf.pack(fill="both", expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_graf)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ---------------- RESUMEN ----------------
        self.label_mejor = ttk.Label(master, text="Mejor método (comparación): -")
        self.label_mejor.pack(padx=10, pady=(0, 10), anchor="w")

    # -------------------------------------------------------
    # UTILIDADES
    # -------------------------------------------------------

    def _parse_vector(self, text: str, nombre: str) -> List[float]:
        vals: List[float] = []
        for line in text.strip().splitlines():
            if not line.strip():
                continue
            vals.append(float(line.strip()))
        if len(vals) < 2:
            raise ValueError(f"Se requieren al menos 2 valores en {nombre}.")
        if len(vals) > 8:
            raise ValueError("El máximo permitido es 8 puntos.")
        return vals

    def _leer_puntos(self):
        x = self._parse_vector(self.text_x.get("1.0", "end"), "x")
        y = self._parse_vector(self.text_y.get("1.0", "end"), "y")
        if len(x) != len(y):
            raise ValueError("x e y deben tener la misma cantidad de datos.")
        x_arr = np.array(x, dtype=float)
        if not np.all(np.diff(x_arr) > 0):
            raise ValueError("Los x_i deben ser estrictamente crecientes.")
        return list(x_arr), list(map(float, y))

    def _mostrar_polinomio(self, texto: str):
        self.text_poly.configure(state="normal")
        self.text_poly.delete("1.0", "end")
        self.text_poly.insert("1.0", texto)
        self.text_poly.configure(state="disabled")

    def _graficar(self, x, y, modelo, tipo: str):
        self.ax.clear()
        self.ax.grid(True)

        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        xmin, xmax = float(np.min(x)), float(np.max(x))
        xs = np.linspace(xmin, xmax, 300)

        if tipo == "polinomio":
            ys = evaluar_polinomio(modelo["coef"], xs)
        elif tipo == "spline_lineal":
            ys = evaluar_spline_lineal(modelo, xs)
        else:  # spline cúbico
            ys = evaluar_spline_cubico(modelo, xs)

        self.ax.plot(xs, ys, label="Interpolante")
        self.ax.scatter(x, y, color="red", label="Datos")
        self.ax.legend()
        self.canvas.draw()

    # -------------------------------------------------------
    # EJECUCIÓN INDIVIDUAL
    # -------------------------------------------------------

    def ejecutar_metodo(self):
        try:
            x, y = self._leer_puntos()
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e))
            return

        metodo = self.metodo.get()

        try:
            if metodo == "Vandermonde":
                res = polinomio_vandermonde(x, y)
                pol_str = polinomio_to_str(res["coef"])
                self._mostrar_polinomio(f"p(x) = {pol_str}")
                self._graficar(x, y, res, "polinomio")

            elif metodo == "Newton interpolante":
                res = polinomio_newton(x, y)
                pol_str = polinomio_to_str(res["coef"])
                self._mostrar_polinomio(f"p_N(x) = {pol_str}")
                self._graficar(x, y, res, "polinomio")

            elif metodo == "Lagrange":
                res = polinomio_lagrange(x, y)
                pol_str = polinomio_to_str(res["coef"])
                self._mostrar_polinomio(f"p_L(x) = {pol_str}")
                self._graficar(x, y, res, "polinomio")

            elif metodo == "Spline lineal":
                res = spline_lineal(x, y)
                s_str = spline_lineal_to_str(res)
                self._mostrar_polinomio(s_str)
                self._graficar(x, y, res, "spline_lineal")

            else:  # Spline cúbico
                res = spline_cubico_natural(x, y)
                s_str = spline_cubico_to_str(res)
                self._mostrar_polinomio(s_str)
                self._graficar(x, y, res, "spline_cubico")

        except Exception as e:
            messagebox.showerror("Error en método", str(e))
            return

    # -------------------------------------------------------
    # EJECUTAR TODOS Y COMPARAR
    # -------------------------------------------------------

    def _error_max_en_puntos(self, modelo, tipo: str, x_arr: np.ndarray, y_arr: np.ndarray) -> float:
        if tipo == "polinomio":
            y_hat = evaluar_polinomio(modelo["coef"], x_arr)
        elif tipo == "spline_lineal":
            y_hat = evaluar_spline_lineal(modelo, x_arr)
        else:
            y_hat = evaluar_spline_cubico(modelo, x_arr)
        err = np.max(np.abs(y_hat - y_arr))
        return float(err)

    def _calcular_todos_metodos(self):
        x, y = self._leer_puntos()
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)

        resultados: Dict[str, Dict[str, Any]] = {}

        # Vandermonde
        try:
            r_v = polinomio_vandermonde(x, y)
            err_v = self._error_max_en_puntos(r_v, "polinomio", x_arr, y_arr)
            resultados["Vandermonde"] = {
                "ejecutado": True,
                "error_max": err_v,
                "descripcion": f"Polinomio de grado {r_v['grado']}",
                "mensaje_error": None,
            }
        except Exception as e:
            resultados["Vandermonde"] = {
                "ejecutado": False,
                "error_max": None,
                "descripcion": "",
                "mensaje_error": str(e),
            }

        # Newton
        try:
            r_n = polinomio_newton(x, y)
            err_n = self._error_max_en_puntos(r_n, "polinomio", x_arr, y_arr)
            resultados["Newton interpolante"] = {
                "ejecutado": True,
                "error_max": err_n,
                "descripcion": f"Polinomio de grado {r_n['grado']}",
                "mensaje_error": None,
            }
        except Exception as e:
            resultados["Newton interpolante"] = {
                "ejecutado": False,
                "error_max": None,
                "descripcion": "",
                "mensaje_error": str(e),
            }

        # Lagrange
        try:
            r_l = polinomio_lagrange(x, y)
            err_l = self._error_max_en_puntos(r_l, "polinomio", x_arr, y_arr)
            resultados["Lagrange"] = {
                "ejecutado": True,
                "error_max": err_l,
                "descripcion": f"Polinomio de grado {r_l['grado']}",
                "mensaje_error": None,
            }
        except Exception as e:
            resultados["Lagrange"] = {
                "ejecutado": False,
                "error_max": None,
                "descripcion": "",
                "mensaje_error": str(e),
            }

        # Spline lineal
        try:
            r_sl = spline_lineal(x, y)
            err_sl = self._error_max_en_puntos(r_sl, "spline_lineal", x_arr, y_arr)
            resultados["Spline lineal"] = {
                "ejecutado": True,
                "error_max": err_sl,
                "descripcion": f"{len(x) - 1} tramos lineales",
                "mensaje_error": None,
            }
        except Exception as e:
            resultados["Spline lineal"] = {
                "ejecutado": False,
                "error_max": None,
                "descripcion": "",
                "mensaje_error": str(e),
            }

        # Spline cúbico
        try:
            r_sc = spline_cubico_natural(x, y)
            err_sc = self._error_max_en_puntos(r_sc, "spline_cubico", x_arr, y_arr)
            resultados["Spline cúbico"] = {
                "ejecutado": True,
                "error_max": err_sc,
                "descripcion": f"{len(x) - 1} tramos cúbicos (spline natural)",
                "mensaje_error": None,
            }
        except Exception as e:
            resultados["Spline cúbico"] = {
                "ejecutado": False,
                "error_max": None,
                "descripcion": "",
                "mensaje_error": str(e),
            }

        # Mejor método: menor error máximo
        mejor = None
        for metodo, res in resultados.items():
            if not res["ejecutado"]:
                continue
            if res["error_max"] is None:
                continue
            if mejor is None:
                mejor = (metodo, res)
            else:
                _, r_act = mejor
                if res["error_max"] < r_act["error_max"]:
                    mejor = (metodo, res)

        return resultados, mejor, x, y

    def ejecutar_todos_y_comparar(self):
        try:
            resultados, mejor, x, y = self._calcular_todos_metodos()
            self.resultados_cap3 = resultados
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if mejor is None:
            self.label_mejor.config(text="Mejor método (comparación): No determinado.")
            messagebox.showinfo("Comparación", "No fue posible determinar un mejor método.")
        else:
            metodo_mejor, res_mejor = mejor
            txt = f"{metodo_mejor} | Error máximo: {res_mejor['error_max']:.3e}"
            self.label_mejor.config(text="Mejor método (comparación): " + txt)
            messagebox.showinfo("Comparación", txt)

    # -------------------------------------------------------
    # INFORME
    # -------------------------------------------------------

    def generar_informe_todos(self):
        if not messagebox.askyesno(
            "Informe",
            "¿Desea ejecutar todos los métodos y generar el informe de comparación (Capítulo 3)?",
        ):
            return

        try:
            resultados, mejor, x, y = self._calcular_todos_metodos()
            self.resultados_cap3 = resultados
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        params = {
            "num_puntos": len(x),
        }

        ruta = escribir_informe_cap3(resultados, x, y, params)
        messagebox.showinfo("Informe generado", f"Informe guardado en:\n{ruta}")
