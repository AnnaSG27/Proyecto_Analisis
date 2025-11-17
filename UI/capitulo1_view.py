# UI/capitulo1_view.py
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from Core.parsing import construir_funcion, construir_derivada
from Core.cap1_no_lineales import (
    biseccion,
    regla_falsa,
    punto_fijo,
    newton,
    secante,
    newton_multiples,
)
from Core.informes import escribir_informe_cap1


class Capitulo1View:
    def __init__(self, master):
        self.master = master
        self.resultados_cap1 = None  # se llenará al ejecutar todos

        # ---------------- PANEL SUPERIOR: ENTRADAS ----------------
        frame_inputs = ttk.LabelFrame(master, text="Entrada de datos - Capítulo 1")
        frame_inputs.pack(fill="x", padx=10, pady=10)

        # Método
        ttk.Label(frame_inputs, text="Método:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.metodo = tk.StringVar(value="Bisección")
        self.combo_metodo = ttk.Combobox(
            frame_inputs,
            textvariable=self.metodo,
            state="readonly",
            values=[
                "Bisección",
                "Regla falsa",
                "Punto fijo",
                "Newton",
                "Secante",
                "Newton raíces múltiples",
            ],
            width=22,
        )
        self.combo_metodo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.combo_metodo.bind("<<ComboboxSelected>>", self._actualizar_visibilidad_campos)

        # f(x)
        ttk.Label(frame_inputs, text="f(x):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.entry_fx = ttk.Entry(frame_inputs, width=40)
        self.entry_fx.grid(row=1, column=1, columnspan=3, sticky="w", padx=5, pady=5)
        self.entry_fx.insert(0, "x**3 - x - 1")

        # g(x) para punto fijo
        ttk.Label(frame_inputs, text="g(x) (punto fijo):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.entry_gx = ttk.Entry(frame_inputs, width=40)
        self.entry_gx.grid(row=2, column=1, columnspan=3, sticky="w", padx=5, pady=5)
        self.entry_gx.insert(0, "cos(x)")  # ejemplo

        # a, b (métodos por intervalo)
        ttk.Label(frame_inputs, text="a:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.entry_a = ttk.Entry(frame_inputs, width=10)
        self.entry_a.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        self.entry_a.insert(0, "1")

        ttk.Label(frame_inputs, text="b:").grid(row=3, column=2, sticky="w", padx=5, pady=5)
        self.entry_b = ttk.Entry(frame_inputs, width=10)
        self.entry_b.grid(row=3, column=3, sticky="w", padx=5, pady=5)
        self.entry_b.insert(0, "2")

        # x0, x1
        ttk.Label(frame_inputs, text="x0:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.entry_x0 = ttk.Entry(frame_inputs, width=10)
        self.entry_x0.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        self.entry_x0.insert(0, "1.5")

        ttk.Label(frame_inputs, text="x1 (secante):").grid(row=4, column=2, sticky="w", padx=5, pady=5)
        self.entry_x1 = ttk.Entry(frame_inputs, width=10)
        self.entry_x1.grid(row=4, column=3, sticky="w", padx=5, pady=5)
        self.entry_x1.insert(0, "1.6")

        # multiplicidad m
        ttk.Label(frame_inputs, text="m (raíces múltiples):").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.entry_m = ttk.Entry(frame_inputs, width=10)
        self.entry_m.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        self.entry_m.insert(0, "2")

        # tolerancia e iteraciones
        ttk.Label(frame_inputs, text="Tolerancia:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.entry_tol = ttk.Entry(frame_inputs, width=10)
        self.entry_tol.grid(row=6, column=1, sticky="w", padx=5, pady=5)
        self.entry_tol.insert(0, "1e-6")

        ttk.Label(frame_inputs, text="Iter máx:").grid(row=6, column=2, sticky="w", padx=5, pady=5)
        self.entry_iter = ttk.Entry(frame_inputs, width=10)
        self.entry_iter.grid(row=6, column=3, sticky="w", padx=5, pady=5)
        self.entry_iter.insert(0, "50")

        # tipo de error
        ttk.Label(frame_inputs, text="Tipo de error:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.tipo_error = tk.StringVar(value="relativo")
        ttk.Radiobutton(frame_inputs, text="Relativo", value="relativo", variable=self.tipo_error).grid(
            row=7, column=1, sticky="w"
        )
        ttk.Radiobutton(frame_inputs, text="Absoluto", value="absoluto", variable=self.tipo_error).grid(
            row=7, column=2, sticky="w"
        )
        ttk.Radiobutton(frame_inputs, text="Condición |f(x)|", value="condicion", variable=self.tipo_error).grid(
            row=7, column=3, sticky="w"
        )

        # botón ejecutar individual
        btn_ejecutar = ttk.Button(frame_inputs, text="Ejecutar método", command=self.ejecutar_metodo)
        btn_ejecutar.grid(row=8, column=0, columnspan=2, pady=10)

        # botón ejecutar todos
        btn_todos = ttk.Button(frame_inputs, text="Ejecutar todos y comparar", command=self.ejecutar_todos_y_comparar)
        btn_todos.grid(row=8, column=2, columnspan=2, pady=10)

        # botón informe
        btn_informe = ttk.Button(frame_inputs, text="Generar informe (todos)", command=self.generar_informe_todos)
        btn_informe.grid(row=9, column=0, columnspan=4, pady=5)

        # ---------------- PANEL TABLA ----------------
        frame_tabla = ttk.LabelFrame(master, text="Tabla de iteraciones")
        frame_tabla.pack(fill="both", expand=True, side="left", padx=10, pady=10)

        cols = ("iter", "a", "b", "xm", "f(xm)", "error")
        self.tree = ttk.Treeview(frame_tabla, columns=cols, show="headings", height=15)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center", width=90)
        self.tree.pack(fill="both", expand=True)

        # ---------------- PANEL GRÁFICA ----------------
        frame_graf = ttk.LabelFrame(master, text="Gráfica f(x)")
        frame_graf.pack(fill="both", expand=True, side="right", padx=10, pady=10)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_graf)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.label_resultado = ttk.Label(master, text="Raíz aproximada: -")
        self.label_resultado.pack(pady=2)

        self.label_mejor = ttk.Label(master, text="Mejor método (comparación): -")
        self.label_mejor.pack(pady=2)

        # Ajustar visibilidad inicial de campos
        self._actualizar_visibilidad_campos()

    # -----------------------------------------------------------------
    def _actualizar_visibilidad_campos(self, event=None):
        metodo = self.metodo.get()
        texto = {
            "Bisección": "Método por intervalos: use [a,b] con f(a)*f(b)<0.",
            "Regla falsa": "Método por intervalos: use [a,b] con f(a)*f(b)<0.",
            "Punto fijo": "Punto fijo: defina g(x) y un x0.",
            "Newton": "Newton: se usa derivada automática de f(x) y x0.",
            "Secante": "Secante: use x0 y x1 cercanos a la raíz.",
            "Newton raíces múltiples": "Newton m: se usa x0 y multiplicidad m>1.",
        }[metodo]
        # Obtener la ventana principal (root) correctamente
        root = self.master.winfo_toplevel()
        root.title(f"Proyecto Métodos Numéricos - {metodo} | {texto}")

    # -----------------------------------------------------------------
    # Ejecutar solo el método seleccionado y llenar tabla + gráfica
    # -----------------------------------------------------------------
    def ejecutar_metodo(self):
        # limpiar tabla
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

        try:
            tol = float(self.entry_tol.get())
            max_iter = int(self.entry_iter.get())
        except ValueError:
            messagebox.showerror("Error", "Verifica tolerancia e iteraciones.")
            return

        metodo = self.metodo.get()
        tipo_err = self.tipo_error.get()

        resultado = None
        xmin, xmax = -2, 2  # por defecto

        try:
            if metodo in ["Bisección", "Regla falsa"]:
                a = float(self.entry_a.get())
                b = float(self.entry_b.get())
                if a >= b:
                    raise ValueError("Se requiere a < b para el intervalo.")
                xmin, xmax = a, b
                if metodo == "Bisección":
                    resultado = biseccion(f, a, b, tol, max_iter, tipo_err)
                else:
                    resultado = regla_falsa(f, a, b, tol, max_iter, tipo_err)

            elif metodo == "Punto fijo":
                gx_str = self.entry_gx.get().strip()
                if not gx_str:
                    raise ValueError("Debe ingresar g(x) para el método de punto fijo.")
                _, g = construir_funcion(gx_str)
                x0 = float(self.entry_x0.get())
                xmin, xmax = x0 - 2, x0 + 2
                resultado = punto_fijo(f, g, x0, tol, max_iter, tipo_err)

            elif metodo == "Newton":
                deriv_expr, df = construir_derivada(fx_str)
                x0 = float(self.entry_x0.get())
                xmin, xmax = x0 - 2, x0 + 2
                resultado = newton(f, df, x0, tol, max_iter, tipo_err)

            elif metodo == "Secante":
                x0 = float(self.entry_x0.get())
                x1 = float(self.entry_x1.get())
                xmin, xmax = min(x0, x1) - 1, max(x0, x1) + 1
                resultado = secante(f, x0, x1, tol, max_iter, tipo_err)

            elif metodo == "Newton raíces múltiples":
                deriv_expr, df = construir_derivada(fx_str)
                x0 = float(self.entry_x0.get())
                m = int(self.entry_m.get())
                xmin, xmax = x0 - 2, x0 + 2
                resultado = newton_multiples(f, df, m, x0, tol, max_iter, tipo_err)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if resultado is None:
            messagebox.showerror("Error", "No se obtuvo resultado.")
            return

        # poblar tabla
        for fila in resultado["tabla"]:
            a_val = "-" if fila["a"] is None else f"{fila['a']:.6g}"
            b_val = "-" if fila["b"] is None else f"{fila['b']:.6g}"
            err_val = "-" if fila["error"] is None else f"{fila['error']:.3e}"
            self.tree.insert("", "end", values=(
                fila["iter"],
                a_val,
                b_val,
                f"{fila['xm']:.6g}",
                f"{fila['f(xm)']:.6g}",
                err_val,
            ))

        raiz = resultado["raiz"]
        self.label_resultado.config(
            text=f"Método: {metodo} | Raíz aproximada: {raiz:.10f}  |  Convergió: {resultado['convergio']}"
        )

        # graficar
        self.graficar_funcion(f, xmin, xmax, raiz)

    # -----------------------------------------------------------------
    # Ejecutar TODOS los métodos con los mismos datos y comparar
    # -----------------------------------------------------------------
    def ejecutar_todos_y_comparar(self):
        try:
            self.resultados_cap1, mejor = self._calcular_todos_metodos()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if mejor is None:
            self.label_mejor.config(text="Mejor método (comparación): Ninguno convergió.")
            messagebox.showinfo("Comparación", "Ningún método convergió con los parámetros dados.")
        else:
            metodo_mejor, res_mejor = mejor
            txt = (
                f"Mejor método: {metodo_mejor} | "
                f"Iteraciones: {res_mejor['iteraciones']} | "
                f"Error final: {res_mejor['error_final']}"
            )
            self.label_mejor.config(text="Mejor método (comparación): " + txt)
            messagebox.showinfo("Comparación", txt)

    # -----------------------------------------------------------------
    def _calcular_todos_metodos(self):
        """
        Ejecuta todos los métodos (dentro de lo posible) y devuelve:
          - resultados: dict
          - mejor: (nombre_metodo, dict_resultado_normalizado) o None
        """
        fx_str = self.entry_fx.get().strip()
        if not fx_str:
            raise ValueError("Debe ingresar f(x).")

        expr_f, f = construir_funcion(fx_str)

        tol = float(self.entry_tol.get())
        max_iter = int(self.entry_iter.get())
        tipo_err = self.tipo_error.get()

        # leemos todos los parámetros
        a = float(self.entry_a.get())
        b = float(self.entry_b.get())
        x0 = float(self.entry_x0.get())
        x1 = float(self.entry_x1.get())
        m = int(self.entry_m.get())
        gx_str = self.entry_gx.get().strip()

        resultados = {}

        # helper para normalizar cada método
        def normalizar_resultado(nombre, res, mensaje_error=None):
            if res is None:
                return {
                    "ejecutado": False,
                    "convergio": False,
                    "iteraciones": None,
                    "raiz": None,
                    "error_final": None,
                    "mensaje_error": mensaje_error,
                }
            # error final: último error de la tabla o |f(raiz)|
            tabla = res.get("tabla", [])
            if tabla:
                ultimo = tabla[-1]
                err_final = ultimo.get("error")
                if err_final is None:
                    err_final = abs(f(res["raiz"]))
            else:
                err_final = abs(f(res["raiz"]))

            return {
                "ejecutado": True,
                "convergio": res.get("convergio", False),
                "iteraciones": res.get("iteraciones"),
                "raiz": res.get("raiz"),
                "error_final": err_final,
                "mensaje_error": mensaje_error,
            }

        # Bisección
        try:
            res_bis = biseccion(f, a, b, tol, max_iter, tipo_err)
            resultados["Bisección"] = normalizar_resultado("Bisección", res_bis)
        except Exception as e:
            resultados["Bisección"] = normalizar_resultado("Bisección", None, str(e))

        # Regla falsa
        try:
            res_rf = regla_falsa(f, a, b, tol, max_iter, tipo_err)
            resultados["Regla falsa"] = normalizar_resultado("Regla falsa", res_rf)
        except Exception as e:
            resultados["Regla falsa"] = normalizar_resultado("Regla falsa", None, str(e))

        # Punto fijo
        try:
            if not gx_str:
                raise ValueError("g(x) no definido.")
            _, g = construir_funcion(gx_str)
            res_pf = punto_fijo(f, g, x0, tol, max_iter, tipo_err)
            resultados["Punto fijo"] = normalizar_resultado("Punto fijo", res_pf)
        except Exception as e:
            resultados["Punto fijo"] = normalizar_resultado("Punto fijo", None, str(e))

        # Newton
        try:
            deriv_expr, df = construir_derivada(fx_str)
            res_new = newton(f, df, x0, tol, max_iter, tipo_err)
            resultados["Newton"] = normalizar_resultado("Newton", res_new)
        except Exception as e:
            resultados["Newton"] = normalizar_resultado("Newton", None, str(e))

        # Secante
        try:
            res_sec = secante(f, x0, x1, tol, max_iter, tipo_err)
            resultados["Secante"] = normalizar_resultado("Secante", res_sec)
        except Exception as e:
            resultados["Secante"] = normalizar_resultado("Secante", None, str(e))

        # Newton raíces múltiples
        try:
            deriv_expr, df = construir_derivada(fx_str)
            res_newm = newton_multiples(f, df, m, x0, tol, max_iter, tipo_err)
            resultados["Newton raíces múltiples"] = normalizar_resultado("Newton raíces múltiples", res_newm)
        except Exception as e:
            resultados["Newton raíces múltiples"] = normalizar_resultado("Newton raíces múltiples", None, str(e))

        # Determinar mejor método
        mejor = None
        for metodo, res_norm in resultados.items():
            if not res_norm["ejecutado"]:
                continue
            if not res_norm["convergio"]:
                continue
            if mejor is None:
                mejor = (metodo, res_norm)
            else:
                _, mejor_res = mejor
                if res_norm["iteraciones"] < mejor_res["iteraciones"]:
                    mejor = (metodo, res_norm)
                elif res_norm["iteraciones"] == mejor_res["iteraciones"]:
                    if res_norm["error_final"] is not None and mejor_res["error_final"] is not None:
                        if res_norm["error_final"] < mejor_res["error_final"]:
                            mejor = (metodo, res_norm)

        return resultados, mejor

    # -----------------------------------------------------------------
    def generar_informe_todos(self):
        """
        Pregunta al usuario y genera el informe de comparación de todos los métodos.
        """
        if not messagebox.askyesno(
            "Informe",
            "¿Desea ejecutar todos los métodos y generar el informe de comparación?",
        ):
            return

        try:
            resultados, mejor = self._calcular_todos_metodos()
            self.resultados_cap1 = resultados
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        fx_str = self.entry_fx.get().strip()
        tipo_err = self.tipo_error.get()

        params = {
            "a": self.entry_a.get(),
            "b": self.entry_b.get(),
            "x0": self.entry_x0.get(),
            "x1": self.entry_x1.get(),
            "m": self.entry_m.get(),
            "tolerancia": self.entry_tol.get(),
            "iter_max": self.entry_iter.get(),
            "g(x)": self.entry_gx.get().strip(),
        }

        ruta = escribir_informe_cap1(resultados, fx_str, tipo_err, params)
        messagebox.showinfo("Informe generado", f"Informe guardado en:\n{ruta}")

    # -----------------------------------------------------------------
    def graficar_funcion(self, f, a, b, raiz):
        self.ax.clear()
        if a >= b:
            a = raiz - 2
            b = raiz + 2
        xs = np.linspace(a, b, 200)
        ys = f(xs)
        self.ax.axhline(0, color="black", linewidth=0.8)
        self.ax.plot(xs, ys, label="f(x)")
        self.ax.plot([raiz], [f(raiz)], "ro", label="raíz aprox.")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()
