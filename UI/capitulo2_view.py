# UI/capitulo2_view.py
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List

import numpy as np

from Core.cap2_iterativos import jacobi, gauss_seidel, sor
from Core.informes import escribir_informe_cap2


class Capitulo2View:
    def __init__(self, master):
        self.master = master
        self.resultados_cap2 = None
        self.A_ultima = None
        self.b_ultima = None

        # ---------------- PANEL SUPERIOR: ENTRADA ----------------
        frame_inputs = ttk.LabelFrame(master, text="Entrada de datos - Capítulo 2 (Sistemas iterativos)")
        frame_inputs.pack(fill="x", padx=10, pady=10)

        # Método
        ttk.Label(frame_inputs, text="Método:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.metodo = tk.StringVar(value="Jacobi")
        self.combo_metodo = ttk.Combobox(
            frame_inputs,
            textvariable=self.metodo,
            state="readonly",
            values=["Jacobi", "Gauss-Seidel", "SOR"],
            width=15,
        )
        self.combo_metodo.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # ω para SOR
        ttk.Label(frame_inputs, text="ω (SOR):").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.entry_omega = ttk.Entry(frame_inputs, width=10)
        self.entry_omega.grid(row=0, column=3, sticky="w", padx=5, pady=5)
        self.entry_omega.insert(0, "1.2")

        # Tolerancia e iteraciones
        ttk.Label(frame_inputs, text="Tolerancia:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.entry_tol = ttk.Entry(frame_inputs, width=10)
        self.entry_tol.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.entry_tol.insert(0, "1e-6")

        ttk.Label(frame_inputs, text="Iter máx:").grid(row=1, column=2, sticky="w", padx=5, pady=5)
        self.entry_iter = ttk.Entry(frame_inputs, width=10)
        self.entry_iter.grid(row=1, column=3, sticky="w", padx=5, pady=5)
        self.entry_iter.insert(0, "100")

        # Ayuda formato
        ttk.Label(
            frame_inputs,
            text="Formato A (máx 7x7): una fila por línea, números separados por espacios.\n"
                 "Formato b y x0: un valor por línea.",
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=5, pady=5)

        # Textos A, b, x0
        frame_textos = ttk.Frame(master)
        frame_textos.pack(fill="x", padx=10, pady=5)

        frame_A = ttk.LabelFrame(frame_textos, text="Matriz A")
        frame_A.pack(side="left", fill="both", expand=True, padx=5)
        self.text_A = tk.Text(frame_A, width=40, height=10)
        self.text_A.pack(fill="both", expand=True, padx=3, pady=3)
        # Ejemplo 4x4
        ejemplo_A = "10 -1 2 0\n-1 11 -1 3\n2 -1 10 -1\n0 3 -1 8"
        self.text_A.insert("1.0", ejemplo_A)

        frame_b = ttk.LabelFrame(frame_textos, text="Vector b")
        frame_b.pack(side="left", fill="both", expand=True, padx=5)
        self.text_b = tk.Text(frame_b, width=15, height=10)
        self.text_b.pack(fill="both", expand=True, padx=3, pady=3)
        ejemplo_b = "6\n25\n-11\n15"
        self.text_b.insert("1.0", ejemplo_b)

        frame_x0 = ttk.LabelFrame(frame_textos, text="Vector x0 (opcional)")
        frame_x0.pack(side="left", fill="both", expand=True, padx=5)
        self.text_x0 = tk.Text(frame_x0, width=15, height=10)
        self.text_x0.pack(fill="both", expand=True, padx=3, pady=3)

        # Botones
        frame_botones = ttk.Frame(master)
        frame_botones.pack(fill="x", padx=10, pady=5)

        btn_ejecutar = ttk.Button(frame_botones, text="Ejecutar método", command=self.ejecutar_metodo)
        btn_ejecutar.pack(side="left", padx=5)

        btn_todos = ttk.Button(frame_botones, text="Ejecutar todos y comparar", command=self.ejecutar_todos_y_comparar)
        btn_todos.pack(side="left", padx=5)

        btn_informe = ttk.Button(frame_botones, text="Generar informe (todos)", command=self.generar_informe_todos)
        btn_informe.pack(side="left", padx=5)

        # ---------------- PANEL INFERIOR: TABLA + INFO ----------------
        frame_inferior = ttk.Frame(master)
        frame_inferior.pack(fill="both", expand=True, padx=10, pady=10)

        frame_tabla = ttk.LabelFrame(frame_inferior, text="Tabla de iteraciones")
        frame_tabla.pack(fill="both", expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(frame_tabla, columns=("iter", "error"), show="headings", height=15)
        self.tree.heading("iter", text="iter")
        self.tree.heading("error", text="error")
        self.tree.column("iter", width=50, anchor="center")
        self.tree.column("error", width=120, anchor="center")
        self.tree.pack(fill="both", expand=True)

        # ---------------- PANEL INFO (DEBAJO DE LA TABLA) ----------------
        frame_info = ttk.LabelFrame(master, text="Información del método")
        frame_info.pack(fill="x", padx=10, pady=(0, 10))

        self.label_radio = ttk.Label(frame_info, text="Radio espectral T: -")
        self.label_radio.pack(anchor="w", pady=2)

        self.label_conv = ttk.Label(frame_info, text="Convergencia teórica (ρ(T)<1): -")
        self.label_conv.pack(anchor="w", pady=2)

        self.label_mejor = ttk.Label(frame_info, text="Mejor método (comparación): -")
        self.label_mejor.pack(anchor="w", pady=2)


    # ---------------------------------------------------------------
    # Utilidades de parseo
    # ---------------------------------------------------------------
    def _parse_matrix(self, text: str) -> List[List[float]]:
        filas = []
        for line in text.strip().splitlines():
            if not line.strip():
                continue
            partes = line.replace(",", " ").split()
            fila = [float(p) for p in partes]
            filas.append(fila)
        if not filas:
            raise ValueError("La matriz A no puede estar vacía.")
        n = len(filas[0])
        for fila in filas:
            if len(fila) != n:
                raise ValueError("Todas las filas de A deben tener el mismo número de columnas.")
        if len(filas) != n:
            raise ValueError("La matriz A debe ser cuadrada.")
        if n > 7:
            raise ValueError("El tamaño máximo permitido es 7x7.")
        return filas

    def _parse_vector(self, text: str, n_esperado: int, nombre: str) -> List[float]:
        vals = []
        for line in text.strip().splitlines():
            if not line.strip():
                continue
            vals.append(float(line.strip()))
        if len(vals) != n_esperado:
            raise ValueError(f"El vector {nombre} debe tener {n_esperado} componentes.")
        return vals

    def _parse_vector_opcional(self, text: str, n_esperado: int) -> List[float] | None:
        if not text.strip():
            return None
        vals = []
        for line in text.strip().splitlines():
            if not line.strip():
                continue
            vals.append(float(line.strip()))
        if len(vals) != n_esperado:
            raise ValueError(f"El vector x0 debe tener {n_esperado} componentes.")
        return vals

    def _configurar_tabla(self, n: int):
        cols = ["iter"] + [f"x{i+1}" for i in range(n)] + ["error"]
        self.tree["columns"] = cols
        self.tree["show"] = "headings"
        for c in cols:
            self.tree.heading(c, text=c)
            width = 60 if c == "iter" else 90
            self.tree.column(c, width=width, anchor="center")
        # limpiar filas
        for item in self.tree.get_children():
            self.tree.delete(item)

    # ---------------------------------------------------------------
    # Ejecución individual
    # ---------------------------------------------------------------
    def ejecutar_metodo(self):
        try:
            A = self._parse_matrix(self.text_A.get("1.0", "end"))
            n = len(A)
            b = self._parse_vector(self.text_b.get("1.0", "end"), n, "b")
            x0 = self._parse_vector_opcional(self.text_x0.get("1.0", "end"), n)
            tol = float(self.entry_tol.get())
            max_iter = int(self.entry_iter.get())
            omega = float(self.entry_omega.get())
        except Exception as e:
            messagebox.showerror("Error de entrada", str(e))
            return

        self.A_ultima = A
        self.b_ultima = b

        metodo = self.metodo.get()
        try:
            if metodo == "Jacobi":
                res = jacobi(A, b, x0, tol, max_iter)
            elif metodo == "Gauss-Seidel":
                res = gauss_seidel(A, b, x0, tol, max_iter)
            else:  # SOR
                res = sor(A, b, x0, omega, tol, max_iter)
        except Exception as e:
            messagebox.showerror("Error en método", str(e))
            return

        self._mostrar_resultado_en_tabla(res)
        self._actualizar_info(res, metodo)

    def _mostrar_resultado_en_tabla(self, res):
        A = np.array(self.A_ultima, dtype=float)
        n = A.shape[0]
        self._configurar_tabla(n)

        for fila in self.tree.get_children():
            self.tree.delete(fila)

        for fila in res["tabla"]:
            vals = [fila["iter"]]
            x_vec = fila["x"]
            vals.extend(f"{xi:.6g}" for xi in x_vec)
            vals.append(f"{fila['error']:.3e}")
            self.tree.insert("", "end", values=vals)

    def _actualizar_info(self, res, nombre_metodo: str):
        rho = res.get("radio_espectral", None)
        if rho is not None:
            self.label_radio.config(text=f"Radio espectral T ({nombre_metodo}): {rho:.6g}")
            conv_teor = "Sí (ρ(T)<1)" if rho < 1 else "No (ρ(T)≥1)"
            self.label_conv.config(text=f"Convergencia teórica (ρ(T)<1): {conv_teor}")
        else:
            self.label_radio.config(text="Radio espectral T: -")
            self.label_conv.config(text="Convergencia teórica (ρ(T)<1): -")

    # ---------------------------------------------------------------
    # Ejecutar TODOS y comparar
    # ---------------------------------------------------------------
    def ejecutar_todos_y_comparar(self):
        try:
            resultados, mejor = self._calcular_todos_metodos()
            self.resultados_cap2 = resultados
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if mejor is None:
            self.label_mejor.config(text="Mejor método (comparación): Ninguno convergió.")
            messagebox.showinfo("Comparación", "Ningún método convergió con los parámetros dados.")
        else:
            metodo_mejor, res_mejor = mejor
            txt = (
                f"{metodo_mejor} | Iteraciones: {res_mejor['iteraciones']} | "
                f"Residuo final: {res_mejor['residuo_final']:.3e}"
            )
            self.label_mejor.config(text="Mejor método (comparación): " + txt)
            messagebox.showinfo("Comparación", txt)

    def _calcular_todos_metodos(self):
        A = self._parse_matrix(self.text_A.get("1.0", "end"))
        n = len(A)
        b = self._parse_vector(self.text_b.get("1.0", "end"), n, "b")
        x0 = self._parse_vector_opcional(self.text_x0.get("1.0", "end"), n)
        tol = float(self.entry_tol.get())
        max_iter = int(self.entry_iter.get())
        omega = float(self.entry_omega.get())

        self.A_ultima = A
        self.b_ultima = b

        A_np = np.array(A, dtype=float)
        b_np = np.array(b, dtype=float)

        resultados = {}

        def normalizar(nombre, res, error_msg=None):
            if res is None:
                return {
                    "ejecutado": False,
                    "convergio": False,
                    "iteraciones": None,
                    "radio_espectral": None,
                    "error_iter_final": None,
                    "residuo_final": None,
                    "mensaje_error": error_msg,
                }
            tabla = res.get("tabla", [])
            if tabla:
                err_iter_final = tabla[-1]["error"]
                x_final = res["sol"]
                residuo_final = float(np.linalg.norm(A_np @ x_final - b_np, ord=np.inf))
            else:
                err_iter_final = None
                x_final = res["sol"]
                residuo_final = float(np.linalg.norm(A_np @ x_final - b_np, ord=np.inf))

            return {
                "ejecutado": True,
                "convergio": res.get("convergio", False),
                "iteraciones": res.get("iteraciones"),
                "radio_espectral": res.get("radio_espectral"),
                "error_iter_final": err_iter_final,
                "residuo_final": residuo_final,
                "mensaje_error": error_msg,
            }

        # Jacobi
        try:
            r_jac = jacobi(A, b, x0, tol, max_iter)
            resultados["Jacobi"] = normalizar("Jacobi", r_jac)
        except Exception as e:
            resultados["Jacobi"] = normalizar("Jacobi", None, str(e))

        # Gauss-Seidel
        try:
            r_gs = gauss_seidel(A, b, x0, tol, max_iter)
            resultados["Gauss-Seidel"] = normalizar("Gauss-Seidel", r_gs)
        except Exception as e:
            resultados["Gauss-Seidel"] = normalizar("Gauss-Seidel", None, str(e))

        # SOR
        try:
            r_sor = sor(A, b, x0, omega, tol, max_iter)
            resultados["SOR"] = normalizar("SOR", r_sor)
        except Exception as e:
            resultados["SOR"] = normalizar("SOR", None, str(e))

        # determinar mejor
        mejor = None
        for metodo, res in resultados.items():
            if not res["ejecutado"]:
                continue
            if not res["convergio"]:
                continue
            if res["residuo_final"] is None:
                continue
            if mejor is None:
                mejor = (metodo, res)
            else:
                _, r_act = mejor
                if res["residuo_final"] < r_act["residuo_final"]:
                    mejor = (metodo, res)
                elif res["residuo_final"] == r_act["residuo_final"]:
                    if res["iteraciones"] < r_act["iteraciones"]:
                        mejor = (metodo, res)

        return resultados, mejor

    # ---------------------------------------------------------------
    # Informe
    # ---------------------------------------------------------------
    def generar_informe_todos(self):
        if not messagebox.askyesno(
            "Informe",
            "¿Desea ejecutar todos los métodos y generar el informe de comparación?",
        ):
            return

        try:
            resultados, mejor = self._calcular_todos_metodos()
            self.resultados_cap2 = resultados
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        A = self._parse_matrix(self.text_A.get("1.0", "end"))
        n = len(A)
        b = self._parse_vector(self.text_b.get("1.0", "end"), n, "b")

        params = {
            "tolerancia": self.entry_tol.get(),
            "iter_max": self.entry_iter.get(),
            "omega_SOR": self.entry_omega.get(),
        }

        ruta = escribir_informe_cap2(resultados, A, b, params)
        messagebox.showinfo("Informe generado", f"Informe guardado en:\n{ruta}")
