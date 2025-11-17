# Core/informes.py
import os
from datetime import datetime
from typing import Dict, Any, List


# ============================================================
# INFORME CAPÍTULO 1
# ============================================================

def escribir_informe_cap1(
    resultados: Dict[str, Dict],
    fx_str: str,
    tipo_error: str,
    params: Dict[str, Any],
) -> str:
    os.makedirs("informes", exist_ok=True)
    nombre = f"informes/cap1_informe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # determinar mejor método
    mejor = None
    for metodo, res in resultados.items():
        if not res.get("ejecutado"):
            continue
        if not res.get("convergio"):
            continue
        if mejor is None:
            mejor = (metodo, res)
        else:
            _, r_act = mejor
            if res["iteraciones"] < r_act["iteraciones"]:
                mejor = (metodo, res)
            elif res["iteraciones"] == r_act["iteraciones"]:
                if res["error_final"] is not None and r_act["error_final"] is not None:
                    if res["error_final"] < r_act["error_final"]:
                        mejor = (metodo, res)

    with open(nombre, "w", encoding="utf-8") as f:
        f.write("INFORME CAPÍTULO 1 – MÉTODOS PARA ECUACIONES NO LINEALES\n\n")
        f.write(f"Función f(x): {fx_str}\n")
        f.write(f"Tipo de error seleccionado: {tipo_error}\n\n")

        f.write("Parámetros de ejecución:\n")
        for k, v in params.items():
            f.write(f"  - {k}: {v}\n")
        f.write("\n")

        f.write("RESUMEN POR MÉTODO\n")
        f.write("-------------------\n\n")
        for metodo, res in resultados.items():
            f.write(f"Método: {metodo}\n")
            if not res.get("ejecutado"):
                f.write("  Estado: NO EJECUTADO\n")
                f.write(f"  Motivo: {res.get('mensaje_error', 'No disponible')}\n\n")
                continue

            f.write("  Ejecutado: Sí\n")
            f.write(f"  Convergió: {res.get('convergio')}\n")
            f.write(f"  Iteraciones: {res.get('iteraciones')}\n")
            f.write(f"  Raíz aproximada: {res.get('raiz')}\n")
            f.write(f"  Error final (según criterio): {res.get('error_final')}\n")
            if res.get("mensaje_error"):
                f.write(f"  Nota: {res['mensaje_error']}\n")
            f.write("\n")

        f.write("COMPARACIÓN GLOBAL\n")
        f.write("------------------\n")
        if mejor is None:
            f.write("No hay ningún método que haya convergido con los parámetros dados.\n")
        else:
            metodo_mejor, res_mejor = mejor
            f.write("Según el criterio de comparación (convergencia, menor número de\n")
            f.write("iteraciones y menor error final), el mejor método en esta ejecución fue:\n\n")
            f.write(f"  -> {metodo_mejor}\n")
            f.write(f"     Iteraciones: {res_mejor['iteraciones']}\n")
            f.write(f"     Error final: {res_mejor['error_final']}\n")
            f.write(f"     Raíz aproximada: {res_mejor['raiz']}\n")

    return nombre


# ============================================================
# INFORME CAPÍTULO 2
# ============================================================

def _formatear_matriz(A: List[List[float]]) -> str:
    lineas = []
    for fila in A:
        lineas.append(" ".join(f"{v:.6g}" for v in fila))
    return "\n".join(lineas)


def _formatear_vector(b: List[float]) -> str:
    return "\n".join(f"{v:.6g}" for v in b)


def escribir_informe_cap2(
    resultados: Dict[str, Dict],
    A: List[List[float]],
    b: List[float],
    params: Dict[str, Any],
) -> str:
    """
    resultados: {
      "Jacobi": {
          "ejecutado": bool,
          "convergio": bool,
          "iteraciones": int | None,
          "radio_espectral": float | None,
          "error_iter_final": float | None,
          "residuo_final": float | None,
          "mensaje_error": str | None,
      },
      ...
    }
    """
    os.makedirs("informes", exist_ok=True)
    nombre = f"informes/cap2_informe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # determinar mejor método: el que converge y tiene menor residuo final
    mejor = None
    for metodo, res in resultados.items():
        if not res.get("ejecutado"):
            continue
        if not res.get("convergio"):
            continue
        if res.get("residuo_final") is None:
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

    with open(nombre, "w", encoding="utf-8") as f:
        f.write("INFORME CAPÍTULO 2 – MÉTODOS ITERATIVOS PARA SISTEMAS LINEALES\n\n")

        f.write("Sistema Ax = b utilizado:\n\n")
        f.write("Matriz A:\n")
        f.write(_formatear_matriz(A))
        f.write("\n\nVector b:\n")
        f.write(_formatear_vector(b))
        f.write("\n\n")

        f.write("Parámetros de ejecución:\n")
        for k, v in params.items():
            f.write(f"  - {k}: {v}\n")
        f.write("\n")

        f.write("RESUMEN POR MÉTODO\n")
        f.write("-------------------\n\n")

        for metodo, res in resultados.items():
            f.write(f"Método: {metodo}\n")
            if not res.get("ejecutado"):
                f.write("  Ejecutado: NO\n")
                f.write(f"  Motivo: {res.get('mensaje_error', 'No disponible')}\n\n")
                continue

            f.write("  Ejecutado: SÍ\n")
            f.write(f"  Convergió (según tolerancia): {res.get('convergio')}\n")
            f.write(f"  Iteraciones: {res.get('iteraciones')}\n")
            f.write(f"  Radio espectral de la matriz de iteración: {res.get('radio_espectral')}\n")
            f.write(f"  Error iterativo final ||x^(k+1)-x^(k)||_∞: {res.get('error_iter_final')}\n")
            f.write(f"  Residuo final ||Ax - b||_∞: {res.get('residuo_final')}\n")
            if res.get("mensaje_error"):
                f.write(f"  Nota: {res['mensaje_error']}\n")
            f.write("\n")

        f.write("COMPARACIÓN GLOBAL\n")
        f.write("------------------\n")
        if mejor is None:
            f.write("No hay ningún método que haya convergido con los parámetros dados.\n")
        else:
            metodo_mejor, res_mejor = mejor
            f.write("Según el criterio de comparación (convergencia, menor residuo final\n")
            f.write("y luego menor número de iteraciones), el mejor método en esta ejecución fue:\n\n")
            f.write(f"  -> {metodo_mejor}\n")
            f.write(f"     Iteraciones: {res_mejor['iteraciones']}\n")
            f.write(f"     Radio espectral: {res_mejor['radio_espectral']}\n")
            f.write(f"     Residuo final: {res_mejor['residuo_final']}\n")

    return nombre


# ============================================================
# INFORME CAPÍTULO 3
# ============================================================

def escribir_informe_cap3(
    resultados: Dict[str, Dict[str, Any]],
    x: List[float],
    y: List[float],
    params: Dict[str, Any],
) -> str:
    """
    resultados: {
      "Vandermonde": {
          "ejecutado": bool,
          "error_max": float | None,
          "descripcion": str,
          "mensaje_error": str | None,
      },
      ...
    }
    """
    os.makedirs("informes", exist_ok=True)
    nombre = f"informes/cap3_informe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # mejor método: menor error_max
    mejor = None
    for metodo, res in resultados.items():
        if not res.get("ejecutado"):
            continue
        if res.get("error_max") is None:
            continue
        if mejor is None:
            mejor = (metodo, res)
        else:
            _, r_act = mejor
            if res["error_max"] < r_act["error_max"]:
                mejor = (metodo, res)

    with open(nombre, "w", encoding="utf-8") as f:
        f.write("INFORME CAPÍTULO 3 – MÉTODOS DE INTERPOLACIÓN\n\n")
        f.write("Puntos (x_i, y_i) usados en la interpolación:\n")
        for xi, yi in zip(x, y):
            f.write(f"  ({xi:.6g}, {yi:.6g})\n")
        f.write("\n")

        f.write("Parámetros de ejecución:\n")
        for k, v in params.items():
            f.write(f"  - {k}: {v}\n")
        f.write("\n")

        f.write("RESUMEN POR MÉTODO\n")
        f.write("-------------------\n\n")
        for metodo, res in resultados.items():
            f.write(f"Método: {metodo}\n")
            if not res.get("ejecutado"):
                f.write("  Ejecutado: NO\n")
                f.write(f"  Motivo: {res.get('mensaje_error', 'No disponible')}\n\n")
                continue
            f.write("  Ejecutado: SÍ\n")
            f.write(f"  Descripción: {res.get('descripcion')}\n")
            f.write(f"  Error máximo en puntos dados: {res.get('error_max')}\n")
            if res.get("mensaje_error"):
                f.write(f"  Nota: {res['mensaje_error']}\n")
            f.write("\n")

        f.write("COMPARACIÓN GLOBAL\n")
        f.write("------------------\n")
        if mejor is None:
            f.write("No fue posible determinar un mejor método (no se ejecutaron\n")
            f.write("correctamente o no hay errores calculados).\n")
        else:
            metodo_mejor, res_mejor = mejor
            f.write("Se toma como mejor método aquel con menor error máximo en los\n")
            f.write("puntos dados.\n\n")
            f.write(f"  -> {metodo_mejor}\n")
            f.write(f"     Descripción: {res_mejor.get('descripcion')}\n")
            f.write(f"     Error máximo: {res_mejor.get('error_max')}\n")

    return nombre
