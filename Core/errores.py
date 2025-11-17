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
