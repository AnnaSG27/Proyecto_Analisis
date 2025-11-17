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
