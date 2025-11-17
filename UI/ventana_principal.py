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
