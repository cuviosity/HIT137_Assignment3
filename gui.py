
# this is the window stuff. i tried to make it look like the example but not too serious :)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional
from explanations import OOP_EXPLANATIONS
from models import registry

# helper because i keep writing json.dumps wrong
def pretty_json(x):
    import json
    return json.dumps(x, indent=2, ensure_ascii=False)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tkinter AI GUI (amateur mode)")
        self.geometry("980x720")

        # i store some state here. globals are bad but also easy
        self.input_mode = tk.StringVar(value="Text")
        self.selected_model_name = tk.StringVar(value=registry.get_model_names()[0])
        self.model = None
        self.image_path: Optional[str] = None

        self._make_menus()
        self._make_top()
        self._make_main()
        self._make_bottom()
        self._show_oop_text()

    def _make_menus(self):
        m = tk.Menu(self); self.config(menu=m)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Exit", command=self.destroy)
        m.add_cascade(label="File", menu=f)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="About", command=lambda: messagebox.showinfo("About","i hope this works on your pc"))
        m.add_cascade(label="Help", menu=h)

    def _make_top(self):
        top = ttk.Frame(self, padding=7); top.pack(fill="x")
        ttk.Label(top, text="Model Selection:").pack(side="left")
        self.dd = ttk.Combobox(top, values=registry.get_model_names(), textvariable=self.selected_model_name, state="readonly", width=40)
        self.dd.pack(side="left", padx=6)
        ttk.Button(top, text="Load Model", command=self.on_load).pack(side="left", padx=4)

    def _make_main(self):
        main = ttk.Frame(self, padding=7); main.pack(fill="both", expand=True)
        left = ttk.Labelframe(main, text="User Input Section", padding=7)
        right = ttk.Labelframe(main, text="Model Output Section", padding=7)
        left.pack(side="left", fill="both", expand=True, padx=(0,7))
        right.pack(side="left", fill="both", expand=True)

        r = ttk.Frame(left); r.pack(fill="x")
        ttk.Radiobutton(r, text="Text", variable=self.input_mode, value="Text").pack(side="left")
        ttk.Radiobutton(r, text="Image", variable=self.input_mode, value="Image").pack(side="left", padx=8)
        ttk.Button(r, text="Browse", command=self.on_browse).pack(side="right")

        self.input_box = tk.Text(left, height=12); self.input_box.pack(fill="both", expand=True, pady=6)

        b = ttk.Frame(left); b.pack(fill="x", pady=4)
        ttk.Button(b, text="Run Model 1", command=lambda: self._run_fixed("Text: Sentiment (easy one)")).pack(side="left")
        ttk.Button(b, text="Run Model 2", command=lambda: self._run_fixed("Vision: Image Classifier (the picture one)")).pack(side="left", padx=6)
        ttk.Button(b, text="Clear", command=self.on_clear).pack(side="right")

        ttk.Label(right, text="Output Display:").pack(anchor="w")
        self.output_box = tk.Text(right, height=12); self.output_box.pack(fill="both", expand=True, pady=6)

        self.left = left
        self.right = right

    def _make_bottom(self):
        bottom = ttk.Labelframe(self, text="Model Information & Explanation", padding=7)
        bottom.pack(fill="both", expand=False, padx=7, pady=7)

        grid = ttk.Frame(bottom); grid.pack(fill="both", expand=True)
        self.model_info = tk.Text(grid, height=10, width=50); self.model_info.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        self.oop_info = tk.Text(grid, height=10, width=50); self.oop_info.grid(row=0, column=1, sticky="nsew")
        grid.columnconfigure(0, weight=1); grid.columnconfigure(1, weight=1)

        ttk.Label(self, text="note: set HF_TOKEN or it will crash (sorry)").pack(fill="x", padx=8, pady=(0,8))

    def _show_oop_text(self):
        self.oop_info.delete("1.0","end")
        for k, v in OOP_EXPLANATIONS.items():
            self.oop_info.insert("end", f"- {k}: {v}\n\n")

    # event handlers (i put them together so i can find them fast)
    def on_browse(self):
        if self.input_mode.get() == "Image":
            p = filedialog.askopenfilename(title="choose picture", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
            if p:
                self.image_path = p
                self.input_box.delete("1.0","end")
                self.input_box.insert("end", f"[image selected] {p}\n")
        else:
            messagebox.showinfo("Browse", "switch to Image first to pick a file")

    def on_clear(self):
        self.input_box.delete("1.0","end")
        self.output_box.delete("1.0","end")
        self.model_info.delete("1.0","end")

    def on_load(self):
        name = self.selected_model_name.get()
        try:
            self.model = registry.create_model(name)
            self.model.load()
            self._fill_model_info(name)
            messagebox.showinfo("ok","model loaded :)")
        except Exception as e:
            messagebox.showerror("oops", str(e))

    def _fill_model_info(self, name):
        self.model_info.delete("1.0","end")
        cat = "Text" if "Text:" in name else "Vision"
        self.model_info.insert("end", f"Model Name: {name}\n")
        self.model_info.insert("end", f"Category: {cat}\n")
        self.model_info.insert("end", f"Hugging Face ID: {self.model.model_id if self.model else '?'}\n")
        self.model_info.insert("end", "Short Description:\n")
        if cat == "Text":
            self.model_info.insert("end","  does sentiment. like positive/negative vibes.\n")
        else:
            self.model_info.insert("end","  guesses what the picture is basically.\n")

    def _get_user_input(self):
        if self.input_mode.get() == "Text":
            return self.input_box.get("1.0","end").strip()
        else:
            return self.image_path

    def _run_fixed(self, fixed_name):
        try:
            mdl = registry.create_model(fixed_name); mdl.load()
        except Exception as e:
            messagebox.showerror("load fail", str(e)); return
        data = self._get_user_input()
        if not data:
            messagebox.showwarning("hmm","type something or pick an image first")
            return
        try:
            result = mdl.run(data)
            self.output_box.delete("1.0","end")
            self.output_box.insert("end", pretty_json(result))
        except Exception as e:
            messagebox.showerror("run fail", str(e))
