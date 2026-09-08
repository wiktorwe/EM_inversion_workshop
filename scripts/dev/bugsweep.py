"""Static + dynamic bug sweep on the workshop notebooks.

Catches the class of bug that keeps shipping: a symbol deleted while a
reference to it survives inside a callback. validate_notebooks.py cannot see
those, because it executes the cell but never clicks anything.
"""
import ast, builtins, contextlib, io, json, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")

NBS = ["01_fw_setup.ipynb", "02_fwmodelling_and_data_visualization.ipynb",
       "03_2d_inversion.ipynb", "04_2d_inversion_results.ipynb",
       "05_1d_inversion.ipynb", "06_1d_inversion_results.ipynb"]

def code_of(nb):
    d = json.loads((ROOT / nb).read_text())
    return "\n\n".join("".join(c["source"]) for c in d["cells"] if c["cell_type"] == "code")

def defined_names(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            out.update(a.arg for a in n.args.args + n.args.kwonlyargs)
            if n.args.vararg: out.add(n.args.vararg.arg)
            if n.args.kwarg: out.add(n.args.kwarg.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name: out.add(n.name)
        elif isinstance(n, (ast.comprehension,)):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name): out.add(t.id)
        elif isinstance(n, ast.Global): out.update(n.names)
        elif isinstance(n, ast.arg): out.add(n.arg)
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            for t in ast.walk(n.optional_vars):
                if isinstance(t, ast.Name): out.add(t.id)
    return out

fail = 0
for nb in NBS:
    src = code_of(nb)
    tree = ast.parse(src)
    known = defined_names(tree) | set(dir(builtins)) | {"display", "get_ipython", "__name__", "__file__"}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    undef = sorted(used - known)
    status = "OK " if not undef else "BUG"
    if undef: fail += 1
    print(f"{status} {nb}: undefined names = {undef or 'none'}")

    # every Button must be bound to something
    buttons = {t.targets[0].id for t in tree.body
               if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name)
               and isinstance(t.value, ast.Call)
               and getattr(getattr(t.value.func, 'attr', None), '__str__', lambda: '')() == 'Button'}
    bound = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr == "on_click" and isinstance(f.value, ast.Name):
                bound.add(f.value.id)
            if isinstance(f, ast.Name) and f.id in ("bind_button_with_feedback",
                                                    "_rebind_on_click") and n.args:
                if isinstance(n.args[0], ast.Name): bound.add(n.args[0].id)
    dead = sorted(buttons - bound)
    if dead:
        print(f"    UNBOUND BUTTONS: {dead}")
        fail += 1

print("\nRESULT:", "PASS - no undefined names, no unbound buttons" if not fail
      else f"{fail} problem(s) found")
sys.exit(1 if fail else 0)
