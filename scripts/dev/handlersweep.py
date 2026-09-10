"""Call EVERY button handler in every notebook and report NameError/AttributeError.

Handlers are where the shipped bugs live: validate_notebooks executes the cell,
never the callbacks. Only NameError/AttributeError count as failures - a
FileNotFoundError from a handler run against an empty workspace is expected.
"""
import contextlib, io, json, sys, traceback, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")
from IPython.display import display as _d

# Plotly's fig.show() OPENS A BROWSER TAB outside a notebook. Calling every
# handler would otherwise spawn a window per figure. Pin a renderer that writes
# nothing and opens nothing.
import plotly.io as pio
pio.renderers.default = "json"
import plotly.graph_objects as _go
_go.Figure.show = lambda self, *a, **k: None

# The two Step 05 tuners are REAL optimiser runs, not UI updates: measured
# 174 s for `on_tune_de_budget` alone on this workspace, and the lambda L-curve
# is longer. That is minutes of differential evolution inside a lint sweep, and
# it is why this sweep started timing out.
#
# They are NOT put in SKIP. `on_tune_lambda_lcurve` is exactly the handler that
# died on `SETUP_META` in front of a user, so it is the last one to stop
# checking. Instead the optimiser underneath them is bound to a single tiny
# budget: the whole handler still executes end to end - every attribute read,
# every plot, every status update - it just does not spend three minutes
# converging to a recommendation nothing reads. This must be patched BEFORE the
# notebooks are exec'd, because they import these by name.
import scripts.modules.inversion_tuning as _tuning

_real_tune_de = _tuning.tune_de_budget
_real_tune_lambda = _tuning.tune_lambda_lcurve


def _cheap_tune_de(cfg, tx_entry, **kw):
    kw.update(budgets=((4, 1),), n_seeds=2, n_jobs=1)
    return _real_tune_de(cfg, tx_entry, **kw)


def _cheap_tune_lambda(cfg, tx_entry, **kw):
    kw.update(lambdas=(0.0, 500.0), n_jobs=1)
    return _real_tune_lambda(cfg, tx_entry, **kw)


_tuning.tune_de_budget = _cheap_tune_de
_tuning.tune_lambda_lcurve = _cheap_tune_lambda

NBS = ["01_fw_setup.ipynb", "02_fwmodelling_and_data_visualization.ipynb",
       "03_2d_inversion.ipynb", "04_2d_inversion_results.ipynb",
       "05_1d_inversion.ipynb", "06_1d_inversion_results.ipynb"]
SKIP = {"on_quit_gui", "on_quit", "on_run_all", "on_run_inversion", "on_run_model",
        "on_calibrate_all", "on_generate_inputs", "on_apply_outputs"}   # spawn work

bad = []
for nb in NBS:
    d = json.loads((ROOT / nb).read_text())
    code = "\n\n".join("".join(c["source"]) for c in d["cells"] if c["cell_type"] == "code")
    g = {"__name__": "__main__", "__file__": str(ROOT / nb), "display": _d}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, nb, "exec"), g)
    handlers = sorted(k for k, v in g.items()
                      if k.startswith(("on_", "update_", "refresh_")) and callable(v)
                      and k not in SKIP)
    hits = []
    for h in handlers:
        try:
            # `observe` callbacks receive a CHANGE DICT, not None - passing None
            # produced four bogus AttributeErrors on the first run of this sweep.
            change = {"name": "value", "old": None, "new": None, "type": "change",
                      "owner": None}
            with contextlib.redirect_stdout(io.StringIO()):
                for arg in (None, change, ()):
                    try:
                        g[h]() if arg == () else g[h](arg)
                        break
                    except TypeError:
                        continue
                    except AttributeError:
                        if arg is None:
                            continue        # retry as an observe callback
                        raise
        except (NameError, AttributeError) as e:
            hits.append(f"{h}: {type(e).__name__}: {e}")
        except Exception:
            pass
    print(f"{'BUG' if hits else 'OK '} {nb}: {len(handlers)} handler(s)"
          + ("" if not hits else "\n     " + "\n     ".join(hits)))
    bad += hits

print("\nRESULT:", "PASS - no NameError/AttributeError in any handler" if not bad
      else f"{len(bad)} broken handler(s)")
sys.exit(1 if bad else 0)
