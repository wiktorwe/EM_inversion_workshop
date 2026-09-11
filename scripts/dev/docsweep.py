#!/usr/bin/env python3
"""Does the PROSE still describe code that exists?

The bug class this exists to catch: the code moves and the prose does not.
A panel can describe a contract the code dropped, the README can repeat it, and
a comment can name a function nobody defines any more - while every other sweep
PASSES, because they check undefined NAMES, cell EXECUTION, handler EXCEPTIONS
and PATH binding. None of them can see a sentence that is false, so "4/4 PASS"
is not evidence the application works.

Three checks:

1. DEAD SYMBOLS - any identifier mentioned in prose, in backticks, that does not
   exist anywhere in the code.
2. TEMPORAL WORDS IN GUI TEXT - "now", "no longer", "used to", "instead of" and
   friends, in anything a USER sees: markdown cells, `ipw.HTML`, `push_message`,
   widget `description=`. GUI text says what a control does and how to use it;
   it is not a changelog, and the reader has no idea what any other version
   said. Rationale belongs in code comments, limitations in `KNOWN_ISSUES.md`,
   parameter detail in `doc/gui_manual.tex`. See RULE 5 in the project skill.
3. SECTIONS MISSING FROM THE INTRO - a notebook section that the notebook's own
   intro cell never mentions. A purpose statement that is true but describes
   half the notebook is invisible to checks 1 and 2, and it is the first thing
   a user reads.

Prose means: markdown cells, `#` comments, docstrings, .md files, and the HTML
strings the notebooks display to the user - the last being where this repo's
stale claims actually live.

A backticked symbol is exempted from check 1 when a past-tense marker appears
in the same sentence: `KNOWN_ISSUES.md` and the design notes have to be able to
name something that is gone, and naming it is not the same as claiming it
exists. The heuristic is deliberately generous - the target is the claim still
being made in the present tense.
"""
from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIPLE_D = chr(34) * 3
TRIPLE_S = chr(39) * 3

# Identifiers that legitimately appear in prose without being code here.
ALLOW = {
    # third-party / external tools referred to by name
    "empymod", "numpy", "scipy", "plotly", "ipywidgets", "voila", "segyio",
    "joblib", "nvidia-smi", "mpirun", "conda", "pip", "make", "git",
    # rockem-suite symbols (a different repo)
    "mpiEmmodTE2d", "mpiEminvTE2d", "mpiEmmodADI3d", "mpiEmmodADITE2d",
    "mpiEminvADITE2d", "mpiEmmodTE2dGpu", "mpiEminvTE2dGpu",
    "magnetic_line_source_fields_layered", "magnetic_z_line_source_fields_layered",
    "layers_to_stack", "build_layered_1d_grid", "LayerSpec", "ModelEmTE2D",
    "greens_layered_2d", "insertPhysicalSource", "recordData", "modint",
    "suggest_time_steps", "stencil_cfl_factor", "_STENCIL_COEFFS", "Der",
    "cflStencilSum", "makeMap", "Geometry2D", "WavesEmTE2D", "der", "utils",
    "uncertainty_from_result", "run_block1d_inversion", "InversionConfig",
    "utils.check_ab", "check_ab",
    # upstream rockem-suite symbols and inv.cfg keys, and the repo itself
    "InversionEmTE2D", "max_linesearch", "EM_inversion_workshop",
    # inv.cfg record/weight keys - config keys the engine reads, not Python names
    "Recordfile_HX", "Recordfile_HZ", "Recordfile_EY",
    "Recordfile_HX_HX", "Recordfile_HX_HZ", "Recordfile_HZ_HX", "Recordfile_HZ_HZ",
    "Dataweightfile", "Dataweightfile_HX", "Dataweightfile_HZ", "source_type",
    # upstream C++ methods named when explaining what the engine does
    "saveResults", "runGrad", "runGrad_adi", "stackImage",
    "differential_evolution", "dual_annealing", "nbformat", "ast",
}
PAST_MARKERS = re.compile(
    r"\b(used to|was|were|had|until|no longer|previously|formerly|historical|"
    r"legacy|old|removed|deleted|gone|before|once|earlier|moved|renamed|"
    r"replaced|retired|superseded|instead of|rather than|not any ?more|"
    r"shipped|trap|bug)\b",
    re.I)
IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`")


def code_tokens() -> set[str]:
    """Every token that appears in CODE (not comments) anywhere in the repo.

    Deliberately broader than "names the AST defines": dict keys, string
    literals and f-string fragments are all legitimate things for prose to
    reference. The question this check asks is the one that actually matters -
    *does this thing still exist anywhere in the code at all?* - and a tighter
    definition produced 70 false positives on a clean tree, which is a check
    nobody would run twice.
    """
    tokens: set[str] = set()
    word = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

    def add_code(src: str) -> None:
        try:
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type == tokenize.COMMENT:
                    continue                      # comments are PROSE
                if tok.type == tokenize.STRING:
                    # DOCSTRINGS ARE PROSE TOO. Counting them as code defeats
                    # the check: a symbol named only in the docstring that
                    # discusses it would look defined, and the sweep would pass
                    # on a poisoned tree.
                    # Triple-quoted or long => prose; short literals stay, since
                    # prose legitimately references string VALUES.
                    body = tok.string.lstrip('rbfuRBFU')
                    if body[:3] in (TRIPLE_D, TRIPLE_S) or len(tok.string) > 120:
                        continue
                tokens.update(word.findall(tok.string))
        except (tokenize.TokenError, IndentationError, SyntaxError):
            tokens.update(word.findall(src))

    for p in sorted(ROOT.rglob("*.py")):
        if "__pycache__" not in p.parts:
            add_code(p.read_text(errors="replace"))
    for p in sorted(ROOT.glob("*.ipynb")):
        for c in json.loads(p.read_text())["cells"]:
            if c["cell_type"] == "code":
                add_code("".join(c["source"]))
    for p in ROOT.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            tokens.add(p.name)
            tokens.add(p.stem)
    return tokens


def prose_chunks():
    """(where, text) for every piece of prose in the repo."""
    for p in sorted(ROOT.rglob("*.md")):
        if any(x in p.parts for x in ("__pycache__", "node_modules")):
            continue
        yield str(p.relative_to(ROOT)), p.read_text(errors="replace")
    for p in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        src = p.read_text(errors="replace")
        try:
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type == tokenize.COMMENT:
                    yield f"{p.relative_to(ROOT)}:{tok.start[0]}", tok.string
                elif tok.type == tokenize.STRING and len(tok.string) > 120:
                    yield f"{p.relative_to(ROOT)}:{tok.start[0]}", tok.string
        except (tokenize.TokenError, IndentationError):
            pass
    for p in sorted(ROOT.glob("*.ipynb")):
        for ci, c in enumerate(json.loads(p.read_text())["cells"]):
            text = "".join(c["source"])
            if c["cell_type"] == "markdown":
                yield f"{p.name} md cell {ci}", text
            else:
                # comments and the long strings the GUI displays
                for m in re.finditer(r"^\s*#.*$", text, flags=re.M):
                    yield f"{p.name} cell {ci}", m.group(0)
                for m in re.finditer(r"'([^'\\]{120,})'|\"([^\"\\]{120,})\"", text):
                    yield f"{p.name} cell {ci}", m.group(0)


TEMPORAL = re.compile(
    r"\b(no longer|used to|previously|formerly|instead of|this replaces|"
    r"replaced by|has changed|is now|are now|now reads|now uses|now computes|"
    r"used to be|rather than before|as before|unlike before)\b", re.I)


def gui_strings():
    """(where, text) for every string a USER sees, from the notebooks."""
    for p in sorted(ROOT.glob("*.ipynb")):
        nb = json.loads(p.read_text())
        for ci, c in enumerate(nb["cells"]):
            if c["cell_type"] == "markdown":
                yield f"{p.name} markdown cell {ci}", "".join(c["source"])
                continue
            try:
                tree = ast.parse("".join(c["source"]))
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.Call):
                    fname = getattr(n.func, "attr", None) or getattr(n.func, "id", None)
                    if fname in ("push_message", "HTML"):
                        for a in ast.walk(n):
                            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                                yield f"{p.name} cell {ci} ({fname})", a.value
                    for k in n.keywords:
                        if k.arg in ("description", "placeholder") and \
                           isinstance(k.value, ast.Constant) and \
                           isinstance(k.value.value, str):
                            yield f"{p.name} cell {ci} (widget label)", k.value.value


# Words too generic to prove that an intro mentions a section: every notebook
# runs, loads and computes something.
GENERIC = {
    "run", "runs", "load", "loads", "compute", "computes", "show", "shows",
    "the", "and", "for", "all", "with", "from", "this", "that", "step",
    "steps", "into", "onto", "each", "its", "own", "set", "sets", "get",
    "gets", "make", "makes", "use", "uses", "view", "views", "data",
}


def section_headings(nb: dict):
    """The numbered section titles a notebook renders, e.g. `3b) Validate ...`."""
    out = []
    for c in nb["cells"]:
        text = "".join(c["source"])
        for m in re.finditer(r"<h[23][^>]*>\s*(\d+[a-z]?\)\s*[^<]+)</h[23]>", text):
            out.append(re.sub(r"\s+", " ", m.group(1)).strip())
    return out


def uncovered_sections():
    """Sections the notebook's intro cell does not mention.

    The intro is the only place that says what the whole step is for, and a
    reader decides from it whether this is the notebook they want. An intro
    that describes two of five sections is not false, so neither the dead-symbol
    check nor the temporal-word check can see it - but it is still wrong.

    A section counts as mentioned when one of its distinctive words appears in
    the intro, matched on a six-character stem so `Visualization` covers
    `visualising`. Generic verbs are not distinctive; `channel-gain` is.
    """
    for p in sorted(ROOT.glob("*.ipynb")):
        nb = json.loads(p.read_text())
        intro = next(("".join(c["source"]) for c in nb["cells"]
                      if c["cell_type"] == "markdown"), "")
        stems = {w[:6] for w in re.findall(r"[A-Za-z]{3,}", intro.lower())}
        for head in section_headings(nb):
            title = head.split(")", 1)[1]
            words = [w for w in re.findall(r"[A-Za-z]{4,}", title.lower())
                     if w not in GENERIC]
            if words and not any(w[:6] in stems for w in words):
                yield p.name, head


def main() -> int:
    known = code_tokens()
    bad: list[str] = []
    for where, text in prose_chunks():
        for sentence in re.split(r"(?<=[.!?])\s+|\n\n", text):
            if PAST_MARKERS.search(sentence):
                continue          # a deliberate historical mention
            for m in IDENT.finditer(sentence):
                token = m.group(1)
                leaf = token.split(".")[-1]
                if leaf in known or token in ALLOW or leaf in ALLOW:
                    continue
                # Dataset directory names (`f2000Hz_hx`) are BUILT at runtime by
                # `headless.dataset_name`, so they never appear as literals.
                if re.fullmatch(r"f\d+Hz_(hx|hz)", leaf):
                    continue
                # Only flag CODE-SHAPED tokens. A backticked English word -
                # `view`, `frequency`, `component`, `global` - is a widget
                # label or a noun, not a symbol, and flagging those is what
                # made the first version of this check unusable.
                code_shaped = ("_" in leaf or "." in token
                               or re.search(r"[a-z][A-Z]", leaf))
                if not code_shaped or len(leaf) < 6 or leaf.isupper():
                    continue
                bad.append(f"{where}: `{token}` is not defined anywhere in the code")

    seen, uniq = set(), []
    for b in bad:
        if b not in seen:
            seen.add(b)
            uniq.append(b)

    unsaid = [f'{nb}: the intro cell does not mention "{head}"'
              for nb, head in uncovered_sections()]

    timeless = []
    for where, text in gui_strings():
        m = TEMPORAL.search(text)
        if m:
            snippet = re.sub(r"\s+", " ", text).strip()[:110]
            timeless.append(f'{where}: "{m.group(0)}" in GUI text -- {snippet}')

    for b in uniq:
        print("STALE " + b)
    for t in dict.fromkeys(timeless):
        print("TEMPORAL " + t)
    for u in unsaid:
        print("UNSAID " + u)
    n_bad = len(uniq) + len(dict.fromkeys(timeless)) + len(unsaid)
    if not n_bad:
        print("\nRESULT: PASS - prose matches the code, GUI text is timeless, "
              "every section is in its notebook's intro")
    else:
        print(f"\nRESULT: {len(uniq)} stale reference(s), "
              f"{len(dict.fromkeys(timeless))} temporal word(s) in GUI text, "
              f"{len(unsaid)} unmentioned section(s)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
