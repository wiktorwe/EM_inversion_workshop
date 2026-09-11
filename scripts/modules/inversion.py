"""Inversion setup helpers for staging inputs and local runs."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from third_party.rockseis.io.rsfile import rsfile


CFG_LINE_RE = re.compile(r'^(\s*([A-Za-z0-9_]+)\s*=\s*")([^"]*)(";\s*(?:#.*)?)$')


def read_cfg_values(cfg_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in Path(cfg_path).read_text().splitlines():
        match = CFG_LINE_RE.match(line)
        if match:
            values[match.group(2)] = match.group(3)
    return values


def update_cfg_values(
    cfg_path: Path,
    updates: Dict[str, str],
    allow_new: Optional[Iterable[str]] = None,
    new_key_comments: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Rewrite `key = "value";` lines in place; raise KeyError for unknown keys.

    The KeyError is a safety property, not an inconvenience: every key the
    workshop writes exists in `scripts/templates/inv.cfg`, so a typo in one of
    them is caught here instead of reaching the engine as a silently-ignored
    setting. `allow_new` is the narrow exception - keys named there are APPENDED
    when absent, everything else still raises.

    The only keys that use it are `Recordfile_<SRC>_<REC>`, which exist only for
    a multi-source (joint) inversion. They are deliberately NOT in the template:
    the engine reads `Recordfile_<SRC>_<REC>` in preference to `Recordfile_<REC>`
    (src/mpiEminvTE2d.cpp:395-402) and only falls back when the pair key is empty
    or absent, so a template that always carried them would either point every
    single-source run at files it never stages, or need inert empty values. The
    upstream config emitter omits them for a single source for the same reason
    (python/rockem/config/config.py:339-361), and a single-source `inv.cfg` from
    this workshop is byte-for-byte what it was before joint inversion existed.
    """
    cfg_path = Path(cfg_path)
    appendable = set(allow_new or ())
    lines = cfg_path.read_text().splitlines()
    seen = set()
    out: List[str] = []
    for line in lines:
        match = CFG_LINE_RE.match(line)
        if not match:
            out.append(line)
            continue
        key = match.group(2)
        if key in updates:
            out.append(f'{match.group(1)}{updates[key]}{match.group(4)}')
            seen.add(key)
        else:
            out.append(line)
    missing = sorted(set(updates.keys()) - seen)
    appended = [k for k in missing if k in appendable]
    rejected = [k for k in missing if k not in appendable]
    if rejected:
        raise KeyError(f"Keys not found in {cfg_path}: {rejected}")
    if appended:
        comments = dict(new_key_comments or {})
        out.append("")
        for key in appended:
            comment = comments.get(key)
            suffix = f' # {comment}' if comment else ""
            out.append(f'{key} = "{updates[key]}";{suffix}')
    cfg_path.write_text("\n".join(out) + "\n")
    return read_cfg_values(cfg_path)


def _ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> Path:
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    _ensure_parent(dst)
    shutil.copyfile(src, dst)
    return dst


# The receiver components every workshop forward run records, and the file each
# one writes. `mod.cfg` sets Hxrecord/Hzrecord true for both regardless of which
# source it fires, which is what makes the 2x2 tensor available from two runs.
RECEIVER_FIELDS = ("HX", "HZ")
FORWARD_RECORD_NAMES = {"EY": "Eyshot.rss", "HX": "Hxshot.rss", "HZ": "Hzshot.rss"}

# Source-field name -> the TE2D engine's numeric source_type (mod.cfg / inv.cfg
# "1=EY 3=HX 5=HZ"; see rockem-suite gotchas - the codes are per-engine and
# non-contiguous, so never hardcode the digit at a call site). This is the ONE
# map; it lives in the module with no heavy dependencies and `headless` imports
# it, not the other way round, which would drag scipy and segyio into every
# consumer of this one.
SOURCE_TYPE_CODES = {"EY": 1, "HX": 3, "HZ": 5}
SOURCE_TYPE_NAMES = {code: name for name, code in SOURCE_TYPE_CODES.items()}


def _source_type_code(source_field: str) -> int:
    """Numeric `source_type` for a field name."""
    name = str(source_field).upper()
    if name not in SOURCE_TYPE_CODES:
        raise ValueError(
            f"source_field must be one of {sorted(SOURCE_TYPE_CODES)}, got {source_field!r}"
        )
    return int(SOURCE_TYPE_CODES[name])


def source_field_from_mod_cfg(fdmodel_dir: Path) -> str:
    """The source component a forward run actually fired, from its own mod.cfg."""
    values = read_cfg_values(Path(fdmodel_dir) / "mod.cfg")
    raw = values.get("source_type")
    if raw is None:
        raise KeyError(f"No source_type in {Path(fdmodel_dir) / 'mod.cfg'}")
    code = int(str(raw).strip())
    if code not in SOURCE_TYPE_NAMES:
        raise ValueError(f"Unsupported source_type {code!r} in {fdmodel_dir}")
    return SOURCE_TYPE_NAMES[code]


def normalise_fdmodel_dirs(fdmodel_dir) -> Dict[str, Path]:
    """`{source_field: forward_dir}` from either a mapping or a single directory.

    A bare path keeps working and is keyed by the source that directory's own
    `mod.cfg` names, so the single-source path needs no caller changes.
    """
    if isinstance(fdmodel_dir, (str, Path)):
        one = Path(fdmodel_dir)
        return {source_field_from_mod_cfg(one): one}
    dirs = {str(k).upper(): Path(v) for k, v in dict(fdmodel_dir).items()}
    if not dirs:
        raise ValueError("At least one forward directory must be given.")
    for name in dirs:
        _source_type_code(name)
    return dirs


def staged_record_name(source_field: str, receiver_field: str, multi_source: bool) -> str:
    """Staged filename for one (source, receiver) record.

    One source keeps the historical `Hx_data.rss` / `Hz_data.rss`, so a
    single-source run stages exactly the files it always did. Several sources
    take the `<Src>_<Rec>` shape of the engine's own `Recordfile_<SRC>_<REC>` key.
    """
    rec = str(receiver_field).upper().title()
    if not multi_source:
        return f"{rec}_data.rss"
    return f"{str(source_field).upper().title()}_{rec}_data.rss"


def recordfile_cfg_key(source_field: str, receiver_field: str, multi_source: bool) -> str:
    """`Recordfile_<REC>` for one source, `Recordfile_<SRC>_<REC>` for several.

    Mirrors the engine's own resolution (src/mpiEminvTE2d.cpp:377-415): the pair
    key wins where present, and the per-receiver key is read ONLY when a single
    source type is named.
    """
    rec = str(receiver_field).upper()
    if not multi_source:
        return f"Recordfile_{rec}"
    return f"Recordfile_{str(source_field).upper()}_{rec}"


def source_fields_from_cfg(directory) -> List[str]:
    """The source components an `inv.cfg` inverts, e.g. `["HX", "HZ"]`.

    Reads `source_type`, which is a comma-separated list for a joint run. An
    empty list when there is no cfg to read, so callers can decide the default.
    """
    cfg = Path(directory) / "inv.cfg"
    if not cfg.exists():
        return []
    raw = str(read_cfg_values(cfg).get("source_type", ""))
    out: List[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) in SOURCE_TYPE_NAMES:
            name = SOURCE_TYPE_NAMES[int(tok)]
            if name not in out:
                out.append(name)
    return out


def find_observed_records(
    directory,
    source_field: Optional[str] = None,
    receiver_fields: Sequence[str] = RECEIVER_FIELDS,
) -> Dict[str, Path]:
    """`{receiver_field: path}` for the observed records staged in a directory.

    ONE reader for both naming forms. A single-source run stages
    `Hx_data.rss`/`Hz_data.rss`; a joint one stages `Hx_Hx_data.rss` ...
    `Hz_Hz_data.rss`, one per (source, receiver). Consumers that want an Hx/Hz
    pair - trace positions, observed-vs-synthetic gains - want ONE source's pair,
    so the source is resolved in this order: the `source_field` argument, then
    the first source named by the directory's own `inv.cfg`, then HX.

    Returns `{}` when no complete set is present, so callers can fall back.
    """
    d = Path(directory)
    recs = [str(r).upper() for r in receiver_fields]
    preferred: List[str] = []
    if source_field:
        preferred.append(str(source_field).upper())
    preferred += source_fields_from_cfg(d)
    preferred += ["HX", "HZ"]

    seen = set()
    for src in [x for x in preferred if not (x in seen or seen.add(x))]:
        for multi in (True, False):
            found = {rec: d / staged_record_name(src, rec, multi) for rec in recs}
            if all(p.exists() for p in found.values()):
                return found
    return {}


def available_synthetic_pairs(
    run_dir,
    source_field: Optional[str] = None,
    kind: str = "mod",
) -> List:
    """`(iteration, hx, hz)` for every modelled/residual pair in a run directory.

    ONE reader for every naming form the engine and this workshop have written,
    newest convention last so `pairs[-1]` prefers it:

      - `data_Hx_mod.rss-<n>` / `data_Hz_mod.rss-<n>`   (written by notebook 04)
      - `data_mod_HX.rss-<n>` / `data_mod_HZ.rss-<n>`   (engine, ONE source)
      - `data_mod_<SRC>_HX.rss-<n>` / `..._HZ.rss-<n>`  (engine, JOINT: the tag
        appears only when source_type names more than one source)

    A joint run writes four modelled files, one per (source, receiver). The pair
    returned is ONE source's - the Kx source by default, which is exactly what
    the untagged names meant on a Kx dataset - so an observed-vs-synthetic
    comparison stays like-for-like. `source_field` picks a different one.

    Iterations are the `-<n>` suffix; the two un-suffixed forms get -2 and -1 so
    they sort before any real iteration and are used only as a last resort.
    `kind` is "mod" or "res".
    """
    run_dir = Path(run_dir)
    pairs: List = []
    seen = set()

    def _add(idx, hx: Path, hz: Path) -> None:
        key = (int(idx), str(hx), str(hz))
        if key in seen or not hx.exists() or not hz.exists():
            return
        seen.add(key)
        pairs.append((int(idx), Path(hx), Path(hz)))

    src = str(source_field).upper() if source_field else "HX"
    # Ordered least- to most-preferred at equal iteration: the sort below is
    # stable, so the last one added wins for `pairs[-1]`.
    stems = [
        ("data_Hx_{k}.rss", "data_Hz_{k}.rss"),
        ("data_{k}_HX.rss", "data_{k}_HZ.rss"),
        ("data_{k}_" + src + "_HX.rss", "data_{k}_" + src + "_HZ.rss"),
    ]
    for plain_idx, (hx_stem, hz_stem) in zip((-2, -1, -1), stems):
        hx_name = hx_stem.format(k=kind)
        hz_name = hz_stem.format(k=kind)
        for hx in run_dir.glob(f"{hx_name}-*"):
            suffix = hx.name[len(hx_name) + 1:]
            if suffix.isdigit():
                _add(int(suffix), hx, run_dir / f"{hz_name}-{suffix}")
        _add(plain_idx, run_dir / hx_name, run_dir / hz_name)
    pairs.sort(key=lambda item: item[0])
    return pairs


def prepare_data_from_fdmodel(
    fdmodel_dir,
    output_dir: Path,
    receiver_fields: Sequence[str] = RECEIVER_FIELDS,
) -> Dict:
    """Stage the observed data for every (source, receiver) pair.

    `fdmodel_dir` is `{source_field: forward_dir}` - one forward run per source
    component, each of which recorded every receiver component. A single Path is
    accepted and behaves exactly as before.

    ONE `Wavelet` is staged and shared by every shot AND every source type. That
    is the intended physics - one magnetic moment amplitude - not a limitation,
    so the wavelets are checked to be identical rather than silently taking the
    first. The engine does not check this; it does check that the record files
    agree on geometry (src/mpiEminvTE2d.cpp:553-585), which they do here because
    `headless.build_forward_matrix` varies only `source_field` and shares the
    survey.

    Returns a dict keyed by `"Wavelet"` and by `(source_field, receiver_field)`
    tuples. With a single source it also carries the historical
    `"Recordfile_HX"` / `"Recordfile_HZ"` keys.
    """
    fdmodel_dirs = normalise_fdmodel_dirs(fdmodel_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = list(fdmodel_dirs)
    multi_source = len(sources) > 1
    recs = [str(r).upper() for r in receiver_fields]

    primary = fdmodel_dirs[sources[0]]
    wavelet = copy_file(primary / "wav2d.rss", output_dir / "wav2d.rss")
    for src in sources[1:]:
        other = fdmodel_dirs[src] / "wav2d.rss"
        if not other.exists():
            raise FileNotFoundError(f"Missing required file: {other}")
        if other.read_bytes() != wavelet.read_bytes():
            raise ValueError(
                f"Wavelet mismatch between forward runs {primary} and {fdmodel_dirs[src]}. "
                "A joint multi-source inversion shares ONE Wavelet across every source "
                "type, so the forward runs must have been driven by the same one."
            )

    targets: Dict = {"Wavelet": wavelet}
    for src in sources:
        for rec in recs:
            staged = copy_file(
                fdmodel_dirs[src] / "Data" / FORWARD_RECORD_NAMES[rec],
                output_dir / staged_record_name(src, rec, multi_source),
            )
            targets[(src, rec)] = staged
            if not multi_source:
                targets[f"Recordfile_{rec}"] = staged
    return targets


def copy_permittivity_model(fdmodel_dir: Path, output_dir: Path) -> Path:
    fdmodel_dir = Path(fdmodel_dir)
    output_dir = Path(output_dir)
    target = output_dir / "ep.rss"
    return copy_file(fdmodel_dir / "ep.rss", target)


def create_initial_sg0_model(
    fdmodel_dir: Path,
    output_dir: Path,
    mode: str = "uniform_resistivity",
    uniform_conductivity: Optional[float] = None,
    uniform_resistivity: Optional[float] = None,
    min_conductivity: float = 1e-8,
) -> Path:
    fdmodel_dir = Path(fdmodel_dir)
    output_dir = Path(output_dir)
    target = output_dir / "sg0.rss"
    ref_sg = fdmodel_dir / "sg.rss"
    if not ref_sg.exists():
        raise FileNotFoundError(f"Missing reference conductivity model: {ref_sg}")

    ref = rsfile()
    ref.read(str(ref_sg))
    arr = np.asarray(ref.data, dtype=np.float64)

    if mode == "uniform_conductivity":
        if uniform_conductivity is None:
            raise ValueError("uniform_conductivity must be provided.")
        sigma = float(uniform_conductivity)
    elif mode == "uniform_resistivity":
        if uniform_resistivity is None:
            raise ValueError("uniform_resistivity must be provided.")
        rho = float(uniform_resistivity)
        if rho <= 0.0:
            raise ValueError("uniform_resistivity must be > 0.")
        sigma = 1.0 / rho
    else:
        raise ValueError(
            f"Unsupported initial model mode: {mode}. "
            "Allowed modes: uniform_conductivity, uniform_resistivity."
        )

    sigma = max(float(min_conductivity), sigma)
    ref.data = np.asfortranarray(np.full(arr.shape, sigma, dtype=arr.dtype))
    _ensure_parent(target)
    ref.write(str(target))
    return target


def _write_uniform_rss_from_reference(
    ref_path: Path,
    target_path: Path,
    value: float,
) -> Path:
    ref_path = Path(ref_path)
    target_path = Path(target_path)
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing reference model file: {ref_path}")

    ref = rsfile()
    ref.read(str(ref_path))
    arr = np.asarray(ref.data, dtype=np.float64)
    ref.data = np.asfortranarray(np.full(arr.shape, float(value), dtype=arr.dtype))
    _ensure_parent(target_path)
    ref.write(str(target_path))
    return target_path


def create_bound_files(
    fdmodel_dir: Path,
    output_dir: Path,
    sg_min: float,
    sg_max: float,
    ep_min: Optional[float] = None,
    ep_max: Optional[float] = None,
) -> Dict[str, Path]:
    fdmodel_dir = Path(fdmodel_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sg_ref = output_dir / "sg0.rss"
    if not sg_ref.exists():
        sg_ref = fdmodel_dir / "sg.rss"
    if not sg_ref.exists():
        raise FileNotFoundError(
            f"Missing conductivity reference model for bound files: {sg_ref}"
        )

    paths = {
        "Lboundsg": _write_uniform_rss_from_reference(
            sg_ref, output_dir / "Sg_min.rss", sg_min
        ),
        "Uboundsg": _write_uniform_rss_from_reference(
            sg_ref, output_dir / "Sg_max.rss", sg_max
        ),
    }

    if ep_min is not None or ep_max is not None:
        if ep_min is None or ep_max is None:
            raise ValueError("Both ep_min and ep_max must be provided together.")
        ep_ref = output_dir / "ep.rss"
        if not ep_ref.exists():
            ep_ref = fdmodel_dir / "ep.rss"
        if not ep_ref.exists():
            raise FileNotFoundError(
                f"Missing permittivity reference model for bound files: {ep_ref}"
            )
        paths["Lboundep"] = _write_uniform_rss_from_reference(
            ep_ref, output_dir / "Ep_min.rss", ep_min
        )
        paths["Uboundep"] = _write_uniform_rss_from_reference(
            ep_ref, output_dir / "Ep_max.rss", ep_max
        )

    return paths


def create_weight_file_from_hx(hx_record_path: Path, weight_path: Path) -> Path:
    hx_record_path = Path(hx_record_path)
    weight_path = Path(weight_path)
    if not hx_record_path.exists():
        raise FileNotFoundError(f"Missing HX record file: {hx_record_path}")
    f = rsfile()
    f.read(str(hx_record_path))
    data = np.asarray(f.data, dtype=np.float64)
    nt, ntr = data.shape
    weight = np.tile(np.hanning(nt)[:, np.newaxis], (1, ntr))
    f.data = np.asfortranarray(weight.astype(data.dtype, copy=False))
    _ensure_parent(weight_path)
    f.write(str(weight_path))
    return weight_path


def write_inv_cfg(
    template_cfg: Path,
    output_cfg: Path,
    max_iterations: int,
    apertx: float,
    dtx: float,
    dtz: float,
    input_file_values: Dict[str, str],
    tik_sgregalpha: Optional[float] = None,
    geps: Optional[float] = None,
    allow_new: Optional[Iterable[str]] = None,
    new_key_comments: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    template_cfg = Path(template_cfg)
    output_cfg = Path(output_cfg)
    copy_file(template_cfg, output_cfg)
    updates = {
        "max_iterations": str(int(max_iterations)),
        "apertx": f"{float(apertx):.6f}",
        "dtx": f"{float(dtx):.6f}",
        "dtz": f"{float(dtz):.6f}",
    }
    if tik_sgregalpha is not None:
        updates["tik_sgregalpha"] = f"{float(tik_sgregalpha):.6g}"
    if geps is not None:
        updates["geps"] = f"{float(geps):.6g}"
    updates.update(input_file_values)
    return update_cfg_values(
        output_cfg, updates, allow_new=allow_new, new_key_comments=new_key_comments
    )


def apply_inversion_controls(
    cfg_path: Path,
    *,
    max_iterations: int,
    apertx: float,
    dtx: float,
    dtz: float,
    tik_sgregalpha: Optional[float] = None,
    geps: Optional[float] = None,
) -> Dict[str, str]:
    """Write the live inversion knobs into an already-staged `inv.cfg`.

    Staging copies `input/<freq>/inv.cfg` into `Run{N}/`. The GUI knobs that
    feed those keys must be applied HERE, at launch, because Generate Inputs
    can have been clicked with different values. Copying the staged file
    unchanged is what makes Max iter look like a dummy: the engine reads
    `Run{N}/inv.cfg`, not the widget.
    """
    updates = {
        "max_iterations": str(int(max_iterations)),
        "apertx": f"{float(apertx):.6f}",
        "dtx": f"{float(dtx):.6f}",
        "dtz": f"{float(dtz):.6f}",
    }
    if tik_sgregalpha is not None:
        updates["tik_sgregalpha"] = f"{float(tik_sgregalpha):.6g}"
    if geps is not None:
        updates["geps"] = f"{float(geps):.6g}"
    # `geps` is new in the template; a staged inv.cfg from before that line
    # existed still has to accept it, or Run dies with KeyError after Generate
    # already succeeded.
    return update_cfg_values(
        Path(cfg_path), updates, allow_new=["geps"] if geps is not None else None
    )


def prepare_inversion_inputs(
    fdmodel_dir,
    template_cfg: Path,
    output_dir: Path,
    max_iterations: int,
    apertx: float,
    dtx: float,
    dtz: float,
    initial_model_mode: str = "uniform_resistivity",
    uniform_conductivity: Optional[float] = None,
    uniform_resistivity: Optional[float] = None,
    tik_sgregalpha: Optional[float] = None,
    geps: Optional[float] = None,
    sg_min: float = 1e-8,
    sg_max: float = 1.0,
    ep_min: Optional[float] = None,
    ep_max: Optional[float] = None,
    constrain: bool = True,
    receiver_fields: Sequence[str] = RECEIVER_FIELDS,
) -> Dict[str, Path]:
    """Stage everything one `mpiEminvTE2d` run needs, from one or more forward runs.

    `fdmodel_dir` is either a single forward directory (unchanged behaviour) or
    `{source_field: forward_dir}`. With several sources the run is JOINT: the
    engine's work list spans (shot x source type) and every source stacks into
    the same gradient, so one model update sees all four tensor components.

    ONE shared `Dataweightfile` for every source, deliberately. `Dataweightfile_<SRC>`
    exists upstream for cases like differing acquisition noise between the source
    types - NOT for equalising component amplitudes. Cxz/Czx are weak on a layered
    model because a 1D earth has no lateral structure for them to sense; that small
    amplitude is correct information. Up-weighting them would amplify noise and
    modelling error in exactly the regime where they carry no signal.
    """
    fdmodel_dirs = normalise_fdmodel_dirs(fdmodel_dir)
    sources = list(fdmodel_dirs)
    multi_source = len(sources) > 1
    primary = fdmodel_dirs[sources[0]]
    recs = [str(r).upper() for r in receiver_fields]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_paths = prepare_data_from_fdmodel(
        fdmodel_dir=fdmodel_dirs, output_dir=output_dir, receiver_fields=recs
    )
    ep_path = copy_permittivity_model(fdmodel_dir=primary, output_dir=output_dir)
    if initial_model_mode not in {"uniform_conductivity", "uniform_resistivity"}:
        raise ValueError(
            "Initial model must be uniform. "
            "Use 'uniform_conductivity' or 'uniform_resistivity'."
        )

    sg0_path = create_initial_sg0_model(
        fdmodel_dir=primary,
        output_dir=output_dir,
        mode=initial_model_mode,
        uniform_conductivity=uniform_conductivity,
        uniform_resistivity=uniform_resistivity,
    )
    # ONE weight for all four components, built from the first source's Hx
    # record. Every record file shares a time axis and trace order (the engine
    # enforces it at startup), so which one it is read from does not matter.
    weight_path = create_weight_file_from_hx(
        hx_record_path=data_paths[(sources[0], recs[0])],
        weight_path=output_dir / "weight.rss",
    )
    bound_paths: Dict[str, Path] = {}
    if constrain:
        bound_paths = create_bound_files(
            fdmodel_dir=primary,
            output_dir=output_dir,
            sg_min=sg_min,
            sg_max=sg_max,
            ep_min=ep_min,
            ep_max=ep_max,
        )
    mod_cfg_values = read_cfg_values(primary / "mod.cfg")
    # Pin order/lpml/PML to the exact forward-run values - a mismatch here
    # (e.g. inv.cfg's own template lpml drifting from mod.cfg's)
    # would make the FWI's re-modelled wavefield subtly inconsistent with the
    # forward data it's fitting, independent of any real model update.
    #
    # source_type is the LIST of the source components being inverted, comma
    # separated, in the order the forward directories were given. It used to be
    # copied out of the one forward run's mod.cfg, because a run had exactly one
    # source; with several it cannot be copied, so the safety that pin provided
    # is asserted directly instead: each forward directory's own mod.cfg must
    # name the source it is filed under. Without that check a mis-filed dataset
    # would fit Kz data with a Kx source - a silent, physically wrong gradient
    # rather than an error.
    #
    # The discretisation keys are pinned from the FIRST directory and every
    # other one must agree, because all the sources share a single inv.cfg and
    # therefore a single grid, stencil and PML.
    shared_keys = ("pml_kmax", "pml_smax", "pml_amax", "order", "lpml")
    for src in sources:
        values = read_cfg_values(fdmodel_dirs[src] / "mod.cfg")
        actual = str(values.get("source_type", "")).strip()
        expected = str(_source_type_code(src))
        if actual != expected:
            raise ValueError(
                f"Forward directory {fdmodel_dirs[src]} is filed under source {src} "
                f"(source_type {expected}) but its mod.cfg says source_type = {actual!r}. "
                "Inverting it would fit one source's data with another source's field."
            )
        for key in shared_keys:
            if values.get(key) != mod_cfg_values.get(key):
                raise ValueError(
                    f"Forward run {fdmodel_dirs[src]} disagrees with {primary} on {key} "
                    f"({values.get(key)!r} vs {mod_cfg_values.get(key)!r}). A joint "
                    "inversion runs every source through one inv.cfg, so they must "
                    "share the grid, stencil and PML."
                )
    pml_updates = {}
    for key in shared_keys:
        value = mod_cfg_values.get(key)
        if value is not None:
            pml_updates[key] = value
    pml_updates["source_type"] = ",".join(str(_source_type_code(s)) for s in sources)

    # Observed-data keys. One source keeps `Recordfile_<REC>`, which is in the
    # template; several need `Recordfile_<SRC>_<REC>`, which is not, and is
    # appended (see update_cfg_values).
    record_updates: Dict[str, str] = {}
    record_comments: Dict[str, str] = {}
    for src in sources:
        for rec in recs:
            key = recordfile_cfg_key(src, rec, multi_source)
            record_updates[key] = data_paths[(src, rec)].name
            moment = {"HX": "Kx", "HZ": "Kz", "EY": "Ey"}[src]
            record_comments[key] = f"{rec} recorded from a {moment} source"
    allow_new = sorted(record_updates) if multi_source else None
    if multi_source:
        # Blank the template's single-source keys rather than leaving them
        # naming files a joint run does not stage. The engine ignores them
        # entirely here (it reads them only `if(single_source)`), so this is
        # for whoever reads the cfg, not for the engine.
        for rec in recs:
            record_updates.setdefault(f"Recordfile_{rec}", "")

    cfg_path = output_dir / "inv.cfg"
    input_file_values: Dict[str, str] = {
        "constrain": "true" if constrain else "false",
        "Sg": "sg0.rss",
        "Ep": "ep.rss",
        "Wavelet": "wav2d.rss",
        "Dataweightfile": "weight.rss",
        **record_updates,
        **pml_updates,
    }
    if constrain:
        input_file_values["Lboundsg"] = "Sg_min.rss"
        input_file_values["Uboundsg"] = "Sg_max.rss"
        if ep_min is not None and ep_max is not None:
            input_file_values["Lboundep"] = "Ep_min.rss"
            input_file_values["Uboundep"] = "Ep_max.rss"
    write_inv_cfg(
        template_cfg=template_cfg,
        output_cfg=cfg_path,
        max_iterations=max_iterations,
        apertx=apertx,
        dtx=dtx,
        dtz=dtz,
        tik_sgregalpha=tik_sgregalpha,
        geps=geps,
        input_file_values=input_file_values,
        allow_new=allow_new,
        new_key_comments=record_comments,
    )
    result: Dict[str, Path] = {
        "inv_cfg": cfg_path,
        "sg0": sg0_path,
        "ep": ep_path,
        "wavelet": data_paths["Wavelet"],
        # The first source's pair, so consumers that know only about one Hx/Hz
        # pair keep resolving. `records` is the full (source, receiver) map.
        "hx": data_paths[(sources[0], recs[0])],
        "hz": data_paths[(sources[0], recs[-1])],
        "weight": weight_path,
        "records": {pair: path for pair, path in data_paths.items() if isinstance(pair, tuple)},
        "source_fields": list(sources),
    }
    if bound_paths:
        result.update(bound_paths)
    return result


def stage_run_directory(
    input_dir: Path,
    run_dir: Path,
    clean: bool = True,
    include_patterns: Optional[Iterable[str]] = None,
) -> List[Path]:
    input_dir = Path(input_dir)
    run_dir = Path(run_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for child in run_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    staged: List[Path] = []
    patterns = list(include_patterns) if include_patterns else ["*"]
    picked = set()
    for pattern in patterns:
        for path in input_dir.glob(pattern):
            if path.is_file():
                picked.add(path)
    for src in sorted(picked):
        dst = run_dir / src.name
        copy_file(src, dst)
        staged.append(dst)
    return staged


__all__ = [
    "RECEIVER_FIELDS",
    "available_synthetic_pairs",
    "SOURCE_TYPE_CODES",
    "SOURCE_TYPE_NAMES",
    "create_bound_files",
    "create_initial_sg0_model",
    "create_weight_file_from_hx",
    "find_observed_records",
    "normalise_fdmodel_dirs",
    "prepare_data_from_fdmodel",
    "apply_inversion_controls",
    "prepare_inversion_inputs",
    "read_cfg_values",
    "recordfile_cfg_key",
    "source_field_from_mod_cfg",
    "source_fields_from_cfg",
    "staged_record_name",
    "stage_run_directory",
    "update_cfg_values",
    "write_inv_cfg",
]
