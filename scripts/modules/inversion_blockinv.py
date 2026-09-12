"""BlockInv 1D layered inversion and DE-start hybrid for the workshop.

The Gauss-Newton block solver glue lives here, not in notebook 05's code cell,
so experiments and the GUI import the same implementation.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from scripts.modules.analytic_1d_forward import ForwardRejected
from scripts.modules.inversion_1d import (
    DEFAULT_COMPONENTS,
    TENSOR_COMPONENTS,
    component_weights,
    forward_tensor_for_tx,
    n_tensor_data,
    resolve_tensor_calibration,
    unpack_model_params,
)
from scripts.modules.inversion_tuning import run_de_once
from third_party.empy_blockinv import (
    InversionConfig,
    pack_model,
    run_block1d_inversion,
    split_model,
    uncertainty_from_result,
)


def normalize_thickness_to_span(thickness, span_target, thk_min=1e-6):
    thk = np.asarray(thickness, dtype=float).reshape(-1)
    if thk.size == 0:
        return thk
    thk = np.clip(thk, float(thk_min), np.inf)
    total = float(np.sum(thk))
    if total <= 0.0:
        return np.full_like(thk, float(span_target) / float(thk.size))
    return thk * (float(span_target) / total)


def params_from_rho_thk(rho, thickness, cfg):
    rho = np.asarray(rho, dtype=float)
    thk = np.asarray(thickness, dtype=float)
    span = float(cfg["z_end_rel"] - cfg["z_start_rel"])
    thk_norm = normalize_thickness_to_span(thk, span_target=span, thk_min=cfg["thk_min"])
    lrho = np.log10(np.clip(rho, 10.0 ** cfg["log10_rho_min"], 10.0 ** cfg["log10_rho_max"]))
    lthk = np.log10(np.clip(thk_norm, 10.0 ** cfg["log10_thk_min"], 10.0 ** cfg["log10_thk_max"]))
    if lrho.size == 0:
        return lthk
    return np.concatenate([lrho, lthk]).astype(float)


def pack_complex(*arrays):
    """Interleave complex blocks into blockinv's real data vector."""
    out = []
    for a in arrays:
        a = np.asarray(a, dtype=complex)
        out.append(np.ravel(np.real(a)))
        out.append(np.ravel(np.imag(a)))
    return np.concatenate(out).astype(float)


def apply_tensor_calibration(pred, cal, components):
    """`C * pred` per component, using that component's SOURCE calibration."""
    out = {}
    for comp in components:
        C = cal["C"].get(TENSOR_COMPONENTS[comp][0])
        scale = np.asarray(C, dtype=complex)[:, None] if C is not None else 1.0
        out[comp] = scale * pred[comp]
    return out


def blockinv_forward(model, ctx):
    n_layers = int(ctx["n_layers"])
    tx_entry = ctx["tx_entry"]
    cfg = ctx["cfg"]
    components = tuple(ctx["components"])
    thk, rho = split_model(np.asarray(model, dtype=float), n_layers)
    params = params_from_rho_thk(rho, thk, cfg)
    try:
        pred = forward_tensor_for_tx(
            params, tx_entry, n_layers, cfg["z_start_rel"], cfg["z_end_rel"], cfg["eps_r"],
            components=components,
            snap_dz=cfg.get("snap_dz"), snap_origin_m=cfg.get("snap_origin_m"),
            n_nodes=int(cfg.get("n_nodes", 120)),
        )
    except ForwardRejected:
        nfreq, nrx = tx_entry["freqs"].size, tx_entry["off_x"].size
        pred = {c: np.full((nfreq, nrx), 1e6, dtype=complex) for c in components}

    cal = ctx["calibration"]
    pred_cal = apply_tensor_calibration(pred, cal, components)
    pred_vec = pack_complex(*(pred_cal[c] for c in components))

    n_reg = int(ctx.get("n_reg", 0))
    reg_scale = float(ctx.get("reg_scale", 0.0))
    if n_reg > 0 and reg_scale > 0.0:
        lrho = np.asarray(params[:n_layers], dtype=float)
        reg_pred = reg_scale * np.diff(lrho)
        pred_vec = np.concatenate([pred_vec, np.asarray(reg_pred, dtype=float)])

    return pred_vec.astype(float)


def blockinv_residual(observed, predicted, error, ctx):
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    err = np.asarray(error, dtype=float)
    res = (pred - obs) / np.clip(err, 1e-12, np.inf)

    i = 0
    for comp, n in zip(ctx.get("components", ()), ctx.get("n_per_component", ())):
        w = np.sqrt(max(float(ctx.get("weights", {}).get(comp, 1.0)), 0.0))
        if i + 2 * int(n) > res.size:
            break
        res[i:i + int(n)] *= w
        res[i + int(n):i + 2 * int(n)] *= w
        i += 2 * int(n)

    return res.astype(float)


def _resolve_calibration(cfg, tx_entry, setup_meta_path):
    cal = cfg.get("tensor_calibration")
    if cal is not None:
        return cal
    if setup_meta_path is None:
        setup_meta_path = cfg.get("setup_meta_path")
    if setup_meta_path is None:
        raise ValueError(
            "BlockInv needs a calibration: set cfg['tensor_calibration'] or pass "
            "setup_meta_path."
        )
    return resolve_tensor_calibration(cfg, setup_meta_path, tx_entry=tx_entry)


def _build_start_model(cfg, n_layers):
    span = float(cfg["z_end_rel"] - cfg["z_start_rel"])
    start_params = cfg.get("start_params")
    if start_params is not None:
        rho, thk, _ = unpack_model_params(
            np.asarray(start_params, dtype=float),
            n_layers,
            cfg["z_start_rel"],
            cfg["z_end_rel"],
        )
        return pack_model(thk, rho)

    rng = np.random.default_rng(int(cfg["seed"]))
    rho_lo_l = float(cfg["log10_rho_min"])
    rho_hi_l = float(cfg["log10_rho_max"])
    start_rho = 10.0 ** rng.uniform(rho_lo_l, rho_hi_l, size=n_layers)

    thk_lo_l = float(cfg["log10_thk_min"])
    thk_hi_l = float(cfg["log10_thk_max"])
    start_thk = 10.0 ** rng.uniform(thk_lo_l, thk_hi_l, size=max(n_layers - 1, 0))
    if start_thk.size > 0:
        start_thk = normalize_thickness_to_span(
            start_thk, span_target=span, thk_min=cfg["thk_min"]
        )
    return pack_model(start_thk, start_rho)


def invert_single_tx_blockinv(tx_entry, cfg, *, setup_meta_path=None):
    """Run BlockInv for one transmitter; optional DE start via cfg['start_params']."""
    n_layers = int(cfg["n_layers"])
    components = tuple(cfg.get("components") or DEFAULT_COMPONENTS)
    cal = _resolve_calibration(cfg, tx_entry, setup_meta_path)
    weights = component_weights(cfg)

    obs = {c: np.asarray(tx_entry["obs"][c], dtype=complex) for c in components}
    obs_vec_data = pack_complex(*(obs[c] for c in components))

    err_blocks = []
    for c in components:
        sig_arr = np.asarray(cal["sigma"][c], dtype=float)
        if sig_arr.ndim == 1:
            sig = np.broadcast_to(sig_arr[:, None], obs[c].shape)
        else:
            sig = np.broadcast_to(sig_arr, obs[c].shape)
        err_blocks += [np.ravel(sig), np.ravel(sig)]
    err_vec_data = np.clip(np.concatenate(err_blocks).astype(float), 1e-300, np.inf)

    n_reg = max(n_layers - 1, 0)
    reg_scale = np.sqrt(max(float(cfg.get("reg_lambda", 0.0)), 0.0) / float(n_reg)) if n_reg > 0 else 0.0
    if n_reg > 0 and reg_scale > 0.0:
        obs_vec = np.concatenate([obs_vec_data, np.zeros(n_reg, dtype=float)])
        err_vec = np.concatenate([err_vec_data, np.ones(n_reg, dtype=float)])
    else:
        obs_vec = obs_vec_data
        err_vec = err_vec_data

    span = float(cfg["z_end_rel"] - cfg["z_start_rel"])
    start_model = _build_start_model(cfg, n_layers)

    callback_context = {
        "n_layers": n_layers,
        "tx_entry": tx_entry,
        "cfg": cfg,
        "calibration": cal,
        "components": components,
        "n_per_component": [int(np.size(obs[c])) for c in components],
        "weights": weights,
        "n_reg": int(n_reg if reg_scale > 0.0 else 0),
        "reg_scale": float(reg_scale),
    }

    inv_cfg = InversionConfig(
        n_layers=n_layers,
        start_model=start_model,
        max_iter=int(cfg.get("block_max_iter", 15)),
        min_dphi_percent=1.0,
        stop_at_chi1=False,
        callback_context=callback_context,
        lower_bound=1e-8,
    )

    result = run_block1d_inversion(
        observed=obs_vec,
        error=err_vec,
        config=inv_cfg,
        forward_fn=blockinv_forward,
        misfit_fn=blockinv_residual,
    )

    thk_opt, rho_opt = split_model(np.asarray(result.model, dtype=float), n_layers)
    thk_opt = normalize_thickness_to_span(thk_opt, span_target=span, thk_min=cfg["thk_min"])
    params_best = params_from_rho_thk(rho_opt, thk_opt, cfg)
    nfreq, nrx = tx_entry["freqs"].size, tx_entry["off_x"].size
    try:
        pred_best = forward_tensor_for_tx(
            params_best, tx_entry, n_layers, cfg["z_start_rel"], cfg["z_end_rel"],
            cfg["eps_r"], components=components,
            snap_dz=cfg.get("snap_dz"), snap_origin_m=cfg.get("snap_origin_m"),
            n_nodes=int(cfg.get("n_nodes", 120)),
        )
    except ForwardRejected:
        pred_best = {c: np.full((nfreq, nrx), np.nan, dtype=complex) for c in components}
    rho, thk, depth_rel = unpack_model_params(
        params_best, n_layers, cfg["z_start_rel"], cfg["z_end_rel"]
    )

    sigma_rho = np.full_like(rho, np.nan, dtype=float)
    sigma_thk = np.full_like(thk, np.nan, dtype=float)
    sigma_depth = np.full_like(depth_rel, np.nan, dtype=float)
    try:
        sigma_abs, _ = uncertainty_from_result(result)
        sigma_abs = np.asarray(sigma_abs, dtype=float)
        n_thk = max(n_layers - 1, 0)
        sigma_thk_raw = sigma_abs[:n_thk] if n_thk else np.array([], dtype=float)
        sigma_rho_raw = sigma_abs[n_thk:]
        if n_thk:
            scale = float(span) / max(
                float(np.sum(np.asarray(result.model[:n_thk], dtype=float))), 1e-12
            )
            sigma_thk = np.asarray(sigma_thk_raw, dtype=float) * scale
            sigma_depth = np.sqrt(np.cumsum(np.maximum(sigma_thk, 0.0) ** 2))
        sigma_rho = np.asarray(sigma_rho_raw[: rho.size], dtype=float)
    except Exception:
        pass

    tx_z = float(tx_entry.get("tx_z", 0.0))
    z_top = tx_z + float(cfg["z_start_rel"])
    z_bottom = tx_z + float(cfg["z_end_rel"])
    depth_abs = tx_z + np.asarray(depth_rel, dtype=float)

    residual = np.asarray(result.residual, dtype=float)
    block_obj = float(np.dot(residual, residual)) if residual.size else np.nan
    _n_reg_rows = int(callback_context.get("n_reg", 0))
    _data_rows = residual[: residual.size - _n_reg_rows] if _n_reg_rows else residual
    data_misfit = float(np.dot(_data_rows, _data_rows)) if _data_rows.size else np.nan
    chi2 = data_misfit / max(n_tensor_data(tx_entry, components, weights=weights), 1)

    nan_block = np.full((nfreq, nrx), np.nan, dtype=complex)
    return {
        "success": True,
        "message": "",
        "misfit": block_obj,
        "data_misfit": float(data_misfit),
        "chi2": float(chi2),
        "params": params_best,
        "rho": rho,
        "thickness": thk,
        "depth": depth_abs,
        "z_top": float(z_top),
        "z_bottom": float(z_bottom),
        "pred": pred_best,
        "pred_hxh": pred_best.get("Cxx", nan_block),
        "pred_hxhz": pred_best.get("Cxz", nan_block),
        "rho_unc": sigma_rho,
        "thickness_unc": sigma_thk,
        "depth_unc": sigma_depth,
    }


def _model_fields_from_params(params, tx_entry, cfg, components):
    n_layers = int(cfg["n_layers"])
    nfreq, nrx = tx_entry["freqs"].size, tx_entry["off_x"].size
    rho, thk, depth_rel = unpack_model_params(
        params, n_layers, cfg["z_start_rel"], cfg["z_end_rel"]
    )
    tx_z = float(tx_entry.get("tx_z", 0.0))
    try:
        pred = forward_tensor_for_tx(
            params, tx_entry, n_layers, cfg["z_start_rel"], cfg["z_end_rel"],
            cfg["eps_r"], components=components,
            snap_dz=cfg.get("snap_dz"), snap_origin_m=cfg.get("snap_origin_m"),
            n_nodes=int(cfg.get("n_nodes", 120)),
        )
    except ForwardRejected:
        pred = {c: np.full((nfreq, nrx), np.nan, dtype=complex) for c in components}
    nan_block = np.full((nfreq, nrx), np.nan, dtype=complex)
    return {
        "params": np.asarray(params, dtype=float),
        "rho": rho,
        "thickness": thk,
        "depth": tx_z + np.asarray(depth_rel, dtype=float),
        "z_top": tx_z + float(cfg["z_start_rel"]),
        "z_bottom": tx_z + float(cfg["z_end_rel"]),
        "pred": pred,
        "pred_hxh": pred.get("Cxx", nan_block),
        "pred_hxhz": pred.get("Cxz", nan_block),
    }


def linearised_uncertainty_at_params(tx_entry, cfg, params, *, setup_meta_path=None):
    """Jacobian covariance at ``params`` without a full BlockInv polish."""
    unc_cfg = dict(cfg, start_params=np.asarray(params, dtype=float), block_max_iter=1)
    out = invert_single_tx_blockinv(tx_entry, unc_cfg, setup_meta_path=setup_meta_path)
    return {
        "rho_unc": out["rho_unc"],
        "thickness_unc": out["thickness_unc"],
        "depth_unc": out["depth_unc"],
    }


def polish_de_with_blockinv(
    tx_entry,
    cfg,
    de_params,
    *,
    de_chi2,
    de_data_misfit=None,
    setup_meta_path=None,
):
    """BlockInv polish from an existing DE model; keep DE if polish worsens χ²."""
    components = tuple(cfg.get("components") or DEFAULT_COMPONENTS)
    de_params = np.asarray(de_params, dtype=float).reshape(-1)
    de_chi2 = float(de_chi2)
    ndata = max(n_tensor_data(tx_entry, components), 1)
    if de_data_misfit is None:
        de_data_misfit = de_chi2 * ndata

    blk_cfg = dict(cfg, start_params=de_params)
    blk_out = invert_single_tx_blockinv(tx_entry, blk_cfg, setup_meta_path=setup_meta_path)
    blk_chi2 = float(blk_out["chi2"])

    if de_chi2 <= blk_chi2:
        unc_cfg = dict(blk_cfg, block_max_iter=1)
        unc_out = invert_single_tx_blockinv(tx_entry, unc_cfg, setup_meta_path=setup_meta_path)
        return {
            **blk_out,
            **_model_fields_from_params(de_params, tx_entry, cfg, components),
            "chi2": de_chi2,
            "data_misfit": float(de_data_misfit),
            "rho_unc": unc_out["rho_unc"],
            "thickness_unc": unc_out["thickness_unc"],
            "depth_unc": unc_out["depth_unc"],
            "polish_reverted": True,
            "de_chi2_data": de_chi2,
            "blockinv_chi2": blk_chi2,
        }

    return {
        **blk_out,
        "polish_reverted": False,
        "de_chi2_data": de_chi2,
        "blockinv_chi2": blk_chi2,
    }


def invert_de_then_blockinv(tx_entry, cfg, *, setup_meta_path=None):
    """Cheap DE start, BlockInv polish, and Jacobian-based parameter uncertainties."""
    seed = int(cfg.get("seed", 42))
    de_out = run_de_once(cfg, tx_entry, seed)
    out = polish_de_with_blockinv(
        tx_entry,
        cfg,
        de_out["params"],
        de_chi2=float(de_out["chi2_data"]),
        de_data_misfit=float(de_out["data_misfit"]),
        setup_meta_path=setup_meta_path,
    )
    out["de_params"] = de_out["params"]
    out["de_total"] = float(de_out["total"])
    out["de_nfev"] = int(de_out["nfev"])
    out["de_nit"] = int(de_out["nit"])
    return out


# Notebook 05 and blockinv_tensor_test.py use the underscore names.
_normalize_thickness_to_span = normalize_thickness_to_span
_params_from_rho_thk = params_from_rho_thk
_pack_complex = pack_complex
_apply_tensor_calibration = apply_tensor_calibration
_blockinv_forward = blockinv_forward
_blockinv_residual = blockinv_residual
_invert_single_tx_blockinv = invert_single_tx_blockinv


__all__ = [
    "normalize_thickness_to_span",
    "params_from_rho_thk",
    "pack_complex",
    "apply_tensor_calibration",
    "blockinv_forward",
    "blockinv_residual",
    "invert_single_tx_blockinv",
    "linearised_uncertainty_at_params",
    "polish_de_with_blockinv",
    "invert_de_then_blockinv",
    "_normalize_thickness_to_span",
    "_params_from_rho_thk",
    "_pack_complex",
    "_apply_tensor_calibration",
    "_blockinv_forward",
    "_blockinv_residual",
    "_invert_single_tx_blockinv",
]
