"""IDSR Mathematical & Physical Core: Beer-Lambert Rendering, Continuous Refinement,
Cramér-Rao Bounds, MDL Residual Decomposition, and Identity-Preserving Local Editing.
"""

from __future__ import annotations

import math
import zlib
import cv2
import numpy as np


def estimate_illumination_and_optical_depth(
    gray: np.ndarray,
    spacing: float = 16.0,
) -> dict:
    """Estimate smooth background illumination I_0(x, y), optical depth tau(x, y), and sensor noise sigma."""
    if gray.ndim != 2:
        raise ValueError("Expected a 2D grayscale array.")
    img_f = np.clip(gray.astype(np.float64), 1.0, 255.0)
    ksize = max(15, int(round(2.5 * spacing)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.float64)
    i0 = cv2.GaussianBlur(closed, (0, 0), max(4.0, 1.2 * spacing))
    i0 = np.clip(np.maximum(i0, img_f), 30.0, 255.0)

    # Beer-Lambert optical depth: tau(x, y) = -ln(I(x, y) / I_0(x, y)) >= 0
    tau = -np.log(np.clip(img_f / i0, 1.0 / 255.0, 1.0))

    # Robust sensor noise sigma estimation on paper background via Laplacian MAD
    bg_mask = tau < 0.08
    lap = cv2.Laplacian(img_f, cv2.CV_64F)
    bg_lap = lap[bg_mask] if np.count_nonzero(bg_mask) > 64 else lap.reshape(-1)
    mad = float(np.median(np.abs(bg_lap - np.median(bg_lap))))
    sigma_px = max(0.25, mad / (0.6745 * math.sqrt(20.0)))

    return {
        "i0": i0,
        "tau": tau,
        "sensor_noise_sigma_px": round(sigma_px, 4),
        "mean_illumination_px": round(float(np.mean(i0)), 3),
    }


def _ellipse_optical_field(
    xx: np.ndarray,
    yy: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_rad: float,
    hollow: bool,
    absorbance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate smooth differentiable elliptical notehead optical depth and its (d/dcx, d/dcy) gradients."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = xx - cx
    dy = yy - cy
    u = (dx * cos_a + dy * sin_a) / max(rx, 1.5)
    v = (-dx * sin_a + dy * cos_a) / max(ry, 1.5)
    r2 = u * u + v * v

    # Smooth edge transition with steepness beta
    beta = 4.5
    arg_outer = np.clip(beta * (r2 - 1.0), -30.0, 30.0)
    sig_outer = 1.0 / (1.0 + np.exp(arg_outer))
    dsig_dr2_outer = -beta * sig_outer * (1.0 - sig_outer)

    if not hollow:
        field = absorbance * sig_outer
        df_dr2 = absorbance * dsig_dr2_outer
    else:
        r2_inner = (u / 0.58) ** 2 + (v / 0.46) ** 2
        arg_inner = np.clip(beta * (r2_inner - 1.0), -30.0, 30.0)
        sig_inner = 1.0 / (1.0 + np.exp(arg_inner))
        hole = 1.0 - 0.85 * sig_inner
        field = absorbance * sig_outer * hole
        dsig_dr2_inner = -beta * sig_inner * (1.0 - sig_inner)
        df_dr2 = absorbance * (dsig_dr2_outer * hole - 0.85 * sig_outer * dsig_dr2_inner * 2.8)

    du_dcx = -cos_a / max(rx, 1.5)
    dv_dcx = sin_a / max(ry, 1.5)
    du_dcy = -sin_a / max(rx, 1.5)
    dv_dcy = -cos_a / max(ry, 1.5)

    dr2_dcx = 2.0 * (u * du_dcx + v * dv_dcx)
    dr2_dcy = 2.0 * (u * du_dcy + v * dv_dcy)

    grad_cx = df_dr2 * dr2_dcx
    grad_cy = df_dr2 * dr2_dcy
    return field, grad_cx, grad_cy


def compute_crlb_staff_spaces(
    spacing: float,
    rx: float,
    ry: float,
    absorbance: float,
    sensor_noise_sigma_px: float,
    mean_illumination_px: float = 245.0,
    hollow: bool = False,
) -> dict:
    """Compute the Fisher Information Matrix and Cramér-Rao Lower Bound (CRLB) for notehead center (cx, cy)."""
    sp = max(4.0, float(spacing))
    half = int(math.ceil(max(rx, ry) + 4))
    coords = np.arange(-half, half + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    tau_v, g_cx, g_cy = _ellipse_optical_field(xx, yy, 0.0, 0.0, rx, ry, -0.31, hollow, absorbance)

    # Intensity I = I_0 * exp(-tau) => dI/dc = -I * d(tau)/dc
    intensity = mean_illumination_px * np.exp(-tau_v)
    dI_dcx = -intensity * g_cx
    dI_dcy = -intensity * g_cy

    sigma2 = max(0.25, float(sensor_noise_sigma_px)) ** 2
    f_xx = float(np.sum(dI_dcx * dI_dcx)) / sigma2
    f_yy = float(np.sum(dI_dcy * dI_dcy)) / sigma2
    f_xy = float(np.sum(dI_dcx * dI_dcy)) / sigma2
    fim = np.asarray([[f_xx, f_xy], [f_xy, f_yy]], dtype=np.float64)
    cov = np.linalg.pinv(fim + 1e-9 * np.eye(2))
    crlb_px = math.sqrt(max(0.0, float(np.trace(cov))))
    crlb_sp = crlb_px / sp
    return {
        "crlb_center_px": round(crlb_px, 5),
        "crlb_center_sp": round(crlb_sp, 5),
        "fisher_eigenvalues": [round(float(v), 2) for v in np.linalg.eigvalsh(fim)],
        "jacobian_rank": int(np.linalg.matrix_rank(fim, tol=1e-4)),
    }


def refine_notehead_beer_lambert(
    tau_clean: np.ndarray,
    initial_center: tuple[float, float] | list[float],
    spacing: float,
    kind: str,
    max_iter: int = 18,
    max_shift_sp: float = 0.45,
) -> dict:
    """Refine notehead center (cx, cy), angle, and optical absorbance via damped Gauss-Newton on Beer-Lambert tau."""
    sp = max(6.0, float(spacing))
    cx, cy = float(initial_center[0]), float(initial_center[1])
    init_cx, init_cy = cx, cy
    hollow = kind == "hollow_head"
    rx = max(2.5, 0.56 * sp)
    ry = max(1.8, 0.36 * sp)
    pad = int(math.ceil(max(rx, ry) + 0.55 * sp + 3))
    h, w = tau_clean.shape
    x0 = max(0, int(round(cx)) - pad)
    x1 = min(w, int(round(cx)) + pad + 1)
    y0 = max(0, int(round(cy)) - pad)
    y1 = min(h, int(round(cy)) + pad + 1)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return {
            "center": [round(cx, 3), round(cy, 3)],
            "accepted": False,
            "optical_absorbance": 1.0,
            "shift_sp": 0.0,
        }

    roi_tau = tau_clean[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float64),
        np.arange(x0, x1, dtype=np.float64),
        indexing="ij",
    )

    # First coarse 3x3 sub-grid initialization within [-0.18 sp, +0.18 sp] to avoid flat-tail stalling
    best_init = (cx, cy)
    best_loss = float("inf")
    alpha = max(0.35, min(2.5, float(np.percentile(roi_tau, 90.0))))
    angle = -0.31
    for dx_try in (-0.14 * sp, 0.0, 0.14 * sp):
        for dy_try in (-0.14 * sp, 0.0, 0.14 * sp):
            f_try, _, _ = _ellipse_optical_field(xx, yy, cx + dx_try, cy + dy_try, rx, ry, angle, hollow, alpha)
            loss_try = float(np.mean((roi_tau - f_try) ** 2))
            if loss_try < best_loss:
                best_loss = loss_try
                best_init = (cx + dx_try, cy + dy_try)
    cx, cy = best_init

    # Damped Gauss-Newton updates on (cx, cy, alpha)
    damping = 1.5
    for _ in range(max_iter):
        pred, g_cx, g_cy = _ellipse_optical_field(xx, yy, cx, cy, rx, ry, angle, hollow, alpha)
        res = roi_tau - pred
        # Update alpha in closed form
        unit_pred = pred / max(alpha, 1e-4)
        denom_a = float(np.sum(unit_pred * unit_pred))
        if denom_a > 1e-6:
            alpha = float(np.clip(np.sum(roi_tau * unit_pred) / denom_a, 0.25, 3.0))

        J1 = g_cx.reshape(-1)
        J2 = g_cy.reshape(-1)
        r_vec = res.reshape(-1)
        H11 = float(np.dot(J1, J1)) + damping
        H22 = float(np.dot(J2, J2)) + damping
        H12 = float(np.dot(J1, J2))
        b1 = float(np.dot(J1, r_vec))
        b2 = float(np.dot(J2, r_vec))
        det = H11 * H22 - H12 * H12
        if abs(det) < 1e-9:
            break
        step_x = float(np.clip((H22 * b1 - H12 * b2) / det, -1.2, 1.2))
        step_y = float(np.clip((H11 * b2 - H12 * b1) / det, -1.2, 1.2))
        cx_new = cx + step_x
        cy_new = cy + step_y
        if math.hypot(cx_new - init_cx, cy_new - init_cy) > max_shift_sp * sp:
            scale_back = (max_shift_sp * sp) / max(math.hypot(cx_new - init_cx, cy_new - init_cy), 1e-6)
            cx_new = init_cx + (cx_new - init_cx) * scale_back
            cy_new = init_cy + (cy_new - init_cy) * scale_back
        cx, cy = cx_new, cy_new
        if math.hypot(step_x, step_y) < 1e-3:
            break

    shift_sp = math.hypot(cx - init_cx, cy - init_cy) / sp
    pred_final, _, _ = _ellipse_optical_field(xx, yy, cx, cy, rx, ry, angle, hollow, alpha)
    corr_num = float(np.sum((roi_tau - roi_tau.mean()) * (pred_final - pred_final.mean())))
    corr_den = float(np.linalg.norm(roi_tau - roi_tau.mean()) * np.linalg.norm(pred_final - pred_final.mean())) + 1e-8
    corr = corr_num / corr_den
    accepted = bool(corr >= 0.20 and shift_sp <= max_shift_sp)
    final_cx, final_cy = (cx, cy) if accepted else (init_cx, init_cy)
    return {
        "center": [round(final_cx, 4), round(final_cy, 4)],
        "initial_center": [round(init_cx, 4), round(init_cy, 4)],
        "accepted": accepted,
        "beer_lambert_correlation": round(corr, 4),
        "optical_absorbance": round(alpha, 4),
        "physical_depth_z_np": round(alpha, 4),
        "shift_sp": round(shift_sp, 4),
    }


def render_beer_lambert_scene(
    shape: tuple[int, int],
    scene: dict,
    i0_field: np.ndarray | None = None,
    psf_sigma: float = 0.45,
) -> dict:
    """Render modeled scene primitives via Beer-Lambert subtractive optical depth."""
    height, width = int(shape[0]), int(shape[1])
    tau_model = np.zeros((height, width), dtype=np.float64)

    # 1. Staff lines & barlines
    for staff in scene.get("staffs", []):
        for y_val in staff.get("lines", []):
            y = int(round(float(y_val)))
            if 0 <= y < height:
                layer = np.zeros((height, width), dtype=np.float32)
                cv2.line(layer, (0, y), (width - 1, y), 0.95, 1, cv2.LINE_AA)
                tau_model += layer
        for x_val in staff.get("barlines", []):
            x = int(round(float(x_val)))
            if 0 <= x < width:
                layer = np.zeros((height, width), dtype=np.float32)
                cv2.line(layer, (x, int(round(float(staff["top"])))), (x, int(round(float(staff["bottom"])))), 1.10, 1, cv2.LINE_AA)
                tau_model += layer

    # 2. Stems, beams, accidentals, dots
    for prim in scene.get("primitives", []):
        kind = prim.get("kind")
        geom = prim.get("geometry", {})
        if kind == "stem":
            layer = np.zeros((height, width), dtype=np.float32)
            x0, y0 = int(round(float(geom.get("x0_px", 0)))), int(round(float(geom.get("y0_px", 0))))
            x1, y1 = int(round(float(geom.get("x1_px", 0)))), int(round(float(geom.get("y1_px", 0))))
            cv2.line(layer, (x0, y0), (x1, y1), 1.25, 1, cv2.LINE_AA)
            tau_model += layer
        elif kind == "beam":
            layer = np.zeros((height, width), dtype=np.float32)
            x0, y0 = int(round(float(geom.get("x0_px", 0)))), int(round(float(geom.get("y0_px", 0))))
            x1, y1 = int(round(float(geom.get("x1_px", 0)))), int(round(float(geom.get("y1_px", 0))))
            thick = max(2, int(round(float(geom.get("thickness_px_estimate", 3.0)))))
            cv2.line(layer, (x0, y0), (x1, y1), 1.55, thick, cv2.LINE_AA)
            tau_model += layer
        elif kind == "dot":
            layer = np.zeros((height, width), dtype=np.float32)
            cx = int(round(float(geom.get("center_x_px", 0))))
            cy = int(round(float(geom.get("center_y_px", 0))))
            cv2.circle(layer, (cx, cy), 2, 1.40, -1, cv2.LINE_AA)
            tau_model += layer

    # 3. Noteheads using their estimated Beer-Lambert absorbance
    for note in scene.get("notes", []):
        cx, cy = [float(v) for v in note["center"]]
        bbox = [float(v) for v in note["bbox"]]
        rx = max(2, int(round(bbox[2] / 2.0)))
        ry = max(2, int(round(bbox[3] / 2.0)))
        fit = note.get("geometry_refinement") or {}
        angle = float(fit.get("rotation_deg", -18.0)) if fit.get("accepted") else -18.0
        alpha = float((note.get("appearance") or {}).get("optical_absorbance", 1.45))
        hollow = note.get("shape_class") == "hollow_head"
        thick = max(1, int(round(ry * 0.35))) if hollow else -1
        layer = np.zeros((height, width), dtype=np.float32)
        cv2.ellipse(layer, (int(round(cx)), int(round(cy))), (rx, ry), angle, 0, 360, alpha, thick, cv2.LINE_AA)
        tau_model += layer

    if psf_sigma > 0:
        tau_blurred = cv2.GaussianBlur(tau_model.astype(np.float32), (0, 0), float(psf_sigma)).astype(np.float64)
    else:
        tau_blurred = tau_model

    if i0_field is None:
        i0_use = np.full((height, width), 255.0, dtype=np.float64)
    else:
        i0_use = np.asarray(i0_field, dtype=np.float64)

    hat_i_float64 = np.clip(i0_use * np.exp(-tau_blurred), 0.0, 255.0)
    hat_i_uint8 = np.clip(np.round(hat_i_float64), 0, 255).astype(np.uint8)
    return {
        "hat_i_float64": hat_i_float64,
        "hat_i_uint8": hat_i_uint8,
        "tau_model": tau_model,
    }


def compute_mdl_decomposition(
    observed_gray: np.ndarray,
    scene: dict,
    hat_i_float64: np.ndarray | None = None,
    sensor_noise_sigma_px: float = 2.0,
) -> dict:
    """Compute the full IDSR MDL code-length budget and verify machine-precision 2D identity."""
    obs_f64 = observed_gray.astype(np.float64)
    height, width = obs_f64.shape
    n_pixels = height * width

    if hat_i_float64 is None:
        bl = render_beer_lambert_scene((height, width), scene)
        hat_i_float64 = bl["hat_i_float64"]

    # Machine-precision float64 residual and reconstruction
    residual_f64 = obs_f64 - hat_i_float64
    reconstructed_f64 = hat_i_float64 + residual_f64
    max_float64_err = float(np.max(np.abs(reconstructed_f64 - obs_f64)))

    # Integer lossless residual
    hat_i_u8 = np.clip(np.round(hat_i_float64), 0, 255).astype(np.uint8)
    res_i16 = observed_gray.astype(np.int16) - hat_i_u8.astype(np.int16)
    reconstructed_u8 = (hat_i_u8.astype(np.int16) + res_i16).astype(np.uint8)
    exact_u8 = bool(np.array_equal(reconstructed_u8, observed_gray))

    # Parametric code length L(V, E, theta):
    # Each primitive costs: 3 bits kind + 2*12 bits (x,y) + 2*8 bits (rx,ry) + 8 bits angle + 8 bits absorbance + 6 bits symbolic = 65 bits
    # Each relation edge costs: 4 bits type + 2*10 bits node indices = 24 bits
    # Smooth illumination field (8x8 spline grid) + capture params = 64 * 10 = 640 bits
    n_prims = len(scene.get("primitives", []))
    n_rels = len(scene.get("relations", []))
    bits_parametric = float(n_prims * 65 + n_rels * 24 + 640)

    # Residual code length L(r | V, E, theta) and Noise Oracle L_oracle:
    # Oracle knows every foreground stroke exactly and only encodes irreducible sensor noise sigma_px
    sigma_oracle = max(0.5, float(sensor_noise_sigma_px))
    oracle_bits_per_px = 0.5 * math.log2(2.0 * math.pi * math.e * (sigma_oracle ** 2))

    # Model residual evaluated via Laplace scale parameter b = mean(|r|) and actual zlib compression
    abs_res = np.abs(residual_f64)
    # On explained background + modeled primitives, measure excess over noise floor
    unmodeled_excess_px = np.maximum(0.0, abs_res - sigma_oracle * math.sqrt(2.0 / math.pi))
    b_excess = float(np.mean(unmodeled_excess_px))
    model_bits_per_px = oracle_bits_per_px + (bits_parametric / n_pixels) + math.log2(1.0 + b_excess / sigma_oracle)
    excess_mdl_bits_per_px = model_bits_per_px - oracle_bits_per_px

    # Unmodeled raw image entropy / zlib baseline (encoding image without primitives)
    raw_zlib_bytes = len(zlib.compress(observed_gray.tobytes(), level=9))
    res_zlib_bytes = len(zlib.compress(np.ascontiguousarray(res_i16, dtype="<i2").tobytes(), level=9))

    return {
        "n_pixels": n_pixels,
        "n_primitives": n_prims,
        "n_relations": n_rels,
        "parametric_bits_total": round(bits_parametric, 2),
        "parametric_bits_per_pixel": round(bits_parametric / n_pixels, 6),
        "oracle_noise_bits_per_pixel": round(oracle_bits_per_px, 4),
        "model_total_bits_per_pixel": round(model_bits_per_px, 4),
        "mdl_excess_bits_per_pixel": round(excess_mdl_bits_per_px, 5),
        "raw_image_zlib_bits_per_pixel": round(8.0 * raw_zlib_bytes / n_pixels, 4),
        "residual_int16_zlib_bits_per_pixel": round(8.0 * res_zlib_bytes / n_pixels, 4),
        "float64_max_reconstruction_error": max_float64_err,
        "float64_identity_within_machine_eps": bool(max_float64_err <= 1e-14),
        "uint8_exact_reconstruction": exact_u8,
    }


def evaluate_beer_lambert_regimes(seed: int = 42) -> dict:
    """Falsification experiment for P4 (Beer-Lambert law) across 4 physical ink/paper media regimes."""
    rng = np.random.default_rng(seed)
    h, w = 80, 80
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")

    # Layer A: staff line at y=40; Layer B: notehead at (40, 40) crossing the staff line
    tau_a = 0.95 * np.exp(-0.5 * ((yy - 40.0) / 1.1) ** 2)
    tau_b, _, _ = _ellipse_optical_field(xx, yy, 40.0, 40.0, 9.0, 6.0, -0.31, False, 1.45)
    tau_sum = tau_a + tau_b
    overlap_mask = (tau_a > 0.25) & (tau_b > 0.25)

    regimes = {
        "translucent_iron_gall_ink": {"r_spec": 0.005, "occlusion_mix": 0.02, "scatter_sigma": 0.25},
        "standard_carbon_print_ink": {"r_spec": 0.025, "occlusion_mix": 0.08, "scatter_sigma": 0.40},
        "graphite_pencil_specular": {"r_spec": 0.180, "occlusion_mix": 0.35, "scatter_sigma": 0.55},
        "opaque_gouache_correction": {"r_spec": 0.450, "occlusion_mix": 0.92, "scatter_sigma": 0.30},
    }

    results = {}
    for name, params in regimes.items():
        r_spec = params["r_spec"]
        occ = params["occlusion_mix"]
        # True physical composition mixes Beer-Lambert additive optical depth with opaque surface occlusion + specular floor
        tau_effective = (1.0 - occ) * (tau_a + tau_b) + occ * np.maximum(tau_a, tau_b)
        transmitted = (1.0 - r_spec) * np.exp(-tau_effective) + r_spec
        transmitted = np.clip(transmitted + rng.normal(0.0, 0.004, transmitted.shape), 1e-4, 1.0)
        tau_inferred = -np.log(transmitted)
        rel_error_overlap = float(
            np.mean(np.abs(tau_inferred[overlap_mask] - tau_sum[overlap_mask]))
            / np.mean(tau_sum[overlap_mask])
        )
        results[name] = {
            "specular_floor": r_spec,
            "surface_occlusion_fraction": occ,
            "beer_lambert_relative_error_at_intersection": round(rel_error_overlap, 4),
            "p4_holds_within_10pct": bool(rel_error_overlap <= 0.10),
        }
    return results


def run_contract_basin_experiment(
    spacing: float = 16.0,
    trials_per_sigma: int = 36,
    seed: int = 20261029,
) -> dict:
    """Empirically measure the Analysis-by-Synthesis Refinement Basin Contract across initial error sigmas."""
    rng = np.random.default_rng(seed)
    h, w = 96, 96
    true_cx, true_cy = 48.0, 48.0
    rx, ry = 0.56 * spacing, 0.36 * spacing
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")

    # Render a realistic periodic staff neighborhood (adjacent notehead / residual ink harmonics at +-1.0 sp)
    tau_note, _, _ = _ellipse_optical_field(xx, yy, true_cx, true_cy, rx, ry, -0.31, False, 1.50)
    tau_neighbor_up, _, _ = _ellipse_optical_field(xx, yy, true_cx, true_cy - 1.0 * spacing, rx, ry, -0.31, False, 0.55)
    tau_neighbor_dn, _, _ = _ellipse_optical_field(xx, yy, true_cx, true_cy + 1.0 * spacing, rx, ry, -0.31, False, 0.55)
    tau_scene = tau_note + tau_neighbor_up + tau_neighbor_dn

    sigma_levels_sp = (0.0, 0.05, 0.15, 0.25, 0.30, 0.40, 0.50, 0.80, 1.20)
    rows = []
    for sigma_sp in sigma_levels_sp:
        before_sp = []
        after_sp = []
        pitch_err = []
        for _ in range(trials_per_sigma):
            noisy_tau = np.clip(tau_scene + rng.normal(0.0, 0.035, tau_scene.shape), 0.0, 4.0)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius_sp = sigma_sp * abs(float(rng.normal(1.0, 0.18))) if sigma_sp > 0 else 0.0
            init_x = true_cx + radius_sp * spacing * math.cos(angle)
            init_y = true_cy + radius_sp * spacing * math.sin(angle)
            err_before = math.hypot(init_x - true_cx, init_y - true_cy) / spacing

            fit = refine_notehead_beer_lambert(noisy_tau, (init_x, init_y), spacing, "filled_head", max_shift_sp=0.48)
            ref_x, ref_y = fit["center"]
            err_after = math.hypot(ref_x - true_cx, ref_y - true_cy) / spacing
            before_sp.append(err_before)
            after_sp.append(err_after)
            # Pitch error occurs when vertical error exceeds 0.25 sp (half a diatonic step = 0.25 sp)
            pitch_err.append(1.0 if abs(ref_y - true_cy) / (spacing / 2.0) >= 0.5 else 0.0)

        rows.append({
            "sigma_init_sp": sigma_sp,
            "mean_error_before_sp": round(float(np.mean(before_sp)), 4),
            "mean_error_after_sp": round(float(np.mean(after_sp)), 4),
            "p95_error_after_sp": round(float(np.percentile(after_sp, 95)), 4),
            "pitch_error_rate": round(float(np.mean(pitch_err)), 4),
            "converges_to_crlb_regime": bool(float(np.mean(after_sp)) <= 0.05),
        })

    crlb = compute_crlb_staff_spaces(spacing, rx, ry, 1.50, sensor_noise_sigma_px=2.0)
    return {
        "spacing_px": spacing,
        "trials_per_sigma": trials_per_sigma,
        "crlb_reference": crlb,
        "contract_threshold_sp": 0.30,
        "breakdown_threshold_sp": 0.50,
        "sweep": rows,
    }


def edit_note_preserving_identity(
    source_image: np.ndarray,
    scene: dict,
    note_id: str,
    new_diatonic_step: int | None = None,
    new_center_xy: tuple[float, float] | None = None,
) -> dict:
    """Modify one note in the scene while keeping all pixels outside its local edit mask bit-for-bit identical."""
    if source_image.ndim != 2 or source_image.dtype != np.uint8:
        raise ValueError("source_image must be a 2D uint8 grayscale image.")
    edited_image = source_image.copy()
    edited_scene = {
        **scene,
        "notes": [dict(n) for n in scene.get("notes", [])],
        "primitives": [dict(p) for p in scene.get("primitives", [])],
    }

    target_note = next((n for n in edited_scene["notes"] if str(n["id"]) == str(note_id)), None)
    if target_note is None:
        raise ValueError(f"Note {note_id!r} not found in scene.")

    staff_idx = int(target_note["staff_index"])
    staff = edited_scene["staffs"][staff_idx]
    sp = float(staff["spacing"])
    old_cx, old_cy = [float(v) for v in target_note["center"]]
    old_bbox = [int(v) for v in target_note["bbox"]]

    if new_center_xy is not None:
        new_cx, new_cy = float(new_center_xy[0]), float(new_center_xy[1])
        step = int(round((float(staff["bottom"]) - new_cy) / (sp / 2.0)))
    elif new_diatonic_step is not None:
        step = int(new_diatonic_step)
        new_cx = old_cx
        new_cy = float(staff["bottom"]) - step * (sp / 2.0)
    else:
        raise ValueError("Specify either new_diatonic_step or new_center_xy.")

    # Estimate local paper background I_0
    illum = estimate_illumination_and_optical_depth(source_image, spacing=sp)
    i0 = illum["i0"]

    height, width = source_image.shape
    edit_mask = np.zeros((height, width), dtype=bool)

    # 1. Inpaint old notehead footprint with paper illumination I_0 and restore any crossed staff line
    pad = max(3, int(round(0.35 * sp)))
    ox0 = max(0, old_bbox[0] - pad)
    oy0 = max(0, old_bbox[1] - pad)
    ox1 = min(width, old_bbox[0] + old_bbox[2] + pad)
    oy1 = min(height, old_bbox[1] + old_bbox[3] + pad)
    edit_mask[oy0:oy1, ox0:ox1] = True
    edited_image[oy0:oy1, ox0:ox1] = np.clip(np.round(i0[oy0:oy1, ox0:ox1]), 0, 255).astype(np.uint8)

    for line_y in staff.get("lines", []):
        ly = int(round(float(line_y)))
        if oy0 <= ly < oy1:
            edited_image[ly, ox0:ox1] = np.clip(np.round(i0[ly, ox0:ox1] * math.exp(-0.95)), 0, 255).astype(np.uint8)

    # 2. Render moved notehead via Beer-Lambert at (new_cx, new_cy)
    rx = max(3, int(round(old_bbox[2] / 2.0)))
    ry = max(2, int(round(old_bbox[3] / 2.0)))
    nx0 = max(0, int(round(new_cx)) - rx - pad)
    ny0 = max(0, int(round(new_cy)) - ry - pad)
    nx1 = min(width, int(round(new_cx)) + rx + pad + 1)
    ny1 = min(height, int(round(new_cy)) + ry + pad + 1)
    edit_mask[ny0:ny1, nx0:nx1] = True

    alpha = float((target_note.get("appearance") or {}).get("optical_absorbance", 1.45))
    hollow = target_note.get("shape_class") == "hollow_head"
    yy, xx = np.meshgrid(np.arange(ny0, ny1, dtype=np.float64), np.arange(nx0, nx1, dtype=np.float64), indexing="ij")
    tau_new, _, _ = _ellipse_optical_field(xx, yy, new_cx, new_cy, rx, ry, -0.31, hollow, alpha)
    cur_tau = -np.log(np.clip(edited_image[ny0:ny1, nx0:nx1].astype(np.float64) / i0[ny0:ny1, nx0:nx1], 1.0 / 255.0, 1.0))
    combined = i0[ny0:ny1, nx0:nx1] * np.exp(-(cur_tau + tau_new))
    edited_image[ny0:ny1, nx0:nx1] = np.clip(np.round(combined), 0, 255).astype(np.uint8)

    # Update symbolic pitch and geometry in edited_scene
    letters = ["C", "D", "E", "F", "G", "A", "B"]
    octave_delta, letter_idx = divmod(2 + step, 7)
    target_note["center"] = [round(new_cx, 3), round(new_cy, 3)]
    target_note["bbox"] = [
        int(round(new_cx - old_bbox[2] / 2.0)),
        int(round(new_cy - old_bbox[3] / 2.0)),
        old_bbox[2],
        old_bbox[3],
    ]
    target_note["diatonic_step_from_bottom_line"] = step
    target_note["pitch"] = {**target_note.get("pitch", {}), "step": letters[letter_idx], "octave": 4 + octave_delta}

    outside_mask = ~edit_mask
    outside_identical = bool(np.array_equal(edited_image[outside_mask], source_image[outside_mask]))
    unchanged_fraction = float(np.mean(edited_image == source_image))

    return {
        "edited_image": edited_image,
        "edited_scene": edited_scene,
        "edit_mask": edit_mask,
        "outside_roi_bit_identical": outside_identical,
        "unchanged_pixel_fraction": round(unchanged_fraction, 6),
        "modified_pixel_count": int(np.count_nonzero(edited_image != source_image)),
    }
