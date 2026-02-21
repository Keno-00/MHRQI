"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Magnitude Hierarchical Representation of Quantum Images            ║
║  Classical Denoiser Comparison: BM3D, NL-Means, SRAD                        ║
║                                                                              ║
║  Author: Keno-00                                                             ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import datetime
import os

import cv2
import numpy as np
import srad
from bm3d import BM3DProfile, BM3DStages, bm3d

import plots

# -----------------------------
# Image domain utilities
# -----------------------------

def to_float01(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
    return np.clip(img, 0.0, 1.0)


def to_uint8(img):
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def extract_roi(img, roi):
    y, x, h, w = roi
    H, W = img.shape
    if y < 0 or x < 0 or y + h > H or x + w > W:
        raise ValueError(f"ROI {roi} out of bounds for image {img.shape}")
    return img[y:y+h, x:x+w]


# -----------------------------
# Denoisers (float in, float out)
# -----------------------------

def denoise_bm3d(img, sigma=0.05, stage=BM3DStages.ALL_STAGES):
    out = bm3d(img, sigma_psd=sigma, stage_arg=stage, profile=BM3DProfile())
    return np.clip(out, 0.0, 1.0)


def denoise_nlmeans(img, h=10, template=7, search=21):
    u8 = to_uint8(img)
    out = cv2.fastNlMeansDenoising(u8, None, h, template, search)
    return out.astype(np.float32) / 255.0


def denoise_srad(img, iters=400, dt=0.65, decay=0.8):
    srad_in = (img * 255.0).astype(np.float32) + 1e-5
    out = srad.SRAD(srad_in, iters, dt, decay)
    return np.clip(out / 255.0, 0.0, 1.0)


# -----------------------------
# Automated ROI selection
# -----------------------------

def auto_homogeneous_roi(img, win=20, stride=10):
    H, W = img.shape
    best_cov = np.inf
    best_roi = None

    eps = 1e-6

    for y in range(0, H - win, stride):
        for x in range(0, W - win, stride):
            patch = img[y:y+win, x:x+win]
            mu = patch.mean()
            if mu < eps or mu > 0.95:  # Skip dark or saturated regions
                continue
            
            sigma = patch.std()
            cov = sigma / mu
            
            # Center-bias: Prefer regions closer to the image center
            dist_to_center = np.sqrt((y + win/2 - H/2)**2 + (x + win/2 - W/2)**2)
            dist_norm = dist_to_center / np.sqrt((H/2)**2 + (W/2)**2)
            
            # Combine COV and center bias (cost = COV + 0.1 * normalized_distance)
            cost = cov + 0.1 * dist_norm
            
            if cost < best_cov:
                best_cov = cost
                best_roi = (y, x, win, win)

    if best_roi is None:
        raise RuntimeError("Failed to find homogeneous ROI")

    return best_roi





# -----------------------------
# Metric wrapper
# -----------------------------

def compute_metrics(ref, test, roi_bg):
    return {
        "SSI": plots.compute_ssi(ref, test, roi_bg),
        "BRISQUE": plots.compute_brisque(test),
        "NIQE": plots.compute_niqe(test),
        "PIQE": plots.compute_piqe(test),
        "SMPI": plots.compute_smpi(ref, test)
    }


# -----------------------------
# Main comparison
# -----------------------------

def compare_to(image_input, proposed_img=None, methods="all",
               plot=True, save=True, save_prefix="denoised", save_dir=None,
               reference_image=None):

    img = to_float01(image_input)

    if reference_image is not None:
        ref_img = to_float01(reference_image)
    else:
        ref_img = img # Default to input if no reference (noisy ref)

    if save and save_dir is None:
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        save_dir = os.path.join("evals", date_str)

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if proposed_img is not None:
        proposed = to_float01(proposed_img)
        if proposed.shape != img.shape:
            raise ValueError("Proposed image shape mismatch")
    else:
        proposed = None

    denoisers = {
        "bm3d": denoise_bm3d,
        "nlmeans": denoise_nlmeans,
        "srad": denoise_srad
    }

    if methods == "all":
        methods_list = list(denoisers.keys())
    elif isinstance(methods, str):
        methods_list = [methods]
    else:
        methods_list = methods

    # Alias for clarity in expanded logic
    noisy_img = img
    # If reference_image was provided, ref_img is likely the Clean Ref.
    # If not, ref_img was set to img (Noisy).
    # We need to know if we really have a clean ref.
    has_clean_ref = (reference_image is not None)
    clean_img = ref_img if has_clean_ref else None

    # Automated ROI from Noisy Image (for SSI)
    try:
        roi = auto_homogeneous_roi(noisy_img)
    except RuntimeError:
        # Fallback if no homogeneous region found
        roi = (0, 0, noisy_img.shape[0], noisy_img.shape[1])

    # Prepare list of methods to process
    methods_to_run = [("Original", None)] # Include original noisy image
    for m_name in methods_list:
        if m_name in denoisers:
            methods_to_run.append((m_name, denoisers[m_name]))
        else:
            print(f"Warning: Unknown denoiser method '{m_name}' skipped.")

    if proposed is not None:
        methods_to_run.append(("proposed", None))

    # -----------------------------
    # Compute Metrics using correct references
    # -----------------------------

    results = []

    for method_name, func in methods_to_run:
        # print(f"  ... processing {method_name}")

        if method_name == "Original":
            res_img = noisy_img
        elif method_name == "proposed" and proposed is not None:
            res_img = proposed
        else:
            res_img = func(noisy_img)

        # -- Compute Metrics --
        m = {}

        # 1. No Reference Metrics (computed but not reported)
        m["NIQE"] = plots.compute_niqe(res_img)

        # 2. Speckle Metrics (vs NOISY Input)
        m["SSI"] = plots.compute_ssi(noisy_img, res_img, roi)
        m["SMPI"] = plots.compute_smpi(noisy_img, res_img)
        m["ENL"] = plots.compute_enl(res_img, roi)  # Higher is better
        cnr_result = plots.compute_cnr(res_img)
        m["CNR"] = cnr_result[0]  # Higher is better
        
        # 3. Structural Metrics
        omqdi_result = plots.compute_omqdi(noisy_img, res_img)
        m["OMQDI"] = omqdi_result[0]  # Combined metric
        m["EPF"] = omqdi_result[1]    # Edge-Preservation Factor (wavelet)
        m["NSF"] = omqdi_result[2]    # Noise-Suppression Factor
        m["EPI"] = plots.compute_epi(noisy_img, res_img)  # Edge Preservation Index (Sobel)

        # 3. Full Reference Metrics (vs CLEAN Reference)
        if has_clean_ref:
            m["FSIM"] = plots.compute_fsim(clean_img, res_img)
            m["SSIM"] = plots.compute_ssim(clean_img, res_img)
        else:
            m["FSIM"] = float('nan')
            m["SSIM"] = float('nan')

        # Store result image for plotting
        res_u8 = to_uint8(res_img)
        if save:
             cv2.imwrite(os.path.join(save_dir, f"{save_prefix}_{method_name}.png"), res_u8)

        results.append({
            "name": method_name,
            "metrics": m,
            "image": res_u8
        })

    # -----------------------------
    # Display Tables (Separated)
    # -----------------------------

    # Filter out "Original" from display if requested?
    # User: "remove original from here... only participants would show in those comparisons"
    # -----------------------------
    display_results = [r for r in results if r["name"] != "Original"]

    # -----------------------------
    # Generate Reports
    # -----------------------------

    # 1. Full Ref Report (vs Clean)
    if has_clean_ref:
        plots.MetricsPlotter.print_summary_text(display_results, ["FSIM", "SSIM"], "Full Reference Metrics (vs Clean Ref)")
        plots.MetricsPlotter.save_summary_report(
            to_uint8(clean_img),
            display_results,  # Only participants, not Original
            ["FSIM", "SSIM"],
            "Full Reference Metrics (vs Clean Ref)",
            "report_full_ref",
            save_dir=save_dir,
            include_original_in_table=False
        )

    # Filter out Original for ranking (unfair baseline comparison)
    denoiser_results = [r for r in results if r["name"] != "Original"]

    # 2. Speckle Reduction Report (includes ENL and CNR)
    plots.MetricsPlotter.print_summary_text(denoiser_results, ["SSI", "SMPI", "NSF", "ENL", "CNR"], "Speckle Reduction Metrics")
    plots.MetricsPlotter.save_summary_report(
        to_uint8(noisy_img),
        denoiser_results,
        ["SSI", "SMPI", "NSF", "ENL", "CNR"],
        "Speckle Reduction Metrics",
        "report_speckle",
        save_dir=save_dir,
        include_original_in_table=False
    )

    # 3. Structural Similarity Report (EPF wavelet + EPI Sobel)
    plots.MetricsPlotter.print_summary_text(denoiser_results, ["EPF", "EPI", "OMQDI"], "Structural Similarity Metrics")
    plots.MetricsPlotter.save_summary_report(
        to_uint8(noisy_img),
        denoiser_results,
        ["EPF", "EPI", "OMQDI"],
        "Structural Similarity Metrics",
        "report_structural",
        save_dir=save_dir,
        include_original_in_table=False
    )

    # Note: Naturalness metrics (NIQE) computed but not reported (biased for medical images)

    if save:
        print(f"Comparison reports saved to {save_dir}")

    return results


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    n = 256
    img_path = os.path.join("resources", "drusen1.jpeg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (n, n))

    proposed = cv2.imread("testme.png", cv2.IMREAD_GRAYSCALE)
    if proposed is not None:
        proposed = cv2.resize(proposed, (n, n))

    compare_to(
        img,
        proposed_img=proposed,
        methods="all",
        plot=True,
        save=True,
        save_prefix="denoised"
    )
