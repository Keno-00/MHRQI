"""
Patch cnv3 in RECENT benchmark data.

Steps:
1. Run the full MHRQI benchmark pipeline on cnv3.jpeg only (n=256).
2. Overwrite the cnv3 entry in RECENT/raw_results.json with fresh results.
3. Regenerate statistical tables and charts in RECENT/.
"""

import json
import os

import cv2
import numpy as np

import compare_to
import main as mhrqi_main
import statistical_benchmark as sb

# ── Config ────────────────────────────────────────────────────────────────────
IMG_PATH   = "resources/cnv3.jpeg"
RECENT_DIR = "RECENT"
N          = 256
STRENGTH   = 1.65
# ─────────────────────────────────────────────────────────────────────────────

RAW_JSON = os.path.join(RECENT_DIR, "raw_results.json")


def run_cnv3(n, strength):
    """Run the MHRQI pipeline + comparisons for cnv3, return metrics dict."""
    img_name = os.path.basename(IMG_PATH).replace(".jpeg", "")
    img_dir  = os.path.join(RECENT_DIR, img_name)
    os.makedirs(img_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {img_name}  (n={n})")
    print(f"Output:     {img_dir}")
    print(f"{'='*60}")

    orig, recon, run_dir = mhrqi_main.main(
        shots=1000,
        n=n,
        d=2,
        denoise=True,
        use_shots=False,
        fast=True,
        verbose_plots=False,
        img_path=IMG_PATH,
        run_comparison=False,
    )

    cv2.imwrite(os.path.join(img_dir, "original.png"), orig)

    noisy_img = compare_to.to_float01(orig)
    comparison_results = compare_to.compare_to(
        noisy_img,
        proposed_img=compare_to.to_float01(recon),
        methods="all",
        plot=True,
        save=True,
        save_prefix="denoised",
        save_dir=img_dir,
        reference_image=None,
    )

    img_results = {r["name"]: r["metrics"] for r in comparison_results}
    return img_name, img_results


def patch_raw_results(img_name, new_img_results):
    """Load RECENT raw_results.json, replace img_name entry, save."""
    with open(RAW_JSON) as f:
        all_results = json.load(f)

    # Overwrite the stale cnv3 entry
    serializable = {}
    for method, metrics in new_img_results.items():
        serializable[method] = {
            k: float(v) if not np.isnan(v) else None
            for k, v in metrics.items()
        }
    all_results[img_name] = serializable

    with open(RAW_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nPatched {img_name} in {RAW_JSON}")
    return all_results


def regenerate_metrics(all_results):
    """Re-run stats + visualisations and save into RECENT/."""
    print("\nRegenerating statistical tables and charts in RECENT/ ...")

    # Convert JSON nulls back to NaN for numpy operations
    results_np = {}
    for img, methods in all_results.items():
        results_np[img] = {}
        for method, metrics in methods.items():
            results_np[img][method] = {
                k: float(v) if v is not None else float("nan")
                for k, v in metrics.items()
            }

    summary, stat_results = sb.create_results_table(results_np, RECENT_DIR)
    sb.create_visualization(results_np, RECENT_DIR)
    sb.create_summary_heatmap(stat_results, RECENT_DIR)

    print(f"\n{'='*60}")
    print(f"Done. All outputs updated in: {RECENT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    img_name, new_img_results = run_cnv3(N, STRENGTH)
    all_results = patch_raw_results(img_name, new_img_results)
    regenerate_metrics(all_results)
