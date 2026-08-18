import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


METHODS = ("bm3d", "nlmeans", "srad", "proposed")
METRICS = ("SSI", "SMPI", "NSF", "ENL", "CNR", "EPF", "EPI", "OMQDI")
CLASSES = ("cnv", "dme", "drusen", "normal")


def finite(values: list[float]) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    return data[np.isfinite(data)]


def confidence_interval(values: np.ndarray, seed: int, resamples: int) -> list[float]:
    if len(values) < 2 or np.all(values == values[0]):
        mean = float(np.mean(values))
        return [mean, mean]
    result = stats.bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=resamples,
        method="BCa",
        random_state=np.random.default_rng(seed),
    )
    return [float(result.confidence_interval.low), float(result.confidence_interval.high)]


def summarize(values: list[float], seed: int, resamples: int) -> dict:
    data = finite(values)
    return {
        "n": int(len(data)),
        "mean": float(np.mean(data)),
        "sample_sd": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        "median": float(np.median(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean_ci95_bca": confidence_interval(data, seed, resamples),
    }


def signed_rank_details(proposed: np.ndarray, baseline: np.ndarray) -> dict:
    differences = proposed - baseline
    differences = differences[np.isfinite(differences)]
    nonzero = differences[differences != 0]
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    w_plus = float(np.sum(ranks[nonzero > 0]))
    w_minus = float(np.sum(ranks[nonzero < 0]))
    denominator = w_plus + w_minus
    rank_biserial = (w_plus - w_minus) / denominator if denominator else 0.0
    result = stats.wilcoxon(
        nonzero,
        alternative="two-sided",
        zero_method="wilcox",
        method="exact",
    )
    return {
        "n_pairs": int(len(differences)),
        "n_nonzero": int(len(nonzero)),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "W_two_sided": float(result.statistic),
        "p_unadjusted": float(result.pvalue),
        "rank_biserial_proposed_minus_baseline": float(rank_biserial),
        "mean_difference_proposed_minus_baseline": float(np.mean(differences)),
        "median_difference_proposed_minus_baseline": float(np.median(differences)),
    }


def holm_adjust(records: list[dict]) -> None:
    order = sorted(range(len(records)), key=lambda i: records[i]["p_unadjusted"])
    running = 0.0
    count = len(records)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * records[index]["p_unadjusted"])
        running = max(running, adjusted)
        records[index]["p_holm"] = running


def analyze(raw: dict, seed: int, resamples: int) -> dict:
    image_names = sorted(raw)
    method_summary = {}
    class_summary = {}
    for method in METHODS:
        method_summary[method] = {}
        for metric in METRICS:
            values = [raw[name][method][metric] for name in image_names]
            method_summary[method][metric] = summarize(values, seed, resamples)
    for category in CLASSES:
        selected = [name for name in image_names if name.startswith(category)]
        class_summary[category] = {}
        for method in METHODS:
            class_summary[category][method] = {
                metric: summarize(
                    [raw[name][method][metric] for name in selected], seed, resamples
                )
                for metric in METRICS
            }
    tests = []
    for metric in METRICS:
        proposed = np.asarray([raw[name]["proposed"][metric] for name in image_names])
        for baseline in METHODS[:-1]:
            comparison = signed_rank_details(
                proposed,
                np.asarray([raw[name][baseline][metric] for name in image_names]),
            )
            comparison.update({"metric": metric, "baseline": baseline})
            tests.append(comparison)
    holm_adjust(tests)
    return {
        "schema_version": 1,
        "input_images": image_names,
        "method_summary": method_summary,
        "pathology_summary": class_summary,
        "paired_wilcoxon": tests,
        "settings": {
            "sample_sd_ddof": 1,
            "bootstrap_seed": seed,
            "bootstrap_resamples": resamples,
            "bootstrap_method": "BCa",
            "wilcoxon_alternative": "two-sided",
            "wilcoxon_zero_method": "wilcox",
            "wilcoxon_method": "exact",
            "multiple_testing": "Holm across 24 comparisons",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(raw, args.seed, args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
