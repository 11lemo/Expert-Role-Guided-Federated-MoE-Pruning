"""Generate Figure 4 from realized per-expert sparsity reports."""

import argparse
import json
from pathlib import Path

ROLE_NAMES = {
    "shared": "Shared",
    "adaptable_active": "Task-active",
    "task-active": "Task-active",
    "task_active": "Task-active",
    "adaptable_inactive": "Task-inactive",
    "task-inactive": "Task-inactive",
    "task_inactive": "Task-inactive",
    "redundant": "Redundant",
}
ROLE_ORDER = ["Shared", "Task-active", "Task-inactive", "Redundant"]


def find_reports(inputs):
    files = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    files = list(dict.fromkeys(path.resolve() for path in files))
    if not files:
        raise ValueError("No JSON sparsity reports were found.")
    return files


def get_ratio(record, allocated):
    if allocated:
        if "allocated_retained_ratio" in record:
            return float(record["allocated_retained_ratio"])
        return 1.0 - float(record["allocated_sparsity"])
    if "retained_ratio" in record:
        return float(record["retained_ratio"])
    if "realized_sparsity" in record:
        return 1.0 - float(record["realized_sparsity"])
    raise KeyError(
        "Missing realized sparsity. Re-run pruning or use "
        "--use-allocated-ratio."
    )


def aggregate(files, allocated=False):
    totals = {
        role: {"weighted": 0.0, "params": 0, "experts": 0}
        for role in ROLE_ORDER
    }
    domains = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        records = report.get("experts")
        if not isinstance(records, list):
            raise ValueError(f"Not a sparsity report: {path}")
        domains.append(str(report.get("domain", path.stem)))
        for record in records:
            raw_role = str(record.get("role", "")).lower()
            if raw_role not in ROLE_NAMES:
                raise ValueError(f"Unknown role {raw_role!r} in {path}")
            role = ROLE_NAMES[raw_role]
            weight = int(
                record.get("selected_params", record.get("n_params", 0))
            )
            ratio = get_ratio(record, allocated)
            if weight <= 0 or not 0.0 <= ratio <= 1.0:
                raise ValueError(f"Invalid expert record in {path}: {record}")
            totals[role]["weighted"] += ratio * weight
            totals[role]["params"] += weight
            totals[role]["experts"] += 1

    summary = {}
    for role, stats in totals.items():
        if stats["params"]:
            summary[role] = {
                "retained_ratio": stats["weighted"] / stats["params"],
                "parameter_count": stats["params"],
                "expert_count": stats["experts"],
            }
    return summary, domains


def draw(summary, png_path, pdf_path="", show=False):
    import matplotlib.pyplot as plt

    roles = [role for role in ROLE_ORDER if role in summary]
    values = [summary[role]["retained_ratio"] for role in roles]
    colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#E45756"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(6.8, 4.2))
    bars = axis.bar(
        roles,
        values,
        width=0.62,
        color=colors[: len(roles)],
        edgecolor="black",
        linewidth=0.6,
    )
    axis.set_title("Role-wise Retained Weight Ratio")
    axis.set_ylabel("Parameter-weighted retained ratio")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(value + 0.025, 0.975),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()

    png = Path(png_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=600, bbox_inches="tight")
    if pdf_path:
        pdf = Path(pdf_path)
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot parameter-weighted retained ratios by expert role."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument(
        "--output", default="fig_rolewise_retained_ratio.png"
    )
    parser.add_argument(
        "--pdf-output", default="fig_rolewise_retained_ratio.pdf"
    )
    parser.add_argument(
        "--summary-output",
        default="fig_rolewise_retained_ratio_summary.json",
    )
    parser.add_argument("--use-allocated-ratio", action="store_true")
    parser.add_argument("--allow-missing-roles", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    files = find_reports(args.input)
    summary, domains = aggregate(files, args.use_allocated_ratio)
    missing = [role for role in ROLE_ORDER if role not in summary]
    if missing and not args.allow_missing_roles:
        raise ValueError("No records for roles: " + ", ".join(missing))

    payload = {
        "definition": (
            "parameter-weighted mean of 1 - sparsity over all "
            "client-expert pairs assigned to each role"
        ),
        "value_source": (
            "normalized_and_clipped_allocation"
            if args.use_allocated_ratio
            else "post_pruning_realized_zero_ratio"
        ),
        "source_files": [str(path) for path in files],
        "domains": domains,
        "roles": summary,
    }
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    draw(summary, args.output, args.pdf_output, args.show)
    for role in ROLE_ORDER:
        if role in summary:
            stats = summary[role]
            print(
                f"{role}: retained_ratio={stats['retained_ratio']:.6f}, "
                f"params={stats['parameter_count']}, "
                f"experts={stats['expert_count']}"
            )
    print(f"Saved PNG: {Path(args.output).resolve()}")
    if args.pdf_output:
        print(f"Saved PDF: {Path(args.pdf_output).resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()