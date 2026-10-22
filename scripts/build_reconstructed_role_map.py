"""Reconstruct a role map from valid-token top-k routing hit counts.

This is a new implementation of the equations in the manuscript, not the
unrecovered program used for the reported experiments. It cannot recover an
original role map from downstream scores. Compare its output against the
retained original JSON before using it for a reproduction claim.

The input is deliberately a compact, auditable statistics file rather than
raw client text. See ``--help`` and README.md for the input schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "role-router-counts-v1"
OUTPUT_SCHEMA = "role-map-reconstruction-v1"


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return result


def _layer_ids(layers: dict[str, Any], label: str) -> list[str]:
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"{label} must be a nonempty object")
    for key in layers:
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError(f"{label} layer IDs must be canonical nonnegative integer strings")
    return sorted(layers, key=int)


def validate_counts(payload: Any) -> tuple[list[str], list[str]]:
    """Validate the collector schema and the paper's top-k sum invariant."""
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"input schema must be {SCHEMA!r}")
    if not isinstance(payload.get("model"), str) or not payload["model"].strip():
        raise ValueError("model must be a nonempty string")
    specs = payload.get("layer_specs")
    ids = _layer_ids(specs, "layer_specs")
    for layer_id in ids:
        spec = specs[layer_id]
        if not isinstance(spec, dict):
            raise ValueError(f"layer_specs[{layer_id}] must be an object")
        num_experts = _positive_int(spec.get("num_experts"), f"layer {layer_id} num_experts")
        top_k = _positive_int(spec.get("top_k"), f"layer {layer_id} top_k")
        if top_k > num_experts:
            raise ValueError(f"layer {layer_id} top_k exceeds num_experts")

    clients = payload.get("clients")
    if not isinstance(clients, list) or not clients:
        raise ValueError("clients must be a nonempty list")
    client_ids: set[str] = set()
    domains: set[str] = set()
    for index, client in enumerate(clients):
        label = f"clients[{index}]"
        if not isinstance(client, dict):
            raise ValueError(f"{label} must be an object")
        client_id = client.get("client_id")
        domain = client.get("domain")
        if not isinstance(client_id, str) or not client_id.strip():
            raise ValueError(f"{label}.client_id must be a nonempty string")
        if client_id in client_ids:
            raise ValueError(f"duplicate client_id {client_id!r}")
        client_ids.add(client_id)
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError(f"{label}.domain must be a nonempty string")
        domains.add(domain)
        n_valid = _positive_int(client.get("n_valid_tokens"), f"{label}.n_valid_tokens")
        layers = client.get("layers")
        if not isinstance(layers, dict) or set(layers) != set(ids):
            raise ValueError(f"{label}.layers must cover exactly {ids}")
        for layer_id in ids:
            counts = layers[layer_id]
            spec = specs[layer_id]
            if not isinstance(counts, list) or len(counts) != spec["num_experts"]:
                raise ValueError(f"{label}.layers[{layer_id}] has wrong expert count")
            if any(
                isinstance(hit, bool) or not isinstance(hit, int) or hit < 0 or hit > n_valid
                for hit in counts
            ):
                raise ValueError(f"{label}.layers[{layer_id}] has an invalid hit count")
            expected = n_valid * spec["top_k"]
            if sum(counts) != expected:
                raise ValueError(
                    f"{label}.layers[{layer_id}] hits sum to {sum(counts)}, "
                    f"expected n_valid_tokens * top_k = {expected}"
                )
    if len(domains) < 2:
        raise ValueError("normalized domain entropy requires at least two distinct domains")
    return ids, sorted(domains)


def linear_quantile(values: list[float], fraction: float) -> float:
    """Linearly interpolate the sorted values at (N-1)*fraction."""
    if not values:
        raise ValueError("quantile needs at least one value")
    fraction = _finite_probability(fraction, "quantile")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _domain_frequencies(
    payload: dict[str, Any], layer_id: str, domains: list[str]
) -> dict[str, list[float]]:
    num_experts = payload["layer_specs"][layer_id]["num_experts"]
    hits = {domain: [0] * num_experts for domain in domains}
    tokens = {domain: 0 for domain in domains}
    for client in payload["clients"]:
        domain = client["domain"]
        tokens[domain] += client["n_valid_tokens"]
        for expert, count in enumerate(client["layers"][layer_id]):
            hits[domain][expert] += count
    return {
        domain: [count / tokens[domain] for count in hits[domain]]
        for domain in domains
    }


def build_role_map(
    payload: dict[str, Any], *, mode: str, tau_h: float, epsilon: float = 1e-12,
    tau_red: float | None = None, tau_g: float | None = None,
) -> dict[str, Any]:
    """Build the existing pruner's global role prior from validated counts.

    ``mode='qwen-quantile'`` uses manuscript quantiles 0.25 and 0.75.
    ``mode='explicit'`` requires both numeric importance thresholds. This is
    necessary for OLMoE because its original redundant cutoff is unavailable.
    """
    ids, domains = validate_counts(payload)
    tau_h = _finite_probability(tau_h, "tau_h")
    if not isinstance(epsilon, (float, int)) or isinstance(epsilon, bool):
        raise ValueError("epsilon must be a positive finite number")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number")
    if mode == "qwen-quantile":
        if tau_red is not None or tau_g is not None:
            raise ValueError("do not pass numeric importance thresholds in qwen-quantile mode")
    elif mode == "explicit":
        if tau_red is None or tau_g is None:
            raise ValueError("explicit mode requires both tau_red and tau_g")
        tau_red = _finite_probability(tau_red, "tau_red")
        tau_g = _finite_probability(tau_g, "tau_g")
        if tau_red > tau_g:
            raise ValueError("tau_red must not exceed tau_g")
    else:
        raise ValueError("mode must be 'qwen-quantile' or 'explicit'")

    layers_out: dict[str, dict[str, dict[str, Any]]] = {}
    thresholds_out: dict[str, dict[str, float]] = {}
    for layer_id in ids:
        specs = payload["layer_specs"][layer_id]
        num_experts = specs["num_experts"]
        active_threshold = specs["top_k"] / num_experts
        domain_freq = _domain_frequencies(payload, layer_id, domains)
        importance = [
            math.fsum(domain_freq[d][e] for d in domains) / len(domains)
            for e in range(num_experts)
        ]
        layer_tau_red = linear_quantile(importance, 0.25) if mode == "qwen-quantile" else tau_red
        layer_tau_g = linear_quantile(importance, 0.75) if mode == "qwen-quantile" else tau_g
        assert layer_tau_red is not None and layer_tau_g is not None
        thresholds_out[layer_id] = {
            "tau_red": layer_tau_red,
            "tau_g": layer_tau_g,
            "tau_a": active_threshold,
            "tau_h": tau_h,
        }

        layer_roles: dict[str, dict[str, Any]] = {}
        for expert_id, global_importance in enumerate(importance):
            values = [domain_freq[d][expert_id] for d in domains]
            denominator = math.fsum(values) + len(domains) * epsilon
            probabilities = [(value + epsilon) / denominator for value in values]
            entropy = -math.fsum(p * math.log(p) for p in probabilities) / math.log(len(domains))
            active_domains = [
                domain for domain in domains
                if domain_freq[domain][expert_id] >= active_threshold
            ]
            if global_importance <= layer_tau_red:
                role = "redundant"
                # Existing get_expert_role() checks domain membership before
                # role == redundant. A redundant expert must have no active domains.
                active_domains = []
            elif global_importance >= layer_tau_g and entropy >= tau_h:
                role = "shared"
            else:
                role = "adaptable"
            layer_roles[str(expert_id)] = {
                "role": role,
                "active_domains": active_domains,
            }
        layers_out[layer_id] = layer_roles

    return {
        "schema": OUTPUT_SCHEMA,
        "model": payload["model"],
        "reconstruction_notice": (
            "New implementation of the manuscript's equations; not the original "
            "as-run role-map generator and not proof of matching reported results."
        ),
        "source_schema": SCHEMA,
        "domains": domains,
        "epsilon": epsilon,
        "threshold_mode": mode,
        "thresholds": thresholds_out,
        "layer_specs": payload["layer_specs"],
        "layers": layers_out,
    }


def compare_maps(candidate: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    """Return per-expert differences, including missing coverage."""
    reference_layers = reference.get("layers")
    if not isinstance(reference_layers, dict):
        raise ValueError("reference map lacks a 'layers' object")
    candidate_layers = candidate["layers"]
    differences: list[str] = []
    for layer_id in sorted(set(candidate_layers) | set(reference_layers), key=int):
        c_layer = candidate_layers.get(layer_id, {})
        r_layer = reference_layers.get(layer_id, {})
        if not isinstance(r_layer, dict):
            differences.append(f"layer {layer_id}: invalid reference layer")
            continue
        for expert_id in sorted(set(c_layer) | set(r_layer), key=int):
            c = c_layer.get(expert_id)
            r = r_layer.get(expert_id)
            if not isinstance(r, dict):
                differences.append(f"layer {layer_id} expert {expert_id}: absent in reference")
            elif c is None:
                differences.append(f"layer {layer_id} expert {expert_id}: absent in candidate")
            elif c.get("role") != r.get("role") or set(c.get("active_domains", [])) != set(r.get("active_domains", [])):
                differences.append(f"layer {layer_id} expert {expert_id}: role/domains differ")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, type=Path, help="valid-token top-k count JSON")
    parser.add_argument("--output", required=True, type=Path, help="new expert role JSON")
    parser.add_argument("--mode", required=True, choices=("qwen-quantile", "explicit"))
    parser.add_argument("--tau-red", type=float, help="explicit low-importance cutoff; never infer from scores")
    parser.add_argument("--tau-g", type=float, help="explicit shared-importance cutoff")
    parser.add_argument("--tau-h", type=float, help="normalized cross-domain entropy cutoff")
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--reference-map", type=Path, help="compare candidate with retained original JSON")
    parser.add_argument("--overwrite", action="store_true", help="allow replacement of existing output")
    args = parser.parse_args()

    if args.tau_h is None:
        if args.mode == "qwen-quantile":
            args.tau_h = 0.75
        else:
            parser.error("explicit mode requires --tau-h")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; use a new path or --overwrite")
    protected_inputs = [args.counts]
    if args.reference_map:
        protected_inputs.append(args.reference_map)
    if any(args.output.resolve() == path.resolve() for path in protected_inputs):
        parser.error("output must be separate from the counts and reference-map files")
    try:
        raw = args.counts.read_bytes()
        payload = json.loads(raw)
        role_map = build_role_map(
            payload, mode=args.mode, tau_h=args.tau_h, epsilon=args.epsilon,
            tau_red=args.tau_red, tau_g=args.tau_g,
        )
        role_map["source_counts_sha256"] = hashlib.sha256(raw).hexdigest()
        if args.reference_map:
            reference = json.loads(args.reference_map.read_text(encoding="utf-8"))
            differences = compare_maps(role_map, reference)
            print(f"Reference comparison: {len(differences)} differing experts")
            for difference in differences[:20]:
                print(f"  {difference}")
            if len(differences) > 20:
                print(f"  ... and {len(differences) - 20} more")
        else:
            differences = []
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(role_map, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote reconstructed map: {args.output}")
        print(f"Coverage: {len(role_map['layers'])} layers, {len(role_map['domains'])} domains")
        return 2 if differences else 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Role-map construction failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
