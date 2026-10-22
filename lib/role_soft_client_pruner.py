import json
import os

import torch

from .prune import (
    find_layers,
    get_layers,
    parse_expert_id_from_name,
    prune_wanda_with_expert_sparsity,
)


def find_experts(layer):
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
        return layer.mlp.experts

    for name, module in layer.named_modules():
        if "experts" in name.lower():
            return module

    return None


def get_expert_role(role_info, layer_id, expert_id, domain):
    layer_roles = role_info["layers"].get(str(layer_id), {})
    info = layer_roles.get(str(expert_id), None)

    if info is None:
        return "redundant"

    role = info.get("role", "redundant")
    active_domains = info.get("active_domains", [])

    if role == "shared":
        return "shared"

    if domain in active_domains:
        return "adaptable_active"

    if role == "adaptable":
        return "adaptable_inactive"

    return "redundant"


def count_expert_params(expert):
    total = 0
    for _, parameter in expert.named_parameters():
        if parameter.dim() >= 2:
            total += parameter.numel()
    return total


def build_role_expert_sparsity_map(
    model,
    role_path,
    domain,
    target_sparsity,
    shared_scale=0.50,
    adaptable_active_scale=0.55,
    adaptable_inactive_scale=1.00,
    redundant_scale=1.30,
    max_local_sparsity=0.95,
    return_report=False,
):
    """Build the final normalized-and-clipped per-expert sparsity map."""
    with open(role_path, "r", encoding="utf-8") as f:
        role_info = json.load(f)

    layers = get_layers(model)
    entries = []
    role_count = {
        "shared": 0,
        "adaptable_active": 0,
        "adaptable_inactive": 0,
        "redundant": 0,
    }

    for layer_id, layer in enumerate(layers):
        experts = find_experts(layer)
        if experts is None or not hasattr(experts, "__len__"):
            continue

        for expert_id in range(len(experts)):
            expert = experts[expert_id]
            n_params = count_expert_params(expert)
            if n_params <= 0:
                continue

            role = get_expert_role(
                role_info=role_info,
                layer_id=layer_id,
                expert_id=expert_id,
                domain=domain,
            )
            role_count[role] += 1

            if role == "shared":
                scale = shared_scale
            elif role == "adaptable_active":
                scale = adaptable_active_scale
            elif role == "adaptable_inactive":
                scale = adaptable_inactive_scale
            else:
                scale = redundant_scale

            entries.append(
                {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "role": role,
                    "scale": float(scale),
                    "n_params": int(n_params),
                }
            )

    if not entries:
        raise RuntimeError("No experts matched. Please check model expert structure.")

    total_params = sum(entry["n_params"] for entry in entries)

    def expected_sparsity(alpha):
        zeros = 0.0
        for entry in entries:
            sparsity = float(target_sparsity) * entry["scale"] * alpha
            sparsity = min(max(sparsity, 0.0), max_local_sparsity)
            zeros += sparsity * entry["n_params"]
        return zeros / total_params

    lo, hi = 0.0, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if expected_sparsity(mid) < target_sparsity:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2.0

    expert_sparsity_map = {}
    role_sparsities = {}
    for entry in entries:
        sparsity = float(target_sparsity) * entry["scale"] * alpha
        sparsity = min(max(sparsity, 0.0), max_local_sparsity)
        entry["allocated_sparsity"] = float(sparsity)
        entry["allocated_retained_ratio"] = float(1.0 - sparsity)

        expert_sparsity_map.setdefault(entry["layer_id"], {})
        expert_sparsity_map[entry["layer_id"]][entry["expert_id"]] = sparsity
        role_sparsities.setdefault(entry["role"], []).append(sparsity)

    print(f"[Client-Aware Role-Soft] domain={domain}")
    print(f"[Client-Aware Role-Soft] target_sparsity={target_sparsity}")
    print(f"[Client-Aware Role-Soft] role_count={role_count}")
    print(f"[Client-Aware Role-Soft] normalized alpha={alpha:.6f}")
    print(
        "[Client-Aware Role-Soft] expected sparsity="
        f"{expected_sparsity(alpha):.6f}"
    )
    for role, values in role_sparsities.items():
        print(
            f"[Client-Aware Role-Soft] role={role}, "
            f"avg_local_sparsity={sum(values) / len(values):.4f}, "
            f"min={min(values):.4f}, max={max(values):.4f}"
        )

    if not return_report:
        return expert_sparsity_map

    allocation_report = {
        "schema_version": 1,
        "domain": str(domain),
        "target_sparsity": float(target_sparsity),
        "max_local_sparsity": float(max_local_sparsity),
        "normalized_alpha": float(alpha),
        "expected_parameter_weighted_sparsity": float(
            expected_sparsity(alpha)
        ),
        "experts": entries,
    }
    return expert_sparsity_map, allocation_report


def build_realized_sparsity_report(model, allocation_report):
    """Attach exact post-pruning zero counts for every indexed expert."""
    records = {
        (int(entry["layer_id"]), int(entry["expert_id"])): entry
        for entry in allocation_report["experts"]
    }
    for record in records.values():
        record["selected_params"] = 0
        record["zero_params"] = 0

    for layer_id, layer in enumerate(get_layers(model)):
        for name, module in find_layers(layer, scope="moe_only").items():
            expert_id = parse_expert_id_from_name(name)
            record = records.get((layer_id, expert_id))
            if record is None:
                continue

            weight = module.weight.detach()
            record["selected_params"] += int(weight.numel())
            record["zero_params"] += int(
                torch.count_nonzero(weight == 0).item()
            )

    for key, record in records.items():
        total = int(record["selected_params"])
        if total <= 0:
            raise RuntimeError(
                "No prunable expert weights were counted for "
                f"layer={key[0]}, expert={key[1]}."
            )
        realized_sparsity = float(record["zero_params"]) / float(total)
        record["realized_sparsity"] = realized_sparsity
        record["retained_ratio"] = float(1.0 - realized_sparsity)

    total_selected = sum(
        int(record["selected_params"]) for record in records.values()
    )
    total_zeros = sum(
        int(record["zero_params"]) for record in records.values()
    )
    allocation_report["realized_parameter_weighted_sparsity"] = (
        float(total_zeros) / float(total_selected)
    )
    allocation_report["retained_ratio_definition"] = (
        "1 - realized_sparsity, parameter-weighted over selected expert "
        "Linear weights"
    )
    return allocation_report


def save_sparsity_report(report, output_path):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[Client-Aware Role-Soft] saved sparsity report: {output_path}")
    return output_path


def apply_clientaware_role_soft_pruning(
    args,
    model,
    tokenizer,
    dataloader,
    device,
    role_path,
    domain,
    target_sparsity,
    prune_n=0,
    prune_m=0,
    sparsity_report_path=None,
):
    """Run client-aware role allocation and local Wanda pruning."""
    if getattr(args, "prune_scope", None) != "moe_only":
        raise ValueError(
            "role_moe_soft_clientaware requires --prune_scope moe_only "
            "so non-expert parameters remain dense."
        )
    expert_sparsity_map, allocation_report = build_role_expert_sparsity_map(
        model=model,
        role_path=role_path,
        domain=domain,
        target_sparsity=target_sparsity,
        return_report=True,
    )

    old_sparsity = args.sparsity_ratio
    args.sparsity_ratio = target_sparsity
    try:
        prune_wanda_with_expert_sparsity(
            args=args,
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            expert_sparsity_map=expert_sparsity_map,
            device=device,
            prune_n=prune_n,
            prune_m=prune_m,
        )
    finally:
        args.sparsity_ratio = old_sparsity

    realized_report = build_realized_sparsity_report(
        model=model,
        allocation_report=allocation_report,
    )
    if sparsity_report_path:
        save_sparsity_report(realized_report, sparsity_report_path)

    torch.cuda.empty_cache()
    return expert_sparsity_map