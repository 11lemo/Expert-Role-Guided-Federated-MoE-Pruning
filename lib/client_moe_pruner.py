import math

import torch
import torch.nn.functional as F


def _get_batch(batch, device):
    if isinstance(batch, dict):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
    else:
        input_ids = batch[0]
        attention_mask = batch[2] if len(batch) >= 3 else None

    input_ids = input_ids.to(device)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)

    return input_ids, attention_mask


@torch.no_grad()
def prune_client_moe_pruner(
    model,
    dataloader,
    sparsity_ratio,
    device,
):
    """
    Client-local adaptation of MoE-Pruner.

    It prunes only gate_proj, up_proj and down_proj inside routed experts.
    Shared experts and router parameters are not modified.
    """

    model.eval()
    base_model = model.model
    layers = base_model.layers

    state = {"valid_mask": None}
    contexts = {}
    statistics = {}
    modules = {}
    handles = []

    for layer_id, layer in enumerate(layers):
        moe = getattr(layer, "mlp", None)

        if (
            moe is None
            or not hasattr(moe, "experts")
            or not hasattr(moe, "gate")
        ):
            continue

        experts = moe.experts

        if not hasattr(experts, "__len__"):
            raise RuntimeError(
                "The installed Transformers version uses fused expert "
                "parameters. This implementation expects indexable experts."
            )

        num_experts = len(experts)
        top_k = int(model.config.num_experts_per_tok)
        normalize_topk = bool(
            getattr(model.config, "norm_topk_prob", True)
        )

        contexts[layer_id] = {"gate_by_expert": {}}

        def make_router_hook(
            current_layer_id,
            current_num_experts,
            current_top_k,
            current_normalize_topk,
        ):
            def router_hook(module, inputs, output):
                # Older implementations return only router logits.
                # Some newer implementations return a tuple.
                if isinstance(output, tuple) and len(output) >= 3:
                    router_logits = output[0]
                    routing_weights = output[1]
                    selected_experts = output[2]
                else:
                    router_logits = output
                    router_logits = router_logits.reshape(
                        -1, current_num_experts
                    )

                    probabilities = F.softmax(
                        router_logits.float(),
                        dim=-1,
                    )

                    routing_weights, selected_experts = torch.topk(
                        probabilities,
                        k=current_top_k,
                        dim=-1,
                    )

                    if current_normalize_topk:
                        routing_weights = routing_weights / (
                            routing_weights.sum(
                                dim=-1,
                                keepdim=True,
                            )
                            + 1e-12
                        )

                routing_weights = routing_weights.float()
                selected_experts = selected_experts.long()

                valid_mask = state["valid_mask"]

                if (
                    valid_mask is not None
                    and valid_mask.numel()
                    == routing_weights.shape[0]
                ):
                    routing_weights = routing_weights * valid_mask.to(
                        routing_weights.device,
                        dtype=routing_weights.dtype,
                    ).reshape(-1, 1)

                # Reproduce the token order used when Qwen sends
                # selected tokens into each expert.
                expert_mask = F.one_hot(
                    selected_experts,
                    num_classes=current_num_experts,
                ).permute(2, 1, 0)

                gate_by_expert = {}

                for expert_id in range(current_num_experts):
                    topk_position, token_index = torch.where(
                        expert_mask[expert_id]
                    )

                    gate_by_expert[expert_id] = routing_weights[
                        token_index,
                        topk_position,
                    ].detach()

                contexts[current_layer_id][
                    "gate_by_expert"
                ] = gate_by_expert

            return router_hook

        handles.append(
            moe.gate.register_forward_hook(
                make_router_hook(
                    layer_id,
                    num_experts,
                    top_k,
                    normalize_topk,
                )
            )
        )

        for expert_id, expert in enumerate(experts):
            for matrix_name in (
                "gate_proj",
                "up_proj",
                "down_proj",
            ):
                if not hasattr(expert, matrix_name):
                    raise RuntimeError(
                        f"Missing {matrix_name} in "
                        f"layer={layer_id}, expert={expert_id}"
                    )

                linear = getattr(expert, matrix_name)
                key = (layer_id, expert_id, matrix_name)

                modules[key] = linear
                statistics[key] = torch.zeros(
                    linear.in_features,
                    dtype=torch.float32,
                    device=linear.weight.device,
                )

                def make_expert_hook(
                    current_key,
                    current_layer_id,
                    current_expert_id,
                ):
                    def expert_hook(module, inputs):
                        x = inputs[0].detach().reshape(
                            -1,
                            module.in_features,
                        )

                        gate = contexts[current_layer_id][
                            "gate_by_expert"
                        ].get(current_expert_id)

                        if gate is None:
                            raise RuntimeError(
                                "Router weights were not collected "
                                f"for {current_key}."
                            )

                        if gate.numel() != x.shape[0]:
                            raise RuntimeError(
                                "Router/expert token mismatch for "
                                f"{current_key}: gate={gate.numel()}, "
                                f"expert_input={x.shape[0]}"
                            )

                        gate = gate.to(
                            x.device,
                            dtype=torch.float32,
                        )

                        weighted_input = (
                            x.float() * gate.reshape(-1, 1)
                        )

                        statistics[current_key].add_(
                            weighted_input.square().sum(dim=0)
                        )

                    return expert_hook

                handles.append(
                    linear.register_forward_pre_hook(
                        make_expert_hook(
                            key,
                            layer_id,
                            expert_id,
                        )
                    )
                )

    if not modules:
        raise RuntimeError(
            "No routed-expert matrices were found."
        )

    # Collect router-weighted activation statistics.
    try:
        for batch in dataloader:
            input_ids, attention_mask = _get_batch(
                batch,
                device,
            )

            state["valid_mask"] = (
                attention_mask.reshape(-1).bool()
            )

            base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    # Row-wise unstructured pruning.
    total_weights = 0
    total_zeros = 0
    inactive_modules = 0

    for key, linear in modules.items():
        activation_norm = torch.sqrt(
            statistics[key].clamp_min(0)
        )

        if torch.count_nonzero(activation_norm).item() == 0:
            inactive_modules += 1

        activation_norm = activation_norm.to(
            linear.weight.device
        )

        score = (
            linear.weight.data.abs().float()
            * activation_norm.reshape(1, -1)
        )

        number_to_prune = math.floor(
            float(sparsity_ratio) * linear.in_features
        )

        number_to_prune = max(
            0,
            min(number_to_prune, linear.in_features),
        )

        if number_to_prune > 0:
            # Stable ordering makes tied scores reproducible.
            indices = torch.argsort(
                score,
                dim=1,
                stable=True,
            )[:, :number_to_prune]

            pruning_mask = torch.zeros_like(
                linear.weight.data,
                dtype=torch.bool,
            )

            pruning_mask.scatter_(
                1,
                indices,
                True,
            )

            linear.weight.data[pruning_mask] = 0

        total_weights += linear.weight.numel()
        total_zeros += torch.count_nonzero(
            linear.weight.data == 0
        ).item()

        del score

    realized_sparsity = total_zeros / total_weights

    print(
        "[Client-MoE-Pruner] "
        f"target sparsity={sparsity_ratio:.4f}"
    )
    print(
        "[Client-MoE-Pruner] "
        f"realized sparsity={realized_sparsity:.6f}"
    )
    print(
        "[Client-MoE-Pruner] "
        f"modules with zero router-weighted statistics="
        f"{inactive_modules}/{len(modules)}"
    )

    torch.cuda.empty_cache()

    return {
        "target_sparsity": float(sparsity_ratio),
        "realized_sparsity": float(realized_sparsity),
        "pruned_parameters": int(total_weights),
        "inactive_modules": int(inactive_modules),
    }