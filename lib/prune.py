import torch
import torch.nn as nn
from .layerwrapper import WrappedGPT
import re


def parse_expert_id_from_name(full_name: str):
    """
    Parse expert id from layer submodule name.
    Expected patterns like:
      mlp.experts.0.gate_proj
      mlp.experts.3.up_proj
      experts.5.down_proj
    """
    m = re.search(r"experts\.([0-9]+)", full_name)
    if m is not None:
        return int(m.group(1))
    return None


def get_hf_device_map(model):
    """Safely get HuggingFace device map (when loaded with device_map='auto')."""
    dm = getattr(model, "hf_device_map", None)
    return dm if isinstance(dm, dict) else {}


def get_layers(model):
    """
    Return transformer block list for common HF causal LMs.
    LLaMA/Mistral/DeepSeek/Qwen2 style: model.model.layers
    GPT-style: model.transformer.h
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("Cannot locate transformer layers. Expected model.model.layers or model.transformer.h")


# ---------- MoE scope helpers ----------
def _is_moe_related_name(full_name: str) -> bool:
    n = full_name.lower()
    return ("moe" in n) or ("expert" in n) or ("experts" in n) or ("router" in n) or ("gate" in n)


def _is_moe_expert_linear_name(full_name: str) -> bool:
    """
    Return True only for Linear modules inside an indexed MoE expert.

    Matching the structural path ``experts.<id>`` is important here. A
    substring rule such as ``"gate" not in name`` incorrectly excludes an
    expert's ``gate_proj`` together with the actual router/gating module.

    Examples:
      - mlp.experts.0.gate_proj  -> True
      - mlp.experts.0.up_proj    -> True
      - mlp.experts.0.down_proj  -> True
      - mlp.gate                 -> False
      - mlp.shared_expert_gate   -> False
      - mlp.router               -> False
    """
    n = full_name.lower()
    return re.search(r"(?:^|\.)experts\.[0-9]+(?:\.|$)", n) is not None


def find_layers(module, layers=(nn.Linear,), name="", scope="all"):
    """
    Recursively find layers (default nn.Linear).

    scope:
      - all: return all matched layers
      - moe_only: return only MoE experts' linear layers (NOT router/gate by default)
    """
    if isinstance(module, layers):
        if scope == "all":
            return {name: module}
        if scope == "moe_only":
            return {name: module} if _is_moe_expert_linear_name(name) else {}
        raise ValueError(f"Unknown scope: {scope}")

    res = {}
    for name1, child in module.named_children():
        child_name = name + "." + name1 if name != "" else name1
        res.update(find_layers(child, layers=layers, name=child_name, scope=scope))
    return res


def check_sparsity(model, scope="all"):
    """
    Check sparsity ratio for the selected scope.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_layers(model)
    count = 0
    total_params = 0

    for i, layer in enumerate(layers):
        subset = find_layers(layer, scope=scope)
        sub_count = 0
        sub_params = 0
        for name, mod in subset.items():
            W = mod.weight.data
            z = (W == 0).sum().item()
            n = W.numel()
            count += z
            total_params += n
            sub_count += z
            sub_params += n

        if sub_params > 0:
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        else:
            # 该层在 moe_only 下可能没有任何线性层被纳入统计
            print(f"layer {i} sparsity N/A (no selected params in scope={scope})")

    model.config.use_cache = use_cache
    if total_params == 0:
        return 0.0
    return float(count) / float(total_params)


def prepare_calibration_input(model, dataloader, device, datasize=128):
    """
    Collect the hidden_states entering the first transformer block.
    Also caches attention_mask / position_ids / position_embeddings when present.

    Returns:
      inps: [N, seqlen, hidden]
      outs: zeros_like(inps)
      attention_mask / position_ids / position_embeddings: may be None depending on model
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_layers(model)
    device_map = get_hf_device_map(model)

    # prefer embedding device if sharded; otherwise use first param device
    if "model.embed_tokens" in device_map:
        device = device_map["model.embed_tokens"]
    else:
        device = next(model.parameters()).device

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((datasize, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad_(False)

    cache = {
        "i": 0,
        "attention_mask": None,
        "position_ids": None,
        "position_embeddings": None,  # for Qwen2/Qwen2-MoE
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            idx = cache["i"]
            if idx < inps.shape[0]:
                x = inp
                # inp: [B, seqlen, hidden] or [seqlen, hidden]
                if x.dim() == 3:
                    x = x[0]  # take first sample
                # now [seqlen, hidden]
                inps[idx].copy_(x)

            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["position_ids"] = kwargs.get("position_ids", None)

            pe = kwargs.get("position_embeddings", None)
            if pe is not None:
                cache["position_embeddings"] = pe

            raise ValueError("Catcher stop")

    layers[0] = Catcher(layers[0])

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= datasize:
                break
            try:
                model(batch[0].to(device))
            except ValueError:
                pass

    layers[0] = layers[0].module

    datasize_eff = min(cache["i"], datasize)
    inps = inps[:datasize_eff]
    outs = torch.zeros_like(inps)

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    model.config.use_cache = use_cache
    return inps, outs, attention_mask, position_ids, position_embeddings


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def _layer_forward(layer, x, attention_mask=None, position_ids=None, position_embeddings=None):
    """
    Call transformer block forward with best-effort kwargs compatibility.
    """
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings

    out = layer(x, **kwargs)
    if isinstance(out, tuple):
        return out[0]
    return out


def prune_wanda(args, model, tokenizer, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    device_map = get_hf_device_map(model)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    nsamples = inps.shape[0]
    if nsamples == 0:
        raise RuntimeError("No calibration samples collected in prune_wanda.")

    layers = get_layers(model)
    scope = getattr(args, "prune_scope", "all")

    for i, layer in enumerate(layers):
        subset = find_layers(layer, scope=scope)

        # move tensors to layer device if sharded
        if f"model.layers.{i}" in device_map:
            dev = device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)
            if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)):
                position_embeddings = tuple(pe.to(dev) for pe in position_embeddings)

        if len(subset) == 0:
            # moe_only 下有的层可能不包含 experts linear
            with torch.no_grad():
                for j in range(nsamples):
                    x = inps[j].unsqueeze(0).contiguous()
                    outs[j] = _layer_forward(layer, x, attention_mask, position_ids, position_embeddings)
            inps, outs = outs, inps
            continue

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def hook_fn(_, inp, out):
                x = inp[0].detach()
                y = out[0].detach() if isinstance(out, tuple) else out.detach()
                wrapped_layers[name].add_batch(x, y)
            return hook_fn

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        with torch.no_grad():
            for j in range(nsamples):
                x = inps[j].unsqueeze(0).contiguous()
                outs[j] = _layer_forward(layer, x, attention_mask, position_ids, position_embeddings)

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        W_mask.scatter_(1, ii + idx, True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if getattr(args, "use_variant", False):
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (
                        abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        and (alpha_hist[1] - alpha_hist[0]) >= 0.001
                    ):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_hist[1] = alpha
                            alpha = (alpha + alpha_hist[0]) / 2
                        else:
                            alpha_hist[0] = alpha
                            alpha = (alpha + alpha_hist[1]) / 2
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    k = int(W_metric.shape[1] * args.sparsity_ratio)
                    indices = sort_res[1][:, :k]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

        with torch.no_grad():
            for j in range(nsamples):
                x = inps[j].unsqueeze(0).contiguous()
                outs[j] = _layer_forward(layer, x, attention_mask, position_ids, position_embeddings)

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_with_expert_sparsity(
    args,
    model,
    tokenizer,
    dataloader,
    expert_sparsity_map,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0
):
    """
    Wanda pruning with per-expert sparsity ratio.
    Only meaningful when scope='moe_only'.
    """
    device_map = get_hf_device_map(model)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    nsamples = inps.shape[0]
    if nsamples == 0:
        raise RuntimeError("No calibration samples collected in prune_wanda_with_expert_sparsity.")

    layers = get_layers(model)
    scope = getattr(args, "prune_scope", "all")

    for i, layer in enumerate(layers):
        subset = find_layers(layer, scope=scope)

        if f"model.layers.{i}" in device_map:
            dev = device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)
            if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)):
                position_embeddings = tuple(pe.to(dev) for pe in position_embeddings)

        if len(subset) == 0:
            with torch.no_grad():
                for j in range(nsamples):
                    x = inps[j].unsqueeze(0).contiguous()
                    outs[j] = _layer_forward(layer, x, attention_mask, position_ids, position_embeddings)
            inps, outs = outs, inps
            continue

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def hook_fn(_, inp, out):
                x = inp[0].detach()
                y = out[0].detach() if isinstance(out, tuple) else out.detach()
                wrapped_layers[name].add_batch(x, y)
            return hook_fn

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        with torch.no_grad():
            for j in range(nsamples):
                x = inps[j].unsqueeze(0).contiguous()
                outs[j] = _layer_forward(layer, x, attention_mask, position_ids, position_embeddings)

        for h in handles:
            h.remove()

        for name in subset:
            print(f"client-aware pruning layer {i} name {name}")

            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            expert_id = parse_expert_id_from_name(name)

            if expert_id is not None and i in expert_sparsity_map and expert_id in expert_sparsity_map[i]:
                local_sparsity = expert_sparsity_map[i][expert_id]
            else:
                local_sparsity = args.sparsity_ratio

            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        W_mask.scatter_(1, ii + idx, True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                k = int(W_metric.shape[1] * local_sparsity)
                k = max(0, min(k, W_metric.shape[1]))
                if k > 0:
                    indices = sort_res[1][:, :k]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

        with torch.no_grad():
            for j in range(nsamples):
                x = inps[j].unsqueeze(0).contiguous()
                outs[j] = _layer_forward(layer, x, attention_mask, position_ids, position_embeddings)

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def prune_wanda_and_update_global_mask(
    args,
    model,
    global_masks,
    tokenizer,
    dataloader,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0
):
    device_map = get_hf_device_map(model)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
            model, dataloader, device
        )

    nsamples = inps.shape[0]
    if nsamples == 0:
        raise RuntimeError("No calibration samples collected in prune_wanda_and_update_global_mask.")

    layers = get_layers(model)
    scope = getattr(args, "prune_scope", "all")

    for i, layer in enumerate(layers):
        subset = find_layers(layer, scope=scope)

        if f"model.layers.{i}" in device_map:
            dev = device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)
            if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)):
                position_embeddings = tuple(pe.to(dev) for pe in position_embeddings)

        # 即使这一层没选中要剪的模块，也要把 outs 算出来以便推进
        if len(subset) == 0:
            with torch.no_grad():
                for j in range(nsamples):
                    x = inps[j].unsqueeze(0).contiguous()
                    outs[j] = _layer_forward(layer, x, attention_mask, position_ids, position_embeddings)
            inps, outs = outs, inps
            continue

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def hook_fn(_, inp, out):
                x = inp[0].detach()
                y = out[0].detach() if isinstance(out, tuple) else out.detach()
                wrapped_layers[name].add_batch(x, y)
            return hook_fn

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        with torch.no_grad():
            for j in range(nsamples):
                x = inps[j].unsqueeze(0).contiguous()
                outs[j] = _layer_forward(layer, x, attention_mask, position_ids, position_embeddings)

        for h in handles:
            h.remove()

        for name in subset:
            print(f"locally pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        W_mask.scatter_(1, ii + idx, True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if getattr(args, "use_variant", False):
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (
                        abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        and (alpha_hist[1] - alpha_hist[0]) >= 0.001
                    ):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_hist[1] = alpha
                            alpha = (alpha + alpha_hist[0]) / 2
                        else:
                            alpha_hist[0] = alpha
                            alpha = (alpha + alpha_hist[1]) / 2
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    k = int(W_metric.shape[1] * args.sparsity_ratio)
                    indices = sort_res[1][:, :k]
                    W_mask.scatter_(1, indices, True)

            # local prune to build vote mask
            with torch.no_grad():
                subset[name].weight.data[W_mask] = 0

            # update global mask (vote accumulation)
            unique_name = f"layer_{i}.{name}"
            global_masks[unique_name] += W_mask.to(global_masks[unique_name].dtype)

        # recompute for next layer swap
        with torch.no_grad():
            for j in range(nsamples):
                x = inps[j].unsqueeze(0).contiguous()
                outs[j] = _layer_forward(layer, x, attention_mask, position_ids, position_embeddings)

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_fedprllm_layer(args, model, global_masks, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = get_layers(model)
    scope = getattr(args, "prune_scope", "all")

    for i, layer in enumerate(layers):
        subset = find_layers(layer, scope=scope)
        if len(subset) == 0:
            continue

        for name in subset:
            unique_name = f"layer_{i}.{name}"
            print(f"server pruning layer {i} name {name}")
            W_metric = -global_masks[unique_name].float()

            if prune_n != 0:
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        W_mask.scatter_(1, ii + idx, True)
            else:
                flat = W_metric.flatten()
                kth = int(W_metric.numel() * args.sparsity_ratio)
                kth = min(max(kth, 0), W_metric.numel() - 1)
                thresh = torch.sort(flat)[0][kth].item()
                W_mask = (W_metric < thresh)

                need = int(W_metric.numel() * args.sparsity_ratio) - int(W_mask.sum().item())
                if need > 0:
                    idxs = (W_metric == thresh).nonzero(as_tuple=False)
                    pick = idxs[torch.randperm(idxs.size(0))[:need]]
                    W_mask[pick[:, 0], pick[:, 1]] = True

            with torch.no_grad():
                subset[name].weight.data[W_mask] = 0

            if getattr(args, "weight_scaling", False):
                with torch.no_grad():
                    subset[name].weight.data = subset[name].weight.data * (
                        (args.client_num - global_masks[unique_name]).to(torch.float16) / args.client_num
                    ).to(subset[name].weight.data.device)


def prune_fedprllm_row_or_column(args, model, global_masks, device=torch.device("cuda:0"), dim=1, prune_n=0, prune_m=0):
    layers = get_layers(model)
    scope = getattr(args, "prune_scope", "all")

    for i, layer in enumerate(layers):
        subset = find_layers(layer, scope=scope)
        if len(subset) == 0:
            continue

        for name in subset:
            unique_name = f"layer_{i}.{name}"
            print(f"server pruning layer {i} name {name}")
            W_metric = -global_masks[unique_name].float()

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        W_mask.scatter_(1, ii + idx, True)
            else:
                if dim == 1:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    k = int(W_metric.shape[1] * args.sparsity_ratio)
                    indices = sort_res[1][:, :k]
                    W_mask.scatter_(1, indices, True)
                elif dim == 0:
                    sort_res = torch.sort(W_metric, dim=0, stable=True)
                    k = int(W_metric.shape[0] * args.sparsity_ratio)
                    indices = sort_res[1][:k, :]
                    W_mask.scatter_(0, indices, True)
                else:
                    raise ValueError("dim should be 0 or 1")

            with torch.no_grad():
                subset[name].weight.data[W_mask] = 0

            if getattr(args, "weight_scaling", False):
                with torch.no_grad():
                    subset[name].weight.data = subset[name].weight.data * (
                        (args.client_num - global_masks[unique_name]).to(torch.float16) / args.client_num
                    ).to(subset[name].weight.data.device)
