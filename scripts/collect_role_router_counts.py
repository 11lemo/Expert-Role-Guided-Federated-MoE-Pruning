"""Reconstructed, valid-token top-k router statistics for role-map discovery.

This is a new implementation of the procedure described in the manuscript, not
the missing original experiment script. It emits counts only; it does not infer
or tune expert roles from reported benchmark scores.

Supported model implementations: Hugging Face Transformers 4.46.3 OLMoE and
Qwen2-MoE. The script requires an already downloaded local model directory.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SCHEMA = "role-router-counts-v1"
SUPPORTED_MODEL_TYPES = {"olmoe", "qwen2_moe"}


def select_client_texts(domain_dir, domains, nsamples_per_client, clients_per_domain, seed):
    """Mirror lib.domain_data's filtering, shuffling, repeating and split."""
    if len(set(domains)) != len(domains):
        raise ValueError("--domains must not contain duplicates")
    if nsamples_per_client < 1 or clients_per_domain < 1:
        raise ValueError("sample and client counts must be positive")

    rng = random.Random(seed)
    selected = []
    for domain in domains:
        source = Path(domain_dir) / f"{domain}.json"
        if not source.is_file():
            raise FileNotFoundError(f"Domain file not found: {source}")
        with source.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, list):
            raise ValueError(f"Expected a JSON list of texts: {source}")
        texts = [str(item).strip() for item in raw if len(str(item).strip()) >= 10]
        if not texts:
            raise ValueError(f"No valid text samples in {source}")
        rng.shuffle(texts)
        required = nsamples_per_client * clients_per_domain
        if len(texts) < required:
            texts = (texts * ((required + len(texts) - 1) // len(texts)))[:required]
        else:
            texts = texts[:required]
        for client_number in range(clients_per_domain):
            start = client_number * nsamples_per_client
            selected.append(
                {
                    "client_id": f"{domain}_{client_number}",
                    "domain": domain,
                    "texts": texts[start : start + nsamples_per_client],
                }
            )
    return selected


def discover_moe_gates(model):
    """Find sparse expert gates, retaining the model's decoder-layer indices."""
    model_type = getattr(model.config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type={model_type!r}; expected {sorted(SUPPORTED_MODEL_TYPES)}"
        )
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Expected model.model.layers in the Hugging Face model")

    gates = {}
    specs = {}
    for layer_id, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        experts = getattr(mlp, "experts", None)
        if gate is None or experts is None:
            continue  # Qwen2-MoE may have ordinary dense decoder layers.
        num_experts = int(getattr(mlp, "num_experts"))
        top_k = int(getattr(mlp, "top_k"))
        if len(experts) != num_experts or not 0 < top_k <= num_experts:
            raise ValueError(f"Invalid MoE dimensions at decoder layer {layer_id}")
        if int(getattr(gate, "out_features", -1)) != num_experts:
            raise ValueError(f"Unexpected gate output width at decoder layer {layer_id}")
        layer_key = str(layer_id)
        gates[layer_key] = gate
        specs[layer_key] = {"num_experts": num_experts, "top_k": top_k}
    if not gates:
        raise ValueError("No supported sparse MoE gates found")
    return backbone, gates, specs


class RouterCounter:
    """Accumulate expert hits from actual gate outputs for the current client."""

    def __init__(self, torch, gates, layer_specs):
        self.torch = torch
        self.layer_specs = layer_specs
        self.handles = [gate.register_forward_hook(self._make_hook(layer_id)) for layer_id, gate in gates.items()]
        self.mask = None
        self.counts = None
        self.seen = None

    def begin_client(self):
        self.counts = {
            layer_id: self.torch.zeros(spec["num_experts"], dtype=self.torch.int64)
            for layer_id, spec in self.layer_specs.items()
        }

    def begin_batch(self, attention_mask):
        if self.counts is None:
            raise RuntimeError("Call begin_client before begin_batch")
        self.mask = attention_mask.reshape(-1).bool()
        self.seen = {layer_id: 0 for layer_id in self.layer_specs}

    def end_batch(self):
        missing_or_repeated = {key: count for key, count in self.seen.items() if count != 1}
        if missing_or_repeated:
            raise RuntimeError(f"Expected one gate call per MoE layer: {missing_or_repeated}")
        self.mask = None

    def _make_hook(self, layer_id):
        spec = self.layer_specs[layer_id]

        def hook(_module, _inputs, router_logits):
            if self.mask is None:
                raise RuntimeError("A router gate ran outside a counted batch")
            if not isinstance(router_logits, self.torch.Tensor):
                raise TypeError(f"Gate output is not a tensor at layer {layer_id}")
            logits = router_logits.reshape(-1, spec["num_experts"])
            if logits.shape[0] != self.mask.numel():
                raise ValueError(
                    f"Layer {layer_id}: {logits.shape[0]} routed positions but "
                    f"{self.mask.numel()} attention-mask positions"
                )
            # Match the v4.46.3 MoE block: float32 softmax, then top-k.
            selected = self.torch.topk(
                self.torch.softmax(logits.float(), dim=-1), spec["top_k"], dim=-1
            ).indices
            valid = self.mask.to(device=selected.device)
            hits = self.torch.bincount(
                selected[valid].reshape(-1).to(self.torch.int64),
                minlength=spec["num_experts"],
            )
            self.counts[layer_id] += hits.to(device="cpu", dtype=self.torch.int64)
            self.seen[layer_id] += 1

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def collect(args, client_texts):
    # Imports are deferred so --dry-run validates inputs without torch/GPU.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not Path(args.model).is_dir():
        raise FileNotFoundError("--model must be an existing local model directory")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    backbone, gates, specs = discover_moe_gates(model)
    input_device = model.get_input_embeddings().weight.device
    if input_device.type == "meta":
        raise RuntimeError("Embedding weights are on meta device; cannot place input tensors")
    counter = RouterCounter(torch, gates, specs)
    clients = []
    try:
        with torch.inference_mode():
            for client in client_texts:
                counter.begin_client()
                n_valid_tokens = 0
                texts = client["texts"]
                for start in range(0, len(texts), args.batch_size):
                    batch = texts[start : start + args.batch_size]
                    encoded = tokenizer(
                        batch,
                        padding="max_length" if args.padding == "max_length" else True,
                        truncation=True,
                        max_length=args.seqlen,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    mask = encoded["attention_mask"]
                    n_valid_tokens += int(mask.sum().item())
                    counter.begin_batch(mask)
                    backbone(
                        input_ids=encoded["input_ids"].to(input_device),
                        attention_mask=mask.to(input_device),
                        use_cache=False,
                        output_router_logits=False,
                    )
                    counter.end_batch()
                if n_valid_tokens <= 0:
                    raise ValueError(f"No valid tokens for {client['client_id']}")
                layer_counts = {
                    layer_id: [int(value) for value in counts.tolist()]
                    for layer_id, counts in counter.counts.items()
                }
                for layer_id, counts in layer_counts.items():
                    expected = n_valid_tokens * specs[layer_id]["top_k"]
                    if sum(counts) != expected:
                        raise RuntimeError(
                            f"Layer {layer_id}, client {client['client_id']}: "
                            f"count sum {sum(counts)} != {expected}"
                        )
                clients.append(
                    {
                        "client_id": client["client_id"],
                        "domain": client["domain"],
                        "n_valid_tokens": n_valid_tokens,
                        "layers": layer_counts,
                    }
                )
                print(f"{client['client_id']}: {len(texts)} samples, {n_valid_tokens} valid tokens", flush=True)
    finally:
        counter.close()

    return {
        "schema": SCHEMA,
        "model": str(Path(args.model).resolve()),
        "layer_specs": specs,
        "clients": clients,
        "provenance": {
            "status": "reconstructed_from_manuscript_not_original_run",
            "model_type": model.config.model_type,
            "transformers_version": transformers.__version__,
            "domains": args.domains,
            "nsamples_per_client": args.nsamples_per_client,
            "clients_per_domain": args.clients_per_domain,
            "seqlen": args.seqlen,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "padding": args.padding,
            "dtype": args.dtype,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing local HF model directory")
    parser.add_argument("--domain-dir", required=True, type=Path)
    parser.add_argument("--domains", required=True, nargs="+", help="Domain JSON file stems")
    parser.add_argument("--nsamples-per-client", type=int, default=64)
    parser.add_argument("--clients-per-domain", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--padding", choices=("max_length", "longest"), default="max_length")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate client data without loading the model")
    parser.add_argument("--output", type=Path, help="Counts JSON destination (required except in --dry-run)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    if args.seqlen < 1 or args.batch_size < 1:
        parser.error("--seqlen and --batch-size must be positive")
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is set")
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output} (use --overwrite to replace)")
    return args


def main():
    args = parse_args()
    selected = select_client_texts(
        args.domain_dir,
        args.domains,
        args.nsamples_per_client,
        args.clients_per_domain,
        args.seed,
    )
    for client in selected:
        print(f"{client['client_id']}: {len(client['texts'])} selected samples")
    if args.dry_run:
        print("Dry run complete: no model was loaded and no file was written")
        return
    result = collect(args, selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved reconstructed router counts to {args.output}")


if __name__ == "__main__":
    main()
