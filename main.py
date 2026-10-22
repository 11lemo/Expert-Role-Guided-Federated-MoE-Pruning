# main.py
import argparse
import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import (
    prune_wanda,
    check_sparsity,
    find_layers,
    prune_wanda_and_update_global_mask,
    prune_fedprllm_layer,
    prune_fedprllm_row_or_column,
    get_hf_device_map,
    get_layers,
)
from lib.eval import eval_ppl
from lib.data import get_loaders


print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus: ", torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    n_gpus = torch.cuda.device_count()

    if n_gpus == 0:
        raise RuntimeError(
            "No CUDA GPU is visible. Please check nvidia-smi and CUDA_VISIBLE_DEVICES."
        )

    max_memory = {}
    for i in range(n_gpus):
        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        usable_gib = int(total_gib * 0.90)
        max_memory[i] = f"{usable_gib}GiB"

    max_memory["cpu"] = "128GiB"

    print(f"[get_llm] visible GPUs: {n_gpus}")
    print(f"[get_llm] max_memory: {max_memory}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )

    if hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings is not None:
        model.seqlen = model.config.max_position_embeddings
    else:
        model.seqlen = 2048

    return model

    if hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings is not None:
        model.seqlen = model.config.max_position_embeddings
    else:
        model.seqlen = 2048

    return model


def pick_input_device(model):
    dm = get_hf_device_map(model)
    for k in ["model.embed_tokens", "embed_tokens", "model.wte", "transformer.wte"]:
        if k in dm:
            return torch.device(dm[k])
    return next(model.parameters()).device


def split_dataloader_for_clients(dataloader, client_num, seed=0, split_mode="iid"):
    dataloader = list(dataloader)
    total = len(dataloader)
    client_num = min(client_num, total)

    if split_mode == "iid":
        rng = random.Random(seed)
        rng.shuffle(dataloader)
    elif split_mode == "block":
        pass
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    per_client = max(1, total // client_num)
    local_dataloaders = [
        dataloader[i * per_client:(i + 1) * per_client]
        for i in range(client_num)
    ]

    rest = dataloader[client_num * per_client:]
    if len(rest) > 0:
        local_dataloaders[-1].extend(rest)

    return local_dataloaders


def save_tabular_log(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        print("\t".join(header), file=f, flush=True)
        for row in rows:
            print("\t".join(str(row[h]) for h in header), file=f, flush=True)




def should_save_this_client(args, domain=None):
    """
    Control whether to save the current pruned model.
    --save_pruned_model: enable saving.
    --save_domain: save a specific domain, e.g. gsm8k; use all or omit to save all.
    """
    if not getattr(args, "save_pruned_model", False):
        return False

    save_domain = getattr(args, "save_domain", None)

    if save_domain is None or save_domain == "all":
        return True

    if domain is None:
        return True

    base_domain = str(domain).split("_")[0]
    return str(domain) == save_domain or base_domain == save_domain


def save_model_for_lm_eval(args, model, tokenizer, method, domain=None):
    """
    Save the pruned model in HuggingFace format for lm-evaluation-harness.
    This function is called after PPL evaluation, so it does not affect PPL.
    """
    if not should_save_this_client(args, domain):
        print(f"[Save] skip saving model, domain={domain}")
        return None

    os.makedirs(args.save_model_dir, exist_ok=True)

    if domain is None:
        save_name = f"{method}_s{args.sparsity_ratio}"
    else:
        save_name = f"{method}_{domain}_s{args.sparsity_ratio}"

    save_dir = os.path.join(args.save_model_dir, save_name)

    print(f"[Save] saving model to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(
        save_dir,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(save_dir)

    print(f"[Save] done: {save_dir}")
    return save_dir


def eval_and_log(args, model, tokenizer, device, sparsity_ratio, method, save_client=None):
    ppl_wiki = eval_ppl(args, model, tokenizer, device, dataset="wikitext2")
    ppl_c4 = eval_ppl(args, model, tokenizer, device, dataset="c4")
    ppl_ptb = eval_ppl(args, model, tokenizer, device, dataset="ptb")

    print(f"wikitext2 perplexity {ppl_wiki}")
    print(f"c4 perplexity {ppl_c4}")
    print(f"ptb perplexity {ppl_ptb}")

    return {
        "method": method,
        "client": save_client if save_client is not None else "-",
        "dataset": "wikitext2",
        "actual_sparsity": f"{sparsity_ratio:.4f}",
        "ppl_test": f"{ppl_wiki:.4f}",
    }, {
        "method": method,
        "client": save_client if save_client is not None else "-",
        "dataset": "c4",
        "actual_sparsity": f"{sparsity_ratio:.4f}",
        "ppl_test": f"{ppl_c4:.4f}",
    }, {
        "method": method,
        "client": save_client if save_client is not None else "-",
        "dataset": "ptb",
        "actual_sparsity": f"{sparsity_ratio:.4f}",
        "ppl_test": f"{ppl_ptb:.4f}",
    }, ppl_wiki, ppl_c4, ppl_ptb


def print_mean_and_fairness(
    tag,
    sp_list,
    wiki_list,
    c4_list,
    ptb_list,
):

    mean_sp = float(np.mean(sp_list))
    mean_wiki = float(np.mean(wiki_list))
    mean_c4 = float(np.mean(c4_list))
    mean_ptb = float(np.mean(ptb_list))

    print(f"\n====== {tag} MEAN ======")
    print(f"mean sparsity {mean_sp:.4f}")
    print(f"mean wiki {mean_wiki}")
    print(f"mean c4 {mean_c4}")
    print(f"mean ptb {mean_ptb}")

    print(f"\n====== {tag} FAIRNESS ======")

    print(f"worst wiki {float(np.max(wiki_list))}")
    print(f"worst c4 {float(np.max(c4_list))}")
    print(f"worst ptb {float(np.max(ptb_list))}")

    print(
        f"wiki gap "
        f"{float(np.max(wiki_list)-np.min(wiki_list))}"
    )

    print(
        f"c4 gap "
        f"{float(np.max(c4_list)-np.min(c4_list))}"
    )

    print(
        f"ptb gap "
        f"{float(np.max(ptb_list)-np.min(ptb_list))}"
    )

    return (
        mean_sp,
        mean_wiki,
        mean_c4,
        mean_ptb,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)

    parser.add_argument("--sparsity_ratio", type=float, default=0.0)
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured"])
    parser.add_argument("--prune_method", type=str, default="wanda", choices=["wanda"])
    parser.add_argument("--use_variant", action="store_true")

    parser.add_argument("--cache_dir", type=str, default="llm_weights")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--save_model", type=str, default=None)

    parser.add_argument(
        "--save_pruned_model",
        action="store_true",
        help="save pruned model for lm_eval downstream task evaluation",
    )

    parser.add_argument(
        "--save_model_dir",
        type=str,
        default=os.environ.get(
            "FEDPRLLM_CHECKPOINT_DIR", "/root/autodl-tmp/FedPrLLM/checkpoints"
        ),
        help="directory to save pruned models",
    )

    parser.add_argument(
        "--save_domain",
        type=str,
        default=None,
        help="only save one domain/client, e.g. gsm8k/humaneval/squad/agnews; use all to save all",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="fedprllm",
        choices=[
            "centralized", # 普通集中式 Wanda 剪枝 baseline
            "client_wanda",
            "fedprllm",
            "role_moe_soft_clientaware",
            "client_moe_pruner",
        ],
    )

    parser.add_argument("--client_num", type=int, default=8)
    parser.add_argument("--comparison_group", type=str, default="layer", choices=["layer", "row", "column"])
    parser.add_argument("--weight_scaling", action="store_true")

    parser.add_argument(
        "--prune_scope",
        type=str,
        default="all",
        choices=["all", "moe_only"],
    )

    parser.add_argument(
        "--split_mode",
        type=str,
        default="iid",
        choices=["iid", "block"],
        help="iid: shuffle then split; block: contiguous split for non-iid simulation",
    )

    parser.add_argument(
        "--role_map_path",
        type=str,
        default=os.environ.get(
            "FEDPRLLM_ROLE_MAP_PATH", "/root/autodl-tmp/FedPrLLM/logs/expert_roles.json"
        ),
    )
    parser.add_argument(
        "--expert_sparsity_log_dir",
        type=str,
        default="logs/expert_sparsity_maps",
        help=(
            "directory for per-client realized expert sparsity reports used "
            "to generate Figure 4; pass an empty string to disable"
        ),
    )
    parser.add_argument(
        "--domain_clients",
        type=str,
        default=None,
        help="comma-separated domain clients",
    )

    parser.add_argument(
        "--nsamples_per_client",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--clients_per_domain",
        type=int,
        default=1,
        help="number of clients generated for each domain",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        raise NotImplementedError("Only unstructured supported.")

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()

    if args.seqlen is not None:
        print(f"Override model seqlen: {model.seqlen} -> {args.seqlen}")
        model.seqlen = args.seqlen

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = pick_input_device(model)
    print("use input device:", device)

    print("loading calibration data")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print(f"dataset loading complete, actual samples: {len(dataloader)}")

    # ---------------- centralized：集中式 MoE-only Wanda baseline，所有校准数据合在一起剪一个全局模型 ----------------
    if args.method == "centralized":
        print("centralized pruning")

        if args.sparsity_ratio != 0:
            prune_wanda(args, model, tokenizer, dataloader, device, prune_n=prune_n, prune_m=prune_m)

        sparsity_ratio = check_sparsity(model, scope=args.prune_scope)
        print(f"sparsity sanity check ({args.prune_scope}) {sparsity_ratio:.4f}")

        r1, r2, r3, _, _, _ = eval_and_log(
            args, model, tokenizer, device, sparsity_ratio, args.method
        )

        if args.save:
            save_tabular_log(
                os.path.join(args.save, f"log_{args.method}_ns_{args.nsamples}.txt"),
                ["method", "client", "dataset", "actual_sparsity", "ppl_test"],
                [r1, r2, r3],
            )

        if args.save_model:
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)

        return

    # ---------------- client_wanda：每个 domain client 用自己的本地数据做 Wanda，不使用 expert role ----------------
    if args.method == "client_wanda":
        from lib.domain_data import get_domain_client_loaders

        print("Client-Wanda pruning: local domain data + Wanda, no expert role")

        if args.domain_clients is None:
            raise ValueError("--domain_clients must be provided for client_wanda")

        domains = [
            x.strip()
            for x in args.domain_clients.split(",")
            if len(x.strip()) > 0
        ]

        with torch.no_grad():
            dense_state = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            }

        local_dataloaders, domain_names = get_domain_client_loaders(
            domains=domains,
            nsamples_per_client=args.nsamples_per_client,
            seed=args.seed,
            seqlen=model.seqlen,
            tokenizer=tokenizer,
            clients_per_domain=args.clients_per_domain,
        )

        rows = []
        sp_list = []
        wiki_list = []
        c4_list = []
        ptb_list = []

        for c, local_dl in enumerate(local_dataloaders):
            domain = domain_names[c]

            print(f"\n[Client-Wanda Client {c}] {domain}")

            model.load_state_dict(dense_state, strict=True)
            torch.cuda.empty_cache()

            if args.sparsity_ratio != 0:
                prune_wanda(
                    args,
                    model,
                    tokenizer,
                    local_dl,
                    device,
                    prune_n=prune_n,
                    prune_m=prune_m,
                )

            sparsity_ratio = check_sparsity(
                model,
                scope=args.prune_scope,
            )

            print(f"sparsity sanity check ({args.prune_scope}) {sparsity_ratio:.4f}")

            r1, r2, r3, pw, pc4, pptb = eval_and_log(
                args,
                model,
                tokenizer,
                device,
                sparsity_ratio,
                "client_wanda",
                domain,
            )

            rows.extend([r1, r2, r3])

            save_model_for_lm_eval(
                args=args,
                model=model,
                tokenizer=tokenizer,
                method="client_wanda",
                domain=domain,
            )

            sp_list.append(sparsity_ratio)
            wiki_list.append(pw)
            c4_list.append(pc4)
            ptb_list.append(pptb)

        print("\n====== CLIENT-WANDA MEAN ======")
        print(f"mean sparsity {float(np.mean(sp_list)):.4f}")
        print(f"mean wiki {float(np.mean(wiki_list))}")
        print(f"mean c4 {float(np.mean(c4_list))}")
        print(f"mean ptb {float(np.mean(ptb_list))}")

        print("\n====== CLIENT-WANDA FAIRNESS ======")
        print(f"worst wiki {float(np.max(wiki_list))}")
        print(f"worst c4 {float(np.max(c4_list))}")
        print(f"worst ptb {float(np.max(ptb_list))}")
        print(f"wiki gap {float(np.max(wiki_list) - np.min(wiki_list))}")
        print(f"c4 gap {float(np.max(c4_list) - np.min(c4_list))}")
        print(f"ptb gap {float(np.max(ptb_list) - np.min(ptb_list))}")

        if args.save:
            os.makedirs(args.save, exist_ok=True)
            save_tabular_log(
                os.path.join(
                    args.save,
                    f"log_{args.method}_ns_{args.nsamples}.txt",
                ),
                ["method", "client", "dataset", "actual_sparsity", "ppl_test"],
                rows,
            )

        return
# ---------------- client_moe_pruner：每个 domain client 用自己的本地数据做 Wanda，不使用 expert role ----------------
    if args.method == "client_moe_pruner":
            from lib.domain_data import get_domain_client_loaders

            print("Client-Wanda pruning: local domain data + Wanda, no expert role")

            if args.domain_clients is None:
                raise ValueError("--domain_clients must be provided for client_moe_pruner")

            domains = [
                x.strip()
                for x in args.domain_clients.split(",")
                if len(x.strip()) > 0
            ]

            with torch.no_grad():
                dense_state = {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                }

            local_dataloaders, domain_names = get_domain_client_loaders(
                domains=domains,
                nsamples_per_client=args.nsamples_per_client,
                seed=args.seed,
                seqlen=model.seqlen,
                tokenizer=tokenizer,
                clients_per_domain=args.clients_per_domain,
            )

            rows = []
            sp_list = []
            wiki_list = []
            c4_list = []
            ptb_list = []

            for c, local_dl in enumerate(local_dataloaders):
                domain = domain_names[c]

                print(f"\n[Client-Wanda Client {c}] {domain}")

                model.load_state_dict(dense_state, strict=True)
                torch.cuda.empty_cache()

                if args.sparsity_ratio != 0:
                    from lib.client_moe_pruner import (
                        prune_client_moe_pruner,
                    )

                    report = prune_client_moe_pruner(
                        model=model,
                        dataloader=local_dl,
                        sparsity_ratio=args.sparsity_ratio,
                        device=device,
                    )

                    print(report)

                sparsity_ratio = check_sparsity(
                    model,
                    scope=args.prune_scope,
                )

                print(f"sparsity sanity check ({args.prune_scope}) {sparsity_ratio:.4f}")

                r1, r2, r3, pw, pc4, pptb = eval_and_log(
                    args,
                    model,
                    tokenizer,
                    device,
                    sparsity_ratio,
                    "client_moe_pruner",
                    domain,
                )

                rows.extend([r1, r2, r3])

                save_model_for_lm_eval(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    method="client_moe_pruner",
                    domain=domain,
                )

                sp_list.append(sparsity_ratio)
                wiki_list.append(pw)
                c4_list.append(pc4)
                ptb_list.append(pptb)

            print("\n====== client_moe_pruner MEAN ======")
            print(f"mean sparsity {float(np.mean(sp_list)):.4f}")
            print(f"mean wiki {float(np.mean(wiki_list))}")
            print(f"mean c4 {float(np.mean(c4_list))}")
            print(f"mean ptb {float(np.mean(ptb_list))}")

            print("\n====== client_moe_pruner FAIRNESS ======")
            print(f"worst wiki {float(np.max(wiki_list))}")
            print(f"worst c4 {float(np.max(c4_list))}")
            print(f"worst ptb {float(np.max(ptb_list))}")
            print(f"wiki gap {float(np.max(wiki_list) - np.min(wiki_list))}")
            print(f"c4 gap {float(np.max(c4_list) - np.min(c4_list))}")
            print(f"ptb gap {float(np.max(ptb_list) - np.min(ptb_list))}")

            if args.save:
                os.makedirs(args.save, exist_ok=True)
                save_tabular_log(
                    os.path.join(
                        args.save,
                        f"log_{args.method}_ns_{args.nsamples}.txt",
                    ),
                    ["method", "client", "dataset", "actual_sparsity", "ppl_test"],
                    rows,
                )

            return


    # ---------------- fedprllm：联邦剪枝 baseline，客户端上传剪枝 mask，服务器聚合后得到一个全局稀疏模型 ----------------
    if args.method == "fedprllm":
        print("fedprllm one-shot pruning")

        with torch.no_grad():
            dense_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        local_dataloaders = split_dataloader_for_clients(
            dataloader,
            args.client_num,
            seed=args.seed,
            split_mode=args.split_mode,
        )

        client_num = len(local_dataloaders)
        print(f"divide data into {client_num} clients, split_mode={args.split_mode}")

        global_masks = {}
        layers = get_layers(model)

        for i, layer in enumerate(layers):
            subset = find_layers(layer, scope=args.prune_scope)
            for name, m in subset.items():
                unique_name = f"layer_{i}.{name}"
                global_masks[unique_name] = torch.zeros(
                    m.weight.data.shape,
                    dtype=torch.int8,
                    device=m.weight.data.device,
                )

        if len(global_masks) == 0:
            raise RuntimeError(f"global_masks is empty under scope={args.prune_scope}")

        for c in range(client_num):
            print(f"use client {c + 1}/{client_num} to update global masks")

            prune_wanda_and_update_global_mask(
                args,
                model,
                global_masks,
                tokenizer,
                local_dataloaders[c],
                device,
                prune_n=prune_n,
                prune_m=prune_m,
            )

            model.load_state_dict(dense_state, strict=True)
            torch.cuda.empty_cache()

        print("server pruning")

        if args.comparison_group == "layer":
            prune_fedprllm_layer(args, model, global_masks, device, prune_n=prune_n, prune_m=prune_m)
        elif args.comparison_group == "row":
            prune_fedprllm_row_or_column(args, model, global_masks, device, dim=1, prune_n=prune_n, prune_m=prune_m)
        elif args.comparison_group == "column":
            prune_fedprllm_row_or_column(args, model, global_masks, device, dim=0, prune_n=prune_n, prune_m=prune_m)

        sparsity_ratio = check_sparsity(model, scope=args.prune_scope)
        print(f"sparsity sanity check ({args.prune_scope}) {sparsity_ratio:.4f}")

        r1, r2, r3, _, _, _ = eval_and_log(
            args, model, tokenizer, device, sparsity_ratio, args.method
        )

        save_model_for_lm_eval(
            args=args,
            model=model,
            tokenizer=tokenizer,
            method=args.method,
            domain=None,
        )

        if args.save:
            save_tabular_log(
                os.path.join(args.save, f"log_{args.method}_ns_{args.nsamples}.txt"),
                ["method", "client", "dataset", "actual_sparsity", "ppl_test"],
                [r1, r2, r3],
            )

        return





    # ---------------- role_moe_soft_clientaware：专家角色 + 客户端 Wanda 重要性 ----------------
    if args.method == "role_moe_soft_clientaware":
        from lib.domain_data import get_domain_client_loaders
        from lib.role_soft_client_pruner import apply_clientaware_role_soft_pruning

        print("Client-Aware Soft Role-MoE")

        if args.domain_clients is None:
            raise ValueError("--domain_clients must be provided for role_moe_soft_clientaware")

        domains = [
            x.strip()
            for x in args.domain_clients.split(",")
            if len(x.strip()) > 0
        ]

        with torch.no_grad():
            dense_state = {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            }

        local_dataloaders, domain_names = get_domain_client_loaders(
            domains=domains,
            nsamples_per_client=args.nsamples_per_client,
            seed=args.seed,
            seqlen=model.seqlen,
            tokenizer=tokenizer,
            clients_per_domain=args.clients_per_domain,
        )

        rows = []
        sp_list = []
        wiki_list = []
        c4_list = []
        ptb_list = []

        for c, local_dl in enumerate(local_dataloaders):
            domain = domain_names[c]
            base_domain = domain.split("_")[0]

            print(f"\n[Client-Aware Soft Role-MoE Client {c}] {domain}")

            model.load_state_dict(dense_state, strict=True)
            torch.cuda.empty_cache()

            client_sparsity = args.sparsity_ratio

            print(
                f"[Client-Aware Soft Role-MoE] domain={domain}, "
                f"base_domain={base_domain}, "
                f"client_sparsity={client_sparsity}"
            )

            sparsity_report_path = None
            if args.expert_sparsity_log_dir:
                safe_domain = str(domain).replace("/", "_").replace("\\", "_")
                sparsity_tag = f"{client_sparsity:.4f}".rstrip("0").rstrip(".")
                sparsity_report_path = os.path.join(
                    args.expert_sparsity_log_dir,
                    f"{safe_domain}_s{sparsity_tag}.json",
                )

            apply_clientaware_role_soft_pruning(
                args=args,
                model=model,
                tokenizer=tokenizer,
                dataloader=local_dl,
                device=device,
                role_path=args.role_map_path,
                domain=base_domain,
                target_sparsity=client_sparsity,
                prune_n=prune_n,
                prune_m=prune_m,
                sparsity_report_path=sparsity_report_path,
            )

            sparsity_ratio = check_sparsity(
                model,
                scope=args.prune_scope,
            )

            r1, r2, r3, pw, pc4, pptb = eval_and_log(
                args,
                model,
                tokenizer,
                device,
                sparsity_ratio,
                "role_moe_soft_clientaware",
                domain,
            )

            rows.extend([r1, r2, r3])

            save_model_for_lm_eval(
                args=args,
                model=model,
                tokenizer=tokenizer,
                method="role_moe_soft_clientaware",
                domain=domain,
            )

            sp_list.append(sparsity_ratio)
            wiki_list.append(pw)
            c4_list.append(pc4)
            ptb_list.append(pptb)

        print("\n====== CLIENT-AWARE SOFT ROLE-MOE MEAN ======")
        print(f"mean sparsity {float(np.mean(sp_list)):.4f}")
        print(f"mean wiki {float(np.mean(wiki_list))}")
        print(f"mean c4 {float(np.mean(c4_list))}")
        print(f"mean ptb {float(np.mean(ptb_list))}")

        print("\n====== CLIENT-AWARE SOFT ROLE-MOE FAIRNESS ======")
        print(f"worst wiki {float(np.max(wiki_list))}")
        print(f"worst c4 {float(np.max(c4_list))}")
        print(f"worst ptb {float(np.max(ptb_list))}")

        print(f"wiki gap {float(np.max(wiki_list) - np.min(wiki_list))}")
        print(f"c4 gap {float(np.max(c4_list) - np.min(c4_list))}")
        print(f"ptb gap {float(np.max(ptb_list) - np.min(ptb_list))}")

        if args.save:
            os.makedirs(args.save, exist_ok=True)
            save_tabular_log(
                os.path.join(
                    args.save,
                    f"log_{args.method}_ns_{args.nsamples}.txt",
                ),
                ["method", "client", "dataset", "actual_sparsity", "ppl_test"],
                rows,
            )

        return


    raise NotImplementedError(f"method {args.method} not implemented")


if __name__ == "__main__":
    main()
