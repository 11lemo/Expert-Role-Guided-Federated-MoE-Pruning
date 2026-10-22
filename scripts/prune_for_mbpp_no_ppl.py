"""Rebuild pruned MBPP checkpoints while explicitly skipping PPL evaluation.

This is a time-saving wrapper used for the missing MBPP checkpoint rebuild,
not a replacement for the paper's PPL evaluation.
"""

import main as fed_main
from lib.domain_data import get_domain_client_loaders


original_get_loaders = fed_main.get_loaders


def domain_union_get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    tokenizer=None,
    **kwargs,
):
    if name == "c4":
        loaders, names = get_domain_client_loaders(
            domains=["squad", "gsm8k", "humaneval"],
            nsamples_per_client=64,
            seed=seed,
            seqlen=seqlen,
            tokenizer=tokenizer,
            clients_per_domain=1,
        )
        merged = [sample for loader in loaders for sample in loader]
        print("[MBPP rebuild] global calibration clients:", names)
        print("[MBPP rebuild] total calibration samples:", len(merged))
        return merged, None

    return original_get_loaders(
        name,
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
        **kwargs,
    )


def skip_ppl(args, model, tokenizer, device, sparsity_ratio, method, save_client=None):
    print("[MBPP rebuild] Skip PPL; rebuild the 50% MBPP checkpoint only")

    def row(dataset):
        return {
            "method": method,
            "client": save_client if save_client is not None else "-",
            "dataset": dataset,
            "actual_sparsity": f"{sparsity_ratio:.4f}",
            "ppl_test": "SKIPPED",
        }

    nan = float("nan")
    return row("wikitext2"), row("c4"), row("ptb"), nan, nan, nan


fed_main.get_loaders = domain_union_get_loaders
fed_main.eval_and_log = skip_ppl
fed_main.main()
