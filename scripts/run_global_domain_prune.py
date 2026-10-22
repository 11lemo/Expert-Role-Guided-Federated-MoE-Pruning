"""Use the three domain clients jointly for global Wanda/FedPrLLM baselines.

This mirrors the calibration wrapper used in the experiments (64 samples per
domain, seed supplied by main.py). Run it from the repository root so that
`main` and `lib` are importable.
"""

import main as fed_main
from lib.domain_data import get_domain_client_loaders


original_get_loaders = fed_main.get_loaders


def domain_get_loaders(name, nsamples, seed, seqlen, tokenizer):
    if name != "c4":
        return original_get_loaders(
            name,
            nsamples=nsamples,
            seed=seed,
            seqlen=seqlen,
            tokenizer=tokenizer,
        )

    loaders, names = get_domain_client_loaders(
        domains=["squad", "gsm8k", "humaneval"],
        nsamples_per_client=64,
        seed=seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
        clients_per_domain=1,
    )
    merged = [sample for loader in loaders for sample in loader]
    print("[Global calibration] domains:", names)
    print("[Global calibration] total samples:", len(merged))
    return merged, None


fed_main.get_loaders = domain_get_loaders
fed_main.main()
