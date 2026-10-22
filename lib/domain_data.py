
import os
import json
import random
import torch


def _make_sample(text, tokenizer, seqlen):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seqlen,
    )

    input_ids = enc.input_ids

    if input_ids.shape[1] < seqlen:
        pad_len = seqlen - input_ids.shape[1]
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        pad = torch.full((1, pad_len), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, pad], dim=1)
    else:
        input_ids = input_ids[:, :seqlen]

    target = input_ids.clone()
    target[:, :-1] = -100

    return input_ids, target


def get_domain_client_loaders(
    domains,
    nsamples_per_client,
    seed,
    seqlen,
    tokenizer,
    clients_per_domain=1,
    domain_dir=None,
):
    """
    Generate real domain clients.

    Example:
      domains = ["gsm8k", "humaneval", "squad", "agnews"]
      clients_per_domain = 16

    Return:
      local_dataloaders:
        [gsm8k_0_loader, ..., gsm8k_15_loader,
         humaneval_0_loader, ..., agnews_15_loader]

      domain_names:
        ["gsm8k_0", ..., "agnews_15"]
    """

    if domain_dir is None:
        domain_dir = os.path.join(
            os.environ.get("FEDPRLLM_DATA_ROOT", "/root/autodl-tmp/FedPrLLM"),
            "domain_clients",
        )

    rng = random.Random(seed)

    local_dataloaders = []
    domain_names = []

    for domain in domains:
        path = os.path.join(domain_dir, f"{domain}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Domain file not found: {path}")

        texts = json.load(open(path, "r", encoding="utf-8"))
        texts = [str(x).strip() for x in texts if len(str(x).strip()) >= 10]

        if len(texts) == 0:
            raise RuntimeError(f"No valid samples found for domain={domain}")

        rng.shuffle(texts)

        need = nsamples_per_client * clients_per_domain

        # 如果某个 domain 样本不足，就循环补齐，保证每个 client 有足够样本
        if len(texts) < need:
            repeat = (need + len(texts) - 1) // len(texts)
            texts = (texts * repeat)[:need]
        else:
            texts = texts[:need]

        for cid in range(clients_per_domain):
            start = cid * nsamples_per_client
            end = (cid + 1) * nsamples_per_client
            client_texts = texts[start:end]

            loader = []
            for text in client_texts:
                loader.append(_make_sample(text, tokenizer, seqlen))

            client_name = f"{domain}_{cid}"

            local_dataloaders.append(loader)
            domain_names.append(client_name)

            print(f"[Domain Client] {client_name}: {len(loader)} samples")

    return local_dataloaders, domain_names
