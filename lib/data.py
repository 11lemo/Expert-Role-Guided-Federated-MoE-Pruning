# lib/data.py
# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import os
import numpy as np
import random
import torch
from datasets import load_dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    set_seed(seed)
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    ids = trainenc.input_ids
    L = ids.shape[1]
    max_start = L - seqlen - 1
    if max_start <= 0:
        raise ValueError(f"wikitext2 tokens too short for seqlen={seqlen}, L={L}. Lower --seqlen.")

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, max_start)
        j = i + seqlen
        inp = ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, tokenizer):
    set_seed(seed)
    local_ptb_dir = os.path.join(
        os.environ.get("FEDPRLLM_DATA_ROOT", "/root/autodl-tmp/FedPrLLM"),
        "local_ptb_data",
    )

    traindata = load_dataset("text", data_files=f"{local_ptb_dir}/ptb.train.txt", split="train")
    testdata = load_dataset("text", data_files=f"{local_ptb_dir}/ptb.test.txt", split="train")

    trainenc = tokenizer("\n".join(traindata["text"]), return_tensors="pt")
    ids = trainenc.input_ids
    L = ids.shape[1]
    max_start = L - seqlen - 1
    if max_start <= 0:
        raise ValueError(f"PTB tokens too short for seqlen={seqlen}, L={L}. Lower --seqlen.")

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, max_start)
        j = i + seqlen
        inp = ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    testenc = tokenizer("\n".join(testdata["text"]), return_tensors="pt")
    testenc = testenc.input_ids[:, :(256 * seqlen)]
    testenc = TokenizerWrapper(testenc)

    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    set_seed(seed)
    local_data_dir = os.path.join(
        os.environ.get("FEDPRLLM_DATA_ROOT", "/root/autodl-tmp/FedPrLLM"),
        "local_c4_data",
    )

    train_files = sorted([
        os.path.join(local_data_dir, f)
        for f in os.listdir(local_data_dir)
        if f.startswith("c4-train") and f.endswith(".json.gz")
    ])
    val_files = sorted([
        os.path.join(local_data_dir, f)
        for f in os.listdir(local_data_dir)
        if f.startswith("c4-validation") and f.endswith(".json.gz")
    ])

    if len(train_files) == 0:
        raise FileNotFoundError(f"No c4-train*.json.gz found in {local_data_dir}")
    if len(val_files) == 0:
        raise FileNotFoundError(f"No c4-validation*.json.gz found in {local_data_dir}")

    traindata = load_dataset("json", data_files=train_files, split="train")
    valdata = load_dataset("json", data_files=val_files, split="train")

    # 拼 token 流再切块，保证长度够
    take_n = min(len(traindata), 5000)
    idx = np.random.choice(len(traindata), size=take_n, replace=False)
    texts = [traindata[int(i)]["text"] for i in idx]

    enc = tokenizer("\n\n".join(texts), return_tensors="pt")
    ids = enc.input_ids
    L = ids.shape[1]

    # 不够就加大拼接量
    need = (nsamples + 1) * seqlen + 1
    if L < need:
        take_n2 = min(len(traindata), 12000)
        idx2 = np.random.choice(len(traindata), size=take_n2, replace=False)
        texts2 = [traindata[int(i)]["text"] for i in idx2]
        enc = tokenizer("\n\n".join(texts2), return_tensors="pt")
        ids = enc.input_ids
        L = ids.shape[1]

    max_start = L - seqlen - 1
    if max_start <= 0:
        raise ValueError(f"C4 tokens too short for seqlen={seqlen}, L={L}. Lower --seqlen or increase shards.")

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, max_start)
        j = i + seqlen
        inp = ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # val
    val_take = min(len(valdata), 1100)
    valenc = tokenizer("\n\n".join(valdata[:val_take]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unknown dataset name: {name}")
