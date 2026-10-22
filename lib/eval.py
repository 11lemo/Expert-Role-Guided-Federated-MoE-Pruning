# lib/eval.py
import torch
import torch.nn as nn
import fnmatch

from .data import get_loaders


def eval_ppl(
    args,
    model,
    tokenizer,
    device=torch.device("cuda:0"),
    dataset="wikitext2",
    max_eval_nsamples=128,
    eval_seed=0,
):
    """
    Evaluate perplexity on a dataset (wikitext2 / c4 / ptb).

    max_eval_nsamples:
      - limits the number of evaluation blocks (each block has length=model.seqlen)
      - makes evaluation stable and comparable across runs
    """
    print(f"evaluating on {dataset}")

    # 注意：这里 get_loaders 返回 (trainloader, testloader)
    # 我们只用 testloader 做 PPL
    _, testloader = get_loaders(
        dataset,
        nsamples=getattr(args, "nsamples", 128),  # 不影响 testloader 的构建逻辑，但保持接口一致
        seed=eval_seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(
            model,
            testloader,
            bs=1,
            device=device,
            max_eval_nsamples=max_eval_nsamples,
        )
    return ppl_test


def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None, max_eval_nsamples=None):
    """
    Evaluate perplexity on a trainloader produced by get_loaders (list of (inp, tar)).
    """
    nsamples = len(trainloader)
    if max_eval_nsamples is not None:
        nsamples = min(nsamples, int(max_eval_nsamples))

    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i + bs, nsamples)

        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

        neg_log_likelihood = loss.float() * model.seqlen * (j - i)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    torch.cuda.empty_cache()
    return ppl.item()


def eval_ppl_wikitext(model, testenc, bs=1, device=None, max_eval_nsamples=None):
    """
    Evaluate perplexity on a testenc returned by get_loaders (TokenizerWrapper or tokenizer output).

    We treat testenc.input_ids as a long token stream, chunk into blocks of length model.seqlen.
    """
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    # 关键：固定评估块数，避免 nsamples 每次变化导致可比性差
    if max_eval_nsamples is not None:
        nsamples = min(nsamples, int(max_eval_nsamples))

    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i + bs, nsamples)

        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)

        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )

        neg_log_likelihood = loss.float() * model.seqlen * (j - i)
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    torch.cuda.empty_cache()
    return ppl.item()


def eval_zero_shot(
    model_name,
    model,
    tokenizer,
    task_list=("boolq", "rte", "hellaswag", "winogrande", "arc_challenge", "arc_easy", "openbookqa"),
    num_fewshot=0,
    use_accelerate=False,
    add_special_tokens=False,
):
    from lm_eval import tasks, evaluator

    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)

    task_names = pattern_match(task_list, tasks.ALL_TASKS)

    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"

    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
    )
    return results
