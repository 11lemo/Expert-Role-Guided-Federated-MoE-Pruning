# Expert Role-Guided Client-Aware Pruning

Code for **Expert Role-Guided Client-Aware Pruning for Distributed Compression of Mixture-of-Experts Large Language Models**. This repository contains the proposed client-aware method and the Wanda, FedPrLLM, Client-Wanda, and MoE-Pruner comparison branches. Model weights, datasets, and pruned checkpoints are not included.

## Setup

The OLMoE experiments used Python 3.12, PyTorch 2.8.0+cu128, Transformers 4.46.3, Accelerate 1.1.1, one CUDA GPU, sequence length 128, seed 0, and 64 calibration samples per domain. Install these packages and the model's tokenizer dependencies in an environment compatible with your GPU. Set `FEDPRLLM_DATA_ROOT` to a directory containing `domain_clients/{squad,gsm8k,humaneval}.json`, the C4 calibration files in `local_c4_data/`, and PTB files in `local_ptb_data/`. WikiText-2 must be available online or in the Hugging Face cache for PPL evaluation.

Ours also needs an expert role map. The OLMoE experiment's saved map is included at `artifacts/olmoe_expert_roles.json`; pass it through `--role_map_path`. The optional utilities in `scripts/collect_role_router_counts.py` and `scripts/build_reconstructed_role_map.py` can regenerate a compatible map from routing counts.

## Pruning

Run from the repository root. Replace the example paths before use:

```bash
export FEDPRLLM_DATA_ROOT=/path/to/data-root
MODEL=/path/to/OLMoE-1B-7B-0924-Instruct
ROLE_MAP=artifacts/olmoe_expert_roles.json
OUT=/path/to/checkpoints/sparsity_50
COMMON="--model $MODEL --sparsity_ratio 0.5 --prune_scope moe_only --seqlen 128 --nsamples_per_client 64 --clients_per_domain 1 --seed 0 --save_pruned_model --save_model_dir $OUT"

python -m scripts.run_global_domain_prune --method centralized --nsamples 192 --save_model "$OUT/wanda_s0.5" $COMMON
python -m scripts.run_global_domain_prune --method fedprllm --nsamples 192 --client_num 3 --split_mode block $COMMON
python main.py --method client_wanda --domain_clients squad,gsm8k,humaneval --nsamples 64 $COMMON
python main.py --method client_moe_pruner --domain_clients squad,gsm8k,humaneval --nsamples 64 $COMMON
python main.py --method role_moe_soft_clientaware --domain_clients squad,gsm8k,humaneval --role_map_path "$ROLE_MAP" --nsamples 64 $COMMON
```

For 60% sparsity, change `--sparsity_ratio` to `0.6` and use a separate output directory. Global baselines use the union of three 64-sample domain sets (192 samples); client-specific methods use 64 samples per domain. The reported sparsity applies to routed-expert projection matrices, not to every model parameter.

## Evaluation

Use the SQuAD-client checkpoint for BoolQ and RTE (zero-shot accuracy), the GSM8K-client checkpoint for GSM8K (five-shot exact match), and the HumanEval-client checkpoint for MBPP (three-shot Pass@1). The recorded MBPP evaluation used `lm-evaluation-harness` v0.4.8. For example:

```bash
python -m lm_eval --model hf \
  --model_args "pretrained=/path/to/checkpoint,dtype=bfloat16,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 5 --batch_size 1 --device cuda:0
```

MBPP executes model-generated code; run it only in an isolated, disposable environment with the harness's explicit code-execution opt-in. `scripts/prune_for_mbpp_no_ppl.py` skips PPL evaluation and must not be used to report PPL results.

## Attribution

The starting codebase was FedPrLLM. The release keeps only the comparison methods used for this paper; it does not include model weights, datasets, or evaluation-harness source code.
