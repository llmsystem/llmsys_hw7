# Assignment 7: Introduction to RLHF

**CMU 11-868 LLM Systems — Fall 2026**

This README contains the updated instructions for Fall 2026. Where it differs from the archived assignment webpage, follow this README.

## Overview

Implement a small reinforcement learning from human feedback (RLHF) pipeline: train a reward model on human preference pairs, then optimize GPT-2 using proximal policy optimization (PPO). The code separates rollout generation from policy updates, taking inspiration from [verl](https://verl.readthedocs.io/en/latest/) and [HybridFlow](https://arxiv.org/abs/2409.19256). It is a single-device PyTorch teaching implementation; it does not use the verl package, distributed workers, or HybridFlow's execution engine.

The goal is to understand reward learning, token-level advantages, and PPO, and to examine the limitations of optimizing a learned reward. A small reward model and a base GPT-2 model do not guarantee helpful or harmless responses.

## Learning objectives and reading

- Train and validate a pairwise preference reward model.
- Implement generalized advantage estimation (GAE), a clipped PPO objective, and an entropy bonus.
- Handle response masks, EOS tokens, and a frozen reference policy correctly.
- Compare held-out reward scores and generated text, including evidence of reward exploitation.

Read the [RLHF introduction](https://huggingface.co/blog/rlhf), [PPO paper](https://arxiv.org/abs/1707.06347), and [GAE paper](https://arxiv.org/abs/1506.02438). The verl documentation and HybridFlow paper provide systems context; implementing their distributed architecture is outside this assignment.

## Setup

Use Python 3.10–3.13. A CUDA GPU is recommended for the full training run. CPU is sufficient for the offline unit tests, but full training is much slower. The default pipeline keeps separate policy, reference, and value backbones in memory; lower the rollout batch size and sequence length if memory is limited. Mixed precision and distributed training are not implemented.

```bash
git clone https://github.com/llmsystem/llmsys_hw7.git
cd llmsys_hw7
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run all commands from the repository root. Initial model and dataset downloads require network access and sufficient local disk space.

## Prepare data

```bash
python scripts/prepare_data.py --dataset Anthropic/hh-rlhf --output_dir data --max_samples 10000 --seed 42
python scripts/prepare_data.py --output_dir data --validate
```

The [Anthropic HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) contains chosen/rejected conversations. Preparation samples from both `helpful-base` and `harmless-base`, requires a shared conversation prefix, and keeps final responses between 10 and 500 characters. It retains one pair per unique prompt, shuffles deterministically, and splits examples into 80% training, 10% validation, and 10% test data (with rounding). `--max_samples` limits the total number of accepted examples before splitting.

| File | Purpose |
| --- | --- |
| `data/preference_data.json` | Reward model training pairs |
| `data/preference_data_val.json` | Reward model validation pairs |
| `data/preference_data_test.json` | Reward model held-out test pairs |
| `data/train_prompts.json` | Policy rollout prompts from the training split |
| `data/val_prompts.json` | Policy validation and checkpoint selection |
| `data/eval_prompts.json` | Final policy evaluation on the test split |

Training scripts require prepared data. Download errors stop preparation rather than silently substituting artificial examples. Do not use toy data or the test prompts for checkpoint selection or hyperparameter tuning. Reward tokenization truncates from the left to retain the final response; long conversation prefixes may be dropped.

## Problem 1: Reward model (40 points)

Complete `RewardModelTrainer.compute_loss` in `src/reward_model.py` (`ASSIGN7_1`). The provided model uses a pretrained DistilBERT encoder, its first-token representation, and a scalar reward head.

1. Compute one reward per chosen and rejected conversation.
2. Use the provided `MarginRankingLoss` with margin 1 and positive targets, so chosen responses should receive higher scores.
3. Return `(loss, chosen_rewards, rejected_rewards)`: a scalar mean loss and two tensors of shape `[batch_size]`. Keep the computation differentiable.

```bash
python scripts/train_reward_model.py
```

The script selects `outputs/reward_model/best_reward_model.pt` by validation preference accuracy, saves the final model separately, and writes metrics and `logs/reward_model_training_curves.png`. Preference accuracy is the fraction of pairs with `chosen_reward > rejected_reward`; ties count as incorrect. Report validation and test accuracy separately. The reward-model experiment must achieve at least 60% validation preference accuracy. Include supporting diagnostics if your run falls below this requirement.

## Problem 2: PPO training (40 points)

Complete the three marked sections in `src/rlhf_trainer.py`:

- `_compute_gae` (`ASSIGN7_2_1`): backward GAE with `ppo_gamma` and `ppo_gae_lambda`.
- `_train_step_custom` (`ASSIGN7_2_2`): clipped surrogate policy loss and entropy bonus.
- `_compute_entropy` (`ASSIGN7_2_3`): categorical entropy from logits, averaged over supplied valid response states.

### Token alignment and GAE contract

For token IDs `x[0:L]`, the hidden state/logits at position `t` predict token `x[t+1]`. If the padded prompt width is `P`, response actions use state positions `P-1` through `L-2`. `_compute_gae` receives full-width values and a **response-state mask**, not the ordinary input attention mask. The caller shifts the generated-token mask one position to the left. The first generated EOS is a valid action; all subsequent padding is excluded.

Put the scalar sequence reward at the last valid response state; other step rewards are zero. Traverse backward with:

```text
delta[t] = reward[t] + gamma * next_valid * value[t+1] - value[t]
advantage[t] = delta[t] + gamma * lambda * next_valid * advantage[t+1]
return[t] = advantage[t] + value[t]
```

Use zero next value/advantage after the final valid state. For this assignment, reaching the generation length limit is also terminal. Return full-width tensors with zeros outside the response-state mask. Do not propagate through padding or prompt states.

### PPO contract

Use `ratio = exp(new_log_prob - old_log_prob)` and minimize the negative mean of the smaller unclipped/clipped advantage surrogate. Compute entropy from `scores[new_log_probs_mask]`, excluding padding and prompt tokens. The surrounding code adds the entropy bonus, optional reference-policy KL penalty, and value regression. Rollout log probabilities, advantages, and returns must remain detached during updates. The reference policy stays frozen.

Training rollouts sample from the unmodified policy distribution (temperature 1, top-p 1, top-k disabled), matching the log probabilities used by PPO. Evaluation may use temperature/top-p sampling. `rollout_max_length` is the total prompt-plus-response token limit; prompts are truncated to half that length. Each rollout batch is updated as a whole for `verl.ppo_epochs`; minibatching, adaptive KL, and gradient accumulation are not implemented. Related legacy configuration fields are inactive.

```bash
python scripts/run_rlhf.py --model_name gpt2
```

The script uses validation prompts to select `outputs/rlhf_model/best_rlhf_model`, saves `final_rlhf_model` and per-epoch training checkpoints, and writes `plots/rlhf_training_curves.png`. Full training checkpoints include optimizer, reference, and value weights and can be substantially larger than inference model directories.

## Problem 3: Evaluation and analysis (20 points)

```bash
python scripts/evaluate.py --base_model gpt2 \
  --rlhf_model outputs/rlhf_model/best_rlhf_model \
  --reward_model outputs/reward_model/best_reward_model.pt \
  --num_samples 3 --max_prompts 100
```

`--rlhf_model` accepts the actual directory containing a Hugging Face model and tokenizer. `--reward_model` accepts the exact reward checkpoint file. Without these arguments, evaluation searches for the best checkpoint and then the final checkpoint in the configured output directories. Without a policy checkpoint, it evaluates only the base model.

Submit a short report with:

- Reward model validation/test accuracy and training settings, including seed and dataset size.
- PPO curves and `plots/reward_comparison.png`, with mean/std reward and response length for both policies on the same held-out prompts and generation settings.
- At least five side-by-side prompt/response examples and a qualitative discussion of helpfulness, harmlessness, coherence, and reward exploitation.
- An explanation of any missing reward improvement, instability, truncation effects, or mismatch between learned reward and text quality.

Evaluation writes sampled outputs and metrics to `evaluation_results/`. Each model generates independent samples; their reward differences are not human preference win rates. Reward units have no universal zero or scale. Higher reward or a shifted histogram alone does not demonstrate better responses. Incoherent high-reward generations should be investigated and discussed, not treated as successful alignment.

## Configuration

All three training/evaluation commands accept `--config path/to/config.yaml`. Explicit command-line arguments override YAML values; omitted arguments preserve them. Configuration sections are defined in `src/config.py`. For example:

```yaml
training:
  reward_num_epochs: 3
  ppo_num_epochs: 2
  ppo_learning_rate: 1.0e-6
verl:
  rollout_batch_size: 2
  rollout_max_length: 128
  ppo_epochs: 4
experiment:
  seed: 42
```

For PPO, `training.ppo_kl_penalty` controls the reference KL coefficient (default 0), `training.ppo_value_coef` weights value regression, and `training.ppo_entropy_coef` controls entropy regularization. `verl.ppo_clip_eps` controls PPO clipping. Record the effective configuration and package versions with your report.

## Tests

```bash
python -m pytest tests/ -v
python scripts/check_implementation.py
```

Tests use small randomly initialized local models and require no model downloads, GPU, or checkpoints. They check numerical contracts and framework behavior, not the 60% accuracy requirement or full-scale convergence. Tests that exercise unfinished student sections raise `NotImplementedError` until those sections are completed; this is expected in the starter repository. Complete the marked sections and run the tests before submitting.

`scripts/check_implementation.py` is the student self-check tool. It runs the public tests with Hugging Face downloads disabled and writes per-test feedback to `self_check.json` and `self_check.log`. It does not assign course points. Exit codes are 0 for passing all tests, 1 for failed/skipped tests, and 2 for setup errors or timeout (default 300 seconds; change with `--timeout`). The starter code will report failed tests until you implement its TODOs.

Staff use a separate, unpublished grading tool and their own tests, along with the submitted experiment evidence and report. Student self-check results are feedback, not an official grade.

## Submission and grading

| Problem | Points | Evidence |
| --- | --- | --- |
| Reward model | 40 = 20 automatic + 20 TA review | Ranking-loss correctness; validation/test accuracy, training settings, curves and diagnostics |
| PPO | 40 = 20 automatic + 20 TA review | GAE/PPO/entropy correctness; reproducible training curves and analysis |
| Evaluation | 20 TA review | Held-out comparison, saved samples/configurations, and substantive analysis |

The assignment has **40 automatic points**: reward loss 20, GAE 10, PPO 5, and entropy 5. Staff run the unpublished grader against submitted source; local self-check results do not determine the official score. The other **60 points** cover the experiments, evidence, and analysis. Passing the public tests does not automatically earn the manual-review points.

Reward improvement is an experimental outcome to analyze, not a required fixed transition such as −0.5 to +0.5. Correctness and an honest explanation matter when reward does not improve. Do not fabricate or selectively omit failed runs.

### Build the submission

**Do not upload model weights, training checkpoints, datasets, or virtual environments.** Keep the selected reward and policy checkpoints locally; staff may request them separately if needed. The regular submission is limited to **50 MiB compressed and uncompressed**, with a **10 MiB limit per file**.

After completing all three experiments, save your report as `report.pdf` or `report.md` and put the exact data preparation, training, and evaluation commands in `commands.txt`, including any overrides. The training/evaluation scripts automatically save the effective configurations as `logs/reward_model_config.yaml`, `logs/rlhf_config.yaml`, and `logs/evaluation_config.yaml`. These include the seed and resolved command-line overrides. Use the default output directories for submission.

```bash
python scripts/submission.py --andrew-id YOUR_ANDREW_ID \
  --report report.pdf --commands commands.txt
```

The script creates `assignment7_YOUR_ANDREW_ID.zip`. Missing experiment evidence stops packaging and identifies the missing file. The script also runs the public student self-check and includes its JSON feedback. Failed implementation tests still allow submission for partial credit; setup errors stop packaging. These checks do not assign course points. The script never deletes local checkpoints or overwrites an existing archive; use `--output another_name.zip` for a new archive.

The archive includes only:

- Python source under `src/`, `scripts/`, and `tests/`, plus the README and requirements.
- Your report and exact commands, generated `package_versions.txt`, and student `self_check.json` feedback.
- Small metrics, configurations, plots, and sample outputs from `logs/`, `plots/`, and `evaluation_results/`.
- A `submission_manifest.json` recording your Andrew ID, file sizes, and SHA-256 hashes.

Required experiment evidence is:

```text
logs/reward_model_metrics.json
logs/rlhf_training_metrics.json
logs/reward_model_config.yaml
logs/rlhf_config.yaml
logs/evaluation_config.yaml
logs/reward_model_training_curves.png
plots/rlhf_training_curves.png
plots/reward_comparison.png
evaluation_results/detailed_comparison.json
evaluation_results/base_model_outputs.json
evaluation_results/rlhf_model_outputs.json
```

`outputs/`, `data/`, hidden files, caches, model weight formats, and environment folders are excluded. Symlinks are rejected. If a plot or JSON file exceeds the limit, reduce image resolution or the number of saved evaluation samples and regenerate the archive. Submit the generated ZIP, not the repository folder or a manually zipped `outputs/` directory.
