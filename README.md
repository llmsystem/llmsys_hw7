# Assignment 7: Introduction to RLHF

**CMU 11-868 LLM Systems — Fall 2026**

## Overview

Implement a small reinforcement learning from human feedback (RLHF) pipeline: train a reward model on human preference pairs, then optimize GPT-2 using proximal policy optimization (PPO). The code separates rollout generation from policy updates, taking inspiration from [verl](https://verl.readthedocs.io/en/latest/) and [HybridFlow](https://arxiv.org/abs/2409.19256). It is a single-device PyTorch teaching implementation; it does not use the verl package, distributed workers, or HybridFlow's execution engine.

The goal is to understand reward learning, token-level advantages, and PPO, and to examine the limitations of optimizing a learned reward. A small reward model and a base GPT-2 model do not guarantee helpful or harmless responses.

## Learning objectives and reading

- Train and validate a pairwise preference reward model.
- Implement the provided token-level advantage calculation, clipped PPO objective, and entropy bonus.
- Handle response masks, EOS tokens, and a frozen reference policy correctly.
- Compare held-out reward scores and generated text, including evidence of reward exploitation.

Read the [RLHF introduction](https://huggingface.co/blog/rlhf) and [PPO paper](https://arxiv.org/abs/1707.06347). The verl documentation and HybridFlow paper provide systems context; implementing their distributed architecture is outside this assignment.

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

## Problem 2: RLHF training with a verl-like framework (40 points)

Complete the three marked sections in `src/rlhf_trainer.py`:

- `_compute_gae` (`ASSIGN7_2_1`): the simplified one-step TD advantage calculation described in the code comments. The method name is retained for compatibility; full recursive GAE is not required.
- `_train_step_custom` (`ASSIGN7_2_2`): clipped surrogate policy loss and entropy bonus.
- `_compute_entropy` (`ASSIGN7_2_3`): categorical entropy from logits, averaged over supplied valid response states.

The original scaffold uses one-step TD targets: each valid response state receives its immediate reward plus a discounted next-state value, and its advantage is the target minus its current value. The final response state receives the sequence reward with zero bootstrap value. `ppo_gae_lambda` is unused in this simplified calculation.

The caller supplies a response-state mask: state position `t` predicts token `t+1`. Include the first EOS action, exclude padding and prompt-only states, and return zeros outside the mask. The other two TODOs implement the standard clipped PPO surrogate and categorical entropy. No additional model or algorithm is required.

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

Inspect `plots/reward_comparison.png` and the generated samples in `evaluation_results/` to compare the base and trained policies. Use the same held-out prompts and generation settings for both models.

You may observe incoherent generations receiving high rewards: this can indicate reward exploitation. A higher learned reward alone does not establish that the responses are more helpful or harmless. Differences between independently sampled rewards are not human preference win rates.

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
| Reward model | 40 | Pass the reward-model tests and achieve at least 60% validation preference accuracy |
| RLHF training | 40 | Complete the trainer TODOs, pass the relevant tests, and demonstrate reward improvement in `plots/rlhf_training_curves.png` |
| Evaluation | 20 | Run the base/trained comparison and submit `plots/reward_comparison.png` with its metrics and generated samples |

Reward improvement is measured relative to your own baseline, rather than against a universal reward value such as −0.5 or +0.5. If the reward does not improve, include a brief explanation with the results so staff can assess the run.

### Build the submission

**Do not upload model weights, training checkpoints, datasets, or virtual environments.** Keep the selected reward and policy checkpoints locally; staff may request them separately if needed. The regular submission is limited to **50 MiB compressed and uncompressed**, with a **10 MiB limit per file**.

After completing the three experiments, run the submission command below. A separate report is not required. The training/evaluation scripts automatically save the effective configurations as `logs/reward_model_config.yaml`, `logs/rlhf_config.yaml`, and `logs/evaluation_config.yaml`. These include the seed and resolved command-line overrides. Use the default output directories for submission.

```bash
python scripts/submission.py --andrew-id YOUR_ANDREW_ID
```

You may optionally attach notes with `--report report.md` (or PDF) and exact commands with `--commands commands.txt`.

The script creates `assignment7_YOUR_ANDREW_ID.zip`. Missing experiment evidence stops packaging and identifies the missing file. The script also runs the public student self-check and includes its JSON feedback. Failed implementation tests still allow submission for partial credit; setup errors stop packaging. These checks do not assign course points. The script never deletes local checkpoints or overwrites an existing archive; use `--output another_name.zip` for a new archive.

The archive includes only:

- Python source under `src/`, `scripts/`, and `tests/`, plus the README and requirements.
- Generated `package_versions.txt` and student `self_check.json` feedback, plus notes or commands if you supplied them.
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
