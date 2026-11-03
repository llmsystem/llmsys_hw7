"""Regression tests for data isolation, policy masking, and PPO contracts."""
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

from scripts import prepare_data
from src.config import get_config
from src.reward_model import create_reward_model
from src.rlhf_trainer import VERLPolicyWrapper, VERLTrainer


@pytest.fixture
def trainer():
    torch.manual_seed(7)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    config = get_config()
    config.verl.rollout_max_length = 12
    config.verl.ppo_epochs = 2
    config.training.ppo_learning_rate = 1e-3
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_embd=16, n_layer=1, n_head=2, n_positions=64,
        bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))
    reward = create_reward_model(hidden_size=16, dropout=0)
    return VERLTrainer(model, reward, tokenizer, config, torch.device("cpu"))


def test_config_isolation_and_validation():
    a, b = get_config(), get_config()
    a.verl.rollout_batch_size = 99
    assert b.verl.rollout_batch_size == 4
    a.training.ppo_mini_batch_size = 0
    with pytest.raises(ValueError, match="positive"):
        a.validate()


def test_shared_preference_prefix_required():
    good = "\n\nHuman: A question\n\nAssistant: A sufficiently long answer"
    bad = "\n\nHuman: Different question\n\nAssistant: A sufficiently bad answer"
    assert prepare_data.process_hh_example({'chosen': good, 'rejected': bad}) is None
    pair = prepare_data.process_hh_example({'chosen': good, 'rejected': good + ' rejected'})
    assert pair['chosen'] == good.strip()


def test_prepared_split_ratios_and_prompt_isolation(monkeypatch, tmp_path):
    pairs = [{'prompt': str(i), 'chosen': str(i) + ' yes', 'rejected': str(i) + ' no'}
             for i in range(20)]
    monkeypatch.setattr(prepare_data, 'load_hh_rlhf_dataset', lambda n: (pairs, []))
    prepare_data.prepare_dataset(output_dir=str(tmp_path), max_samples=20)
    def read(name):
        return json.loads((tmp_path / name).read_text())
    assert [len(read(name)) for name in ['preference_data.json', 'preference_data_val.json',
                                        'preference_data_test.json']] == [16, 2, 2]
    prompts = [{p['prompt'] for p in read(name)} for name in
               ['train_prompts.json', 'val_prompts.json', 'eval_prompts.json']]
    assert not prompts[0] & prompts[1]
    assert not prompts[0] & prompts[2]
    assert not prompts[1] & prompts[2]
    assert prepare_data.validate_data(str(tmp_path))


def test_download_errors_are_not_synthetic_success(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError('network unavailable')
    monkeypatch.setattr(prepare_data, 'load_dataset', unavailable)
    with pytest.raises(RuntimeError, match='network unavailable'):
        prepare_data.load_hh_rlhf_dataset(20)


def test_both_preference_sources_sampled(monkeypatch):
    class Rows(list):
        def shuffle(self, seed):
            return self
    def load(name, data_dir, split):
        return Rows([{'chosen': f'\n\nHuman: {data_dir} {i}\n\nAssistant: Chosen long response',
                      'rejected': f'\n\nHuman: {data_dir} {i}\n\nAssistant: Rejected long response'}
                     for i in range(30)])
    monkeypatch.setattr(prepare_data, 'load_dataset', load)
    rows, _ = prepare_data.load_hh_rlhf_dataset(10)
    assert sum('helpful-base' in row['prompt'] for row in rows) == 5
    assert sum('harmless-base' in row['prompt'] for row in rows) == 5


def test_left_padding_does_not_change_policy_probabilities(trainer):
    ids = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[3, 3, 5, 6, 7]])
    trainer.policy.eval()
    a = trainer.policy(ids, torch.ones_like(ids)).logits
    b = trainer.policy(padded, torch.tensor([[0, 0, 1, 1, 1]])).logits
    torch.testing.assert_close(a, b[:, 2:], atol=1e-6, rtol=1e-5)


def test_eos_kept_once_in_rollout_mask(trainer, monkeypatch):
    def generate(input_ids, **kwargs):
        suffix = torch.tensor([[3, 3, 3], [5, 6, 3]])
        return SimpleNamespace(sequences=torch.cat([input_ids, suffix], dim=1),
                               scores=[None] * 3)
    monkeypatch.setattr(trainer.policy.model, 'generate', generate)
    _, log_probs, width, ids, mask = trainer.policy.generate(['this', 'this is'])
    assert mask[:, width:].tolist() == [[1, 0, 0], [1, 1, 1]]
    assert [len(p) for p in log_probs] == [1, 3]
    assert all(not p.requires_grad for p in log_probs)


def test_gae_terminal_reward_and_padding(trainer):
    trainer.config.training.ppo_gamma = 1
    trainer.config.training.ppo_gae_lambda = 1
    values = torch.tensor([[99., .2, .4, 99., 99.], [99., .1, .3, .5, 99.]])
    mask = torch.tensor([[0, 1, 1, 0, 0], [0, 1, 1, 1, 0]])
    adv, returns = trainer._compute_gae(torch.tensor([1., 2.]), values, mask)
    expected = torch.tensor([[0., .4, 1., 0., 0.], [0., .3, .5, 2., 0.]])
    torch.testing.assert_close(returns, expected)
    torch.testing.assert_close(adv, (expected - values) * mask)


def test_td_uses_gamma_without_recursive_gae(trainer):
    trainer.config.training.ppo_gamma = .9
    trainer.config.training.ppo_gae_lambda = .5
    adv, returns = trainer._compute_gae(
        torch.tensor([1.]), torch.tensor([[.2, .4, 0.]]), torch.tensor([[1, 1, 0]]))
    torch.testing.assert_close(adv, torch.tensor([[.16, .6, 0.]]))
    torch.testing.assert_close(returns, torch.tensor([[.36, 1., 0.]]))


def test_entropy_value_and_gradient(trainer):
    logits = torch.zeros(3, 4, requires_grad=True)
    entropy = trainer._compute_entropy(logits)
    torch.testing.assert_close(entropy, torch.tensor(4.).log())
    entropy.backward()
    assert torch.isfinite(logits.grad).all()


def test_ppo_update_freezes_reference_and_retains_old_probs(trainer):
    rollout = trainer.generate_rollouts(['this is', 'another'])
    original = [p.clone() for p in rollout.log_probs]
    assert not rollout.advantages.requires_grad
    assert not rollout.returns.requires_grad
    scores = trainer.policy(rollout.full_seq, rollout.full_attn_mask).logits[:, rollout.prompt_len - 1:-1]
    current, _ = trainer.policy._compute_log_probs_tensor(rollout.full_seq, scores, rollout.prompt_len)
    torch.testing.assert_close(torch.cat(current), torch.cat(original))
    reference = {k: v.clone() for k, v in trainer.ref_policy.state_dict().items()}
    policy = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    metrics = trainer.train_step(rollout)
    assert all(torch.isfinite(torch.tensor(v)) for v in vars(metrics).values())
    assert any(not torch.equal(v, trainer.policy.state_dict()[k]) for k, v in policy.items())
    for key, value in reference.items():
        torch.testing.assert_close(value, trainer.ref_policy.state_dict()[key], rtol=0, atol=0)
    for old, new in zip(original, rollout.log_probs):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    assert all(p.grad is None for p in trainer.reward_model.parameters())


def test_checkpoint_restores_original_reference(trainer, tmp_path):
    path = str(tmp_path / 'checkpoint.pt')
    original = next(trainer.ref_policy.parameters()).detach().clone()
    trainer.save_checkpoint(path, 1, {'reward': .5})
    with torch.no_grad():
        next(trainer.ref_policy.parameters()).add_(1)
    trainer.load_checkpoint(path)
    torch.testing.assert_close(next(trainer.ref_policy.parameters()), original)


def test_ppo_clips_positive_and_negative_advantages(trainer):
    trainer.config.verl.ppo_epochs = 1
    trainer.config.training.ppo_entropy_coef = 0
    trainer.config.training.ppo_kl_penalty = 0
    rollout = trainer.generate_rollouts(['this is', 'another'])
    scores = trainer.policy(rollout.full_seq, rollout.full_attn_mask).logits[:, rollout.prompt_len - 1:-1]
    _, mask = trainer.policy._compute_log_probs_tensor(rollout.full_seq, scores, rollout.prompt_len)
    actions = torch.arange(mask.sum()) % 2 * 2. - 1.
    rollout.advantages.zero_()
    rollout.advantages[:, rollout.prompt_len - 1:-1][mask] = actions
    rollout.log_probs = [p - torch.tensor(2.).log() for p in rollout.log_probs]
    expected = -torch.minimum(2 * actions, 1.2 * actions).mean()
    metrics = trainer.train_step(rollout)
    assert metrics.policy_loss == pytest.approx(expected.item(), abs=1e-5)


def test_nonuniform_entropy(trainer):
    logits = torch.tensor([[0., 2.]], requires_grad=True)
    entropy = trainer._compute_entropy(logits)
    assert entropy.item() == pytest.approx(0.36533385, abs=1e-6)
    entropy.backward()
    assert logits.grad[0, 0] > 0
    assert logits.grad[0, 1] < 0


@pytest.mark.parametrize('module_name,entrypoint,args', [
    ('train_reward_model', 'train_reward_model', []),
    ('run_rlhf', 'run_rlhf_training', ['--reward_model_path', '/tmp/selected.pt']),
    ('evaluate', 'evaluate_models', ['--reward_model', '/tmp/selected.pt',
                                   '--rlhf_model', '/tmp/selected-policy', '--num_samples', '7']),
])
def test_cli_preserves_yaml_and_honors_exact_paths(monkeypatch, tmp_path, module_name, entrypoint, args):
    import importlib
    module = importlib.import_module('scripts.' + module_name)
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('model:\n  model_name: custom-policy\n  reward_model_name: custom-reward\n'
                           'training:\n  reward_num_epochs: 8\n  ppo_num_epochs: 9\n')
    captured = []
    monkeypatch.setattr(module, entrypoint, captured.append)
    monkeypatch.setattr('sys.argv', [module_name, '--config', str(config_file), *args])
    module.main()
    config = captured[0]
    assert config.model.model_name == 'custom-policy'
    assert config.model.reward_model_name == 'custom-reward'
    assert config.training.reward_num_epochs == 8
    assert config.training.ppo_num_epochs == 9
    if module_name != 'train_reward_model':
        assert config.system.reward_model_path == '/tmp/selected.pt'
    if module_name == 'evaluate':
        assert config.system.rlhf_model_path == '/tmp/selected-policy'
        assert config.experiment.eval_num_samples == 7
