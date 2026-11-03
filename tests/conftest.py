"""Small local transformer fixtures; tests do not download model weights."""
import sys
from pathlib import Path

import pytest
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertTokenizerFast, DistilBertConfig, DistilBertModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def tiny_encoder(tmp_path_factory):
    path = tmp_path_factory.mktemp("encoder")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab += "this is a good bad helpful response another test text short not and with several words .".split()
    (path / "vocab.txt").write_text("\n".join(dict.fromkeys(vocab)))
    BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    config = DistilBertConfig(vocab_size=len(set(vocab)), dim=24, hidden_dim=32,
                             n_layers=1, n_heads=2, max_position_embeddings=512)
    DistilBertModel(config).save_pretrained(path)
    return str(path)


@pytest.fixture(autouse=True)
def local_reward_backbone(monkeypatch, tiny_encoder):
    for factory in (AutoConfig, AutoModel, AutoTokenizer):
        original = factory.from_pretrained
        def load(name, *args, _original=original, **kwargs):
            if name == "distilbert-base-uncased":
                name = tiny_encoder
            return _original(name, *args, **kwargs)
        monkeypatch.setattr(factory, "from_pretrained", load)
