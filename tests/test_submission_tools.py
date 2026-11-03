"""Submission size limits and archive contents."""
import zipfile

import pytest

from scripts import submission


@pytest.fixture
def bundle_inputs(tmp_path):
    root = tmp_path / 'project'
    for name in [*submission.SOURCE_FILES, 'README.md', 'requirements.txt',
                 *submission.REQUIRED_EVIDENCE]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    (root / 'scripts').mkdir()
    (root / 'scripts/train.py').write_text('print("training")')
    (root / 'tests').mkdir()
    (root / 'tests/test_example.py').write_text('')
    report = tmp_path / 'report.md'
    report.write_text('Experiment report')
    commands = tmp_path / 'commands.txt'
    commands.write_text('python scripts/train_reward_model.py')
    return root, report, commands, tmp_path / 'submission.zip'


def build(inputs):
    root, report, commands, output = inputs
    return submission.build_submission(root, 'student1', report, commands, output)


def test_package_excludes_gigabyte_weights_and_hidden_files(bundle_inputs):
    root, _, _, output = bundle_inputs
    for name in ['outputs/model.safetensors', 'data/train.json', '.venv/weights.pt']:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as file:
            file.truncate(1024**3)
    (root / 'logs/weights.pt').write_bytes(b'weights')
    (root / 'src/.secret.py').write_text('private')
    (root / 'scripts/autograde.py').write_text('private tool')
    (root / 'logs/grading_results.json').write_text('{}')
    manifest = build(bundle_inputs)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert 'src/reward_model.py' in names
        assert 'report.md' in names
        assert 'grading_results.json' not in names
        assert 'package_versions.txt' in names
        assert not any(name.startswith(('outputs/', 'data/', '.venv/')) for name in names)
        assert not any(name.endswith(('.pt', '.safetensors')) for name in names)
        assert 'src/.secret.py' not in names
        assert 'scripts/autograde.py' not in names
        assert 'logs/grading_results.json' not in names
        assert manifest['assignment'] == 6
        assert manifest['files']['report.md']['bytes'] == len('Experiment report')
    assert output.stat().st_size < 1024**2


def test_missing_evidence_stops_packaging(bundle_inputs):
    root, *_ = bundle_inputs
    (root / submission.REQUIRED_EVIDENCE[0]).unlink()
    with pytest.raises(ValueError, match='Regular files'):
        build(bundle_inputs)


def test_oversized_evidence_stops_packaging(bundle_inputs):
    root, *_ = bundle_inputs
    with (root / 'plots/huge.png').open('wb') as file:
        file.truncate(submission.MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match='10 MiB'):
        build(bundle_inputs)


def test_total_limit_applies_to_uncompressed_files(bundle_inputs, monkeypatch):
    monkeypatch.setattr(submission, 'MAX_TOTAL_BYTES', 10)
    with pytest.raises(ValueError, match='uncompressed'):
        build(bundle_inputs)


def test_submission_does_not_overwrite_existing_archive(bundle_inputs):
    output = bundle_inputs[-1]
    output.write_bytes(b'keep me')
    with pytest.raises(ValueError, match='already exists'):
        build(bundle_inputs)
    assert output.read_bytes() == b'keep me'


def test_symlinks_are_rejected(bundle_inputs):
    root, report, *_ = bundle_inputs
    (root / 'plots/linked.md').symlink_to(report)
    with pytest.raises(ValueError, match='Symlink'):
        build(bundle_inputs)


def test_notes_and_commands_are_optional(bundle_inputs):
    root, _, _, output = bundle_inputs
    submission.build_submission(root, 'student1', None, None, output)
    with zipfile.ZipFile(output) as archive:
        assert 'report.md' not in archive.namelist()
        assert 'commands.txt' not in archive.namelist()
