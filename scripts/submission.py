#!/usr/bin/env python3
"""Build a small, reproducible submission without model or dataset files."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024
SOURCE_FILES = ('src/config.py', 'src/utils.py', 'src/reward_model.py', 'src/rlhf_trainer.py')
REQUIRED_EVIDENCE = (
    'logs/reward_model_metrics.json', 'logs/rlhf_training_metrics.json',
    'logs/reward_model_config.yaml', 'logs/rlhf_config.yaml', 'logs/evaluation_config.yaml',
    'logs/reward_model_training_curves.png', 'plots/rlhf_training_curves.png',
    'plots/reward_comparison.png', 'evaluation_results/detailed_comparison.json',
    'evaluation_results/base_model_outputs.json', 'evaluation_results/rlhf_model_outputs.json',
)
EVIDENCE_SUFFIXES = {'.json', '.png', '.jpg', '.jpeg', '.csv', '.yaml', '.yml', '.txt', '.md', '.pdf'}


def checked_bytes(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Regular files only: {path}')
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f'File exceeds 10 MiB: {path}')
    data = path.read_bytes()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f'File exceeds 10 MiB: {path}')
    return data


def build_submission(root, andrew_id, report, commands, output, check_path=None):
    root, output = Path(root), Path(output)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', andrew_id):
        raise ValueError('Use your Andrew ID (letters, digits, underscore, or hyphen)')
    if Path(report).suffix.lower() not in {'.pdf', '.md'}:
        raise ValueError('The report must be a PDF or Markdown file')
    required = [*SOURCE_FILES, 'README.md', 'requirements.txt', *REQUIRED_EVIDENCE]
    files = {}
    total_bytes = 0
    def add(name, data):
        nonlocal total_bytes
        total_bytes += len(data) - len(files.get(name, b''))
        if total_bytes > MAX_TOTAL_BYTES:
            raise ValueError('Submission exceeds 50 MiB uncompressed; reduce plots or sample outputs')
        files[name] = data

    for relative in required:
        add(relative, checked_bytes(root / relative))
    # Only explicit source/evidence roots are eligible. Never walk outputs or data.
    for directory, suffixes in [('src', {'.py'}), ('scripts', {'.py'}), ('tests', {'.py'}),
                                 ('logs', EVIDENCE_SUFFIXES), ('plots', EVIDENCE_SUFFIXES),
                                 ('evaluation_results', EVIDENCE_SUFFIXES)]:
        base = root / directory
        if base.is_symlink():
            raise ValueError(f'Symlink directory is not allowed: {base}')
        for path in sorted(base.rglob('*')):
            relative = path.relative_to(root)
            if (path.name in {'autograde.py', 'grading_results.json', 'grading_results.log'}
                    or any(part.startswith('.') or part == '__pycache__' for part in relative.parts)):
                continue
            if path.is_symlink():
                raise ValueError(f'Symlink is not allowed: {relative}')
            if path.is_file() and path.suffix.lower() in suffixes:
                add(relative.as_posix(), checked_bytes(path))
    add('report' + Path(report).suffix.lower(), checked_bytes(Path(report)))
    add('commands.txt', checked_bytes(Path(commands)))
    if check_path is not None:
        check_data = checked_bytes(Path(check_path))
        check = json.loads(check_data)
        if check.get('status') not in {'passed', 'failed'}:
            raise ValueError('Student self-check could not finish; fix its setup and retry')
        add('self_check.json', check_data)
    versions = sorted({f'{dist.metadata["Name"]}=={dist.version}'
                       for dist in importlib.metadata.distributions() if dist.metadata['Name']})
    add('package_versions.txt', ('\n'.join(versions) + '\n').encode())
    manifest = {
        'assignment': 7, 'term': 'Fall 2026', 'andrew_id': andrew_id,
        'checkpoints_included': False,
        'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                  for name, data in sorted(files.items())},
    }
    add('submission_manifest.json', json.dumps(manifest, indent=2).encode())
    if output.exists():
        raise ValueError(f'Output already exists: {output}; choose another filename')
    if output.is_symlink():
        raise ValueError('Output must not be a symlink')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        archive = Path(temp) / 'submission.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, data in sorted(files.items()):
                bundle.writestr(name, data)
        if archive.stat().st_size > MAX_TOTAL_BYTES:
            raise ValueError('Compressed submission exceeds 50 MiB')
        # Exclusive creation prevents an existing archive from being overwritten.
        with output.open('xb') as destination:
            destination.write(archive.read_bytes())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--andrew-id', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--commands', type=Path, required=True, help='Text file with exact experiment commands')
    parser.add_argument('--output', type=Path, help='Default: assignment7_ANDREW_ID.zip')
    args = parser.parse_args()
    try:
        for name in REQUIRED_EVIDENCE:
            checked_bytes(ROOT / name)
        checked_bytes(args.report)
        checked_bytes(args.commands)
        output = args.output or Path(f'assignment7_{args.andrew_id}.zip')
        with tempfile.TemporaryDirectory() as temporary:
            check_path = Path(temporary) / 'self_check.json'
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_implementation.py'),
                                     '--output', str(check_path)], cwd=ROOT)
            if result.returncode not in (0, 1):
                raise ValueError('Student self-check setup failed; see its output')
            manifest = build_submission(ROOT, args.andrew_id, args.report, args.commands,
                                        output, check_path=check_path)
        print(f'Created {output.resolve()} ({output.stat().st_size / 1024**2:.2f} MiB; '
              f'{len(manifest["files"]) + 1} files). No checkpoints or datasets included.')
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f'Submission error: {error}\n')


if __name__ == '__main__':
    main()
