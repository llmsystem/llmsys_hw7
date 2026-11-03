#!/usr/bin/env python3
"""Run the offline implementation tests and save feedback."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def summarize(xml_path, exit_code):
    counts = {'passed': 0, 'failed': 0, 'errors': 0, 'skipped': 0}
    cases = []
    for case in ET.parse(xml_path).getroot().iter('testcase'):
        status = 'passed'
        for tag, outcome in [('failure', 'failed'), ('error', 'errors'), ('skipped', 'skipped')]:
            if case.find(tag) is not None:
                status = outcome
                break
        counts[status] += 1
        cases.append({'name': case.attrib.get('classname', '') + '::' + case.attrib.get('name', ''),
                      'status': status})
    collection_errors = sum(1 for _ in ET.parse(xml_path).getroot().iter('error'))
    status = ('error' if exit_code not in (0, 1) or counts['errors'] or collection_errors
              else 'failed' if exit_code == 1 or counts['failed'] or counts['skipped']
              else 'passed' if cases else 'error')
    return {'status': status, 'counts': counts, 'tests': cases,
            'note': 'Implementation test results.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('self_check.json'))
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    log = ''
    try:
        with tempfile.TemporaryDirectory(prefix='hw6-check-') as temporary:
            directory = Path(temporary).resolve()
            environment = os.environ.copy()
            environment.update({'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                                'TRANSFORMERS_OFFLINE': '1', 'MPLBACKEND': 'Agg',
                                'MPLCONFIGDIR': str(directory / 'mpl')})
            xml_path = directory / 'tests.xml'
            result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-q',
                                     '--junitxml', str(xml_path)], cwd=ROOT, env=environment,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    timeout=args.timeout)
            log = result.stdout
            report = summarize(xml_path, result.returncode)
    except (OSError, ValueError, ET.ParseError, subprocess.TimeoutExpired) as error:
        report = {'status': 'error', 'message': str(error),
                  'note': 'Self-check did not finish. Check dependencies and the companion log.'}
        if isinstance(error, subprocess.TimeoutExpired):
            captured = error.stdout or b''
            log = captured.decode(errors='replace') if isinstance(captured, bytes) else captured
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    args.output.with_suffix('.log').write_text(log)
    print(f'Self-check: {report["status"]}; {report.get("counts", {})}')
    print(f'Results: {args.output.resolve()}\nLog: {args.output.with_suffix(".log").resolve()}')
    raise SystemExit({'passed': 0, 'failed': 1, 'error': 2}[report['status']])


if __name__ == '__main__':
    main()
