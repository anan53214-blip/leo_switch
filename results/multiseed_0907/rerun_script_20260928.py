"""Reproduce the two anomalous HAN+PDQN training runs without touching originals."""
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/pjpjq/LEO_switch')
OUT = ROOT / 'results/multiseed_0907_rerun_20260928'
TASKS = ((43, 20), (45, 30))


def stamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def save(status):
    target = OUT / 'status.json'
    temporary = OUT / 'status.json.tmp'
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(target)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command_for(seed, users):
    original = ROOT / f'results/multiseed_0907/seed{seed}/logs/u{users}_seed{seed}_system_train.log'
    with original.open(encoding='utf-8') as stream:
        first_line = next((line.strip() for line in stream if line.strip()), '')
    if not first_line.startswith('$ '):
        raise ValueError(f'Missing original training command in {original}')
    command = shlex.split(first_line[2:])
    if '--load_path' in command:
        raise ValueError('Expected a fresh training command, not checkpoint continuation')
    run_dir = OUT / f'seed{seed}/u{users}/learned_baselines/han_pdqn'
    log_dir = OUT / f'seed{seed}/logs'
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Rerun directory already contains files: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for flag, value in (('--save_path', run_dir), ('--log_path', log_dir)):
        command[command.index(flag) + 1] = str(value)
    pretrained = Path(command[command.index('--pretrained_han_path') + 1])
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    return command, run_dir, pretrained


def original_reward(seed, users):
    path = ROOT / f'results/multiseed_0907/seed{seed}/u{users}/comparison_summary.csv'
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['display_name'] == 'HAN+PDQN':
                return float(row['mean_reward'])
    raise ValueError(f'HAN+PDQN missing from {path}')


status = {'started_at': stamp(), 'state': 'running', 'tasks': []}
save(status)
for seed, users in TASKS:
    task = {'seed': seed, 'num_users': users, 'state': 'preparing'}
    status['tasks'].append(task)
    try:
        command, run_dir, pretrained = command_for(seed, users)
        task.update(state='running', started_at=stamp(), command=command,
                    pretrained_han_sha256=sha256(pretrained),
                    original_reward=original_reward(seed, users))
        save(status)
        log_path = OUT / f'seed{seed}_u{users}_train.log'
        with log_path.open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        task['exit_code'] = result.returncode
        task['finished_at'] = stamp()
        if result.returncode:
            raise RuntimeError(f'Training exited with code {result.returncode}; see {log_path}')
        history = json.loads((run_dir / 'training_history.json').read_text(encoding='utf-8'))
        task['total_steps'] = history['summary']['total_steps']
        task['best_reward'] = history['summary']['best_reward']
        task['reward_change'] = task['best_reward'] - task['original_reward']
        task['best_model_sha256'] = sha256(run_dir / 'best_model.pt')
        task['state'] = 'complete'
        save(status)
    except Exception as exc:
        task['state'] = 'failed'
        task['error'] = str(exc)
        status['state'] = 'failed'
        status['finished_at'] = stamp()
        save(status)
        raise
status['state'] = 'complete'
status['finished_at'] = stamp()
save(status)
