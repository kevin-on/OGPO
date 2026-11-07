"""Run an official task script with regional GCS checkpoint persistence."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from google.cloud import storage


def download_file(blob, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(path) + '.tmp')
    os.replace(str(path) + '.tmp', path)


def restore(bucket, prefix, checkpoint_dir):
    pointer = bucket.blob(prefix + '/latest.json')
    if not pointer.exists():
        return
    snapshot = json.loads(pointer.download_as_text())
    # Publish metadata locally only after both payloads have been downloaded.
    for item in snapshot['files']:
        blob = bucket.blob(item['object'], generation=int(item['generation']))
        path = checkpoint_dir / item['name']
        download_file(blob, path)
        if path.stat().st_size != item['bytes']:
            raise RuntimeError(f'Incomplete checkpoint: {path}')
    print('RESTORED', snapshot['step'], flush=True)


def upload_checkpoint(bucket, prefix, checkpoint_dir):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metas = list(checkpoint_dir.glob('*_meta.json'))
        if not metas:
            return
        meta = max(metas, key=lambda p: int(p.name.split('_')[0]))
        step = int(meta.name.split('_')[0])
        pointer = bucket.blob(prefix + '/latest.json')
        previous = json.loads(pointer.download_as_text()) if pointer.exists() else None
        if previous and previous['step'] >= step:
            return
        snapshot_prefix = f'{prefix}/snapshots/{step}-{uuid.uuid4().hex}'
        files = []
        for suffix in ('agent.pkl', 'buffer.pkl', 'meta.json'):
            path = checkpoint_dir / f'{step}_{suffix}'
            blob = bucket.blob(snapshot_prefix + '/' + path.name)
            blob.upload_from_filename(str(path), if_generation_match=0, timeout=600)
            files.append(dict(name=path.name, object=blob.name,
                              generation=blob.generation, bytes=path.stat().st_size))
        snapshot = dict(step=step, files=files, prefix=snapshot_prefix,
                        previous=previous['prefix'] if previous else None)
        pointer.upload_from_string(json.dumps(snapshot), content_type='application/json')
        # Keep the latest and its predecessor; also clean abandoned partial uploads.
        keep = (snapshot_prefix + '/', (snapshot['previous'] or '') + '/')
        for blob in bucket.list_blobs(prefix=prefix + '/snapshots/'):
            if not blob.name.startswith(keep):
                blob.delete()
        print('UPLOADED_CHECKPOINT', step, flush=True)


def upload_artifacts(bucket, prefix, exp_dir, log_path, uploaded):
    # save_agent publishes params files by rename. The final BC file is retained
    # as an artifact; it is deliberately not used as a resume checkpoint.
    for path in list(exp_dir.rglob('params_*.pkl')) + list(exp_dir.rglob('flags.json')) + list(exp_dir.rglob('*.csv')):
        stat = (path.stat().st_size, path.stat().st_mtime_ns)
        if uploaded.get(path) != stat:
            bucket.blob(prefix + '/artifacts/' + str(path.relative_to(exp_dir))).upload_from_filename(str(path), timeout=600)
            uploaded[path] = stat
    if log_path.exists():
        bucket.blob(prefix + '/logs/' + log_path.name).upload_from_filename(str(log_path), timeout=600)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['square', 'toolhang', 'transport'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--data-prefix', required=True)
    parser.add_argument('overrides', nargs='*')
    args, overrides = parser.parse_known_args()
    overrides = args.overrides + overrides
    client = storage.Client()
    bucket_name, prefix = args.prefix.removeprefix('gs://').split('/', 1)
    bucket = client.bucket(bucket_name)
    data_bucket, data_prefix = args.data_prefix.removeprefix('gs://').split('/', 1)
    data_root = Path.home() / '.cache/ogpo-datasets'
    manifest = json.loads(Path('scripts/tpu/datasets.json').read_text())
    task = {'square': 'square/mh', 'toolhang': 'tool_hang/ph', 'transport': 'transport/mh'}[args.task]
    item = next(x for x in manifest['files'] if x['path'].startswith(task + '/'))
    data_path = data_root / item['path']
    data_root.mkdir(parents=True, exist_ok=True)
    with (data_root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not data_path.exists():
            download_file(client.bucket(data_bucket).blob(data_prefix + '/' + item['path']), data_path)
        with data_path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == item['sha256']
    work = Path(os.environ['TPUQ_TASK_ID'] if 'TPUQ_TASK_ID' in os.environ else args.run_id).resolve()
    work.mkdir(parents=True, exist_ok=True)
    env_name = task.replace('/', '-') + '-low_dim'
    checkpoint_root = work / 'checkpoints'
    checkpoint_dir = checkpoint_root / 'ogpo' / f'{args.seed}_{env_name}'
    restore(bucket, prefix, checkpoint_dir)
    exp_dir = work / 'exp'
    log_path = work / f'run-{time.time_ns()}.log'
    env = dict(os.environ, ROBOMIMIC_DATASET_ROOT=str(data_root),
               OGPO_CHECKPOINT_ROOT=str(checkpoint_root), OGPO_EXP_ROOT=str(exp_dir),
               WANDB_ENTITY='kevinon-stanford-university', WANDB_RUN_ID=args.run_id,
               WANDB_RESUME='allow')
    command = ['bash', f'scripts/ogpo/{args.task}.sh', f'--seed={args.seed}',
               '--project=expos', f'--run_group={args.group}',
               '--preemption_backup_interval=10000'] + overrides
    uploaded = {}
    print('TRAINING_LOG', log_path, flush=True)
    with log_path.open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                time.sleep(30)
                upload_checkpoint(bucket, prefix, checkpoint_dir)
                upload_artifacts(bucket, prefix, exp_dir, log_path, uploaded)
            upload_checkpoint(bucket, prefix, checkpoint_dir)
            upload_artifacts(bucket, prefix, exp_dir, log_path, uploaded)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
    print('TRAINING_EXIT', process.returncode, flush=True)
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
