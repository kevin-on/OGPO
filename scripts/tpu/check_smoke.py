"""Verify real learner updates, evaluation, and a persisted replay after a smoke phase."""
import csv
import json
import os
from pathlib import Path
import pickle
import sys

import jax
import numpy as np
from google.cloud import storage

expected_online_step = int(sys.argv[1])
root = Path(os.environ['TPUQ_TASK_ID'])
meta = max(root.glob('checkpoints/ogpo/*/*_meta.json'), key=lambda p: int(p.name.split('_')[0]))
metadata = json.loads(meta.read_text())
assert metadata['online_loop_step'] == expected_online_step, metadata
step = int(meta.name.split('_')[0])
with meta.with_name(f'{step}_agent.pkl').open('rb') as f:
    state = pickle.load(f)
with meta.with_name(f'{step}_buffer.pkl').open('rb') as f:
    replay = pickle.load(f)
for leaf in jax.tree.leaves(state):
    assert np.isfinite(np.asarray(leaf)).all(), 'nonfinite model/optimizer state'
assert replay.size >= 1024, replay.size
for field, array in replay._dict.items():
    assert np.isfinite(array[:replay.size]).all(), field
assert int(state['actor_network']['step']) > 1
assert int(state['critic_network']['step']) > 1
for filename in ('online_agent.csv', 'eval.csv', 'eval_sde.csv'):
    path = next(root.glob('exp/**/' + filename))
    rows = list(csv.DictReader(path.open()))
    assert rows and int(rows[-1]['step']) == step, (filename, rows[-1:])
    for key, value in rows[-1].items():
        try:
            value = float(value)
        except ValueError:
            continue
        assert np.isfinite(value), (filename, key, value)
assert list(root.glob('exp/**/params_1.pkl')), 'final BC artifact missing'
evidence = dict(online_step=expected_online_step, global_step=step, replay_size=replay.size,
                actor_step=int(state['actor_network']['step']), critic_step=int(state['critic_network']['step']),
                backend=jax.default_backend(), devices=[str(d) for d in jax.devices()], passed=True)
print('SMOKE_PHASE_PASSED', json.dumps(evidence), flush=True)
bucket, prefix = os.environ['SMOKE_PREFIX'].removeprefix('gs://').split('/', 1)
storage.Client().bucket(bucket).blob(prefix + f'/verified-{expected_online_step}.json').upload_from_string(json.dumps(evidence))
