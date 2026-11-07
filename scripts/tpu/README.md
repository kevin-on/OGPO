# TPUQ execution

Run the official `scripts/ogpo/{square,toolhang,transport}.sh` settings on a TPU VM.
Source `scripts/tpu/runtime.sh`, then invoke `scripts/tpu/run.py` with a task,
seed, stable W&B run ID/group, a per-run regional GCS prefix, and the staged
regional dataset prefix. The launcher uses W&B entity
`kevinon-stanford-university`, project `expos`.

The runtime uses the upstream `uv.lock` plus `jax[tpu]==0.10.1` and
`google-cloud-storage==3.7.0`. It keeps TPUQ's device visibility. OSMesa handles
CPU rendering. Reserve a whole four-chip VM per run for the official 32 eval
workers; the learner uses one TPU device in the official `single` mode.

Production changes only operational settings: logging destination, seed, paths,
and rolling model+replay backup every 10,000 online environment steps. Official
BC/online training lengths, early stop, eval counts/intervals, networks, losses,
optimizers, schedules, and datasets are unchanged. `datasets.json` pins the
three official robomimic v1.5 HDF5 files and SHA256 hashes.

## Persistence

- Local rolling files are written to temporary paths; metadata is published last.
  A file lock prevents rolling cleanup during external upload.
- The launcher uploads a complete snapshot before publishing GCS `latest.json`.
  Downloads address exact object generations. Only the latest and predecessor
  snapshots remain; interrupted unpublished uploads do not become resumable.
- Online optimizer transformations are reconstructed before loading optimizer
  state. Replay RNG uses `bit_generator.state`. Corrupt committed checkpoints
  fail instead of silently restarting BC.
- Final BC `params_<step>.pkl` is saved by the official training flow and copied
  to GCS `artifacts/`. It remains available for future experiments. It is not
  connected to automatic BC resume. If Spot interrupts BC before online backup,
  BC starts again; any already uploaded BC artifact remains in GCS.
- Online resume restarts the environment at an episode boundary. Environment
  simulator state, partial trajectories, and rollout RNG are not checkpointed;
  this is recovery of learner/replay state, not an identical rollout replay.
- CSV, flags, and per-attempt stdout are uploaded for diagnostics. A failed sync
  terminates the learner so TPUQ can retry from the last published snapshot.

## Validation

Local persistence tests:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m unittest discover -s tests -p test_checkpoint.py -v
```

For each task, `smoke.sh` uses the full official dataset, model, batch, replay
capacity and 32-worker/64-episode evaluation. Only training duration, warmup,
logging and checkpoint/eval cadence are shortened. It checks BC update, real
rollout, online update, eval, and save; deletes its local rolling files; then
starts a second process that restores from GCS and performs further updates and
evaluation. `check_smoke.py` verifies finite model/optimizer/replay/metrics and
publishes a separate evidence object for each phase. Smoke W&B is disabled.

The campaign manifest outside this checkout records canonical run IDs, pinned
commit, submitted job/task IDs, smoke evidence and launch state. Do not submit
production until all three task smokes have passed at the intended commit.
