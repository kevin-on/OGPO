import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from ogpo.utils.main_helper import save_rolling_checkpoint, load_rolling_checkpoint, get_checkpoint_dir
from ogpo.utils.datasets import ReplayBuffer
from ogpo.agents.modules.flax_utils import save_agent


class CheckpointTest(unittest.TestCase):
    def test_interrupted_save_and_replay_rng(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, OGPO_CHECKPOINT_ROOT=root):
            agent = {'weights': jnp.array([1., 2.])}
            replay = ReplayBuffer.create({'observations': np.ones(2), 'actions': np.ones(1),
                                          'rewards': np.float32(0), 'masks': np.float32(1),
                                          'terminals': np.float32(0)}, 8, numpy_rng=np.random.default_rng(9))
            replay.size = replay.pointer = 3
            save_rolling_checkpoint(agent, replay, 10, 5, 1, 'test', 'run-id')
            original_replace = os.replace
            def interrupt(src, dst):
                if str(dst).endswith('20_meta.json'):
                    raise OSError('simulated preemption before commit')
                original_replace(src, dst)
            with patch('os.replace', side_effect=interrupt), self.assertRaises(OSError):
                save_rolling_checkpoint(agent, replay, 20, 15, 1, 'test', 'run-id')
            loaded, restored, step, online, run_id = load_rolling_checkpoint(1, 'test', agent)
            self.assertEqual((step, online, run_id), (10, 5, 'run-id'))
            np.testing.assert_array_equal(loaded['weights'], agent['weights'])
            self.assertEqual((restored.size, restored.pointer), (3, 3))
            np.testing.assert_array_equal(restored.numpy_rng.integers(100, size=20), replay.numpy_rng.integers(100, size=20))
            save_rolling_checkpoint(agent, replay, 30, 25, 1, 'test', 'run-id')
            directory = Path(get_checkpoint_dir(1, 'test'))
            self.assertEqual(sorted(p.name for p in directory.glob('*.pkl')), ['30_agent.pkl', '30_buffer.pkl'])
            (directory / '30_buffer.pkl').unlink()
            with self.assertRaises(FileNotFoundError):
                load_rolling_checkpoint(1, 'test', agent)

    def test_online_optimizer_next_update_matches_after_restore(self):
        from ogpo.agents.ogpo import OGPOAgent
        from ogpo.utils.config_loader import load_config, config_to_agent_dict
        config = config_to_agent_dict(load_config('ogpo'))
        config.update(actor_hidden_dims=(16, 16), value_hidden_dims=(16, 16),
                      num_qs=2, q_min=-100., q_max=0.)
        bc_agent = OGPOAgent.create(1, jnp.zeros(23), jnp.zeros(7), config, config.adv_clip_min)
        agent = bc_agent.reset_optimizers_with_lr()
        grads = jax.tree.map(jnp.ones_like, agent.actor_network.params)
        critic_grads = jax.tree.map(jnp.ones_like, agent.critic_network.params)
        for _ in range(5):
            agent = agent.replace(actor_network=agent.actor_network.apply_gradients(grads=grads),
                                  critic_network=agent.critic_network.apply_gradients(grads=critic_grads))
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, OGPO_CHECKPOINT_ROOT=root):
            save_rolling_checkpoint(agent, {}, 10, 5, 1, 'test')
            restored, _, _, _, _ = load_rolling_checkpoint(1, 'test', bc_agent)
        for network, gradient in [('actor_network', grads), ('critic_network', critic_grads)]:
            expected = getattr(agent, network).apply_gradients(grads=gradient)
            actual = getattr(restored, network).apply_gradients(grads=gradient)
            for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
                np.testing.assert_array_equal(a, b)

    def test_bc_artifact_survives_later_save(self):
        with tempfile.TemporaryDirectory() as root:
            save_agent({'weights': jnp.ones(2)}, root, 5000)
            save_agent({'weights': jnp.zeros(2)}, root, 15000)
            self.assertTrue((Path(root) / 'params_5000.pkl').exists())
            self.assertTrue((Path(root) / 'params_15000.pkl').exists())
            self.assertFalse(list(Path(root).glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
