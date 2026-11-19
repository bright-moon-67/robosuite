"""Small interface smoke tests for the RSSM and deployment filter."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from defense_task.dataset import save_episode, vectorize_observation
from defense_task.integration import LatentFilter
from defense_task.worldmodel_stage import (
    DefenseRSSM,
    device_stats,
    fit_stats,
    load_checkpoint,
    load_episode,
    sample_batch,
    split_episodes,
)
from defense_task.worldmodel_eval import openloop_report, event_summary


def _episode(length=6):
    rows = []
    for t in range(length):
        robot = np.zeros(20, dtype=np.float32)
        robot[14:17] = [0.0, 0.0, 0.94]
        blocks = np.zeros((5, 6), dtype=np.float32)
        blocks[:, :3] = [0.35 - t * 0.001, 0.0, -0.118]
        blocks[:, 3] = 1
        status = np.zeros(5, dtype=np.float32)
        before = dict(
            defense_robot=robot,
            defense_blocks=blocks.ravel(),
            defense_status=status,
            defense_clock=[t / 1400],
            defense_eef_quat=[0, 0, 0, 1],
        )
        robot_next = robot.copy()
        blocks_next = blocks.copy()
        blocks_next[:, 0] -= 0.001
        after = dict(
            defense_robot=robot_next,
            defense_blocks=blocks_next.ravel(),
            defense_status=status,
            defense_clock=[(t + 1) / 1400],
            defense_eef_quat=[0, 0, 0, 1],
        )
        before_vec, after_vec = vectorize_observation(before), vectorize_observation(after)
        pos = blocks_next[:, :3] + robot_next[14:17]
        prev = blocks[:, :3] + robot[14:17]
        done = t == length - 1
        rows.append(
            dict(
                obs=before_vec,
                next_obs=after_vec,
                action=np.zeros(7, dtype=np.float32),
                reward=-0.2,
                r_disc=0.0,
                r_diff=-0.2,
                done=done,
                terminated=False,
                truncated=done,
                continuation=1.0,
                is_first=t == 0,
                is_last=done,
                status=status,
                cleared_count=0,
                breached_count=0,
                block_pos_gt=pos,
                block_vel_gt=np.zeros((5, 3)),
                prev_block_pos_gt=prev,
                prev_block_vel_gt=np.zeros((5, 3)),
                desired_vel_gt=np.zeros((5, 2)),
                time=t * 0.05,
                next_time=(t + 1) * 0.05,
                action_source=0,
            )
        )
    return rows


def test_rssm_train_forecast_and_filter_smoke(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "episode_00000.npz"
    save_episode(
        path,
        _episode(),
        {
            "seed": 1,
            "action_dim": 7,
            "control_freq": 20,
            "status_codes": {"attacking": 0, "cleared": 1, "breached": 2},
        },
    )
    ep = load_episode(path)
    stats_np = fit_stats([ep])
    stats = device_stats(stats_np, "cpu")
    model = DefenseRSSM(deter=16, latent=6, embed=24)
    batch = sample_batch([ep], 2, 4, np.random.default_rng(42), burn=2)
    np.testing.assert_array_equal(batch["states"][0].numpy(), ep["states"])
    assert set(batch) == {"states", "actions", "reward", "cont"}
    loss, _ = model.batch_loss(batch, stats, burn=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.optim.Adam(model.parameters(), lr=3e-4).step()

    model.eval()
    context = torch.from_numpy(ep["states"][:3])[None]
    context_actions = torch.from_numpy(ep["action"][:2])[None]
    future_actions = torch.from_numpy(ep["action"][2:4])[None]
    positions, statuses, rewards, continuation = model.forecast(context, context_actions, future_actions, stats)
    assert positions.shape == (1, 2, 5, 3)
    assert statuses.shape == (1, 2, 5, 3)
    assert rewards.shape == continuation.shape == (1, 2)
    assert torch.isfinite(positions).all()
    torch.testing.assert_close(statuses.sum(-1), torch.ones(1, 2, 5))
    # Mutating only unobserved future frames cannot change prior predictions.
    ep["states"][3:, 14] += 100
    torch.testing.assert_close(model.forecast(context, context_actions, future_actions, stats)[0], positions)
    ep["states"][3:, 14] -= 100

    latent_filter = LatentFilter(model, stats_np)
    assert latent_filter.reset(ep["states"][0]).shape == (1, 22)
    assert latent_filter.update(ep["action"][0], ep["states"][1]).shape == (1, 22)
    imagined = latent_filter.imagine(ep["action"][:2])
    assert imagined[0].shape == (1, 2, 5, 3)
    features = latent_filter.features().clone()
    latent_filter.imagine(ep["action"][:2])
    torch.testing.assert_close(features, latent_filter.features())

    checkpoint_path = tmp_path / "world_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "stats": stats,
            "architecture": {"deter": 16, "latent": 6, "embed": 24},
        },
        checkpoint_path,
    )
    restored, restored_stats, _ = load_checkpoint(checkpoint_path)
    restored_output = restored.forecast(context, context_actions, future_actions, restored_stats)[0]
    torch.testing.assert_close(restored_output, positions)
    assert LatentFilter.from_checkpoint(checkpoint_path).reset(ep["states"][0]).shape == (1, 22)
    report = openloop_report(restored, [ep], restored_stats, horizons=(2, 10), burn=2, windows_per_ep=1)
    assert report["2"]["windows"] == 1
    assert report["10"]["windows"] == 0
    assert report["10"]["endpoint"]["all"]["wm"]["mean_m"] is None
    assert report["2"]["breach_window"]["recall"] is None


def test_data_and_event_boundaries_smoke(tmp_path):
    import json

    paths = list(range(100))
    train, val, test = split_episodes(paths)
    assert [len(train), len(val), len(test)] == [70, 15, 15]
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    path = tmp_path / "episode_00000.npz"
    metadata = {
        "seed": 1,
        "action_dim": 7,
        "control_freq": 20,
        "status_codes": {"attacking": 0, "cleared": 1, "breached": 2},
    }
    save_episode(path, _episode(), metadata)
    ep = load_episode(path)
    assert ep["continuation"][-1] == 1
    with np.load(path) as source:
        arrays = {k: source[k] for k in source.files}
    arrays["continuation"][-1] = 0
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Continuation"):
        load_episode(path)
    arrays["truncated"][-1] = False
    arrays["terminated"][-1] = True
    np.savez(path, **arrays)
    assert load_episode(path)["continuation"][-1] == 0
    schema = json.loads(path.with_suffix(".json").read_text())
    schema["schema"]["fields"]["defense_status"]["start"] = 49
    path.with_suffix(".json").write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="schema"):
        load_episode(path)
    metrics = event_summary([1, 0, 1], [0.9, 0.7, 0.1])
    assert metrics["recall"] == 0.5 and metrics["precision"] == 0.5
    assert metrics["pr_auc_average_precision"] == pytest.approx(5 / 6)
