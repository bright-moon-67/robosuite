"""python -m defense_task.cli smoke|demo|collect ..."""
import argparse
from pathlib import Path
import json
import numpy as np

from .dataset import vectorize_observation, save_episode, OBS_KEYS
from .policy import DefenseHeuristic


def make_env(args, render=False):
    import robosuite as suite
    # Import class first to trigger robosuite's auto registration via EnvMeta.
    from .environment import DefenseTask  # noqa: F401

    if hasattr(suite, "load_composite_controller_config"):
        controller = suite.load_composite_controller_config(controller="BASIC", robot="Panda")
    else:
        raise RuntimeError("This package targets robosuite 1.5.x composite controllers.")
    # BASIC defaults may change, so explicitly set OSC_POSE for Panda.
    for item in controller.get("body_parts", {}).values():
        if isinstance(item, dict) and item.get("type") in ("OSC_POSITION", "OSC_POSE"):
            item["type"] = "OSC_POSE"
    env = suite.make(
        env_name="DefenseTask", robots="Panda", controller_configs=controller,
        num_blocks=args.blocks, motion_mode=args.motion,
        attack_speed=(args.min_speed, args.max_speed),
        stagger_seconds=args.stagger, seed=args.seed,
        has_renderer=render, has_offscreen_renderer=False,
        horizon=args.horizon, control_freq=20,
    )
    if env.action_dim != 7:
        raise RuntimeError(
            f"Expected OSC_POSE+1 gripper action (7D); got {env.action_dim}. "
            "Inspect controller body_parts for Panda and set OSC_POSE.")
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("smoke", "demo", "collect", "evaluate"))
    p.add_argument("--blocks", type=int, default=1)
    p.add_argument("--motion", choices=("linear", "sine", "pulsed"), default="linear")
    p.add_argument("--min-speed", type=float, default=0.008)
    p.add_argument("--max-speed", type=float, default=0.016)
    p.add_argument("--stagger", type=float, default=6.0)
    p.add_argument("--horizon", type=int, default=1400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.15,
                   help="Probability of random action for exploration data")
    p.add_argument("--out", default="datasets/defense")
    args = p.parse_args()
    if args.min_speed <= 0 or args.min_speed > args.max_speed:
        p.error("invalid speed interval")
    if not 0 <= args.epsilon <= 1:
        p.error("epsilon must be in [0,1]")
    if args.blocks < 1:
        p.error("blocks must be positive")
    env = make_env(args, render=args.mode == "demo")
    rng = np.random.default_rng(args.seed)
    try:
        totals = {"success":0, "breach":0, "cleared":0, "steps":0}
        n_episodes = args.episodes if args.mode in ("collect","evaluate") else 1
        out = Path(args.out)
        for e in range(n_episodes):
            obs = env.reset()
            policy = DefenseHeuristic(
                env.n_blocks, table_top=env.table_top,
                defense_center=env.defense_center, defense_radius=env.defense_radius,
                dt=1./env.control_freq)
            transitions = []
            done, info = False, {}
            for t in range(args.horizon):
                action = policy.act(obs)
                if args.mode == "smoke":
                    action = np.zeros(env.action_dim)
                    action[-1] = -1
                elif args.mode == "collect" and rng.random() < args.epsilon:
                    action = rng.uniform(-0.5, 0.5, env.action_dim)
                    action[-1] = rng.choice([-1., 1.])
                before = vectorize_observation(obs)
                new_obs, rew, done, info = env.step(action)
                after = vectorize_observation(new_obs)
                transitions.append(dict(
                    obs=before, action=np.asarray(action, dtype=np.float32),
                    next_obs=after, reward=rew,
                    r_disc=info["r_disc"], r_diff=info["r_diff"],
                    done=done, status=np.asarray(info["status"]),
                    cleared_count=info["cleared_count"],
                    breached_count=info["breached_count"],
                    block_pos_gt=info["block_pos_gt"],
                    block_vel_gt=info["block_vel_gt"]))
                obs = new_obs
                if args.mode == "demo":
                    env.render()
                if done:
                    break
            if args.mode == "collect":
                meta = dict(
                    base_seed=args.seed, episode_index=e, mode=args.motion,
                    block_count=env.n_blocks,
                    obs_keys=list(OBS_KEYS), control_freq=env.control_freq,
                    action_dim=env.action_dim, episode_idx=e,
                    status_codes={"attacking":0,"cleared":1,"breached":2},
                )
                path = out/f"episode_{e:05d}.npz"
                ep = save_episode(path, transitions, meta)
                print(f"Saved {path} : {len(ep['action'])} steps")
            totals["success"] += int(info.get("success", False))
            totals["breach"] += int(info.get("failed", False))
            totals["cleared"] += int(info.get("cleared_count", 0))
            totals["steps"] += len(transitions)
            print(f"episode={e} steps={len(transitions)} cleared={info.get('cleared_count')} "
                  f"breached={info.get('breached_count')} success={info.get('success')}")
        if args.mode in ("collect","evaluate"):
            print(f"summary: episodes={n_episodes}, wins={totals['success']}, "
                  f"breached_episodes={totals['breach']}, "
                  f"cleared_blocks={totals['cleared']}, "
                  f"mean_steps={totals['steps']/max(n_episodes,1):.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
