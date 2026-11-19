"""Run smoke/demo/evaluate/collect/validate for the pre-world-model stage."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .dataset import OBS_KEYS, ReplayBuffer, load_episode, save_episode, vectorize_observation
from .policy import DefenseHeuristic


def make_env(args, render=False):
    import robosuite as suite

    from .environment import DefenseTask  # registers only when this extension is imported

    env = suite.make(
        "DefenseTask",
        robots="Panda",
        num_blocks=args.blocks,
        motion_mode=args.motion,
        attack_speed=(args.min_speed, args.max_speed),
        stagger_seconds=args.stagger,
        seed=args.seed,
        has_renderer=render,
        has_offscreen_renderer=bool(args.video),
        renderer="mjviewer" if render else "mujoco",
        horizon=args.horizon,
        control_freq=20,
    )
    if env.action_dim != 7:
        env.close()
        raise RuntimeError("Expected normalized 7D world-frame OSC_POSE delta + gripper")
    return env


def diagnostics(env):
    return (
        np.stack([env._block_pos(i) for i in range(env.n_blocks)]),
        np.stack([env._block_v(i) for i in range(env.n_blocks)]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("smoke", "demo", "collect", "evaluate", "validate"))
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--motion", choices=("linear", "sine", "pulsed"), default="linear")
    p.add_argument("--min-speed", type=float, default=0.008)
    p.add_argument("--max-speed", type=float, default=0.016)
    p.add_argument("--stagger", type=float, default=6.0)
    p.add_argument("--horizon", type=int, default=1400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.15, help="Random-action probability in collect only")
    p.add_argument("--out", type=Path, default=Path("datasets/defense"))
    p.add_argument("--report", type=Path, help="JSON evaluation/collection report")
    p.add_argument("--video", type=Path, help="Headless MP4; demo/smoke only, original simulation speed")
    args = p.parse_args()
    if not (
        np.isfinite([args.min_speed, args.max_speed, args.stagger, args.epsilon]).all()
        and 0 <= args.min_speed <= args.max_speed
        and args.stagger >= 0
        and 0 <= args.epsilon <= 1
    ):
        p.error("Invalid speed interval, stagger or epsilon")
    if not 1 <= args.blocks <= 12 or args.episodes < 1 or args.horizon < 1 or args.seed < 0:
        p.error("Invalid blocks, episodes, horizon or seed")
    if args.video and args.mode not in ("smoke", "demo"):
        p.error("--video is for a single smoke/demo episode")
    if args.report and args.report.exists():
        p.error("Report already exists; choose a new path")
    if args.video and args.video.exists():
        p.error("Video already exists; choose a new path")
    if args.mode == "validate":
        paths = sorted(args.out.glob("episode_*.npz"))
        if not paths:
            p.error("No episodes found")
        replay = ReplayBuffer(seed=args.seed)
        lengths = []
        for path in paths:
            episode = load_episode(path)
            replay.add(episode)
            lengths.append(len(episode["action"]))
        sample = replay.sample(4, min(32, min(lengths)))
        print(
            json.dumps(
                {
                    "valid_episodes": len(paths),
                    "transitions": sum(lengths),
                    "sample_state_shape": list(sample["state"].shape),
                    "sample_action_shape": list(sample["action"].shape),
                    "model_keys": list(sample),
                },
                indent=2,
            )
        )
        return

    env = make_env(args, render=args.mode == "demo" and not args.video)
    writer = None
    started = time.monotonic()
    reports = []
    next_file_index = 0
    if args.mode == "collect":
        args.out.mkdir(parents=True, exist_ok=True)
        existing = list(args.out.glob("episode_*.npz")) + list(args.out.glob("episode_*.json"))
        if existing:
            next_file_index = max(int(path.stem.split("_")[-1]) for path in existing) + 1
    try:
        if args.video:
            import imageio.v2 as imageio

            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=20, codec="libx264", quality=8)
        count = args.episodes if args.mode in ("collect", "evaluate") else 1
        for episode_index in range(count):
            episode_seed = args.seed + episode_index
            env.rng = np.random.default_rng(episode_seed)
            rng = np.random.default_rng(episode_seed)
            obs = env.reset()
            policy = DefenseHeuristic(
                env.n_blocks,
                table_top=env.table_top,
                defense_center=env.defense_center,
                defense_radius=env.defense_radius,
                dt=1 / env.control_freq,
            )
            rows, speed_errors = [], []
            force_peaks = 0.0
            initial_positions, _ = diagnostics(env)
            for t in range(args.horizon):
                action = policy.act(obs)
                random_action = args.mode == "collect" and rng.random() < args.epsilon
                if args.mode == "smoke":
                    action = np.zeros(7)
                    action[-1] = -1
                elif random_action:
                    action = rng.uniform(-0.5, 0.5, 7)
                    action[-1] = rng.choice([-1.0, 1.0])
                # Use the same float32 values for execution and storage.
                action = np.asarray(action, dtype=np.float32)
                before = vectorize_observation(obs)
                prev_pos, prev_vel = diagnostics(env)
                before_time = float(env.sim.data.time)
                new_obs, reward, done, info = env.step(action)
                rows.append(
                    dict(
                        obs=before,
                        action=action,
                        next_obs=vectorize_observation(new_obs),
                        reward=reward,
                        r_disc=info["r_disc"],
                        r_diff=info["r_diff"],
                        done=done,
                        terminated=info["terminated"],
                        truncated=info["truncated"],
                        continuation=info["continuation"],
                        is_first=t == 0,
                        is_last=done,
                        status=info["status"],
                        cleared_count=info["cleared_count"],
                        breached_count=info["breached_count"],
                        block_pos_gt=info["block_pos_gt"],
                        block_vel_gt=info["block_vel_gt"],
                        prev_block_pos_gt=prev_pos,
                        prev_block_vel_gt=prev_vel,
                        desired_vel_gt=info["desired_vel_gt"],
                        time=before_time,
                        next_time=info["time"],
                        action_source=int(random_action),
                    )
                )
                # Diagnostics only: exclude touching/grasped or airborne blocks from free-motion speed error.
                for i in range(env.n_blocks):
                    if (
                        env.status[i] == 0
                        and not env.grasp_flags[i]
                        and env.sim.data.time > env._motion[i]["start"] + 0.3
                        and info["block_pos_gt"][i, 2] < env.table_top + 0.03
                        and np.linalg.norm(info["block_pos_gt"][i] - env._eef()) > 0.09
                    ):
                        speed_errors.append(
                            float(np.linalg.norm(info["block_vel_gt"][i, :2] - info["desired_vel_gt"][i]))
                        )
                force_peaks = max(
                    force_peaks, float(np.linalg.norm(env.sim.data.xfrc_applied[env.block_body_ids, :2], axis=1).max())
                )
                obs = new_obs
                if writer:
                    from PIL import Image, ImageDraw

                    frame = Image.fromarray(env.sim.render(960, 720, camera_name="agentview")[::-1])
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 960, 44), fill=(20, 25, 35))
                    draw.text(
                        (15, 14),
                        f"DefenseTask | {args.motion} | t={info['time']:.2f}s | cleared={info['cleared_count']}/{env.n_blocks} | breached={info['breached_count']} | {policy.stage}",
                        fill="white",
                    )
                    writer.append_data(np.asarray(frame))
                elif args.mode == "demo":
                    env.render()
                if done:
                    break
            meta = dict(
                seed=episode_seed,
                episode_index=episode_index,
                block_count=env.n_blocks,
                obs_keys=list(OBS_KEYS),
                control_freq=env.control_freq,
                action_dim=env.action_dim,
                config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                initial_positions_gt=initial_positions.tolist(),
                hidden_motion_diagnostics=[
                    {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in cfg.items()} for cfg in env._motion
                ],
                status_codes={"attacking": 0, "cleared": 1, "breached": 2},
            )
            if args.mode == "collect":
                path = args.out / f"episode_{next_file_index + episode_index:05d}.npz"
                save_episode(path, rows, meta)
            else:
                # Evaluate alignment on actual rollouts, too; no need to persist their arrays.
                from .dataset import DIAGNOSTIC_KEYS, MODEL_KEYS, validate_episode

                validate_episode({k: np.asarray([row[k] for row in rows]) for k in MODEL_KEYS + DIAGNOSTIC_KEYS})
            result = dict(
                seed=episode_seed,
                steps=len(rows),
                success=info["success"],
                failed=info["failed"],
                truncated=info["truncated"],
                cleared=info["cleared_count"],
                breached=info["breached_count"],
                return_total=float(sum(row["reward"] for row in rows)),
                events=env.events,
                speed_rmse=float(np.sqrt(np.mean(np.square(speed_errors)))) if speed_errors else None,
                free_motion_samples=len(speed_errors),
                sampled_force_peak=force_peaks,
                initial_positions_gt=initial_positions.tolist(),
            )
            reports.append(result)
            print(
                f"seed={episode_seed} steps={len(rows)} cleared={result['cleared']}/{env.n_blocks} breached={result['breached']} success={result['success']} truncated={result['truncated']}",
                flush=True,
            )
        summary = dict(
            episodes=len(reports),
            wins=sum(r["success"] for r in reports),
            breached_episodes=sum(r["failed"] for r in reports),
            timeouts=sum(r["truncated"] for r in reports),
            cleared_blocks=sum(r["cleared"] for r in reports),
            total_blocks=len(reports) * env.n_blocks,
            mean_steps=float(np.mean([r["steps"] for r in reports])),
            elapsed_seconds=time.monotonic() - started,
        )
        print(json.dumps(summary, indent=2), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps({"config": meta["config"], "summary": summary, "episodes": reports}, indent=2),
                encoding="utf8",
            )
    finally:
        if writer:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
