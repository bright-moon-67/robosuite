"""Render TableDefense with an idle arm or a simple, explicitly scripted preview.

MUJOCO_GL=egl python -m robosuite.demos.demo_table_defense --mode scripted --gif
MUJOCO_GL=egl python -m robosuite.demos.demo_table_defense --mode incoming

The script sends ordinary OSC commands and closes the physical gripper. It does
not move object poses, attach objects, or stand in for a trained defense policy.
Requires imageio and imageio-ffmpeg in addition to robosuite dependencies.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

import robosuite as suite


class ScriptedPreview:
    """Fixed-order pick/lift/release sequence for checking scene feasibility only."""

    PHASES = (("approach", 2.0), ("descend", 1.0), ("close", 0.8),
              ("lift", 1.2), ("carry", 1.0), ("release", 0.6))

    def __init__(self, env):
        self.env = env
        self.site = env.robots[0].eef_site_id["right"]
        self.orientation = Rotation.from_matrix(env.sim.data.site_xmat[self.site].reshape(3, 3)).as_rotvec()
        self.home = env.sim.data.site_xpos[self.site].copy()
        self.phase_index, self.object_index = 0, 0
        self.phase_start = 2.0
        self.phase = "observe incoming objects"
        self.phase_origin = self.home.copy()
        self.lift_xy = None
        self.failed_picks = []

    def action(self):
        env, now = self.env, self.env.sim.data.time
        if now < 2.0 or self.object_index >= env.num_attackers:
            if self.object_index >= env.num_attackers:
                self.phase = "preview complete"
            return np.r_[self.home, self.orientation, -1]
        name, duration = self.PHASES[self.phase_index]
        if now - self.phase_start >= duration - 1e-8:
            if name == "lift" and not env.removed[self.object_index]:
                self.failed_picks.append(self.object_index)
            self.phase_index += 1
            if self.phase_index == len(self.PHASES):
                self.phase_index = 0
                self.object_index += 1
                if self.object_index >= env.num_attackers:
                    self.phase = "preview complete"
                    return np.r_[self.home, self.orientation, -1]
            self.phase_start = now
            self.phase_origin = env.sim.data.site_xpos[self.site].copy()
            self.lift_xy = env.attacker_positions[min(self.object_index, env.num_attackers - 1), :2].copy()
            name, duration = self.PHASES[self.phase_index]
        self.phase = f"{name}  /  block {self.object_index + 1}"
        pos = env.attacker_positions[self.object_index]
        progress = np.clip((now - self.phase_start) / (duration * 0.68), 0, 1)
        smooth = progress * progress * (3 - 2 * progress)
        grip = -1 if name in {"approach", "descend", "release"} else 1
        if name == "approach":
            target = (1 - smooth) * self.phase_origin + smooth * np.r_[pos[:2], 1.04]
            target[2] = max(target[2], 1.04)
        elif name == "descend":
            target = np.r_[pos[:2], (1 - smooth) * self.phase_origin[2] + smooth * 0.824]
        elif name == "close":
            target = np.r_[pos[:2], 0.824]
        elif name == "lift":
            target = np.r_[self.lift_xy, (1 - smooth) * self.phase_origin[2] + smooth * 1.08]
        else:
            drop = np.array([-0.30, -0.43 + 0.065 * self.object_index, 1.08])
            target = (1 - smooth) * self.phase_origin + smooth * drop if name == "carry" else drop
        return np.r_[target, self.orientation, grip]


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def compose_frame(env, args, phase):
    canvas = Image.new("RGB", (1280, 800), (17, 23, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 15), "TABLE DEFENSE", fill="white", font=font(30, True))
    subtitle = "PHYSICAL GRASP PREVIEW | scripted OSC, no learned policy" if args.mode == "scripted" else "INCOMING OBJECTS | idle robot, breach demonstration"
    draw.text((28, 55), subtitle, fill=(173, 192, 213), font=font(17))
    draw.text((995, 22), f"t = {env.sim.data.time:05.1f} s", fill="white", font=font(25, True))
    draw.text((995, 58), f"Playback {args.playback:g}x", fill=(173, 192, 213), font=font(16))
    main = env.sim.render(width=832, height=624, camera_name="defenseview")[::-1]
    top = env.sim.render(width=400, height=400, camera_name="defensetop")[::-1]
    canvas.paste(Image.fromarray(main), (24, 100))
    canvas.paste(Image.fromarray(top), (872, 100))
    draw.text((888, 110), "TOP VIEW", fill="white", font=font(15, True))
    info = env.task_info()
    color = (255, 105, 94) if info["failed"] else (76, 224, 207)
    label = "BREACH DETECTED" if info["failed"] else ("ALL OBJECTS REMOVED" if info["success"] else "DEFENSE ZONE CLEAR")
    draw.text((888, 520), label, fill=color, font=font(19, True))
    draw.text((888, 555), f"Removed {info['removed_count']}/{env.num_attackers}    Breaches {info['breach_count']}", fill="white", font=font(18))
    draw.text((888, 590), f"Radius  {env.defense_radius:.2f} m", fill=(187, 204, 221), font=font(17))
    draw.text((888, 619), f"Speed   {args.speed * 100:.1f} cm/s", fill=(187, 204, 221), font=font(17))
    draw.text((888, 648), f"Launch interval  {args.launch_interval:g} s", fill=(187, 204, 221), font=font(17))
    draw.text((888, 677), f"Motion  {args.motion}", fill=(187, 204, 221), font=font(17))
    draw.text((28, 743), phase.upper(), fill="white", font=font(18, True))
    draw.text((28, 774), "Cyan: defense boundary   |   Lift bottom > 12 cm for 0.15 s = removed   |   Units: meters", fill=(154, 173, 194), font=font(13))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["incoming", "scripted"], default="scripted")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--speed", type=float, default=0.008)
    parser.add_argument("--launch-interval", type=float, default=4.0)
    parser.add_argument("--motion", choices=["linear", "weave", "mixed"], default="linear")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--playback", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--no-video", action="store_true", help="Run physical preview and save diagnostics only")
    args = parser.parse_args()
    if args.playback <= 0 or args.fps <= 0 or (args.seconds is not None and args.seconds <= 0):
        parser.error("Playback speed, fps and duration must be positive")
    if args.fps > 20 * args.playback:
        parser.error("fps must not exceed control_freq (20) times playback speed")
    args.output = args.output or Path(f"artifacts/table_defense/defense_{args.mode}.mp4")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = suite.load_composite_controller_config(robot="Panda")
    config["body_parts"]["right"]["input_type"] = "absolute"
    config["body_parts"]["right"]["input_ref_frame"] = "world"
    env = suite.make("TableDefense", num_attackers=args.count, defense_radius=args.radius,
        attack_speed=args.speed, motion=args.motion, launch_interval=args.launch_interval,
        controller_configs=config, renderer="mujoco", has_renderer=False,
        has_offscreen_renderer=not args.no_video, use_camera_obs=False,
        hard_reset=False, ignore_done=True, seed=7)
    writer = None
    frames, records = [], []
    try:
        env.reset()
        preview = ScriptedPreview(env)
        seconds = args.seconds or (4 + args.count * sum(p[1] for p in preview.PHASES) if args.mode == "scripted" else 40)
        if not args.no_video:
            writer = imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", quality=8,
                                        macro_block_size=16, ffmpeg_log_level="error")
        next_frame, next_log = 0.0, 0.0
        previous_phase = ""
        for step in range(int(seconds * env.control_freq) + 1):
            phase = preview.phase if args.mode == "scripted" else "watch the slow approach and boundary crossing"
            if phase != previous_phase:
                print(f"t={env.sim.data.time:.2f}s  {phase}  removed={env.removed.sum()} breached={env.breached.sum()}", flush=True)
                previous_phase = phase
            if writer and env.sim.data.time >= next_frame - 1e-8:
                frame = compose_frame(env, args, phase)
                writer.append_data(np.asarray(frame))
                if args.gif and round(next_frame * args.fps / args.playback) % 2 == 0:
                    frames.append(frame.resize((640, 400), Image.Resampling.LANCZOS))
                if step == 0:
                    frame.save(args.output.with_suffix(".png"))
                next_frame += args.playback / args.fps
            if env.sim.data.time >= next_log - 1e-8:
                info = env.task_info()
                records.append({"time": info["time"], "positions": env.attacker_positions.tolist(),
                                "removed_count": info["removed_count"], "breach_count": info["breach_count"],
                                "phase": phase, "clearances": info["clearances"].tolist()})
                next_log += 0.5
            if step == int(seconds * env.control_freq):
                break
            action = preview.action() if args.mode == "scripted" else np.r_[preview.home, preview.orientation, -1]
            env.step(action)
        if writer:
            compose_frame(env, args, phase).save(args.output.with_name(args.output.stem + "_final.png"))
        result = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "events": env.events, "success": env._check_success(),
                  "removed_count": int(env.removed.sum()), "breach_count": int(env.breached.sum()),
                  "failed_scripted_picks": preview.failed_picks, "samples": records}
        args.output.with_suffix(".json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in {"samples", "config"}}, indent=2), flush=True)
    finally:
        if writer:
            writer.close()
        env.close()
    if frames:
        palette = frames[0].quantize(colors=128)
        indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        indexed[0].save(args.output.with_suffix(".gif"), save_all=True, append_images=indexed[1:],
                       duration=round(2000 / args.fps), loop=0, optimize=True)
    if not args.no_video:
        print(f"Saved {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
