"""Explicit action convention shared by the environment and observation-only baseline."""


def panda_osc_config():
    from robosuite import load_composite_controller_config

    config = load_composite_controller_config(robot="Panda")
    arm = config["body_parts"]["right"]
    arm.update(
        type="OSC_POSE",
        input_type="delta",
        input_ref_frame="world",
        input_min=-1,
        input_max=1,
        output_min=[-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        output_max=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    )
    return config
