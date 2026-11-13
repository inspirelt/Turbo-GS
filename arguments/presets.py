"""Presets for native-4K Mip-NeRF 360 scenes, applied by train.py via set_defaults (explicit flags win).
The class defaults in arguments/__init__.py stay unchanged so saved cfg_args keep their meaning."""
from argparse import SUPPRESS

DEFAULT_PRESET = "15k"

PRESETS = {
    "10k": {"iterations": 10000, "update_until": 5000, "max_anchors": 3000000},
    "15k": {"iterations": 15000, "update_until": 7500, "max_anchors": 2500000},
    "25k": {"iterations": 25000, "update_until": 12500, "max_anchors": 3000000},
}

SHARED = {
    "densify_color_threshold": 5e-6,
    "mlp_cov_lr_final": 4e-5,
    "densify_grad_threshold": 5e-5,
    "budget_select": "random",
    "dilation": 2,
    "feat_dim": 56,
    "start_stat": 100,
    "update_from": 300,
    "update_interval": 20,
    "lod": 0,
    "voxel_size": 0.001,
    "update_init_factor": 16,
    "appearance_dim": 0,
    "ratio": 1,
    "data_device": "cuda",
    "test_iterations": [99999],
}

LR_MAX_STEPS = (
    "position_lr_max_steps",
    "offset_lr_max_steps",
    "mlp_opacity_lr_max_steps",
    "mlp_cov_lr_max_steps",
    "mlp_color_lr_max_steps",
    "mlp_featurebank_lr_max_steps",
    "appearance_lr_max_steps",
)


def preset_defaults(name):
    defaults = dict(SHARED, **PRESETS[name])
    defaults.update({key: defaults["iterations"] for key in LR_MAX_STEPS})
    defaults["save_iterations"] = [defaults["iterations"]]
    return defaults


def explicit_flags(parser, argv):
    """Names of the options present on argv, however they were spelled (--x=1, abbreviations)."""
    action_defaults = [action.default for action in parser._actions]
    parser_defaults, parser._defaults = parser._defaults, {}
    for action in parser._actions:
        action.default = SUPPRESS
    given = parser.parse_known_args(argv)[0]
    for action, default in zip(parser._actions, action_defaults):
        action.default = default
    parser._defaults = parser_defaults
    return set(vars(given))


def derive_schedule(args, explicit):
    """An explicit --iterations moves whatever follows from it, unless that was given too.
    Otherwise --iterations alone would leave the LR schedules and densification on the preset."""
    if "iterations" not in explicit:
        return
    if "update_until" not in explicit:
        args.update_until = args.iterations // 2
    if "save_iterations" not in explicit:
        args.save_iterations = [args.iterations]
    for key in LR_MAX_STEPS:
        if key not in explicit:
            setattr(args, key, args.iterations)


def preset_summary(args):
    steps = sorted({getattr(args, key) for key in LR_MAX_STEPS})
    return "[PRESET] name={} iterations={} update_until={} max_anchors={} lr_max_steps={}".format(
        args.preset, args.iterations, args.update_until, args.max_anchors, ",".join(map(str, steps)))
