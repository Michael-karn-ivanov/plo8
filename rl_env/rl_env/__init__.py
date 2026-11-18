"""rl_env: the reinforcement-learning environment for fixed-limit Omaha Hi-Lo 8-or-better.

It drives `//engine` in step mode and gives the learning side tables, decisions,
observation vectors, action masks, results and training records. No poker rule
lives here: betting, pots, showdown and hand evaluation stay in the engine.

Hidden information leaves only through the critic input (`critic.py`), a
separate function and array that the observation code never imports.
"""

from .actions import (
    ACTION_NAMES,
    BET_RAISE,
    CHECK_CALL,
    FOLD,
    N_ACTIONS,
    IllegalMaskedActionError,
    action_kind,
    action_mask,
    mask_array,
    to_engine,
)
from .stacks import HandSampler, HandSpec, StackConfig, draw_stacks
from .tracker import HandTracker

__all__ = [
    # actions
    "FOLD",
    "CHECK_CALL",
    "BET_RAISE",
    "N_ACTIONS",
    "ACTION_NAMES",
    "IllegalMaskedActionError",
    "action_mask",
    "mask_array",
    "to_engine",
    "action_kind",
    # stacks
    "StackConfig",
    "draw_stacks",
    "HandSpec",
    "HandSampler",
    # observation
    "HandTracker",
]
