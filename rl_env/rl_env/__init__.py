"""rl_env: the reinforcement-learning environment for fixed-limit Omaha Hi-Lo 8-or-better.

It drives `//engine` in step mode and gives the learning side tables, decisions,
observation vectors, action masks, results and training records. No poker rule
lives here: betting, pots, showdown and hand evaluation stay in the engine.

Hidden information leaves through two separate doors, never through a
controller: during a hand only the critic input (`critic.py`, a separate
function and array that the observation code never imports); after a hand only
`finished_hole_cards` (for evaluation tools), readable until the table's next
hand starts. Both go through `Table.privileged_hole_cards`.
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
from .batch import Controller, DecisionBatch, FinishedHand, TableBatch
from .controllers import RandomController
from .critic import NO_CARD, PRIV_DIM, critic_cards, expand_critic
from .encoder import ENCODER_VERSION, OBS_DIM, OBS_LAYOUT, encode, encode_view
from .recorder import Recorder, TrainingRecords
from .stacks import HandSampler, HandSpec, StackConfig, draw_stacks
from .table import NoHandError, Table
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
    # table
    "Table",
    "NoHandError",
    # stacks
    "StackConfig",
    "draw_stacks",
    "HandSpec",
    "HandSampler",
    # observation
    "HandTracker",
    "ENCODER_VERSION",
    "OBS_DIM",
    "OBS_LAYOUT",
    "encode",
    "encode_view",
    # critic (privileged)
    "NO_CARD",
    "PRIV_DIM",
    "critic_cards",
    "expand_critic",
    # batch
    "Controller",
    "DecisionBatch",
    "FinishedHand",
    "TableBatch",
    "RandomController",
    "Recorder",
    "TrainingRecords",
]
