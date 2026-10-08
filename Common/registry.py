"""
Common/registry.py

One list of every model in the study, and how to build each one. Tuning,
main.py and the experiments all read from here, so they can never disagree
about which models exist or how they are set up.

  Bandits (6)      EpsilonGreedy, LinUCB, LinTS x {cost_sensitive, label_matching}
  Supervised (3)   LogisticRegression, RandomForest, XGBoost
  Reference (3)    Oracle, FullInfoOnline, PartialInfoOnline   (Experiment 2)

Each entry (ModelSpec) records:
  kind            how the runner drives it:
                    "bandit"      streams all data, learns from its own rewards
                    "online_prob" streams all data, learns from TRUE labels,
                                  outputs P(fraud) -> cost-aware rule
                    "supervised"  trained once, frozen, outputs P(fraud)
                    "oracle"      knows the labels
  deterministic   True -> identical results for every seed, so run once
  depends_on_C_a  True -> what it learns changes with C_a, so it is re-run for
                  every C_a. False -> its decisions (0/1 bandits) or
                  probabilities (supervised, Full-Info) are computed once and
                  re-scored/re-thresholded per C_a.
  grid            its tuning grid (config); None for the Oracle
"""

from dataclasses import dataclass
from typing import Callable, Optional

from Common.config import (
    BANDIT_GRIDS, CONTEXT_FEATURE_COLS, REFERENCE_GRIDS, REWARD_TYPES, SL_GRIDS,
)

FAMILY_BANDIT = "Bandit"
FAMILY_SUPERVISED = "Supervised"
FAMILY_REFERENCE = "Reference"

REWARD_SHORT = {"cost_sensitive": "CS", "label_matching": "LM"}


@dataclass(frozen=True)
class ModelSpec:
    name: str                       # "LinTS", "XGBoost", "FullInfoOnline", ...
    kind: str                       # "bandit" | "online_prob" | "supervised" | "oracle"
    family: str                     # Bandit | Supervised | Reference
    reward_type: Optional[str]      # bandits only
    deterministic: bool
    depends_on_C_a: bool
    grid: Optional[dict]
    builder: Optional[Callable]     # builder(seed, hyperparameters, data, C_a) -> model

    @property
    def id(self):
        """Unique id, e.g. 'CS_LinTS', 'LM_LinTS', 'XGBoost' (cache files, logs)."""
        return f"{REWARD_SHORT[self.reward_type]}_{self.name}" if self.reward_type else self.name

    @property
    def label(self):
        """Readable name, e.g. 'LinTS (cost-sensitive)'."""
        if self.reward_type:
            return f"{self.name} ({'cost-sensitive' if self.reward_type == 'cost_sensitive' else '0/1'})"
        return self.name

    def build(self, seed, hyperparameters, data, C_a):
        return self.builder(seed, dict(hyperparameters), data, C_a)


# =====================================================================
# Builders (imports kept local so loading the registry stays light)
# =====================================================================
def _bandit_builder(algorithm, reward_type):
    def build(seed, hp, data, C_a):
        from Contextual_Bandits.EpsilonGreedy import EpsilonGreedy
        from Contextual_Bandits.LinTS import LinTS
        from Contextual_Bandits.LinUCB import LinUCB
        cls = {"EpsilonGreedy": EpsilonGreedy, "LinUCB": LinUCB, "LinTS": LinTS}[algorithm]
        return cls(data.n_features + 1, seed, reward_type, **hp)
    return build


def _supervised_builder(name):
    def build(seed, hp, data, C_a):
        import importlib
        return importlib.import_module(f"Supervised_Learning.{name}").build(seed, **hp)
    return build


def _full_info_builder(seed, hp, data, C_a):
    from Experiments.reference_policies import FullInfoOnline
    return FullInfoOnline(data.n_features + 1, **hp)


def _partial_info_builder(seed, hp, data, C_a):
    from Experiments.reference_policies import PartialInfoOnline
    idx = CONTEXT_FEATURE_COLS.index("log_amount")
    return PartialInfoOnline(data.n_features + 1, C_a, idx, data.mu[idx], data.sigma[idx], **hp)


def _supervised_deterministic(name):
    import importlib
    return importlib.import_module(f"Supervised_Learning.{name}").DETERMINISTIC


# =====================================================================
# The registry
# =====================================================================
def bandit_specs():
    return [ModelSpec(name=a, kind="bandit", family=FAMILY_BANDIT, reward_type=r,
                      deterministic=False,
                      # the 0/1 reward never involves C_a; the dollar reward does
                      depends_on_C_a=(r == "cost_sensitive"),
                      grid=BANDIT_GRIDS[a], builder=_bandit_builder(a, r))
            for a in BANDIT_GRIDS for r in REWARD_TYPES]


def supervised_specs():
    return [ModelSpec(name=n, kind="supervised", family=FAMILY_SUPERVISED, reward_type=None,
                      deterministic=_supervised_deterministic(n), depends_on_C_a=False,
                      grid=SL_GRIDS[n], builder=_supervised_builder(n))
            for n in SL_GRIDS]


def reference_specs():
    return [
        ModelSpec("Oracle", "oracle", FAMILY_REFERENCE, None, True, True, None, None),
        ModelSpec("FullInfoOnline", "online_prob", FAMILY_REFERENCE, None, True, False,
                  REFERENCE_GRIDS["FullInfoOnline"], _full_info_builder),
        ModelSpec("PartialInfoOnline", "bandit", FAMILY_REFERENCE, None, True, True,
                  REFERENCE_GRIDS["PartialInfoOnline"], _partial_info_builder),
    ]


def all_specs():
    return bandit_specs() + supervised_specs() + reference_specs()


def tuned_specs():
    """Every model that has a tuning grid (all but the Oracle)."""
    return [s for s in all_specs() if s.grid is not None]


def get_spec(model_id):
    for s in all_specs():
        if s.id == model_id:
            return s
    raise KeyError(f"unknown model id {model_id!r}")