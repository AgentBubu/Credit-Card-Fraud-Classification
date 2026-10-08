"""
Common/registry.py

One list of every model in the study, and how to build each one. Tuning,
main.py and the experiments all read from here, so they can never disagree
about which models exist or how they are set up.

  Bandits (6)       CB_CS_<algorithm>   cost-sensitive reward
                    CB_LM_<algorithm>   0/1 (label-matching) reward
                    for EpsilonGreedy, LinUCB, LinTS
  Supervised (6)    SL_CSD_<model>      cost-sensitive DECISION (learns the
                                        label; cost in the decision rule)
                    SL_CSL_<model>      cost-sensitive LEARNING (learns from
                                        cost-weighted examples)
                    for LogisticRegression, RandomForest, XGBoost
  Reference (3)     Oracle, FullInfoOnline, PartialInfoOnline   (Experiment 2)

Each entry (ModelSpec) records:
  id              unique name used in every results file, e.g. 'SL_CSL_XGBoost'
  name            the algorithm, e.g. 'XGBoost'
  variant         'CS' / 'LM' (bandits), 'CSD' / 'CSL' (supervised), '' (reference)
  kind            how the runner drives it:
                    "bandit"         streams all data, learns from its own rewards
                    "online_prob"    streams all data, learns from TRUE labels,
                                     outputs P(fraud) -> cost-aware rule
                    "supervised"     (CSD) trained once, frozen, outputs P(fraud)
                                     -> cost-aware rule
                    "supervised_csl" (CSL) trained once PER C_a on cost-weighted
                                     examples, frozen, blocks if P > 0.5
                    "oracle"         knows the labels
  deterministic   True -> identical results for every seed, so run once
  depends_on_C_a  True -> what it learns changes with C_a, so it is re-run for
                  every C_a. False -> its decisions (0/1 bandits) or
                  probabilities (CSD supervised, Full-Info) are computed once
                  and re-scored / re-thresholded per C_a.
  grid            its tuning grid (config); None for the Oracle
  cache_id        name its saved runs are filed under in Results/cache. Models
                  that existed before the renaming keep their OLD name here
                  (e.g. 'CS_LinTS', 'XGBoost'), so their runs are never recomputed.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from Common.config import (
    BANDIT_GRIDS, CONTEXT_FEATURE_COLS, REFERENCE_GRIDS, REWARD_TYPES, SL_CSD_GRIDS,
    SL_CSL_GRIDS,
)

FAMILY_BANDIT = "Bandit"
FAMILY_SUPERVISED = "Supervised"
FAMILY_REFERENCE = "Reference"

REWARD_SHORT = {"cost_sensitive": "CS", "label_matching": "LM"}
VARIANT_LABEL = {"CS": "cost-sensitive", "LM": "0/1", "CSD": "CSD", "CSL": "CSL", "": ""}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str                       # "LinTS", "XGBoost", "FullInfoOnline", ...
    variant: str                    # "CS" | "LM" | "CSD" | "CSL" | ""
    kind: str
    family: str                     # Bandit | Supervised | Reference
    reward_type: Optional[str]      # bandits only
    deterministic: bool
    depends_on_C_a: bool
    grid: Optional[dict]
    builder: Optional[Callable]     # builder(seed, hyperparameters, data, C_a) -> model
    cache_id: str = ""

    def __post_init__(self):
        if not self.cache_id:
            object.__setattr__(self, "cache_id", self.id)

    @property
    def label(self):
        """Readable name, e.g. 'LinTS (cost-sensitive)', 'XGBoost (CSL)'."""
        return f"{self.name} ({VARIANT_LABEL[self.variant]})" if self.variant else self.name

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


def _sl_module(name):
    import importlib
    return importlib.import_module(f"Supervised_Learning.{name}")


def _csd_builder(name):
    def build(seed, hp, data, C_a):
        return _sl_module(name).build(seed, **hp)
    return build


def _csl_builder(name):
    def build(seed, hp, data, C_a):
        if C_a is None:
            raise ValueError("a CSL model is trained for one specific C_a")
        return _sl_module(name).build_csl(seed, C_a, **hp)
    return build


def _full_info_builder(seed, hp, data, C_a):
    from Experiments.reference_policies import FullInfoOnline
    return FullInfoOnline(data.n_features + 1, **hp)


def _partial_info_builder(seed, hp, data, C_a):
    from Experiments.reference_policies import PartialInfoOnline
    idx = CONTEXT_FEATURE_COLS.index("log_amount")
    return PartialInfoOnline(data.n_features + 1, C_a, idx, data.mu[idx], data.sigma[idx], **hp)


# =====================================================================
# The registry
# =====================================================================
def bandit_specs():
    specs = []
    for a in BANDIT_GRIDS:
        for r in REWARD_TYPES:
            short = REWARD_SHORT[r]
            specs.append(ModelSpec(
                id=f"CB_{short}_{a}", name=a, variant=short, kind="bandit",
                family=FAMILY_BANDIT, reward_type=r, deterministic=False,
                depends_on_C_a=(r == "cost_sensitive"),     # the 0/1 reward never involves C_a
                grid=BANDIT_GRIDS[a], builder=_bandit_builder(a, r),
                cache_id=f"{short}_{a}"))                   # name before the renaming
    return specs


def supervised_specs():
    specs = []
    for n in SL_CSD_GRIDS:
        det = _sl_module(n).DETERMINISTIC
        specs.append(ModelSpec(
            id=f"SL_CSD_{n}", name=n, variant="CSD", kind="supervised",
            family=FAMILY_SUPERVISED, reward_type=None, deterministic=det,
            depends_on_C_a=False, grid=SL_CSD_GRIDS[n], builder=_csd_builder(n),
            cache_id=n))                                    # name before the renaming
    for n in SL_CSL_GRIDS:
        specs.append(ModelSpec(
            id=f"SL_CSL_{n}", name=n, variant="CSL", kind="supervised_csl",
            family=FAMILY_SUPERVISED, reward_type=None,
            deterministic=_sl_module(n).DETERMINISTIC,
            depends_on_C_a=True,                            # the weights contain C_a
            grid=SL_CSL_GRIDS[n], builder=_csl_builder(n)))
    return specs


def reference_specs():
    return [
        ModelSpec("Oracle", "Oracle", "", "oracle", FAMILY_REFERENCE, None, True, True,
                  None, None),
        ModelSpec("FullInfoOnline", "FullInfoOnline", "", "online_prob", FAMILY_REFERENCE,
                  None, True, False, REFERENCE_GRIDS["FullInfoOnline"], _full_info_builder),
        ModelSpec("PartialInfoOnline", "PartialInfoOnline", "", "bandit", FAMILY_REFERENCE,
                  None, True, True, REFERENCE_GRIDS["PartialInfoOnline"],
                  _partial_info_builder),
    ]


def all_specs():
    return bandit_specs() + supervised_specs() + reference_specs()


def tuned_specs():
    """Every model that has a tuning grid (all but the Oracle)."""
    return [s for s in all_specs() if s.grid is not None]


def all_ids():
    return [s.id for s in all_specs()]


def get_spec(model_id):
    for s in all_specs():
        if s.id == model_id:
            return s
    raise KeyError(f"unknown model id {model_id!r}")