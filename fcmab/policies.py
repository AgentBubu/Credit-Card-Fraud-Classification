"""Decision policies.

Every policy chooses one action per transaction: APPROVE, MFA or REVIEW.

Bandits (standard disjoint linear contextual bandits, one ridge-regression reward model per action):
  eps_greedy, linucb, lints. Reward = money saved compared with approving, / REWARD_SCALE.
Supervised (cost-sensitive through the minimum-expected-cost decision rule, trained on 2019
and never retrained): rf_frozen, xgb_frozen.
References (not competitors):
  approve_all (baseline for the savings measure), oracle (knows every true label).
"""
import numpy as np

from . import config as C

try:  # real XGBoost on the experiment machine; a close stand-in only for development
    import xgboost
    HAVE_XGB = True
except ImportError:  # pragma: no cover
    HAVE_XGB = False


# ============================================================================= rewards and costs
def cost_of(action, fraud, amt, bypass):
    """Financial cost (US$) of one decision, from the simplified cost matrix."""
    if action == C.APPROVE:
        return amt if fraud else 0.0
    if action == C.MFA:
        return C.C_MFA + (amt if (fraud and bypass) else 0.0)
    if action == C.REVIEW:
        return C.C_REV
    return 0.0  # DECLINED


def savings_reward(action, fraud, amt, bypass):
    """Bandit reward: cost of approving minus cost of the chosen action, / REWARD_SCALE."""
    return (cost_of(C.APPROVE, fraud, amt, False) - cost_of(action, fraud, amt, bypass)) / C.REWARD_SCALE


def counterfactual_rewards(y, amt, rng):
    """Rewards of all three actions for fully labelled history rows (bandit warm start)."""
    n = len(y)
    bypass = rng.random(n) < C.MFA_BYPASS
    r = np.zeros((n, 3))
    r[:, C.MFA] = np.where(y == 1, np.where(bypass, -C.C_MFA, amt - C.C_MFA), -C.C_MFA)
    r[:, C.REVIEW] = np.where(y == 1, amt - C.C_REV, -C.C_REV)
    return r / C.REWARD_SCALE


# ============================================================================= base
class Policy:
    name = "policy"
    is_bandit = False
    batch = True            # True: decide a whole day at once (no learning inside the day)
    monthly = False         # True: retrained on the first day of each month

    def fit_history(self, X, y, amt, rng):
        pass

    def decide_batch(self, X, amt, y):
        raise NotImplementedError

    def retrain(self, X, y, rng):
        pass


class ApproveAll(Policy):
    name = "approve_all"

    def decide_batch(self, X, amt, y):
        return np.full(len(amt), C.APPROVE, np.int8)


class Oracle(Policy):
    """Knows the true label: approves legitimate transactions; stops fraud with the
    cheaper of MFA (expected cost C_MFA + bypass x amount) and review."""
    name = "oracle"

    def decide_batch(self, X, amt, y):
        stop = np.where(C.C_MFA + C.MFA_BYPASS * amt < C.C_REV, C.MFA, C.REVIEW)
        return np.where(y == 1, stop, C.APPROVE).astype(np.int8)


# ============================================================================= bandits
class LinearBandit(Policy):
    """Disjoint linear contextual bandit (Li et al., 2010 style), one ridge model per action.

    kind = "eps_greedy": best estimated reward; a random action with probability EPSILON.
    kind = "linucb":     estimated reward + ALPHA * sqrt(z' A^-1 z).
    kind = "lints":      linear Thompson sampling with posterior N(theta, TS_SCALE^2 A^-1).
                          Each action's decision depends only on the sampled score z'theta~,
                          whose exact distribution is N(z'theta, TS_SCALE^2 z'A^-1 z), so that
                          scalar is sampled directly (same distribution, much cheaper).
    Context z = [standardized features (standardization statistics from history), bias].
    Optional (config.USE_LEAF_FEATURES, off by default): z also gets leaf indicators (0/1) and
    leaf indicators x amount/100 from a small random forest fitted once on history (tree-leaf
    features, He et al., 2014). Leaf columns are left unstandardized so z stays sparse.
    """
    is_bandit = True
    batch = False

    def __init__(self, kind, d, rng):
        self.name = kind
        self.kind = kind
        self.rng = rng
        self.d_base = d
        self.leaf_model = None

    # --- context
    def _leaf_onehot(self, Xb):
        L = self.leaf_model.apply(np.asarray(Xb, dtype=np.float32))          # (n, trees) node ids
        cols = np.column_stack([self.leaf_maps[k][L[:, k]] for k in range(L.shape[1])])
        return cols                                                          # column index per tree

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        Z = np.zeros((n, self.d))
        Z[:, :self.d_base] = (X - self.mu) / self.sd
        Z[:, -1] = 1.0
        if self.leaf_model is None:
            return Z
        cols = self._leaf_onehot(X)
        rows = np.repeat(np.arange(n), cols.shape[1])
        amt = np.expm1(X[:, 0]) / C.REWARD_SCALE                             # column 0 = log(1 + amount)
        Z[rows, self.d_base + cols.ravel()] = 1.0
        Z[rows, self.d_base + self.n_leaves + cols.ravel()] = np.repeat(amt, cols.shape[1])
        return Z

    # --- learning
    def fit_history(self, X, y, amt, rng):
        """1) fit the leaf forest, 2) standardization statistics, 3) warm start from WARM_N
        random history transactions, where full labels give the reward of every action.
        Each step uses its own random sample of the history."""
        n = len(y)
        self.n_leaves = 0
        if C.USE_LEAF_FEATURES:
            from sklearn.ensemble import RandomForestClassifier
            i_leaf = rng.choice(n, min(C.LEAF_FIT_N, n), replace=False)
            self.leaf_model = RandomForestClassifier(
                n_estimators=C.LEAF_TREES, max_leaf_nodes=C.LEAF_MAX_LEAVES, min_samples_leaf=C.LEAF_MIN_SAMPLES,
                class_weight="balanced", n_jobs=1, random_state=int(rng.integers(1 << 30)),
            ).fit(np.asarray(X[i_leaf], dtype=np.float32), y[i_leaf])
            self.leaf_maps, off = [], 0
            for est in self.leaf_model.estimators_:
                t = est.tree_
                leaves = np.flatnonzero(t.children_left == -1)
                m = np.full(t.node_count, -1, np.int64)
                m[leaves] = off + np.arange(len(leaves))
                off += len(leaves)
                self.leaf_maps.append(m)
            self.n_leaves = off
        self.d = self.d_base + 2 * self.n_leaves + 1
        i_sc = rng.choice(n, min(C.SCALER_N, n), replace=False)
        Xs = np.asarray(X[i_sc], dtype=np.float64)
        self.mu, sd = Xs.mean(0), Xs.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.Ainv = np.stack([np.eye(self.d) / C.RIDGE_LAMBDA] * 3)
        self.b = np.zeros((3, self.d))
        self.theta = np.zeros((3, self.d))
        m = min(C.WARM_N, n)
        if m <= 0:
            return
        idx = rng.choice(n, m, replace=False)
        Z = self.transform(X[idx])
        R = counterfactual_rewards(y[idx], amt[idx], rng)
        Ainv = np.linalg.inv(Z.T @ Z + C.RIDGE_LAMBDA * np.eye(self.d))
        for a in range(3):
            self.Ainv[a] = Ainv
            self.b[a] = Z.T @ R[:, a]
            self.theta[a] = Ainv @ self.b[a]

    def act(self, z):
        nz = np.flatnonzero(z)                     # z is sparse: use only its non-zero entries
        zn = z[nz]
        mean = self.theta[:, nz] @ zn
        if self.kind == "eps_greedy":
            if self.rng.random() < C.EPSILON:
                return int(self.rng.integers(3))
            return int(np.argmax(mean))
        width = np.sqrt(np.maximum(np.einsum("aij,i,j->a", self.Ainv[:, nz][:, :, nz], zn, zn), 0.0))
        if self.kind == "linucb":
            return int(np.argmax(mean + C.ALPHA * width))
        return int(np.argmax(mean + C.TS_SCALE * width * self.rng.standard_normal(3)))

    def update(self, a, z, r):
        Ai = self.Ainv[a]
        nz = np.flatnonzero(z)
        Az = Ai[:, nz] @ z[nz]
        Ai -= np.outer(Az, Az) / (1.0 + z[nz] @ Az[nz])          # Sherman-Morrison rank-one update
        self.b[a, nz] += r * z[nz]
        self.theta[a] = Ai @ self.b[a]


# ============================================================================= supervised
class Supervised(Policy):
    """RF or XGBoost trained on labelled history, probabilities calibrated, decisions by
    minimum expected cost. Costs are used only in the decision step."""

    def __init__(self, model, mode, seed):
        self.model_kind = model
        self.monthly = mode in ("monthly", "quarterly")        # retrained during the test period
        self.retrain_every = {"monthly": 1, "quarterly": 3}.get(mode, 0)   # months
        self.name = f"{model}_{mode}"
        self.seed = seed
        self.model = None

    def _base(self):
        if self.model_kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**C.RF_PARAMS, n_jobs=1, random_state=self.seed)
        if HAVE_XGB:
            return xgboost.XGBClassifier(**C.XGB_PARAMS, n_jobs=1, random_state=self.seed,
                                         eval_metric="logloss")
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**C.HGB_FALLBACK_PARAMS, random_state=self.seed)

    def _fit(self, X, y):
        """X, y are already undersampled (all fraud + UNDERSAMPLE_LEGIT of legitimate)."""
        from sklearn.calibration import CalibratedClassifierCV
        m = CalibratedClassifierCV(self._base(), method="isotonic", cv=C.CALIBRATION_FOLDS)
        m.fit(np.asarray(X), y)
        self.model = m

    def fit_history(self, X, y, amt, rng):
        keep = (y == 1) | (rng.random(len(y)) < C.UNDERSAMPLE_LEGIT)
        self._fit(X[keep], y[keep])

    def retrain(self, X, y, rng):
        """Called by the simulator with undersampled rows and the labels received so far."""
        if self.monthly:
            self._fit(X, y)

    def predict(self, X):
        ps = self.model.predict_proba(np.asarray(X))[:, 1]
        b = C.UNDERSAMPLE_LEGIT          # undo the undersampling shift (Dal Pozzolo et al., 2015)
        return b * ps / (b * ps - ps + 1.0)

    def decide_batch(self, X, amt, y):
        p = self.predict(X)
        exp_cost = np.column_stack([p * amt,                                  # approve
                                    C.C_MFA + C.MFA_BYPASS * p * amt,         # MFA
                                    np.full(len(amt), C.C_REV)])              # review
        return np.argmin(exp_cost, axis=1).astype(np.int8)


# ============================================================================= factory
def make(name, d, seed):
    rng = np.random.default_rng([seed, 7, sum(map(ord, name))])
    if name == "approve_all":
        return ApproveAll()
    if name == "oracle":
        return Oracle()
    if name in C.BANDITS:
        return LinearBandit(name, d, rng)
    if name in C.SUPERVISED + C.SUPERVISED_EXTRA:
        model, mode = name.split("_")
        return Supervised(model, mode, seed)
    raise ValueError(name)
