import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXCLUDE = {"LCTSC-Train-S3-012"}
Q, N_PERM = 0.75, 200
rng = np.random.default_rng(0)

# ---- same data preparation as triage.py ----
df = pd.read_csv("results/features.csv")
df = df[(df.source == "auto") & (~df.patient.isin(EXCLUDE))].reset_index(drop=True)
z = lambda s: (s - s.mean()) / s.std(ddof=0)
df["risk"] = np.nan
for side, g in df.groupby("side"):
    parts = [z(g.lung_in_field_frac), z(g.cld_max_mm)]
    if side == "left":
        parts += [z(g.heart_in_field_frac), z(g.mhd_max_mm)]
    df.loc[g.index, "risk"] = sum(parts) / len(parts)
y = np.zeros(len(df), int)
for side, g in df.groupby("side"):
    y[g.index] = (g.risk >= g.risk.quantile(Q)).astype(int)
df["side_left"] = (df.side == "left").astype(int)
df["heart_lung_ratio"] = df.heart_ml / df.lung_ipsi_ml
df["ap_width_ratio"] = df.chest_ap_mm / df.chest_width_mm
FEATS = ["side_left", "chest_ap_mm", "chest_width_mm", "heart_ml",
         "lung_ipsi_ml", "heart_lung_ratio", "ap_width_ratio"]
X = df[FEATS].values
models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
}

def cv_auc(mdl, yy, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(yy))
    for tr, te in cv.split(X, yy, groups=df.patient):
        p[te] = clone(mdl).fit(X[tr], yy[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(yy, p)

# ---- 1) permutation test: shuffle labels within each side ----
for name, mdl in models.items():
    obs = np.mean([cv_auc(mdl, y, s) for s in range(5)])
    null = []
    for i in range(N_PERM):
        yp = y.copy()
        for side in ("left", "right"):
            idx = np.where(df.side == side)[0]
            yp[idx] = rng.permutation(y[idx])
        null.append(cv_auc(mdl, yp, i))
    null = np.array(null)
    pval = (1 + (null >= obs).sum()) / (1 + N_PERM)
    print(f"{name}: observed AUROC {obs:.3f} | null mean {null.mean():.3f} "
          f"(95th pct {np.percentile(null, 95):.3f}) | permutation p = {pval:.3f}")

# ---- 2) patient-level bootstrap CI on out-of-fold predictions ----
oof = pd.read_csv("results/triage_oof_predictions.csv")
by = {p: g for p, g in oof.groupby("patient")}
pats = np.array(list(by))
for col in ["p_logistic", "p_gradient_boosting"]:
    aucs = []
    for _ in range(1000):
        d = pd.concat([by[p] for p in rng.choice(pats, len(pats))])
        if d.label_q75.nunique() == 2:
            aucs.append(roc_auc_score(d.label_q75, d[col]))
    print(f"{col}: AUROC {roc_auc_score(oof.label_q75, oof[col]):.3f}, "
          f"95% CI {np.percentile(aucs, 2.5):.3f} to {np.percentile(aucs, 97.5):.3f}")