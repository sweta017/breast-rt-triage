import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXCLUDE = {"LCTSC-Train-S3-012"}
SRC = "auto"                    # main analysis uses the automatic segmentations
QS = [0.70, 0.75, 0.80]         # label thresholds (sensitivity analysis); 0.75 is the main one
REPEATS = 5

df = pd.read_csv("results/features.csv")
df = df[(df.source == SRC) & (~df.patient.isin(EXCLUDE))].reset_index(drop=True)
print(len(df), "rows,", df.patient.nunique(), "patients")

def z(s):
    return (s - s.mean()) / s.std(ddof=0)

# surrogate risk score, standardised within each side
df["risk"] = np.nan
for side, g in df.groupby("side"):
    parts = [z(g.lung_in_field_frac), z(g.cld_max_mm)]
    if side == "left":
        parts += [z(g.heart_in_field_frac), z(g.mhd_max_mm)]
    df.loc[g.index, "risk"] = sum(parts) / len(parts)

def make_label(q):
    y = np.zeros(len(df), int)
    for side, g in df.groupby("side"):
        y[g.index] = (g.risk >= g.risk.quantile(q)).astype(int)
    return y

# inputs: global anatomy only (none of the quantities used for the label)
df["side_left"] = (df.side == "left").astype(int)
df["heart_lung_ratio"] = df.heart_ml / df.lung_ipsi_ml
df["ap_width_ratio"] = df.chest_ap_mm / df.chest_width_mm
FEATS = ["side_left", "chest_ap_mm", "chest_width_mm", "heart_ml",
         "lung_ipsi_ml", "heart_lung_ratio", "ap_width_ratio"]

models = {
    "side_only (baseline)": (["side_left"], make_pipeline(StandardScaler(), LogisticRegression())),
    "logistic": (FEATS, make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    "gradient_boosting": (FEATS, GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)),
}

rows, oof_main = [], {}
for q in QS:
    y = make_label(q)
    print(f"\nq={q}: {y.sum()} 'complex' of {len(y)} patient-sides")
    for name, (cols, mdl) in models.items():
        X = df[cols].values
        for rep in range(REPEATS):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
            p = np.zeros(len(df))
            for tr, te in cv.split(X, y, groups=df.patient):
                m = clone(mdl).fit(X[tr], y[tr])
                p[te] = m.predict_proba(X[te])[:, 1]
            rows.append({"q": q, "model": name, "rep": rep,
                         "auroc": roc_auc_score(y, p),
                         "avg_precision": average_precision_score(y, p),
                         "brier": brier_score_loss(y, p)})
            if q == 0.75 and rep == 0:
                oof_main[name] = p

res = pd.DataFrame(rows)
res.to_csv("results/triage_cv_runs.csv", index=False)
summary = (res.groupby(["q", "model"])[["auroc", "avg_precision", "brier"]]
              .agg(["mean", "std"]).round(3))
print("\n", summary)
summary.to_csv("results/triage_summary.csv")

out = df[["patient", "side", "risk"]].copy()
out["label_q75"] = make_label(0.75)
for name, p in oof_main.items():
    out["p_" + name.split(" ")[0]] = np.round(p, 4)
out.to_csv("results/triage_oof_predictions.csv", index=False)
print("\nsaved results/triage_summary.csv and results/triage_oof_predictions.csv")