import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXCLUDE = {"LCTSC-Train-S3-012"}
Q = 0.75

# same data preparation as triage.py
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

imps = []
for rep in range(3):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rep)
    for tr, te in cv.split(X, y, groups=df.patient):
        m = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                                       subsample=0.8, random_state=0).fit(X[tr], y[tr])
        r = permutation_importance(m, X[te], y[te], scoring="roc_auc",
                                   n_repeats=10, random_state=0)
        imps.append(r.importances_mean)
imp = pd.DataFrame({"feature": FEATS, "auroc_drop": np.mean(imps, axis=0),
                    "sd_over_folds": np.std(imps, axis=0)}).sort_values("auroc_drop", ascending=False)
print("Permutation importance (drop in held-out AUROC when the feature is shuffled):")
print(imp.round(3).to_string(index=False))
imp.to_csv("results/importance_gb.csv", index=False)

lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
coef = pd.DataFrame({"feature": FEATS, "std_coef": lr[-1].coef_[0]}).sort_values("std_coef")
print("\nLogistic regression standardised coefficients (fit on all data, descriptive only):")
print(coef.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
s = imp.sort_values("auroc_drop")
ax[0].barh(s.feature, s.auroc_drop, xerr=s.sd_over_folds)
ax[0].set_title("Gradient boosting: permutation importance")
ax[0].set_xlabel("drop in AUROC")
ax[1].barh(coef.feature, coef.std_coef)
ax[1].set_title("Logistic: standardised coefficients")
plt.tight_layout()
plt.savefig("results/importance.png", dpi=120)
print("saved results/importance.png")