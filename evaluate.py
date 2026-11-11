import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

rng = np.random.default_rng(0)
MAIN, F_MAIN = "p_gradient_boosting", 0.25      # pre-specified: route the top 25%
FRACS = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]
CUT = chi2.ppf(0.975, df=5)

oof = pd.read_csv("results/triage_oof_predictions.csv")
ood = pd.read_csv("results/ood_scores.csv")
y, p = oof.label_q75.values, oof[MAIN].values

def op(y, p, f):
    thr = np.quantile(p, 1 - f)
    r = p >= thr
    sens = r[y == 1].mean() if (y == 1).any() else np.nan
    spec = 1 - r[y == 0].mean() if (y == 0).any() else np.nan
    ppv = y[r].mean() if r.any() else np.nan
    return sens, spec, ppv

# ---- operating points with patient-level bootstrap CIs ----
by = {k: g for k, g in oof.groupby("patient")}
pats = np.array(list(by))
rows = []
for f in FRACS:
    s0, sp0, pv0 = op(y, p, f)
    bs = []
    for _ in range(500):
        d = pd.concat([by[k] for k in rng.choice(pats, len(pats))])
        if d.label_q75.nunique() == 2:
            bs.append(op(d.label_q75.values, d[MAIN].values, f)[0])
    rows.append({"route_top_fraction": f, "sensitivity": round(s0, 3),
                 "sens_CI_low": round(np.percentile(bs, 2.5), 3),
                 "sens_CI_high": round(np.percentile(bs, 97.5), 3),
                 "specificity": round(sp0, 3), "PPV": round(pv0, 3)})
ops = pd.DataFrame(rows)
print("Operating points (gradient boosting, label = top 25% risk):")
print(ops.to_string(index=False))
ops.to_csv("results/operating_points.csv", index=False)

# ---- calibration and overall numbers ----
base = y.mean() * (1 - y.mean())
print(f"\nAUROC {roc_auc_score(y, p):.3f} | Brier {brier_score_loss(y, p):.3f} "
      f"| Brier of always predicting the base rate {base:.3f}")

# ---- three-way decision ----
thr = np.quantile(p, 1 - F_MAIN)
flag = ood.set_index("patient").flagged.to_dict()
oof["decision"] = np.where(oof.patient.map(flag).fillna(False), "uncertain",
                           np.where(p >= thr, "complex", "standard"))
extra = ood[ood.flagged & ~ood.patient.isin(oof.patient)]
rows_extra = pd.DataFrame({"patient": extra.patient, "side": "(both)", "decision": "uncertain"})
dec = pd.concat([oof[["patient", "side", "decision"]], rows_extra])
dec.to_csv("results/decisions.csv", index=False)
print("\nThree-way decisions (patient-sides):")
print(dec.decision.value_counts().to_string())
print("\nOf the model's 'complex' calls, true label = complex for",
      int(oof[oof.decision == "complex"].label_q75.sum()), "of",
      int((oof.decision == "complex").sum()))

# ---- summary figure ----
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
for col, name in [("p_gradient_boosting", "gradient boosting"), ("p_logistic", "logistic")]:
    fpr, tpr, _ = roc_curve(y, oof[col])
    ax[0, 0].plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y, oof[col]):.2f})")
ax[0, 0].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[0, 0].set_title("ROC (out-of-fold)"); ax[0, 0].set_xlabel("1 - specificity")
ax[0, 0].set_ylabel("sensitivity"); ax[0, 0].legend()

fp, mp = calibration_curve(y, p, n_bins=5, strategy="quantile")
ax[0, 1].plot(mp, fp, "o-"); ax[0, 1].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[0, 1].set_title("Reliability (5 quantile bins)")
ax[0, 1].set_xlabel("mean predicted probability"); ax[0, 1].set_ylabel("observed fraction complex")

n = ood[~ood.patient.isin(extra.patient)]
ax[1, 0].hist(n.mahalanobis2, bins=15, color="steelblue")
ax[1, 0].axvline(CUT, color="red", ls="--", label=f"cut-off {CUT:.1f}")
for _, r in extra.iterrows():
    ax[1, 0].axvline(r.mahalanobis2, color="black", label=f"{r.patient} ({r.mahalanobis2:.0f})")
ax[1, 0].set_title("Safety flag: anatomy distance"); ax[1, 0].set_xlabel("squared Mahalanobis distance")
ax[1, 0].legend(fontsize=7)

ax[1, 1].errorbar(ops.route_top_fraction, ops.sensitivity,
                  yerr=[ops.sensitivity - ops.sens_CI_low, ops.sens_CI_high - ops.sensitivity],
                  fmt="o-", capsize=3)
ax[1, 1].plot([0, 0.5], [0, 0.5], "k--", lw=0.8, label="no better than random")
ax[1, 1].set_title("Sensitivity vs fraction routed to 'complex'")
ax[1, 1].set_xlabel("fraction routed"); ax[1, 1].set_ylabel("sensitivity"); ax[1, 1].legend()
plt.tight_layout()
plt.savefig("results/fig_summary.png", dpi=130)
print("\nsaved results/fig_summary.png, operating_points.csv, decisions.csv")