import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf

KNOWN_ODD = "LCTSC-Train-S3-012"
CUT = chi2.ppf(0.975, df=5)           # about 12.8: flag the most extreme 2.5% under a Gaussian model

df = pd.read_csv("results/features.csv")
df = df[df.source == "auto"]          # only automatic masks: available in deployment
rows = []
for pid, g in df.groupby("patient"):
    L = g[g.side == "left"].iloc[0]
    R = g[g.side == "right"].iloc[0]
    rows.append({"patient": pid, "heart_ml": L.heart_ml,
                 "lung_L_ml": L.lung_ipsi_ml, "lung_R_ml": R.lung_ipsi_ml,
                 "chest_ap_mm": (L.chest_ap_mm + R.chest_ap_mm) / 2,
                 "chest_width_mm": (L.chest_width_mm + R.chest_width_mm) / 2})
P = pd.DataFrame(rows)

def feats(d):
    return np.log(d[["heart_ml", "lung_L_ml", "lung_R_ml",
                     "chest_ap_mm", "chest_width_mm"]].clip(lower=1).values.astype(float))

def dist2(train_df, test_df):
    lw = LedoitWolf().fit(feats(train_df))
    return lw.mahalanobis(feats(test_df))        # squared Mahalanobis distance

def rule_flag(d):
    r = d.lung_L_ml / d.lung_R_ml
    return ((d.lung_L_ml < 300) | (d.lung_R_ml < 300) | (d.heart_ml < 300)
            | (r < 0.25) | (r > 4))

normals = P[P.patient != KNOWN_ODD].reset_index(drop=True)
odd = P[P.patient == KNOWN_ODD]

# leave-one-out scores for normal patients
d_norm = np.array([dist2(normals.drop(i), normals.iloc[[i]])[0] for i in range(len(normals))])
d_odd = dist2(normals, odd)[0]

print(f"cut-off (squared distance): {CUT:.1f}")
print(f"normal patients flagged: {(d_norm > CUT).sum()} of {len(d_norm)} "
      f"({100 * (d_norm > CUT).mean():.1f}%)")
print(f"median normal distance: {np.median(d_norm):.1f}, 95th percentile: {np.percentile(d_norm, 95):.1f}")
print(f"\nS3-012 (collapsed left lung): distance {d_odd:.1f} -> "
      f"{'FLAGGED' if d_odd > CUT else 'not flagged'}; "
      f"rank {1 + (d_norm > d_odd).sum()} of {len(d_norm) + 1} (1 = most unusual)")
print("rule-based flag for S3-012:", bool(rule_flag(odd).iloc[0]),
      "| normals flagged by rules:", int(rule_flag(normals).sum()))

# simulated abnormalities on feature values (not real images), scored leave-one-out
tests = {
    "left lung collapse (volume x0.1)": lambda d: d.assign(lung_L_ml=d.lung_L_ml * 0.1),
    "left lung partly collapsed (x0.5)": lambda d: d.assign(lung_L_ml=d.lung_L_ml * 0.5),
    "chest AP +40% (bolus/habitus-like)": lambda d: d.assign(chest_ap_mm=d.chest_ap_mm * 1.4),
}
print("\nSimulated abnormalities (fraction of normal patients flagged):")
sim_rows = []
for name, fn in tests.items():
    hits = [dist2(normals.drop(i), fn(normals.iloc[[i]]))[0] > CUT for i in range(len(normals))]
    print(f"  {name}: {np.mean(hits):.2f}")
    sim_rows.append({"test": name, "fraction_flagged": round(float(np.mean(hits)), 3)})

out = normals[["patient"]].copy()
out["mahalanobis2"] = np.round(d_norm, 2)
out["flagged"] = d_norm > CUT
out = pd.concat([out, pd.DataFrame([{"patient": KNOWN_ODD, "mahalanobis2": round(d_odd, 2),
                                     "flagged": d_odd > CUT}])])
out.to_csv("results/ood_scores.csv", index=False)
pd.DataFrame(sim_rows).to_csv("results/ood_simulated.csv", index=False)
print("\nsaved results/ood_scores.csv and results/ood_simulated.csv")