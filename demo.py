import os, sys
import pandas as pd
from features import plot_qc          # reuse the overlay + field drawing

pid = sys.argv[1] if len(sys.argv) > 1 else "LCTSC-Test-S1-101"
side = sys.argv[2] if len(sys.argv) > 2 else "left"

feat = pd.read_csv("results/features.csv")
feat = feat[(feat.patient == pid) & (feat.side == side) & (feat.source == "auto")]
oof = pd.read_csv("results/triage_oof_predictions.csv")
oof = oof[(oof.patient == pid) & (oof.side == side)]
dec = pd.read_csv("results/decisions.csv")
dec = dec[dec.patient == pid]
ood = pd.read_csv("results/ood_scores.csv")
ood = ood[ood.patient == pid]

print(f"\n=== {pid} ({side} side) ===")
if len(feat):
    f = feat.iloc[0]
    print(f"Simulated field: lung depth (CLD) {f.cld_max_mm} mm, heart depth (MHD) {f.mhd_max_mm} mm")
    print(f"Lung in field: {100 * f.lung_in_field_frac:.0f}%   Heart in field: {100 * f.heart_in_field_frac:.0f}%")
if len(ood):
    print(f"Anatomy distance: {ood.iloc[0].mahalanobis2} (cut-off 12.8) -> "
          f"{'FLAGGED, send to a human' if ood.iloc[0].flagged else 'within normal range'}")
if len(oof):
    print(f"Triage model score: {oof.iloc[0].p_gradient_boosting:.2f} (use for ranking, not as a risk)")
if len(dec):
    print("Decision:", ", ".join(sorted(set(dec.decision))))

plot_qc(pid, side, "auto")
img = f"results/qc_field_{pid}_{side}_auto.png"
if os.name == "nt" and os.path.exists(img):
    os.startfile(img)
    