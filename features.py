import os, sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage as ndi

CT_DIR, AUTO_DIR, EXP_DIR = "data/nifti", "results/seg_full", "data/masks"
Z_FRAC = (0.2, 0.8)              # central part of the ipsilateral lung's z-range
SOURCES = ["auto", "expert"]     # where heart/lung masks come from
SIDES = ["left", "right"]
LUNG_L = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
LUNG_R = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]

def rd(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def auto_mask(pid, names):
    m = None
    for n in names:
        a = rd(f"{AUTO_DIR}/{pid}/{n}.nii.gz").astype(bool)
        m = a if m is None else (m | a)
    return m

def body_slice(ct2d):
    b = ct2d > -300
    lab, n = ndi.label(b)
    if n == 0:
        return b
    sizes = ndi.sum(b, lab, range(1, n + 1))
    return ndi.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))

def depth(m2d, M, d, dn, n, sx, sy):
    """perpendicular distance (mm) of mask pixels lying anterior to the line, between M and L"""
    r, c = np.nonzero(m2d)
    if len(r) == 0:
        return np.zeros(0)
    P = np.stack([c * sx, r * sy], 1) - M
    t = P @ d / dn ** 2
    dist = P @ n
    return dist[(t >= 0) & (t <= 1) & (dist > 0)]

def process(pid):
    ct_img = sitk.ReadImage(f"{CT_DIR}/{pid}.nii.gz")
    D = ct_img.GetDirection()
    if D[4] <= 0:
        print("unexpected orientation, skipping", pid); return [], {}, None
    left_is_pos_x = D[0] > 0
    sx, sy, sz = ct_img.GetSpacing()
    ct = sitk.GetArrayFromImage(ct_img)
    vox_ml = sx * sy * sz / 1000

    sternum = auto_mask(pid, ["sternum"])
    chest = auto_mask(pid, ["sternum", "costal_cartilages"])
    zs = np.where(sternum.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        print("no sternum found:", pid); return [], {}, None
    x_mid = float(np.median([np.nonzero(sternum[k])[1].mean() for k in zs]))

    # posterior edge of sternum/cartilage near the midline, per slice
    ysp = np.full(ct.shape[0], np.nan)
    for k in range(ct.shape[0]):
        rr, cc = np.nonzero(chest[k])
        sel = np.abs(cc - x_mid) * sx <= 20
        if sel.any():
            ysp[k] = rr[sel].max()
    if np.isnan(ysp).all():
        print("no midline chest wall:", pid); return [], {}, None
    ysp = np.where(np.isnan(ysp), np.nanmedian(ysp), ysp)

    masks = {"auto": (auto_mask(pid, ["heart"]), auto_mask(pid, LUNG_L), auto_mask(pid, LUNG_R))}
    e = rd(f"{EXP_DIR}/{pid}_expert.nii.gz")
    masks["expert"] = (e == 1, e == 2, e == 3)

    rows, qcs = [], {}
    for src in SOURCES:
        heart, lungL, lungR = masks[src]
        for side in SIDES:
            ipsi = lungL if side == "left" else lungR
            lat = 1 if ((side == "left") == left_is_pos_x) else -1
            zr = np.where(ipsi.any(axis=(1, 2)))[0]
            if len(zr) < 10:
                continue
            z0, z1 = zr.min(), zr.max()
            k_lo = int(z0 + Z_FRAC[0] * (z1 - z0)); k_hi = int(z0 + Z_FRAC[1] * (z1 - z0))
            cld, mhd, aps, wds = [], [], [], []
            lung_vox = heart_vox = 0
            k_mid = (k_lo + k_hi) // 2
            for k in range(k_lo, k_hi + 1):
                rr, cc = np.nonzero(ipsi[k])
                if len(rr) == 0:
                    continue
                xl = cc.max() if lat > 0 else cc.min()
                yl = rr[cc == xl].mean()
                M = np.array([x_mid * sx, ysp[k] * sy]); L = np.array([xl * sx, yl * sy])
                d = L - M; dn = np.linalg.norm(d)
                if dn < 20:
                    continue
                nrm = np.array([-d[1], d[0]]) / dn
                if nrm[1] > 0:                 # anterior = smaller row index
                    nrm = -nrm
                dl = depth(ipsi[k], M, d, dn, nrm, sx, sy)
                dh = depth(heart[k], M, d, dn, nrm, sx, sy)
                cld.append(dl.max() if len(dl) else 0.0)
                mhd.append(dh.max() if len(dh) else 0.0)
                lung_vox += len(dl); heart_vox += len(dh)
                # chest size
                b = body_slice(ct[k])
                col = b[:, int(round(x_mid))]
                both = lungL[k] | lungR[k]
                if col.any() and both.any():
                    r2, c2 = np.nonzero(both)
                    aps.append((r2.max() - np.argmax(col)) * sy)
                    wds.append((c2.max() - c2.min()) * sx)
                if k == k_mid:
                    qcs[(src, side)] = {"k": k, "M": M, "L": L, "sx": sx, "sy": sy}
            if not cld:
                continue
            rows.append({
                "patient": pid, "side": side, "source": src,
                "cld_max_mm": round(max(cld), 1), "cld_mean_mm": round(float(np.mean(cld)), 1),
                "mhd_max_mm": round(max(mhd), 1), "mhd_mean_mm": round(float(np.mean(mhd)), 1),
                "lung_in_field_frac": round(lung_vox / max(ipsi.sum(), 1), 4),
                "heart_in_field_frac": round(heart_vox / max(heart.sum(), 1), 4),
                "lung_in_field_ml": round(lung_vox * vox_ml), "heart_in_field_ml": round(heart_vox * vox_ml),
                "heart_ml": round(heart.sum() * vox_ml), "lung_ipsi_ml": round(ipsi.sum() * vox_ml),
                "chest_ap_mm": round(float(np.median(aps)), 1) if aps else np.nan,
                "chest_width_mm": round(float(np.median(wds)), 1) if wds else np.nan,
            })
    return rows, qcs, (ct, masks)

def plot_qc(pid, side="left", src="auto"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, qcs, ctx = process(pid)
    q = qcs[(src, side)]
    ct, masks = ctx
    heart, lungL, lungR = masks[src]
    k = q["k"]
    plt.figure(figsize=(7, 7))
    plt.imshow(ct[k], cmap="gray", vmin=-1000, vmax=400)
    for m, col in [(heart[k], "red"), (lungL[k], "yellow"), (lungR[k], "dodgerblue")]:
        plt.contour(m, levels=[0.5], colors=col, linewidths=1)
    plt.plot([q["M"][0] / q["sx"], q["L"][0] / q["sx"]], [q["M"][1] / q["sy"], q["L"][1] / q["sy"]],
             "w-", lw=2)
    plt.title(f"{pid} {side} {src} (white line = deep field edge)")
    plt.axis("off")
    out = f"results/qc_field_{pid}_{side}_{src}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_qc(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "left")
    else:
        all_rows = []
        for pid in sorted(os.listdir(AUTO_DIR)):
            if not os.path.exists(f"{EXP_DIR}/{pid}_expert.nii.gz"):
                continue
            rows, _, _ = process(pid)
            all_rows += rows
            print(pid, len(rows), "rows", flush=True)
        df = pd.DataFrame(all_rows)
        df.to_csv("results/features.csv", index=False)
        cols = ["cld_max_mm", "mhd_max_mm", "lung_in_field_frac", "heart_in_field_frac"]
        print(df.groupby(["source", "side"])[cols].mean().round(2))