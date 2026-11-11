import sys
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pid = sys.argv[1]
side = sys.argv[2] if len(sys.argv) > 2 else "L"        # L or R
lobes = (["lung_upper_lobe_left", "lung_lower_lobe_left"] if side == "L"
         else ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"])
label = 2 if side == "L" else 3

img = sitk.ReadImage(f"data/nifti/{pid}.nii.gz")
sx, sy, sz = img.GetSpacing()
ct = sitk.GetArrayFromImage(img)
exp = sitk.GetArrayFromImage(sitk.ReadImage(f"data/masks/{pid}_expert.nii.gz")) == label
auto = np.zeros_like(exp)
for n in lobes:
    auto |= sitk.GetArrayFromImage(sitk.ReadImage(f"results/seg_full/{pid}/{n}.nii.gz")).astype(bool)

v = sx * sy * sz / 1000
print(f"{pid} {side} lung: automatic {auto.sum()*v:.0f} mL, expert {exp.sum()*v:.0f} mL")

diff = auto ^ exp
k = int(np.argmax(diff.sum(axis=(1, 2))))      # axial slice with most disagreement
j = int(np.argmax(diff.sum(axis=(0, 2))))      # coronal slice with most disagreement

fig, ax = plt.subplots(1, 2, figsize=(13, 6))
ax[0].imshow(ct[k], cmap="gray", vmin=-1000, vmax=400)
ax[0].contour(exp[k], levels=[0.5], colors="lime", linewidths=1.2)
ax[0].contour(auto[k], levels=[0.5], colors="red", linewidths=1.2)
ax[0].set_title("axial")
ax[1].imshow(ct[:, j, :], cmap="gray", vmin=-1000, vmax=400, origin="lower", aspect=sz / sx)
ax[1].contour(exp[:, j, :], levels=[0.5], colors="lime", linewidths=1.2)
ax[1].contour(auto[:, j, :], levels=[0.5], colors="red", linewidths=1.2)
ax[1].set_title("coronal")
for a in ax:
    a.axis("off")
fig.suptitle(f"{pid} {side} lung: green = expert, red = automatic")
plt.tight_layout()
plt.savefig(f"results/compare_{pid}_{side}.png", dpi=120)
print(f"saved results/compare_{pid}_{side}.png")