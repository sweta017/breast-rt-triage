import sys
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

pid = sys.argv[1] if len(sys.argv) > 1 else "LCTSC-Test-S1-101"
ct_img = sitk.ReadImage(f"data/nifti/{pid}.nii.gz")
ct = sitk.GetArrayFromImage(ct_img).astype(float)               # (z, y, x)
m = sitk.GetArrayFromImage(sitk.ReadImage(f"data/masks/{pid}_expert.nii.gz"))
sx, sy, sz = ct_img.GetSpacing()

cmap = ListedColormap(["red", "yellow", "dodgerblue", "lime", "cyan"])  # 1..5
def over(a):
    return np.ma.masked_where(a == 0, a)

k = int(np.argmax((m == 1).sum(axis=(1, 2))))    # axial slice with most heart
j = int(np.argmax((m > 0).sum(axis=(0, 2))))     # coronal slice with most mask
lo, hi = -1000, 400                              # display window in HU

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(ct[k], cmap="gray", vmin=lo, vmax=hi)
ax[0].imshow(over(m[k]), cmap=cmap, vmin=1, vmax=5, alpha=0.45)
ax[0].set_title(f"{pid} axial")
ax[1].imshow(ct[:, j, :], cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect=sz / sx)
ax[1].imshow(over(m[:, j, :]), cmap=cmap, vmin=1, vmax=5, alpha=0.45, origin="lower", aspect=sz / sx)
ax[1].set_title("coronal")
for a in ax:
    a.axis("off")
fig.suptitle("red heart | yellow left lung | blue right lung | green esophagus | cyan cord")
plt.tight_layout()
plt.savefig(f"results/overlay_{pid}.png", dpi=120)
print("saved results/overlay_" + pid + ".png")