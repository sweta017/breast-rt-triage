import os
import numpy as np
import SimpleITK as sitk

d = "results/test_seg"
ref = sitk.ReadImage(f"{d}/heart.nii.gz")
sp = ref.GetSpacing()
voxel_ml = sp[0] * sp[1] * sp[2] / 1000

def load(name):
    return sitk.GetArrayFromImage(sitk.ReadImage(f"{d}/{name}.nii.gz")).astype(bool)

heart = load("heart")
lung_l = load("lung_upper_lobe_left") | load("lung_lower_lobe_left")
lung_r = (load("lung_upper_lobe_right") | load("lung_middle_lobe_right")
          | load("lung_lower_lobe_right"))

for name, m in [("heart", heart), ("lung_left", lung_l), ("lung_right", lung_r)]:
    print(f"{name:12s} {m.sum() * voxel_ml:8.0f} mL")

# one labelmap for easy viewing: 1 = heart, 2 = left lung, 3 = right lung
lab = np.zeros(heart.shape, dtype=np.uint8)
lab[heart] = 1
lab[lung_l] = 2
lab[lung_r] = 3
out = sitk.GetImageFromArray(lab)
out.CopyInformation(ref)
sitk.WriteImage(out, f"{d}/merged_labels.nii.gz")
print("saved", f"{d}/merged_labels.nii.gz")