import glob, os
import numpy as np
import pandas as pd
import SimpleITK as sitk

rows = []
for p in sorted(glob.glob("data/masks/*_expert.nii.gz")):
    pid = os.path.basename(p).replace("_expert.nii.gz", "")
    img = sitk.ReadImage(p)
    a = sitk.GetArrayFromImage(img)
    sp = img.GetSpacing()
    v = sp[0] * sp[1] * sp[2] / 1000
    rows.append({"patient": pid, "heart": int((a == 1).sum() * v),
                 "lung_L": int((a == 2).sum() * v), "lung_R": int((a == 3).sum() * v),
                 "esoph": int((a == 4).sum() * v), "cord": int((a == 5).sum() * v)})

df = pd.DataFrame(rows)
df.to_csv("results/expert_volumes.csv", index=False)
print(len(df), "masks")
print(df.describe().loc[["min", "mean", "max"]].round(0))
print("\nSuspicious (any of heart/lungs under 300 mL or zero):")
print(df[(df.heart < 300) | (df.lung_L < 300) | (df.lung_R < 300)])