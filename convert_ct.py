import os
import pydicom
import pandas as pd
import SimpleITK as sitk

root = "data/raw/manifest-1557326747206/LCTSC"
os.makedirs("data/nifti", exist_ok=True)
os.makedirs("results", exist_ok=True)

reader = sitk.ImageSeriesReader()

# 1) find every CT series, keep the one with the most slices per patient
best = {}   # patient -> (n_slices, folder, series_uid, sex)
for dp, _, files in os.walk(root):
    if not any(f.lower().endswith(".dcm") for f in files):
        continue
    for uid in reader.GetGDCMSeriesIDs(dp) or []:
        names = reader.GetGDCMSeriesFileNames(dp, uid)
        ds = pydicom.dcmread(names[0], stop_before_pixels=True)
        if ds.Modality != "CT":
            continue
        pid = str(ds.PatientID)
        if pid not in best or len(names) > best[pid][0]:
            best[pid] = (len(names), dp, uid, getattr(ds, "PatientSex", ""))

print("CT series found:", len(best))

# 2) convert
rows = []
for pid, (n, dp, uid, sex) in sorted(best.items()):
    try:
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(dp, uid))
        img = reader.Execute()
        sitk.WriteImage(img, f"data/nifti/{pid}.nii.gz")
        arr = sitk.GetArrayFromImage(img)
        sp, sz = img.GetSpacing(), img.GetSize()
        rows.append({"patient": pid, "sex": sex, "n_slices": n,
                     "size_xyz": sz, "spacing_xyz": tuple(round(s, 2) for s in sp),
                     "z_extent_mm": round(sz[2] * sp[2], 1),
                     "hu_min": int(arr.min()), "hu_max": int(arr.max()),
                     "status": "ok"})
        print("ok", pid, sz)
    except Exception as e:
        rows.append({"patient": pid, "sex": sex, "status": f"FAILED: {e}"})
        print("FAILED", pid, repr(e))

pd.DataFrame(rows).to_csv("results/qc_table.csv", index=False)
print("Done. See results/qc_table.csv")