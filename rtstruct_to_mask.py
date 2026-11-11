import os
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon

root = "data/raw/manifest-1557326747206/LCTSC"
out_dir = "data/masks"
os.makedirs(out_dir, exist_ok=True)

# label values in the output mask
LABELS = {"heart": 1, "lung_l": 2, "lung_r": 3, "esophagus": 4, "spinalcord": 5}

def norm(name):
    return name.lower().replace(" ", "").replace("-", "_")

# 1) find the RTSTRUCT file of each patient
rt_files = {}
for dp, _, files in os.walk(root):
    for f in files:
        if not f.lower().endswith(".dcm"):
            continue
        p = os.path.join(dp, f)
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        if ds.Modality == "RTSTRUCT":
            rt_files[str(ds.PatientID)] = p
print(len(rt_files), "RTSTRUCT files found", flush=True)

# 2) rasterise the contours onto the CT grid
for pid, p in sorted(rt_files.items()):
    nii = f"data/nifti/{pid}.nii.gz"
    if not os.path.exists(nii):
        print("no CT for", pid)
        continue
    img = sitk.ReadImage(nii)
    nx, ny, nz = img.GetSize()
    mask = np.zeros((nz, ny, nx), np.uint8)

    ds = pydicom.dcmread(p)
    roi_name = {r.ROINumber: norm(r.ROIName) for r in ds.StructureSetROISequence}

    for rc in ds.ROIContourSequence:
        name = roi_name.get(rc.ReferencedROINumber, "")
        if name not in LABELS or not hasattr(rc, "ContourSequence"):
            continue
        lab = LABELS[name]
        layer = {}                       # slice index -> boolean mask
        for c in rc.ContourSequence:
            pts = np.array(c.ContourData, float).reshape(-1, 3)
            idx = np.array([img.TransformPhysicalPointToContinuousIndex(tuple(q)) for q in pts])
            k = int(round(idx[:, 2].mean()))
            if not 0 <= k < nz:
                continue
            rr, cc = polygon(idx[:, 1], idx[:, 0], shape=(ny, nx))
            m = layer.setdefault(k, np.zeros((ny, nx), bool))
            m[rr, cc] ^= True            # XOR handles holes
        for k, m in layer.items():
            mask[k][m] = lab

    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(img)
    sitk.WriteImage(out, f"{out_dir}/{pid}_expert.nii.gz")
    sp = img.GetSpacing()
    v = sp[0] * sp[1] * sp[2] / 1000
    print(pid, {n: int((mask == l).sum() * v) for n, l in LABELS.items()}, "mL", flush=True)