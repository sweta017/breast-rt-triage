import os, pydicom, pandas as pd
from collections import defaultdict

root = "data/raw/manifest-1557326747206/LCTSC"
os.makedirs("results", exist_ok=True)

series = defaultdict(lambda: {"files": []})
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.lower().endswith(".dcm"):
            p = os.path.join(dp, f)
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            s = series[ds.SeriesInstanceUID]
            s["files"].append(p)
            s.update(
                patient=ds.PatientID,
                modality=ds.Modality,
                sex=getattr(ds, "PatientSex", ""),
                thickness=getattr(ds, "SliceThickness", ""),
                spacing=str(getattr(ds, "PixelSpacing", "")),
                desc=getattr(ds, "SeriesDescription", ""),
            )

rows = [{"series_uid": k,
         "n_files": len(v["files"]),
         "folder": os.path.dirname(v["files"][0]),
         **{x: v[x] for x in ["patient", "modality", "sex", "thickness", "spacing", "desc"]}}
        for k, v in series.items()]

df = pd.DataFrame(rows)

print(df.modality.value_counts())
print("Patients:", df.patient.nunique())
print(df[df.modality == "CT"].sex.value_counts())