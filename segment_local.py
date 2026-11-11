import os, glob, subprocess

keep = (["heart", "lung_upper_lobe_left", "lung_lower_lobe_left",
         "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
         "sternum", "costal_cartilages", "spinal_cord", "esophagus", "trachea"]
        + [f"rib_left_{i}" for i in range(1, 13)]
        + [f"rib_right_{i}" for i in range(1, 13)])

out_root = "results/seg_fast"
os.makedirs(out_root, exist_ok=True)
files = sorted(glob.glob("data/nifti/*.nii.gz"))
print(len(files), "scans", flush=True)

for i, f in enumerate(files, 1):
    pid = os.path.basename(f).replace(".nii.gz", "")
    out = f"{out_root}/{pid}"
    if os.path.exists(f"{out}/heart.nii.gz"):
        print(i, pid, "already done", flush=True)
        continue
    print(i, pid, "running...", flush=True)
    cmd = ["TotalSegmentator", "-i", f, "-o", out, "--fast", "--device", "cpu"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED", pid, "\n", r.stderr[-1500:], flush=True)
        break          # stop at the first failure so we can read the error
    # keep only the structures we need
    for g in glob.glob(f"{out}/*.nii.gz"):
        if os.path.basename(g).replace(".nii.gz", "") not in keep:
            os.remove(g)