import os, glob
import SimpleITK as sitk

os.makedirs("data/resampled", exist_ok=True)
spacing = (2.0, 2.0, 2.0)

def resample(img, spacing, is_mask=False):
    old_sp, old_sz = img.GetSpacing(), img.GetSize()
    new_sz = [int(round(old_sz[i] * old_sp[i] / spacing[i])) for i in range(3)]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(spacing)
    f.SetSize(new_sz)
    f.SetOutputDirection(img.GetDirection())
    f.SetOutputOrigin(img.GetOrigin())
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    f.SetDefaultPixelValue(0 if is_mask else -1000)
    return f.Execute(img)

for p in sorted(glob.glob("data/nifti/*.nii.gz")):
    out = os.path.join("data/resampled", os.path.basename(p))
    img = sitk.ReadImage(p)
    sitk.WriteImage(resample(img, spacing), out)
    print("ok", os.path.basename(p))