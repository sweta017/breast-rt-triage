import pandas as pd

df = pd.read_csv("results/features.csv")
print(len(df), "rows;", df.patient.nunique(), "patients")
print("missing values:\n", df.isna().sum()[lambda s: s > 0])

cols = ["cld_max_mm", "mhd_max_mm", "lung_in_field_frac", "heart_in_field_frac"]
print("\nMeans by source and side:")
print(df.groupby(["source", "side"])[cols].mean().round(2))

w = df.pivot_table(index=["patient", "side"], columns="source", values=cols)
print("\nAgreement between automatic and expert masks (correlation):")
for c in cols:
    print(f"  {c}: {w[(c, 'auto')].corr(w[(c, 'expert')]):.2f}")

print("\nS3-012 (collapsed left lung):")
print(df[df.patient == "LCTSC-Train-S3-012"][["side", "source"] + cols])