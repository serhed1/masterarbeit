#################################################################
# HOW TO RUN PYTHON IN VENV
# python3 -m venv myenv
# source myenv/bin/activate
# pip install ...
# python ... .py
#################################################################

from datasets import load_dataset
import pandas as pd

# 1) Laden (zieht Arrow/Parquet-Backend aus HF Hub; cached lokal)
ds = load_dataset("datastax/linkedin_job_listings", split="train")  # ~124k rows

# 2) Schema anzeigen
print(ds)                  # Größe/Split
print(ds.features)         # Spaltennamen + dtypes

# 3) Kleine Vorschau als Pandas
df = ds.to_pandas()        # oder ds.to_polars() falls polars installiert
print(df.head(3))

# 4) Praktische Typkonvertierungen
import numpy as np
epoch_cols = ["listed_time", "original_listed_time", "closed_time", "expiry"]
for c in epoch_cols:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], unit="ms", errors="coerce")

# remote_allowed kommt als float (0/1) → bool
if "remote_allowed" in df.columns:
    df["remote_allowed"] = df["remote_allowed"].astype("float").round().astype("Int64").astype("boolean")

# 5) Nur Felder, die wir für Textanalyse sicher brauchen
keep = [
    "job_id", "company_name", "title", "description", "skills_desc",
    "min_salary","med_salary","max_salary","currency","pay_period",
    "compensation_type","work_type","formatted_work_type","formatted_experience_level",
    "remote_allowed","location","job_posting_url","listed_time"
]
df_small = df[[c for c in keep if c in df.columns]].copy()
print(df_small.sample(5, random_state=42))

# 6) Optional lokal als Parquet speichern (schnell & komprimiert)
df_small.to_parquet("linkedin_job_listings_small.parquet", index=False)