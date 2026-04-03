#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import pandas as pd

EXTRACTIONS_PATHS = [
    "outputs/extractions.json",
    "./outputs/extractions.json",
]
PARQUET_PATHS = [
    "linkedin_job_listings_small.parquet",
    "./linkedin_job_listings_small.parquet",
]
OUT_PATH = "extractions_with_descriptions_sample.json"  # neue Datei

def find(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def main():
    extr_path = find(EXTRACTIONS_PATHS)
    parq_path = find(PARQUET_PATHS)

    if not extr_path or not parq_path:
        raise FileNotFoundError(
            "Quelle nicht gefunden. Erwartet:\n"
            f"- extractions.json unter {EXTRACTIONS_PATHS}\n"
            f"- linkedin_job_listings_small.parquet unter {PARQUET_PATHS}"
        )

    # Extractions laden
    with open(extr_path, "r", encoding="utf-8") as f:
        extr = json.load(f)
    ex_df = pd.DataFrame.from_dict(extr, orient="index")
    ex_df.index.name = "job_id"
    ex_df.reset_index(inplace=True)

    # Parquet laden
    df = pd.read_parquet(parq_path)
    if "job_id" not in df.columns or "description" not in df.columns:
        raise KeyError("Im Parquet müssen Spalten 'job_id' und 'description' enthalten sein.")

    # Typanpassung & Merge
    ex_df["job_id"] = ex_df["job_id"].astype(str)
    df["job_id"] = df["job_id"].astype(str)
    merged = ex_df.merge(df[["job_id", "description"]], on="job_id", how="inner")

    # 100 Beispiele (reproduzierbar)
    sample = merged.sample(n=min(100, len(merged)), random_state=42)

    # Zurück in dict-Struktur wie extractions.json + description ergänzen
    out = {}
    for _, row in sample.iterrows():
        job_id = row["job_id"]
        entry = row.drop(labels=["job_id", "description"]).to_dict()
        # Falls job_tasks versehentlich als String gespeichert wurde, versuchen zu parsen
        if isinstance(entry.get("job_tasks"), str):
            try:
                entry["job_tasks"] = json.loads(entry["job_tasks"])
            except Exception:
                pass
        entry["description"] = row["description"]
        out[job_id] = entry

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Fertig. Datei gespeichert: {OUT_PATH} (Einträge: {len(out)})")

if __name__ == "__main__":
    main()