#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pathlib

# ===== Datei laden =====
PARQUET_PATH = "linkedin_job_listings_small.parquet"
if not pathlib.Path(PARQUET_PATH).exists():
    raise FileNotFoundError(f"{PARQUET_PATH} not found")

df = pd.read_parquet(PARQUET_PATH)

print("\n📊 --- Grundlegende Infos ---")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print("\nSpalten:")
for col in df.columns:
    print(f"  - {col}")

print("\n\n📄 --- Datentypen ---")
print(df.dtypes)

print("\n\n🔍 --- Beispielzeilen ---")
print(df.head(3).to_string())

print("\n\n📈 --- Fehlende Werte (Top 10) ---")
missing = df.isna().sum().sort_values(ascending=False)
print(missing.head(10))

print("\n\n💰 --- Spalten, die evtl. für Compensation relevant sind ---")
for c in df.columns:
    if any(k in c.lower() for k in ["salary", "pay", "compensation", "currency"]):
        nonnull = df[c].notna().sum()
        unique = df[c].nunique(dropna=True)
        print(f"{c:<25} → {nonnull} non-null, {unique} unique")
        print(df[c].dropna().unique()[:5], "\n")

print("\n🏠 --- Spalten, die evtl. Remote-Infos enthalten ---")
for c in df.columns:
    if any(k in c.lower() for k in ["remote", "location", "work_type", "onsite"]):
        nonnull = df[c].notna().sum()
        unique = df[c].nunique(dropna=True)
        print(f"{c:<25} → {nonnull} non-null, {unique} unique")
        print(df[c].dropna().unique()[:5], "\n")

print("\n✅ Analyse abgeschlossen.")