#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.stats import f_oneway, levene, kruskal
import numpy as np

# ========== Setup ==========
DATA_PATH = "outputs/extractions.json"
SAVE_DIR = "analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== Load ==========
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} nicht gefunden.")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# In DataFrame umwandeln
df = pd.DataFrame.from_dict(data, orient="index")

# Nur relevante Spalten
df = df[["task_count", "compensation", "remote"]].dropna()
df["compensation"] = df["compensation"].astype(str)

print("=== Head of Data ===")
print(df.head(), "\n")

# ========== Grundlegende Statistik ==========
print("=== Anzahl Einträge pro Compensation-Typ ===")
print(df["compensation"].value_counts(), "\n")

print("=== Durchschnittliche Task-Anzahl pro Compensation-Typ ===")
print(df.groupby("compensation")["task_count"].describe(), "\n")

# ========== Visualisierung: Boxplot ==========
plt.figure(figsize=(8, 5))
order = sorted(df["compensation"].unique())
sns.boxplot(x="compensation", y="task_count", data=df, order=order)
plt.title("Verteilung der Task-Anzahl nach Compensation Type")
plt.xlabel("Compensation Type")
plt.ylabel("Task Count")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "task_count_by_compensation.png"), dpi=300)
plt.close()

# ========== Regression / Korrelation (symbolisch) ==========
comp_map = {c: i for i, c in enumerate(df["compensation"].unique())}
df["compensation_code"] = df["compensation"].map(comp_map)

corr = df["task_count"].corr(df["compensation_code"])
print(f"Korrelation (TaskCount vs CompensationType): {corr:.3f}")

plt.figure(figsize=(7, 5))
sns.regplot(x="task_count", y="compensation_code", data=df, logistic=False)
plt.yticks(list(comp_map.values()), list(comp_map.keys()))
plt.title("Regression: Task Count vs Compensation Type")
plt.xlabel("Task Count")
plt.ylabel("Compensation Type (encoded)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "regression_taskcount_vs_compensation.png"), dpi=300)
plt.close()

# ========== Heatmap: Compensation × Remote ==========
avg_tasks_remote = df.groupby(["compensation", "remote"])["task_count"].mean().unstack()
print("=== Durchschnittliche Task-Anzahl nach Compensation & Remote-Typ ===")
print(avg_tasks_remote.round(2))

plt.figure(figsize=(9, 5))
sns.heatmap(avg_tasks_remote, annot=True, fmt=".1f", cmap="Blues")
plt.title("Durchschnittliche Task-Anzahl nach Compensation & Remote-Typ")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "heatmap_compensation_remote.png"), dpi=300)
plt.close()

# ========== INFERENZ-TEIL: ANOVA + Effektgrößen ==========
# Gruppen bauen
groups_dict = {c: df.loc[df["compensation"] == c, "task_count"].values for c in order}
groups = [v for v in groups_dict.values() if len(v) > 0]

# Levene-Test (Varianzgleichheit)
lev_stat, lev_p = levene(*groups, center="median")
print(f"\nLevene-Test (Varianzgleichheit): W = {lev_stat:.3f}, p = {lev_p:.4f} "
      f"{'(Varianzen ungleich)' if lev_p < 0.05 else '(Varianzen ~gleich)'}")

# One-way ANOVA
f_stat, p_val = f_oneway(*groups)
print(f"ANOVA (one-way): F = {f_stat:.3f}, p = {p_val:.6f}")

# Effektgrößen berechnen
# Gesamtmittel
grand_mean = df["task_count"].mean()
# SS_between
ss_between = 0.0
for c, vals in groups_dict.items():
    n_i = len(vals)
    if n_i == 0:
        continue
    ss_between += n_i * (np.mean(vals) - grand_mean) ** 2
# SS_total
ss_total = np.sum((df["task_count"] - grand_mean) ** 2)
# Freiheitsgrade
k = len(groups)                    # Anzahl Gruppen
N = len(df)                        # Gesamtbeobachtungen
df_between = k - 1
df_within = N - k
# eta squared & omega squared
eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
# Für omega² brauchen wir MS_within = (SS_total - SS_between) / df_within
ms_within = (ss_total - ss_between) / df_within
omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within)

print(f"Effektgrößen: eta² = {eta_sq:.4f}, omega² = {omega_sq:.4f}")

# ========== Robuste Alternative: Kruskal-Wallis ==========
# (nicht-parametrischer Test, falls Normalität/Varianzgleichheit fraglich)
kw_stat, kw_p = kruskal(*groups)
print(f"Kruskal-Wallis: H = {kw_stat:.3f}, p = {kw_p:.6f}")


print(f"\n✅ Analyse abgeschlossen. Grafiken & Ergebnisse gespeichert in: {os.path.abspath(SAVE_DIR)}")