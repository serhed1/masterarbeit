#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.stats import f_oneway, levene, kruskal

# ========== Setup ==========
DATA_PATH = "outputs/extractions.json"
SAVE_DIR = "analysis_evolution"
os.makedirs(SAVE_DIR, exist_ok=True)

# Subdirectories for each variable type
COMP_DIR = os.path.join(SAVE_DIR, "compensation")
REM_DIR = os.path.join(SAVE_DIR, "remote")
os.makedirs(COMP_DIR, exist_ok=True)
os.makedirs(REM_DIR, exist_ok=True)

# ========== Load Ordered Data ==========
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found.")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    od = json.load(f, object_pairs_hook=OrderedDict)

# Build DataFrame preserving order
rows = []
for i, (job_id, rec) in enumerate(od.items()):
    rows.append({
        "seq": i,
        "job_id": job_id,
        "task_count": rec.get("task_count"),
        "compensation": str(rec.get("compensation")),
        "remote": str(rec.get("remote")),
    })

df = pd.DataFrame(rows).dropna(subset=["task_count"])
df["task_count"] = pd.to_numeric(df["task_count"], errors="coerce")
df = df.dropna(subset=["task_count"]).reset_index(drop=True)

# ========== Helper: Statistical Tests ==========
def run_stats(df_prefix, category_col, order):
    """Run Levene, ANOVA, Kruskal-Wallis + effect sizes."""
    groups_dict = {
        c: df_prefix.loc[df_prefix[category_col] == c, "task_count"].values
        for c in order
    }
    groups = [v for v in groups_dict.values() if len(v) > 0 and np.isfinite(v).all()]
    if len(groups) < 2:
        return {"levene_p": np.nan, "anova_p": np.nan, "kruskal_p": np.nan,
                "eta_sq": np.nan, "omega_sq": np.nan}

    lev_stat, lev_p = levene(*groups, center="median")
    f_stat, p_anova = f_oneway(*groups)
    kw_stat, p_kruskal = kruskal(*groups)

    grand_mean = df_prefix["task_count"].mean()
    ss_between = sum(len(vals) * (np.mean(vals) - grand_mean) ** 2 for vals in groups)
    ss_total = np.sum((df_prefix["task_count"] - grand_mean) ** 2)
    k = len(groups)
    N = len(df_prefix)
    df_between = k - 1
    df_within = max(N - k, 1)
    ms_within = (ss_total - ss_between) / df_within if df_within > 0 else np.nan
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
    omega_sq = ((ss_between - df_between * ms_within) /
                (ss_total + ms_within)) if (ss_total + ms_within) > 0 else np.nan

    return {"levene_p": lev_p, "anova_p": p_anova, "kruskal_p": p_kruskal,
            "eta_sq": eta_sq, "omega_sq": omega_sq}

# ========== Core Function ==========
def analyze_category(df, category_col, save_dir):
    """
    Run full prefix-based evolution analysis for one category column.
    """
    order = sorted(df[category_col].dropna().unique())

    # ----- Boxplots -----
    boxplot_sizes = [100, 1000, 2000, 3000, 4000, 5000]
    for n in boxplot_sizes:
        df_n = df.sort_values("seq").iloc[:min(n, len(df))].copy()
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=category_col, y="task_count", data=df_n, order=order)
        plt.title(f"Task Count Distribution by {category_col.title()} (prefix n={len(df_n)})")
        plt.xlabel(category_col.title())
        plt.ylabel("Task Count")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"task_count_by_{category_col}_n{len(df_n)}.png"), dpi=300)
        plt.close()

    # ----- Fine-grained prefix significance (1000–3000 step=100) -----
    start_n, end_n, step = 1000, min(5000, len(df)), 100
    records = []
    for n in range(start_n, end_n + 1, step):
        df_n = df.sort_values("seq").iloc[:n].copy()
        stats = run_stats(df_n, category_col, order)
        stats["n"] = n
        records.append(stats)
        print(f"[{category_col} | n={n}] ANOVA p={stats['anova_p']:.6f}, "
              f"Kruskal p={stats['kruskal_p']:.6f}")

    results_df = pd.DataFrame(records).sort_values("n")
    results_df.to_csv(os.path.join(save_dir, f"pvalues_overview_{category_col}_prefix_by100.csv"), index=False)

    # ----- Significance evolution plot -----
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["n"], results_df["anova_p"], marker="o", linewidth=1.5, label="ANOVA p-value")
    plt.plot(results_df["n"], results_df["kruskal_p"], marker="s", linewidth=1.5, label="Kruskal-Wallis p-value")
    plt.axhline(0.05, color="red", linestyle="--", linewidth=1, label="5% significance threshold")
    plt.axhline(0.10, color="orange", linestyle=":", linewidth=1, label="10% significance threshold")
    plt.title(f"P-value Trend for {category_col.title()} (prefix n=1000–5000, step=100)")
    plt.xlabel("Number of Job Postings (prefix n)")
    plt.ylabel("p-value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pvalues_vs_prefix_{category_col}_n1000_5000_step100.png"), dpi=300)
    plt.close()

# ========== Run Both Analyses ==========
print("=== Analyzing Compensation ===")
analyze_category(df, "compensation", COMP_DIR)

print("\n=== Analyzing Remote Type ===")
analyze_category(df, "remote", REM_DIR)

print(f"\n✅ Analysis complete. All results saved to: {os.path.abspath(SAVE_DIR)}")