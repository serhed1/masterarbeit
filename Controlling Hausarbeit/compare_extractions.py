#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

# === Pfade ===
OLD_PATH = Path("outputs_141025_1812/extractions.json")
NEW_PATH = Path("outputs/extractions.json")

# === Laden ===
with open(OLD_PATH, "r", encoding="utf-8") as f:
    old_data = json.load(f)

with open(NEW_PATH, "r", encoding="utf-8") as f:
    new_data = json.load(f)

# === Nur die ersten 100 Einträge betrachten ===
job_ids = list(new_data.keys())[:100]

changes = []
unchanged = 0
missing_in_old = 0

for job_id in job_ids:
    old_entry = old_data.get(job_id)
    new_entry = new_data.get(job_id)

    if not old_entry:
        missing_in_old += 1
        continue

    old_comp = old_entry.get("compensation")
    new_comp = new_entry.get("compensation")

    old_remote = old_entry.get("remote")
    new_remote = new_entry.get("remote")

    if old_comp != new_comp or old_remote != new_remote:
        changes.append({
            "job_id": job_id,
            "compensation_old": old_comp,
            "compensation_new": new_comp,
            "remote_old": old_remote,
            "remote_new": new_remote,
        })
    else:
        unchanged += 1

# === Zusammenfassung ===
print("=== Vergleichszusammenfassung (erste 100 Einträge) ===")
print(f"Gesamt überprüft: {len(job_ids)}")
print(f"Unverändert:      {unchanged}")
print(f"Geändert:         {len(changes)}")
print(f"Fehlend (alt):    {missing_in_old}")
print("")

if changes:
    print("Beispiele für Änderungen:")
    for c in changes[:10]:  # nur die ersten 10 zeigen
        print(f"- job_id {c['job_id']}:")
        print(f"    compensation: {c['compensation_old']} → {c['compensation_new']}")
        print(f"    remote:       {c['remote_old']} → {c['remote_new']}")
else:
    print("Keine Unterschiede gefunden.")

# === Optional: Änderungen als JSON speichern ===
with open("outputs/changes_summary.json", "w", encoding="utf-8") as f:
    json.dump(changes, f, ensure_ascii=False, indent=2)

print("\nÄnderungen gespeichert unter: outputs/changes_summary.json")