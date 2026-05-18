"""
Build label distribution table from labels_train.csv.

Shows labels with >=1% share in at least one profile, plus an aggregate row
for the rest and a total class count.
"""

from pathlib import Path
import pandas as pd

SPLITS = Path("data/splits")
OUT_DIR = Path("data/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILES = ["FAST", "BALANCED", "COMPRESS", "ARCHIVE", "WEB"]
THRESHOLD_PCT = 1.0

labels = pd.read_csv(SPLITS / "labels_train.csv")

dist = {}
for prof in PROFILES:
    vc = labels[prof].value_counts(normalize=True) * 100
    dist[prof] = vc

dist_df = pd.DataFrame(dist).fillna(0.0)

show = dist_df[dist_df.max(axis=1) >= THRESHOLD_PCT].index.tolist()
# SKIP first, then by mean share descending
show_sorted = sorted(show, key=lambda l: (l != "SKIP", -dist_df.loc[l].mean()))

rows = []
for lbl in show_sorted:
    row = {"Klase": lbl}
    for prof in PROFILES:
        row[prof] = f"{dist_df.loc[lbl, prof]:.1f}%"
    rows.append(row)

other_labels = [l for l in dist_df.index if l not in show]
other_row = {"Klase": "Pārējās klases"}
for prof in PROFILES:
    total_other = dist_df.loc[other_labels, prof].sum() if other_labels else 0.0
    other_row[prof] = f"<{round(total_other + 1):.0f}%" if total_other > 0 else "—"
rows.append(other_row)

count_row = {"Klase": "Klašu skaits"}
for prof in PROFILES:
    count_row[prof] = str(labels[prof].nunique())
rows.append(count_row)

result = pd.DataFrame(rows)

result.to_csv(OUT_DIR / "label_distribution.csv", index=False)
print(f"Saved: {OUT_DIR}/label_distribution.csv")

md_lines = ["| Klase | " + " | ".join(PROFILES) + " |"]
md_lines.append("| --- | " + " | ".join(["---"] * len(PROFILES)) + " |")
for _, row in result.iterrows():
    md_lines.append("| " + " | ".join(str(row[c]) for c in ["Klase"] + PROFILES) + " |")

md_path = OUT_DIR / "label_distribution.md"
md_path.write_text("\n".join(md_lines), encoding="utf-8")
print(f"Saved: {OUT_DIR}/label_distribution.md")

print("\n" + "\n".join(md_lines).encode("ascii", errors="replace").decode("ascii"))
