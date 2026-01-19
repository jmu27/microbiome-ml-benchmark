#!/usr/bin/env python3
"""
Analyze genus_lineage.csv to find the most abundant phyla
"""

import re
import pandas as pd
from collections import Counter
from pathlib import Path

# Known phylum taxids (common bacterial/archaeal phyla)
# This is a curated list of major phyla
KNOWN_PHYLA = {
    # Bacteria
    1239: 'Bacillota (Firmicutes)',
    1224: 'Pseudomonadota (Proteobacteria)',
    201174: 'Actinomycetota (Actinobacteria)',
    976: 'Bacteroidota (Bacteroidetes)',
    32066: 'Fusobacteriota',
    40117: 'Spirochaetota',
    74201: 'Verrucomicrobiota',
    200783: 'Chloroflexota',
    1297: 'Deinococcota',
    200918: 'Thermotogota',

    # Archaea
    28890: 'Euryarchaeota',
    2172: 'Methanobacteriota',
    1935183: 'Thermoplasmatota',
    1783275: 'Asgardarchaeota',
}

# Path
DATA_DIR = Path("/ua/jmu27/Micro_bench/data/gpt_embedding")
PATH_LINEAGE = DATA_DIR / "genus_lineage.csv"

# Parse lineage
def parse_lineage_row(raw: str):
    """Parse a single lineage row"""
    line = raw.strip()
    if not line or line.lower().startswith("taxid"):
        return None

    parts = line.split(",|,")
    if len(parts) < 3:
        parts = re.split(r"\s*,\|\s*,", line)
    if len(parts) < 3:
        return None

    taxid_str = parts[0].strip(" ,;")
    taxname = parts[1].strip(" ,;")
    lineage_blob = parts[2]

    lineage_blob = re.sub(r"[^\d,]", "", lineage_blob)
    lineage_ids = []
    for tok in lineage_blob.split(","):
        tok = tok.strip()
        if tok.isdigit():
            lineage_ids.append(int(tok))

    taxid = None
    if taxid_str.isdigit():
        taxid = int(taxid_str)

    if taxid is None or not taxname:
        return None
    return taxid, taxname, lineage_ids

# Load and parse
print("Loading genus_lineage.csv...")
lineage_rows = []
with PATH_LINEAGE.open("r", encoding="utf-8") as f:
    for row in f:
        rec = parse_lineage_row(row)
        if rec:
            lineage_rows.append(rec)

df_lineage = pd.DataFrame(lineage_rows, columns=["taxid", "taxname", "lineage_ids"])
print(f"Loaded {len(df_lineage)} genera\n")

# Count phyla
print("Analyzing phylum distribution...")
phylum_counts = Counter()

for lineage_ids in df_lineage["lineage_ids"]:
    lineage_set = set(lineage_ids or [])
    # Check which known phyla this genus belongs to
    for phylum_id in KNOWN_PHYLA.keys():
        if phylum_id in lineage_set:
            phylum_counts[phylum_id] += 1
            break  # Only count once per genus

# Print results
print("=" * 80)
print("Top Phyla by Number of Genera:")
print("=" * 80)
print(f"{'Rank':<6} {'Taxid':<10} {'Count':<8} {'Phylum Name'}")
print("-" * 80)

for rank, (phylum_id, count) in enumerate(phylum_counts.most_common(15), 1):
    phylum_name = KNOWN_PHYLA.get(phylum_id, f"Unknown ({phylum_id})")
    print(f"{rank:<6} {phylum_id:<10} {count:<8} {phylum_name}")

print("-" * 80)
total_classified = sum(phylum_counts.values())
print(f"Total classified genera: {total_classified}/{len(df_lineage)} ({100*total_classified/len(df_lineage):.1f}%)")
print(f"Unclassified genera: {len(df_lineage) - total_classified}")

# Recommend top 3-4 phyla
print("\n" + "=" * 80)
print("RECOMMENDATION for UMAP visualization:")
print("=" * 80)
top_phyla = phylum_counts.most_common(4)
print("\nTop 4 most abundant phyla:")
for rank, (phylum_id, count) in enumerate(top_phyla, 1):
    phylum_name = KNOWN_PHYLA.get(phylum_id, f"Unknown ({phylum_id})")
    pct = 100 * count / len(df_lineage)
    print(f"  {rank}. {phylum_name}")
    print(f"     Taxid: {phylum_id}, Count: {count} ({pct:.1f}%)")

print("\n" + "=" * 80)
