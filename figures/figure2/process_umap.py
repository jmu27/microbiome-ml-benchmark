import re
import ast
import pandas as pd
from pathlib import Path

# ---------- Paths (edit if needed) ----------
DATA_DIR = Path("/ua/jmu27/Micro_bench/data/gpt_embedding")
PATH_LINEAGE = DATA_DIR / "genus_lineage.csv"
PATH_GENUSNAMES = DATA_DIR / "genus_id_names.csv"          # optional, used if available
PATH_EMBED = DATA_DIR / "combined_embedding_genus.pkl" # input
PATH_OUT = DATA_DIR / "combined_embedding_genus_tagged.pkl"  # output

# ---------- Helper: parse a single strange lineage row ----------
def parse_lineage_row(raw: str):
    """
    Expected raw example (with weird separators and trailing punctuation):
        '1678,|,Bifidobacterium,|,1678,31953,85004,1760,201174,1783272,2,131567,,,;'
    Returns: (taxid:int, taxname:str, lineage_ids:List[int])
    """
    line = raw.strip()
    if not line or line.lower().startswith("taxid"):
        return None

    # Split into three parts by the literal ",|," pattern
    parts = line.split(",|,")
    if len(parts) < 3:
        # Try a fallback: sometimes there could be spaces around | or inconsistent commas
        parts = re.split(r"\s*,\|\s*,", line)
    if len(parts) < 3:
        return None  # skip malformed lines

    taxid_str = parts[0].strip(" ,;")
    taxname = parts[1].strip(" ,;")
    lineage_blob = parts[2]

    # Keep only digits and commas from the lineage blob, then split
    # (this removes stray semicolons and extra commas)
    lineage_blob = re.sub(r"[^\d,]", "", lineage_blob)
    lineage_ids = []
    for tok in lineage_blob.split(","):
        tok = tok.strip()
        if tok.isdigit():
            lineage_ids.append(int(tok))

    # taxid
    taxid = None
    if taxid_str.isdigit():
        taxid = int(taxid_str)

    if taxid is None or not taxname:
        return None
    return taxid, taxname, lineage_ids

# ---------- Load and parse genus_lineage.csv ----------
lineage_rows = []
with PATH_LINEAGE.open("r", encoding="utf-8") as f:
    for row in f:
        rec = parse_lineage_row(row)
        if rec:
            lineage_rows.append(rec)

df_lineage = pd.DataFrame(lineage_rows, columns=["taxid", "taxname", "lineage_ids"])
# Normalize names for robust joining
df_lineage["taxname_norm"] = df_lineage["taxname"].str.strip().str.replace(r"\s+", " ", regex=True).str.lower()

# ---------- Tagging logic ----------
# Using the 4 most abundant phyla for better visualization:
# 1. Pseudomonadota (602 genera, 34.8%)
# 2. Bacillota (299 genera, 17.3%)
# 3. Actinomycetota (243 genera, 14.1%)
# 4. Bacteroidota (182 genera, 10.5%)
TARGETS = [
    (1224, 'Pseudomonadota'),     # Proteobacteria - most abundant
    (1239, 'Bacillota'),          # Firmicutes
    (201174, 'Actinomycetota'),   # Actinobacteria
    (976, 'Bacteroidota'),        # Bacteroidetes
]

def compute_tag(lineage_ids):
    s = set(lineage_ids or [])
    for target_id, tag in TARGETS:
        if target_id in s:
            return tag
    return 'Others'  # default tag for unmatched

df_lineage["tag"] = df_lineage["lineage_ids"].apply(compute_tag)

# If there are duplicate taxnames with different tags, choose the first by priority order above.
# We'll keep the *max priority* which we already enforced via compute_tag order
priority = {'Pseudomonadota': 4, 'Bacillota': 3, 'Actinomycetota': 2, 'Bacteroidota': 1, 'Others': 0}
df_lineage["_prio"] = df_lineage["tag"].map(priority).fillna(0)
df_lineage = (
    df_lineage.sort_values(["taxname_norm", "_prio"], ascending=[True, False])
              .drop_duplicates(subset=["taxname_norm"], keep="first")
              .drop(columns=["_prio"])
)

# ---------- Load genus_name.csv if present (best name match) ----------
# We expect at least a column named 'Genus' (case-insensitive).
df_gnames = None
if PATH_GENUSNAMES.exists():
    df_gnames = pd.read_csv(PATH_GENUSNAMES)
    # Standardize column name
    cols_lower = {c.lower(): c for c in df_gnames.columns}
    if "genus" in cols_lower:
        col_genus = cols_lower["genus"]
        df_gnames = df_gnames[[col_genus]].rename(columns={col_genus: "Genus"})
    else:
        # If there's no clear 'Genus' column, fall back to a generic single-column interpretation
        if df_gnames.shape[1] == 1:
            df_gnames.columns = ["Genus"]
        else:
            # Try to coerce first column into 'Genus'
            first_col = df_gnames.columns[0]
            df_gnames = df_gnames[[first_col]].rename(columns={first_col: "Genus"})
    df_gnames["Genus_norm"] = df_gnames["Genus"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()

# ---------- Load embeddings ----------
emb = pd.read_pickle(PATH_EMBED)
# Expect a 'Genus' column
if "Genus" not in emb.columns:
    raise ValueError("combined_embedding_genus.pkl must contain a 'Genus' column.")
emb["Genus_norm"] = emb["Genus"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()

# ---------- Build name→tag mapping ----------
# 1) Directly from df_lineage taxname
name_to_tag = df_lineage.set_index("taxname_norm")["tag"].to_dict()

# 2) If genus_name.csv exists, use it to (optionally) refine matches:
#    - If a Genus is in genus_name.csv, we assume its name is the authoritative form for matching.
if df_gnames is not None and not df_gnames.empty:
    # Just ensure these names exist in lineage; if not, we still fall back to direct name matching.
    # Nothing special to do here beyond preferring the normalized Genus names already present.
    pass

# ---------- Apply tags to the embeddings ----------
emb["tag"] = emb["Genus_norm"].map(name_to_tag)

# If some Genus didn't match by name (NaN tag), try a last-ditch fuzzy-ish cleanup:
# Strip common suffixes like " sp." / " gen." etc. (light heuristic)
def light_cleanup(name: str) -> str:
    if not isinstance(name, str):
        return name
    n = re.sub(r"\b(sp|gen)\.?$", "", name.strip(), flags=re.IGNORECASE)
    n = re.sub(r"\s+", " ", n).strip()
    return n.lower()

missing_mask = emb["tag"].isna()
if missing_mask.any():
    emb.loc[missing_mask, "tag"] = emb.loc[missing_mask, "Genus_norm"].map(
        lambda x: name_to_tag.get(light_cleanup(x))
    )

# ---------- Save result ----------
emb.to_pickle(PATH_OUT)
print(f"Tagged embeddings saved to: {PATH_OUT.resolve()}")
print("Tag distribution:\n", emb["tag"].value_counts(dropna=False))
