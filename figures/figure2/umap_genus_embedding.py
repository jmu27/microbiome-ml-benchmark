#!/usr/bin/env python3
"""
UMAP visualization of genus embeddings colored by taxonomic tags
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import warnings

try:
    import umap
except ImportError as e:
    raise ImportError("Please install umap-learn: pip install umap-learn") from e

warnings.filterwarnings("ignore")

# ---------- Font Setup ----------
def check_helvetica_font():
    """Check if Helvetica or similar font is available"""
    fm._load_fontmanager(try_read_cache=False)
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_preferences = ['Helvetica', 'Helvetica Neue', 'HelveticaNeue', 'Liberation Sans', 'Arial']

    for font in font_preferences:
        if font in available_fonts:
            print(f"✓ Using font: {font}")
            return font

    print("⚠ No preferred font available, using default sans-serif")
    return 'DejaVu Sans'

best_font = check_helvetica_font()

# Set matplotlib parameters for beautiful plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [best_font, 'Liberation Sans', 'Helvetica', 'Helvetica Neue', 'Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# ---------- Paths ----------
DATA_DIR = Path("/ua/jmu27/Micro_bench/data/gpt_embedding")
OUTPUT_DIR = Path("/ua/jmu27/Micro_bench/figures/figure2")
EMBED_FILE = DATA_DIR / "combined_embedding_genus_tagged.pkl"

# ---------- Load Data ----------
print("Loading genus embeddings...")
df = pd.read_pickle(EMBED_FILE)

print(f"Loaded {len(df)} genera")
print(f"Columns: {df.columns.tolist()}")

# Find embedding columns (assuming they start with specific patterns or are numeric)
# Usually embeddings are in columns like 'emb_0', 'emb_1', ... or similar
embedding_cols = [col for col in df.columns if col not in ['Genus', 'Genus_norm', 'tag', 'taxname_norm']]

if not embedding_cols:
    # Try to find numeric columns
    embedding_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'tag' in embedding_cols:
        embedding_cols.remove('tag')

print(f"Found {len(embedding_cols)} embedding dimensions")

# Extract embeddings
X = df[embedding_cols].values
tags = df['tag'].fillna('Others').values  # Use 'Others' for missing tags

print(f"\nTag distribution:")
print(df['tag'].value_counts(dropna=False))

# ---------- UMAP ----------
print("\nRunning UMAP...")
# Using optimized parameters for better class separation
reducer = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=2,
    metric='cosine',
    random_state=42
)
embedding_2d = reducer.fit_transform(X)

# ---------- Plot ----------
print("Creating plot...")

# Define colors for each taxonomic group
# Using high-contrast, vivid colors for the 4 major phyla
color_map = {
    'Pseudomonadota': '#1F77B4',     # Strong Blue - Most abundant (594)
    'Bacillota': '#9467BD',          # Strong Purple - 2nd most (296)
    'Actinomycetota': '#FF7F0E',     # Strong Orange - 3rd most (238)
    'Bacteroidota': '#2CA02C',       # Strong Green - 4th most (179)
    'Others': '#CCCCCC',             # Light gray - Others
    'None': '#7F7F7F'                # Medium gray - No lineage info
}

label_map = {
    'Pseudomonadota': 'Pseudomonadota',
    'Bacillota': 'Bacillota',
    'Actinomycetota': 'Actinomycetota',
    'Bacteroidota': 'Bacteroidota',
    'Others': 'Others',
    'None': 'Unclassified'
}

fig, ax = plt.subplots(figsize=(10, 8))

# Plot each group separately for better legend control
# Order: main groups first (by abundance), then Others
plot_order = ['Pseudomonadota', 'Bacillota', 'Actinomycetota', 'Bacteroidota', 'Others', 'None']
unique_tags = [t for t in plot_order if t in np.unique(tags)]

for tag_value in unique_tags:
    mask = tags == tag_value
    color = color_map.get(tag_value, '#CCCCCC')
    label = label_map.get(tag_value, tag_value)

    ax.scatter(
        embedding_2d[mask, 0],
        embedding_2d[mask, 1],
        c=color,
        s=30,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5,
        label=label
    )

# Beautify plot
ax.set_title('UMAP of Genus Embeddings by Phylum',
             fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('UMAP-1', fontsize=18, fontweight='normal')
ax.set_ylabel('UMAP-2', fontsize=18, fontweight='normal')

# Add grid
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='best', frameon=True, fancybox=False,
         edgecolor='black', framealpha=0.95, fontsize=12)

# Clean up spines
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.tight_layout()

# Save
output_base = OUTPUT_DIR / "genus_embedding_umap_by_taxonomy"
plt.savefig(str(output_base) + ".png", dpi=300, bbox_inches='tight')
plt.savefig(str(output_base) + ".pdf", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Saved:")
print(f"  - {output_base}.png")
print(f"  - {output_base}.pdf")

# ---------- Also create a version with main 4 groups only ----------
print("\nCreating plot for main taxonomic groups only...")

# Only include the four main phyla
main_groups = ['Pseudomonadota', 'Bacillota', 'Actinomycetota', 'Bacteroidota']
main_mask = np.isin(tags, main_groups)
X_main = embedding_2d[main_mask]
tags_main = tags[main_mask]

fig, ax = plt.subplots(figsize=(10, 8))

for tag_value in main_groups:
    if tag_value in tags_main:
        mask = tags_main == tag_value
        color = color_map.get(tag_value, '#CCCCCC')
        label = label_map.get(tag_value, tag_value)

        ax.scatter(
            X_main[mask, 0],
            X_main[mask, 1],
            c=color,
            s=35,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5,
            label=label
        )

ax.set_title('UMAP of Genus Embeddings by Phylum (Main Groups)',
             fontsize=20, fontweight='bold', pad=20)
ax.set_xlabel('UMAP-1', fontsize=18, fontweight='normal')
ax.set_ylabel('UMAP-2', fontsize=18, fontweight='normal')

ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

ax.legend(loc='best', frameon=True, fancybox=False,
         edgecolor='black', framealpha=0.95, fontsize=12)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('black')

plt.tight_layout()

output_base_main = OUTPUT_DIR / "genus_embedding_umap_main_groups"
plt.savefig(str(output_base_main) + ".png", dpi=300, bbox_inches='tight')
plt.savefig(str(output_base_main) + ".pdf", dpi=300, bbox_inches='tight')
plt.close()

print(f"  - {output_base_main}.png")
print(f"  - {output_base_main}.pdf")

print("\n✅ Done!")
