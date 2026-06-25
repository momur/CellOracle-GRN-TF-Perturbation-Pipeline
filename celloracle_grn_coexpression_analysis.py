#!/usr/bin/env python3
"""
celloracle_grn_coexpression_analysis.py
----------------------------------------
Post-hoc analysis: cross-reference the CellOracle base GRN with TF motif
scan results and GRNBoost2 co-expression scores to identify TFs regulating
a gene of interest.

Workflow:
  1. Load TF motif scan results and base GRN
  2. Merge peak-to-gene links with TF binding scores
  3. Load GRNBoost2 co-expression output
  4. For a given GOI, find TFs that both:
       - bind near the GOI locus (motif evidence), AND
       - co-express with the GOI (GRNBoost2 score > threshold)

Usage:
    python celloracle_grn_coexpression_analysis.py

Edit the Configuration section below before running.
"""

import os
import pandas as pd
import celloracle as co

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR     = "/scratch/st-wasser-1/momur_mono/celloracle_islets_inte"
TFINFO_FILE  = "test1.celloracle.tfinfo"
BASE_GRN     = "base_GRN_dataframe.parquet"
GRNBOOST_TSV = os.path.join(DATA_DIR, "GRNboost2_multiomics_betacelltypes.tsv")

GENE_OF_INTEREST   = "SOX9"         # gene whose upstream regulators you want
GRNBOOST_THRESHOLD = 10             # min co-expression score to consider a TF

# ── 1. Load TF motif scan ─────────────────────────────────────────────────────
print("Loading TF motif scan...")
tf_info  = co.load_hdf5(TFINFO_FILE)
tfi_scan = tf_info.scanned_df
tfi_scan.to_csv("TFinfo_scanned_dataframe.csv", index=False)
print(f"  TF scan rows: {len(tfi_scan)}")

# ── 2. Merge peak-to-gene links with TF binding scores ───────────────────────
print("Merging base GRN with TF scan...")
base_grn_df = pd.read_parquet(BASE_GRN, engine="pyarrow")

# Reduce base GRN to peak → gene mapping and rename for merge
peak_gene = (
    base_grn_df[["peak_id", "gene_short_name"]]
    .rename(columns={"peak_id": "seqname"})
)

merged = tfi_scan.merge(peak_gene, on="seqname")
print(f"  Merged rows: {len(merged)}")

# Subset to the gene of interest
goi_merged = merged[merged["gene_short_name"] == GENE_OF_INTEREST].copy()
print(f"  Peaks linked to {GENE_OF_INTEREST}: {len(goi_merged)}")

# ── 3. Load GRNBoost2 co-expression scores ────────────────────────────────────
print("Loading GRNBoost2 output...")
grnboost = pd.read_csv(GRNBOOST_TSV, sep="\t", header=None,
                       names=["TF", "TG", "score"])
grnboost_filtered = grnboost[grnboost["score"] > GRNBOOST_THRESHOLD]

# Top GRNBoost2 TFs for the GOI
grnboost_goi = grnboost_filtered[grnboost_filtered["TF"] == GENE_OF_INTEREST]
coexpress_targets = set(grnboost_goi["TG"])
print(f"  GRNBoost2 targets of {GENE_OF_INTEREST} (score>{GRNBOOST_THRESHOLD}): {len(coexpress_targets)}")

# ── 4. Annotate motif scan with co-expression evidence ────────────────────────
def check_tf_in_coexpress(row, gene_set):
    """Return whether any TF binding this peak is in the co-expression set."""
    direct   = set(str(row.get("factors_direct",   "")).split(", "))
    indirect = set(str(row.get("factors_indirect", "")).split(", "))
    if direct & gene_set:
        return "direct"
    if indirect & gene_set:
        return "indirect"
    return "none"

goi_merged["coexpression_evidence"] = goi_merged.apply(
    check_tf_in_coexpress, gene_set=coexpress_targets, axis=1
)

# ── 5. Save results ───────────────────────────────────────────────────────────
out_file = f"GRN_coexpression_analysis_{GENE_OF_INTEREST}.csv"
goi_merged.sort_values("coexpression_evidence").to_csv(out_file, index=False)
print(f"\nSaved results to {out_file}")

summary = goi_merged["coexpression_evidence"].value_counts()
print(f"\nSummary for {GENE_OF_INTEREST}:\n{summary.to_string()}")

direct_hits = goi_merged[goi_merged["coexpression_evidence"] == "direct"]
print(f"\nDirect co-expression hits ({len(direct_hits)} rows):")
print(direct_hits[["seqname", "factors_direct", "score"]].head(20).to_string(index=False))
