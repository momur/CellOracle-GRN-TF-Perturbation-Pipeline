#!/usr/bin/env python3
"""
celloracle_pipeline.py
----------------------
CellOracle GRN inference and TF perturbation pipeline for EC/Beta cell dataset.

Workflow:
  1. Pre-process AnnData (normalise, HVG, PCA, clustering)
  2. Calculate pseudotime (via CellOracle Pseudotime_calculator)
  3. Build GRN base from ATAC peaks + cicero connections
  4. TF motif scan
  5. Construct Oracle object + KNN imputation
  6. GRN calculation (run on SLURM — slow)
  7. Export filtered/raw GRNs and network score plots
  8. TF perturbation simulation (loop over GOI list)
  9. Pseudotime gradient + perturbation scoring

Usage:
    conda activate celloracle_env
    python celloracle_pipeline.py

Edit the Configuration section below before running.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm
import celloracle as co
from celloracle import motif_analysis as ma
from celloracle.applications import (
    Pseudotime_calculator,
    Oracle_development_module,
    Gradient_calculator,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR    = "/scratch/st-wasser-1/momur_mono/celloracle_EC_beta_inte"
GENOME_DIR  = DATA_DIR                          # where hg38.fa is stored
REF_GENOME  = "hg38"
SAVE_DIR    = os.getcwd()

# Input files
H5AD_RAW        = os.path.join(DATA_DIR, "islets_integ_celloracle.h5ad")
PEAKS_CSV       = os.path.join(DATA_DIR, "islets_cicero_for_celloracle_all_peaks.csv")
CICERO_CSV      = os.path.join(DATA_DIR, "cicero_connections.csv")
GRNBOOST_TSV    = os.path.join(DATA_DIR, "GRNboost2_multiomics_betacelltypes.tsv")

# Oracle / links checkpoints
ORACLE_FILE         = "islets_subset.celloracle.oracle"
ORACLE_PSEUDO_FILE  = "islets_subset_same_oracle_w_pseudotime_monocle.celloracle.oracle"
LINKS_FILE          = "links.celloracle.links"
TFINFO_FILE         = "test1.celloracle.tfinfo"
GRADIENT_FILE       = "Gradient_monocle_pseudotime.celloracle.gradient"
BASE_GRN_PARQUET    = "base_GRN_dataframe.parquet"
PROCESSED_PEAKS_CSV = "processed_peak_file.csv"

# Cell type labels present in this dataset
CELL_TYPES = [
    "Beta cells-1",
    "Hi INS Beta cells-2",
    "EC cells-1",
    "EC cells-2",
    "EC cells-3",
    "EC cells-4",
    "EC cells-5",
]

# Genes of interest for perturbation simulation
GENES_OF_INTEREST = ["SOX9", "KLF9", "MAF", "MAFA", "MAFB", "ONECUT2"]

# Simulation parameters
SCALE_QUIVER     = 50
SCALE_SIMULATION = 30
N_GRID           = 40
MIN_MASS         = 10
VM               = 0.5
SCALE_DEV        = 40

os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. Pre-processing ─────────────────────────────────────────────────────────
print("=== Step 1: Pre-processing ===")
adata = sc.read_h5ad(H5AD_RAW)
print(f"Cells: {adata.shape[0]}  Genes: {adata.shape[1]}")

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.highly_variable_genes(adata, subset=True, inplace=True, n_top_genes=3000)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=0.3)

sc.pl.umap(adata, color="leiden", legend_loc="on data")
plt.savefig(os.path.join(SAVE_DIR, "umap_leiden_hvg.png"), dpi=300)
plt.close()

adata.write(os.path.join(DATA_DIR, "adata_processed.h5ad"))

# ── 2. Pseudotime ─────────────────────────────────────────────────────────────
print("=== Step 2: Pseudotime ===")
adata = sc.read_h5ad(os.path.join(DATA_DIR, "adata_processed.h5ad"))

pt = Pseudotime_calculator(
    adata=adata,
    obsm_key="X_umap",
    cluster_column_name="leiden",
)

# Lineage definition — edit to match your cell type structure
clusters_in_beta_lineage      = ["B-1", "HiB-2"]
clusters_in_immature_lineage  = ["E-4", "E-5", "E-1"]
clusters_in_mature_lineage    = ["E", "E-3"]

lineage_dictionary = {
    "beta_lineage":  clusters_in_beta_lineage,
    "immature":   clusters_in_immature_lineage,
    "mature":     clusters_in_mature_lineage,
}
pt.set_lineage(lineage_dictionary=lineage_dictionary)
pt.plot_lineages()
plt.savefig(os.path.join(SAVE_DIR, "set_lineage.png"), dpi=300)
plt.close()

# Root cells — update barcodes to match your dataset
root_cells = {
    "beta_lineage": "D50_TACAAGCTCCTTAATC-1",
    "immature":  "D50_CGTAGTTAGGACCAGG-1",
    "mature":    "D50_GAGCTAGCACAAGCG-1",
}
pt.set_root_cells(root_cells=root_cells)
pt.plot_root_cells()
plt.savefig(os.path.join(SAVE_DIR, "root_cells.png"), dpi=300)
plt.close()

sc.tl.diffmap(pt.adata)
pt.get_pseudotime_per_each_lineage()
pt.plot_pseudotime(cmap="rainbow")
plt.savefig(os.path.join(SAVE_DIR, "pseudotime_rainbow.png"), dpi=300)
plt.close()

adata.write(os.path.join(DATA_DIR, "adata_pseudotime_included.h5ad"))

# ── 3. Build GRN base from ATAC peaks ────────────────────────────────────────
print("=== Step 3: GRN base from ATAC peaks ===")
peaks = pd.read_csv(PEAKS_CSV, index_col=0)
peaks["x"] = "chr" + peaks["x"].astype(str).str.replace("-", "_")
peaks = peaks["x"].values

cicero_connections = pd.read_csv(CICERO_CSV, index_col=0)
for col in ["Peak1", "Peak2"]:
    cicero_connections[col] = (
        "chr" + cicero_connections[col].astype(str).str.replace("-", "_")
    )

tss_annotated = ma.get_tss_info(peak_str_list=peaks, ref_genome=REF_GENOME)
integrated = ma.integrate_tss_peak_with_cicero(
    tss_peak=tss_annotated, cicero_connections=cicero_connections
)

peak = integrated[integrated.coaccess >= 0.8][["peak_id", "gene_short_name"]].reset_index(drop=True)
peak.to_csv(PROCESSED_PEAKS_CSV)
print(f"Saved processed peaks: {PROCESSED_PEAKS_CSV}  ({len(peak)} rows)")

# ── 4. TF motif scan ──────────────────────────────────────────────────────────
print("=== Step 4: TF motif scan ===")
# Note: hg38.fa must already be downloaded to GENOME_DIR
# To download: genomepy.install_genome(name="hg38", provider="UCSC", genomes_dir=GENOME_DIR)

peaks_df = pd.read_csv(PROCESSED_PEAKS_CSV, index_col=0)
peaks_df = ma.check_peak_format(peaks_df, REF_GENOME, genomes_dir=GENOME_DIR)

tfi = ma.TFinfo(
    peak_data_frame=peaks_df,
    ref_genome=REF_GENOME,
    genomes_dir=GENOME_DIR,
)
tfi.scan(fpr=0.02, motifs=None, verbose=True)
tfi.to_hdf5(file_path=TFINFO_FILE)

# ── 5. Oracle object + KNN imputation ─────────────────────────────────────────
print("=== Step 5: Oracle object + KNN imputation ===")
adata = sc.read_h5ad(os.path.join(DATA_DIR, "adata_pseudotime_included.h5ad"))
print(f"Cells: {adata.shape[0]}  Genes: {adata.shape[1]}")

adata.layers["counts"] = adata.X.copy()
adata.X = adata.layers["counts"].copy()

oracle = co.Oracle()
oracle.import_anndata_as_raw_count(
    adata=adata,
    cluster_column_name="celltype",
    embedding_name="X_umap",
)

base_GRN = pd.read_parquet(BASE_GRN_PARQUET, engine="pyarrow")
oracle.import_TF_data(TF_info_matrix=base_GRN)

oracle.perform_PCA()
plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
plt.savefig(os.path.join(SAVE_DIR, "pca_variance.png"), dpi=300)
plt.close()

n_comps = np.where(
    np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_)) > 0.002)
)[0][0]
n_comps = min(n_comps, 50)
print(f"Selected PCs: {n_comps}")

n_cell = oracle.adata.shape[0]
k = int(0.025 * n_cell)
print(f"KNN k = {k}")

oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True,
                      b_sight=k * 8, b_maxl=k * 4, n_jobs=4)
oracle.to_hdf5(ORACLE_FILE)

# ── 6. GRN calculation ────────────────────────────────────────────────────────
# NOTE: This step is slow (~30 min). Run on SLURM if possible.
print("=== Step 6: GRN calculation ===")
oracle = co.load_hdf5(ORACLE_FILE)
links = oracle.get_links(
    cluster_name_for_GRN_unit="celltype", alpha=10, verbose_level=10
)
links.filter_links(p=0.001, weight="coef_abs", threshold_number=2000)
links.get_network_score()
links.to_hdf5(file_path=LINKS_FILE)

# ── 7. Export GRNs and network score plots ────────────────────────────────────
print("=== Step 7: Export GRNs and network score plots ===")
links = co.load_hdf5(file_path=LINKS_FILE)
links.filter_links(p=0.001, weight="coef_abs", threshold_number=2000)

for ct in CELL_TYPES:
    safe = ct.replace(" ", "_")
    links.filtered_links[ct].to_csv(f"filtered_GRN_for_{safe}.csv")
    links.links_dict[ct].to_csv(f"raw_GRN_for_{safe}.csv")
    links.plot_scores_as_rank(
        cluster=ct, n_gene=30,
        save=os.path.join(SAVE_DIR, f"ranked_score_{safe}"),
    )
    plt.close()

# ── 8. TF perturbation simulation ─────────────────────────────────────────────
print("=== Step 8: TF perturbation simulation ===")
oracle = co.load_hdf5(ORACLE_PSEUDO_FILE)
links  = co.load_hdf5(file_path=LINKS_FILE)

links.filter_links()
oracle.get_cluster_specific_TFdict_from_Links(links_object=links)
oracle.fit_GRN_for_simulation(alpha=10, use_cluster_specific_TFdict=True)

for goi in GENES_OF_INTEREST:
    print(f"  Simulating KO: {goi}")
    out = os.path.join(SAVE_DIR, f"simulation_{goi}_results")
    os.makedirs(out, exist_ok=True)

    # Gene expression histogram
    sc.get.obs_df(oracle.adata, keys=[goi], layer="imputed_count").hist(figsize=(10, 6))
    plt.savefig(os.path.join(out, f"gene_exp_{goi}.png"), dpi=300)
    plt.close()

    # UMAP of imputed expression
    sc.pl.umap(oracle.adata, color=[goi], layer="imputed_count")
    plt.savefig(os.path.join(out, f"imputed_count_UMAP_{goi}.png"), dpi=300)
    plt.close()

    # Simulate KO
    oracle.simulate_shift(perturb_condition={goi: 0.0}, n_propagation=3)
    oracle.estimate_transition_prob(n_neighbors=200, knn_random=True, sampled_fraction=1)
    oracle.calculate_embedding_shift(sigma_corr=0.05)

    # Quiver plot (raw vectors)
    fig, ax = plt.subplots(1, 2, figsize=[13, 6])
    oracle.plot_quiver(scale=SCALE_QUIVER, ax=ax[0])
    ax[0].set_title(f"Simulated shift: {goi} KO")
    oracle.plot_quiver_random(scale=SCALE_QUIVER, ax=ax[1])
    ax[1].set_title("Randomized simulation vector")
    plt.savefig(os.path.join(out, f"quiver_{goi}.png"), dpi=300)
    plt.close()

    # Mass filter
    oracle.calculate_p_mass(smooth=0.8, n_grid=N_GRID, n_neighbors=200)
    oracle.suggest_mass_thresholds(n_suggestion=12)
    plt.savefig(os.path.join(out, f"mass_threshold_{goi}.png"), dpi=300)
    plt.close()

    oracle.calculate_mass_filter(min_mass=MIN_MASS, plot=True)
    plt.savefig(os.path.join(out, f"mass_{MIN_MASS}_threshold_{goi}.png"), dpi=300)
    plt.close()

    # Grid flow
    fig, ax = plt.subplots(1, 2, figsize=[13, 6])
    oracle.plot_simulation_flow_on_grid(scale=SCALE_SIMULATION, ax=ax[0])
    ax[0].set_title(f"Simulated shift: {goi} KO")
    oracle.plot_simulation_flow_random_on_grid(scale=SCALE_SIMULATION, ax=ax[1])
    ax[1].set_title("Randomized simulation vector")
    plt.savefig(os.path.join(out, f"arrow_on_grid_{goi}.png"), dpi=300)
    plt.close()

    # Grid flow with cell cluster colouring
    fig, ax = plt.subplots(figsize=[8, 8])
    oracle.plot_cluster_whole(ax=ax, s=10)
    oracle.plot_simulation_flow_on_grid(scale=SCALE_SIMULATION, ax=ax, show_background=False)
    plt.savefig(os.path.join(out, f"arrow_on_grid_celltype_{goi}.png"), dpi=300)
    plt.close()

# ── 9. Pseudotime gradient + perturbation scoring ─────────────────────────────
print("=== Step 9: Pseudotime gradient + perturbation scoring ===")
oracle   = co.load_hdf5(ORACLE_PSEUDO_FILE)
gradient = co.load_hdf5(GRADIENT_FILE)

dev = Oracle_development_module()
dev.load_differentiation_reference_data(gradient_object=gradient)
dev.load_perturb_simulation_data(oracle_object=oracle)
dev.calculate_inner_product()
dev.calculate_digitized_ip(n_bins=10)

# Perturbation score grid
fig, ax = plt.subplots(1, 2, figsize=[12, 6])
dev.plot_inner_product_on_grid(vm=VM, s=50, ax=ax[0])
ax[0].set_title("Perturbation Score")
dev.plot_inner_product_random_on_grid(vm=VM, s=50, ax=ax[1])
ax[1].set_title("PS (randomized vector)")
plt.savefig(os.path.join(SAVE_DIR, "perturbation_score_grid.png"), dpi=300)
plt.close()

# PS with simulation vector field overlay
fig, ax = plt.subplots(figsize=[6, 6])
dev.plot_inner_product_on_grid(vm=VM, s=50, ax=ax)
dev.plot_simulation_flow_on_grid(scale=SCALE_SIMULATION, show_background=False, ax=ax)
plt.savefig(os.path.join(SAVE_DIR, "perturbation_score_with_flow.png"), dpi=300)
plt.close()

# Full development module layout
dev.visualize_development_module_layout_0(
    s=5,
    scale_for_simulation=SCALE_SIMULATION,
    s_grid=40,
    scale_for_pseudotime=SCALE_DEV,
    vm=VM,
)
plt.savefig(os.path.join(SAVE_DIR, "development_module_layout.png"), dpi=300)
plt.close()

print("Pipeline complete.")
