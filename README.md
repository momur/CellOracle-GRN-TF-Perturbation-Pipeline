# CellOracle GRN + TF Perturbation Pipeline

Gene regulatory network (GRN) inference and transcription factor (TF) perturbation
simulation using [CellOracle](https://celloracle.readthedocs.io/) on 
single-cell multi-omics data.

## Workflow

```
islets_integ_celloracle.h5ad
        │
        ▼
  1. Pre-process (normalise, HVG, PCA, leiden)
        │
        ▼
  2. Pseudotime (CellOracle Pseudotime_calculator or Monocle)
        │
        ▼
  3. GRN base (ATAC peaks + cicero connections → TSS annotation)
        │
        ▼
  4. TF motif scan (gimmemotifs / JASPAR via CellOracle)
        │
        ▼
  5. Oracle object + KNN imputation
        │
        ▼
  6. GRN calculation — get_links() [slow: run on SLURM]
        │
        ├──▶  7. Export filtered/raw GRNs + network score plots
        │
        └──▶  8. TF perturbation simulation (loop over GOI list)
                        │
                        ▼
               9. Pseudotime gradient + perturbation scoring
```

## Setup

```bash
conda activate celloracle_env
pip install celloracle scanpy tqdm
```

The `hg38` genome must be downloaded once before step 4:

```python
import genomepy
genomepy.install_genome(name="hg38", provider="UCSC", genomes_dir="/your/genome/dir")
```

## Files

| File | Description |
|---|---|
| `celloracle_pipeline.py` | Main pipeline (steps 1–9) |
| `celloracle_grn_coexpression_analysis.py` | Cross-reference TF motif scan with GRNBoost2 scores |

## Usage

Edit the **Configuration** section at the top of each script to set your paths,
then run:

```bash
# Full pipeline
python celloracle_pipeline.py

# GRN co-expression analysis (after pipeline completes)
python celloracle_grn_coexpression_analysis.py
```

## Key configuration variables

| Variable | Description |
|---|---|
| `DATA_DIR` | Directory containing input h5ad, peaks, cicero CSVs |
| `GENOME_DIR` | Where hg38.fa is installed |
| `GENES_OF_INTEREST` | List of TFs to simulate KO for |
| `CELL_TYPES` | Cell type labels in `adata.obs.celltype` |
| `MIN_MASS` | Mass filter threshold for velocity grid (start with 10, tune with `suggest_mass_thresholds`) |

## Output files

| File | Description |
|---|---|
| `adata_processed.h5ad` | Pre-processed AnnData |
| `adata_pseudotime_included.h5ad` | AnnData with pseudotime |
| `processed_peak_file.csv` | Filtered ATAC peaks with TSS annotation |
| `test1.celloracle.tfinfo` | TF motif scan results |
| `base_GRN_dataframe.parquet` | Base GRN matrix |
| `islets_subset.celloracle.oracle` | Oracle object |
| `links.celloracle.links` | GRN links per cell type |
| `filtered_GRN_for_<celltype>.csv` | Filtered GRN per cell type |
| `raw_GRN_for_<celltype>.csv` | Raw GRN per cell type |
| `simulation_<GOI>_results/` | Per-gene perturbation figures |
| `perturbation_score_grid.png` | Perturbation score on UMAP grid |
| `GRN_coexpression_analysis_<GOI>.csv` | Motif + co-expression merged table |

## Notes

- Step 6 (`get_links`) can take 30+ min — submit as a SLURM job if running on an HPC.
- The perturbation loop (step 8) reuses the same fitted Oracle object for all GOIs,
  so the order of genes in `GENES_OF_INTEREST` does not matter.
- `scale_for_simulation` and `min_mass` are dataset-specific — check
  `suggest_mass_thresholds()` output before hardcoding `MIN_MASS`.

## Citation

> Kamimoto et al. (2023). Dissecting cell identity via network inference and in
> silico gene perturbation. *Nature*.
