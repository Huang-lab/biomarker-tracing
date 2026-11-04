# Do the cell-type results point to the disease's own organ?

**Short answer.** Sometimes, and only at the very top of the ranking. For about 1 in 8 diseases, the
top-ranked cell type is in the disease's own organ. That is twice what chance gives (p < 0.001, both atlases).
The signal is concentrated in **stomach, lung and heart**, where the own organ beats chance in every
disease, and it is absent for **kidney** (0 of 9 diseases), adipose and pancreas. Below the top 5 the
rankings no longer follow the organ. The cell types the pipeline calls *disease-specific* are 3–4× enriched
for the disease's own organ, but only 25–41 diseases have any.

Atlases: HPA v25.1 (`hpa700`, 700 cell types) and `pp229` (229 cell types). Diseases: UK Biobank
Proteome-Phenome Atlas, prevalent cases (268 diseases, of which 191 / 184 have a primary organ present in the
atlas and are evaluated).

---

## 1. Methods

```mermaid
flowchart LR
  A["Disease sumstats<br/>~2,900 plasma proteins<br/>(HR/OR, P)"] --> B["Per disease, per cell type:<br/>OLS of protein z on<br/>cell-type specificity<br/>→ t-statistic"]
  S["Atlas specificity<br/>(seismic score)<br/>genes × cell types"] --> B
  B --> C["specificity_z<br/>t relative to the same<br/>cell type in all 268 diseases"]
  D["Disease → organ<br/>(ICD-10 codes)"] --> E{"Is the disease's<br/>own organ near the<br/>top of its ranking?"}
  C --> E
```

### 1a. Which organ does each disease belong to? (`build_disease_organ_map.py`)
Each Atlas disease is a FinnGen endpoint. Its ICD-10 codes come from the endpoint definition on Risteys.
Each 3-character ICD-10 category maps to organs through a hand-written block table
(`icd10_block_to_organs.tsv`). An organ is *primary* when at least 25% of the disease's organ-relevant
codes point to it. Symptom, injury and external-cause codes (R, S, T, V–Z) are ignored. Diseases with a
primary organ present in the atlas are evaluated. A disease can have more than one primary organ (e.g.
gastric ulcer: stomach and small intestine).

The assignments were cross-checked against MONDO/UBERON where MONDO records a location (~8% of terms). The
ICD-10 chapter the Atlas files the disease under was also compared with the disease's own codes.
Details: [README.md](README.md).

### 1b. How each cell type is scored for a disease (`univar_association_testing.py`)
For one disease and one cell type *c*, across all measured proteins:

    z_protein  ~  specificity_c(protein's gene)  +  pleiotropy(protein)

* `z_protein`: signed z of the protein–disease association. |z| comes from the P-value; the sign is
  + when HR/OR > 1.
* `specificity_c`: the atlas's seismic specificity score of the gene for cell type *c*.
* `pleiotropy`: the protein's mean signed z across diseases in **other** ICD-10 chapters. It absorbs
  proteins that go up in almost every disease (GDF15 and others).
* The *t*-statistic of the specificity slope is the cell type's association with the disease.
  Significance comes from 2,000 Freedman–Lane permutations stratified by pleiotropy decile, with max-T
  family-wise control across cell types (`pval_maxT_one_side`).

### 1c. Removing each cell type's baseline (`cross_disease_summary.py`)
Some cell types (mostly macrophages and monocytes) score high for nearly every disease. Each t is therefore
re-expressed relative to the same cell type across all 268 diseases:

    specificity_z(d, c) = ( t(d, c) − median over diseases of t(·, c) ) / ( 1.4826 × MAD )

**All organ evaluations below rank cell types by `specificity_z`.** A cell type is *disease-specific* when
it is significant (max-T p ≤ 0.05) *and* its specificity_z has FDR ≤ 0.05.

### 1d. Which organ does each cell type belong to?
The organ of the tissue it was sampled from (`organ_vocabulary.tsv`). This includes the tissue's immune,
endothelial and stromal cells: a "kidney macrophage" counts as kidney.

### 1e. What counts as an organ match: the evaluations
No single cut-off defines a hit, so the results are reported several ways. All of them use the same null:
**organ assignments shuffled across diseases** (2,000×). The shuffle keeps each disease's ranking and each
organ's number of cell types, so organs with many cell types (brain: 84 in hpa700) get no advantage.

| Evaluation | Question | Uses significance? | Script |
|---|---|---|---|
| Mean-rank AUC | Do the own organ's cell types rank above the rest on average? | no | `plot_organ_heatmaps.py` |
| Top-10 share | What fraction of the top 10 cell types are own-organ? | no | `plot_organ_heatmaps.py` |
| **Hit rate at depth k** (new) | Is ≥1 own-organ cell type in the top k, for every k = 1…50? | no | `evaluate_organ_hits.py` |
| **Organ-defining cells only** (new) | Same, counting only parenchymal cells (no immune, endothelial, stromal, blood) | no | `evaluate_organ_hits.py` |
| **Best own-organ rank** (new) | Per disease: does the organ's best cell type beat that organ's typical best rank? | no | `evaluate_organ_hits.py` |
| **Significant hits** (new) | Of the cell types called significant / disease-specific, what share is own-organ? | **yes** | `evaluate_organ_hits.py` |

---

## 2. Results

### What a hit looks like
![Top 15 cell types for four diseases](figures/evaluation/fig1_examples.png)

Gastric ulcer is a clean hit: chief and parietal cells rank first and second. Asthma finds lung epithelium
(ranks 4, 5 and 7) among bone-marrow myeloid cells. Heart failure is led by smooth muscle and pericytes from
many tissues, which is a lineage signal, not an organ signal. Chronic kidney disease has no kidney cell type
in its top 15.

### Top 10 or top 50? The signal is at rank 1
![Hit rate by depth](figures/evaluation/fig2_hit_rate_by_depth.png)

| | hpa700 | pp229 |
|---|---|---|
| Own organ ranked #1 | **13% vs 6%** shuffled (p < 0.001) | **12% vs 6%** (p < 0.001) |
| ≥1 own-organ cell type in top 10 | 29% vs 27% (p = 0.27) | 31% vs 32% (p = 0.59) |
| ≥1 own-organ cell type in top 50 | 61% vs 64% (p = 0.83) | 70% vs 75% (p = 0.97) |
| Organ-defining cells only: #1 | 10% vs 4% (p < 0.001) | 9% vs 3% (p < 0.001) |
| Organ-defining cells only: top 10 | 23% vs 17% (p = 0.026) | 23% vs 18% (p = 0.056) |

Two consequences for how hits are defined:
* **A deeper cut-off adds noise, not hits.** By k = 10 the curve has merged with the shuffled band. A top-50
  definition would report "matches" for 61–70% of diseases, but chance alone gives 64–75%.
* **Counting only organ-defining cells sharpens the test.** Immune and stromal cells appear in every tissue
  and dilute the own-organ set.

The existing top-10 *share* (11% vs 6% for hpa700, p < 0.001) is significant while "≥1 in top 10" is
not. That is because a few diseases contribute many own-organ cell types each (gastric ulcer: 11 of 15),
not because many diseases have one.

### Which organs work
![Best own-organ rank per disease](figures/evaluation/fig3_best_rank_per_disease.png)

Overall, 48% (hpa700) and 47% (pp229) of diseases beat the chance rank, so most diseases look random. The
organ-level split is consistent across the two atlases:

* **Works:** stomach (4/4 in both), lung (11/11, 10/11), heart muscle (7/7 in both), urinary bladder (5/6, hpa700 only).
* **Mixed:** skin, vasculature, esophagus, eye, brain (8–9 of 25), liver (4/7 hpa700, 2/7 pp229).
* **Fails:** kidney (0/9 in both), adipose (0/2), pancreas, breast.

### Significant hits
| | hpa700 | pp229 |
|---|---|---|
| Significant (max-T p ≤ 0.05): hits / diseases | 3,734 / 179 | 1,776 / 171 |
| … share in own organ (chance) | 8.2% (5.3%) — 1.6× | 7.9% (5.3%) — 1.5× |
| Disease-specific (also specificity FDR ≤ 0.05): hits / diseases | 245 / 41 | 101 / 25 |
| … share in own organ (chance) | **19.2% (5.2%) — 3.7×** | **19.8% (6.3%) — 3.1×** |
| Diseases with ≥1 own-organ disease-specific hit | 19 | 14 |

Applying the pipeline's significance calls raises organ precision 3–4×. Coverage is low, though: 79% (hpa700)
and 86% (pp229) of evaluated diseases have no disease-specific hit at all.

---

## 3. Recommended way to report organ concordance
1. **Headline: hit rate at rank 1 and top 5, against the shuffled-organ null.** This is where the signal
   is. Show the full curve (Figure 2) rather than one cut-off, so the choice of k is visible.
2. **Count organ-defining cells as the own organ;** report "any cell type in the tissue" as secondary.
3. **For significance-based claims, use disease-specific hits**, not max-T significance alone. Report the
   precision (19–20%, 3–4× chance) together with the coverage (25–41 diseases).
4. **Report per organ (Figure 3), not only pooled.** The pooled AUC (0.50) averages organs that work against
   organs that don't, and hides both.
5. **Limit the headline set to diseases where an organ match is expected.** The 25 "brain" diseases are
   mostly psychiatric and substance use, and brain proteins rarely reach plasma.

## 4. Limitations
* Organ labels come from ICD-10 and are only partly validated (MONDO has locations for ~8% of terms).
* "Organ-defining" uses an approximate regular-expression lineage call on cell-type names.
* Diseases are not independent. Related endpoints (e.g. four duodenal and gastric ulcer definitions) share
  most cases, so per-organ counts overstate the evidence.
* A shared signal across diseases remains even after the pleiotropy covariate: the first principal component
  of the protein × disease z-matrix explains 40% of the variance. Removing more of it, and scoring tissue
  *within* lineage, are the next methodological steps.

## Reproduce
```bash
python organ_mapping/plot_organ_heatmaps.py --atlas both   # AUC, top-10 share, clustermaps
python organ_mapping/evaluate_organ_hits.py                 # this report's figures and tables
```
Tables behind the figures are in `figures/evaluation/`: `hit_rate_by_depth.tsv`, `<atlas>_best_rank_per_disease.tsv`
and `significant_hits_in_own_organ.tsv`.
