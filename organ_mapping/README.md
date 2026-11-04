# Disease → organ mapping

Assigns each Proteome-Phenome Atlas disease the atlas tissues ("organs") it is expected to affect, so that
cell-type hits from the pipeline can be checked for biological sense. Organ names are the ones the two atlases
share (`organ_vocabulary.tsv`).

## Sources, from coarse to fine

| Level | Source | File | Independent of expression data? |
|---|---|---|---|
| 1 | Atlas `Disease_category` (ICD-10 chapter) → organ system | `chapter_to_organs.tsv` | yes |
| 2 | FinnGen ICD-10 definition of the disease → organs by ICD-10 block | `disease_icd10.tsv`, `icd10_block_to_organs.tsv` | yes |
| manual | Diseases the sources cannot resolve, with the reason | `manual_overrides.tsv` | yes |

The Atlas names diseases with FinnGen endpoint codes (`K11_ALCOLIV`). `fetch_finngen_icd10.py` reads each
endpoint's definition from [Risteys](https://risteys.finngen.fi) (ICD-10 codes per registry, included
endpoints resolved recursively). The endpoint name on Risteys matched the Atlas disease name for every
disease found, which confirms the codes belong to the right disease.

## How an assignment is made (`build_disease_organ_map.py`)

* Every ICD-10 3-character category of the disease is mapped through the **narrowest matching block** to
  primary and secondary organs.
* `organ_support` = share of the disease's organ-relevant categories pointing to an organ. Symptom, injury
  and external-cause codes (R, S, T, V–Z) are ignored. An organ needs support ≥ 0.25 (primary *and* secondary)
  so that one stray cross-chapter code cannot define a disease.
* `status`: `scorable` (has primary organs), `secondary_only`, or `not_scorable` (the organ is not in either
  atlas — joints, bone, thyroid, ear, peripheral nerve, abdominal wall — or the disease is not organ-based).
* `confidence`: `high` (levels 1 and 2 agree), `medium` (level 1 uninformative: infections, neoplasms,
  endocrine), `review` (manual assignment, broad disease, or levels disagree), `n/a` (not scorable).
* `atlas_chapter_mismatch`: the Atlas `Disease_category` is not a chapter of the disease's own ICD-10 codes
  (e.g. depression filed under "Endocrine").
* `lineage = immune`: immune-mediated disease, where hits in immune cells of any tissue are also plausible.

## Files

| File | Content |
|---|---|
| `disease_organ_map.tsv` | **The result**, one row per disease |
| `disease_icd10.tsv` | ICD-10 codes per disease (output of the fetch script) |
| `icd10_block_to_organs.tsv` | Level 2 table; edit this to change what a code means |
| `chapter_to_organs.tsv` | Level 1 table |
| `organ_vocabulary.tsv` | Atlas tissue → shared organ name, and how many columns each atlas has per organ |
| `manual_overrides.tsv` | Manual assignments with reasons |

## Independent cross-check (`mondo_crosscheck.py`)

The tables above were written from ICD-10 knowledge, so they are checked against two ontologies that were
not: [MONDO](https://mondo.monarchinitiative.org) (diseases, with `disease_has_location`) and
[UBERON](https://obophenotype.github.io/uberon/) (anatomy). Each ICD-10 category of a disease is matched to MONDO
terms carrying that exact code; their locations (own and inherited from parent classes) are rolled up through
UBERON to the atlas organs (`organ_to_uberon.tsv`) and compared with `disease_organ_map.tsv`.

```bash
mkdir -p data/ontologies && cd data/ontologies
curl -sLO http://purl.obolibrary.org/obo/mondo.obo && curl -sLO http://purl.obolibrary.org/obo/uberon.obo   # versions used: MONDO 2026-11-03, UBERON 2026-10-01
cd ../.. && python organ_mapping/mondo_crosscheck.py
```

`mondo_crosscheck.tsv` gives per disease the MONDO organs and a `comparison`: `agree` (shares a primary organ),
`agree_secondary_only`, `disagree`, `mondo_has_organ_mine_none`, `no_mondo_location` (matched, but MONDO records
no location) or `no_mondo_match`. **MONDO is silent for most diseases** (it has an explicit location for ~8% of
terms), so this validates a subset; it says nothing about the diseases it has no location for. Matching through
sub-code terms is off by default: where MONDO has no term for the category itself, those are rare syndromes
(Barth, Smith-Lemli-Opitz) that misrepresent it.

## Do the results follow the organ? (`plot_organ_heatmaps.py`)

```bash
python organ_mapping/plot_organ_heatmaps.py --atlas both
```

Uses every `scorable` disease (any confidence) and the cross-disease `specificity_z` of each run (a cell type's
association with the disease relative to its association with a typical disease; 0 = typical). Per atlas, in `figures/`:

| File | Shows |
|---|---|
| `<atlas>_clustermap.png/.pdf` | diseases × cell types, both axes clustered. Left bar = the disease's primary organ system, top bar = the cell type's tissue system (same colours); disease labels list the exact primary organs |
| `<atlas>_organ_by_tissue.png/.pdf` | organ × tissue summary (mean z); same organ = outlined cell; right panel = per-organ AUC of the own-organ cell types |
| `<atlas>_stats.tsv` | mean-rank concordance (AUC), focal top-10 concordance, and silhouette scores (cell types by tissue and by approximate lineage, diseases by organ system), each against a label-shuffling null |
| `<atlas>_organ_topk.tsv`, `<atlas>_organ_auc_per_disease.tsv`, `<atlas>_organ_by_tissue_mean_z.tsv` | the numbers behind the figures |

The large clustered matrices (the table twin of the clustermap) go to `data/organ_heatmaps/` (gitignored).
Colour bars use eight organ *systems* (the palette allows at most eight categorical colours), so identity is also
carried by the legend, the text labels and these tables; a disease with several primary organs is coloured by the
system with most support (ties: the earlier legend entry) and counts toward each of its organs in the summary matrix.

Rebuild after editing a table: `python organ_mapping/build_disease_organ_map.py`.
Re-fetch codes (pages are cached in `data/finngen_cache/`): `python organ_mapping/fetch_finngen_icd10.py`.
