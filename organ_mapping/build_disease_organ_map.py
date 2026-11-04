"""
Step 2 of the disease -> organ mapping: combine the sources into one organ assignment per disease.

Inputs (all in this folder):
  disease_icd10.tsv           ICD-10 codes of every disease (from fetch_finngen_icd10.py)
  icd10_block_to_organs.tsv   Level 2: ICD-10 category range -> primary / secondary atlas organs
  chapter_to_organs.tsv       Level 1: ICD-10 chapter (the Atlas "Disease_category") -> organ system
  organ_vocabulary.tsv        the organ names the two atlases share (used to validate the tables)
  manual_overrides.tsv        diseases the sources cannot resolve, with the reason

For each disease:
  * every ICD-10 category it contains is mapped to organs through the narrowest matching block;
  * `organ_support` is the share of the disease's organ-bearing categories that point to each organ;
  * `organs_primary` are organs with support >= --min-support (default 0.25); the rest of the
    primary and all secondary organs go to `organs_secondary`;
  * Level 1 and Level 2 are compared, and disagreements, very broad diseases and manual
    assignments are listed in `review_reasons` for a person to check.

Output: disease_organ_map.tsv, plus a printed summary of where the sources agree and disagree.

Usage (from the repository root):
    python organ_mapping/build_disease_organ_map.py
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# ICD-10 chapter of a 3-character category, to check the Atlas "Disease_category" label against the codes
CHAPTER_RANGES = [("A00", "B99", "I"), ("C00", "D48", "II"), ("D50", "D89", "III"), ("E00", "E90", "IV"),
                  ("F00", "F99", "V"), ("G00", "G99", "VI"), ("H00", "H59", "VII"), ("H60", "H95", "VIII"),
                  ("I00", "I99", "IX"), ("J00", "J99", "X"), ("K00", "K93", "XI"), ("L00", "L99", "XII"),
                  ("M00", "M99", "XIII"), ("N00", "N99", "XIV"), ("O00", "O99", "XV")]


def icd_chapter(cat):
    k = (cat[0], int(cat[1:3]))
    for lo, hi, ch in CHAPTER_RANGES:
        if (lo[0], int(lo[1:3])) <= k <= (hi[0], int(hi[1:3])): return ch
    return None
NON_ORGAN_CHAPTER_LETTERS = set("RSTVWXYZ")
BROAD_N_ORGANS = 6          # a disease pointing to this many organs is "broad": matching it is nearly chance


def read_tsv(name):
    with open(os.path.join(HERE, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def split(s):
    return [x for x in (s or "").split(";") if x]


def cat_key(cat):
    """ICD-10 3-character category 'K70' -> sortable (letter, number)."""
    return cat[0], int(cat[1:3])


def load_blocks(valid_organs):
    blocks = []
    for r in read_tsv("icd10_block_to_organs.tsv"):
        for o in split(r["organs_primary"]) + split(r["organs_secondary"]):
            if o not in valid_organs:
                raise ValueError(f"icd10_block_to_organs.tsv: unknown organ '{o}' in block {r['first']}-{r['last']}")
        lo, hi = cat_key(r["first"]), cat_key(r["last"])
        # width across letters is irrelevant: ranges never span letters except S-T and V-Y, which carry no organs
        width = (ord(hi[0]) - ord(lo[0])) * 100 + hi[1] - lo[1]
        blocks.append({**r, "lo": lo, "hi": hi, "width": width,
                       "primary": split(r["organs_primary"]), "secondary": split(r["organs_secondary"])})
    return blocks


def block_for(cat, blocks):
    """The narrowest block containing the category (later rows win ties)."""
    k = cat_key(cat)
    hits = [b for b in blocks if b["lo"] <= k <= b["hi"]]
    return min(hits, key=lambda b: (b["width"], -blocks.index(b))) if hits else None


def main(args):
    vocab = read_tsv("organ_vocabulary.tsv")
    valid_organs = {r["organ"] for r in vocab}
    chapters = {r["chapter"]: r for r in read_tsv("chapter_to_organs.tsv")}
    for r in chapters.values():
        for o in split(r["organs"]):
            if o not in valid_organs: raise ValueError(f"chapter_to_organs.tsv: unknown organ '{o}'")
    blocks = load_blocks(valid_organs)
    manual = {r["Disease_code"]: r for r in read_tsv("manual_overrides.tsv")}
    for r in manual.values():
        for o in split(r["organs_primary"]) + split(r["organs_secondary"]):
            if o not in valid_organs: raise ValueError(f"manual_overrides.tsv: unknown organ '{o}'")

    out_rows = []
    for d in read_tsv("disease_icd10.tsv"):
        chap = re.match(r"Chapter (\w+)", d["Disease_category"]).group(1)
        c1 = chapters[chap]
        l1_organs = set(split(c1["organs"]))
        cats = sorted({c[:3] for c in d["icd10_all"].split(",") if re.match(r"[A-Z]\d{2}", c)}, key=cat_key)
        own_chapters = {icd_chapter(c[:3]) for c in d["icd10_own"].split(",") if re.match(r"[A-Z]\d{2}", c)} - {None}
        mismatch = f"{chap} vs {'/'.join(sorted(own_chapters))}" if own_chapters and chap not in own_chapters else ""

        row = {"Disease": d["Disease"], "Disease_code": d["Disease_code"], "Disease_category": d["Disease_category"],
               "chapter": chap, "level1_type": c1["level1_type"], "level1_organs": ";".join(sorted(l1_organs)),
               "n_icd10_categories": len(cats), "atlas_chapter_mismatch": mismatch,
               "icd10_categories": ",".join(cats) if len(cats) <= 12 else ",".join(cats[:12]) + f",...(+{len(cats) - 12})",
               "organs_primary": "", "organ_support": "", "organs_secondary": "", "lineage": "",
               "status": "", "confidence": "", "review_reasons": "", "source": "icd10_blocks", "notes": ""}
        reasons = []

        if d["Disease_code"] in manual:
            m = manual[d["Disease_code"]]
            primary, secondary, lineage = split(m["organs_primary"]), split(m["organs_secondary"]), m["lineage"]
            row.update(source="manual", notes=m["reason"])
            reasons.append("manual")
            support = {o: 1.0 for o in primary}
        else:
            prim_count, sec_count = Counter(), Counter()
            n_with_organ, n_immune, notes = 0, 0, []
            for c in cats:
                b = block_for(c, blocks)
                if not b: continue
                # Denominator: every category that is about a disease of the body, whether or not its
                # organ is in the atlas (peripheral nerve, joints, ear...). Symptom, injury and external
                # cause codes (R, S, T, V-Z) say nothing about organ, so they do not count.
                if c[0] not in NON_ORGAN_CHAPTER_LETTERS: n_with_organ += 1
                prim_count.update(b["primary"])
                sec_count.update(b["secondary"])
                if b["lineage"] == "immune": n_immune += 1
                if b["note"]: notes.append(b["note"])
            support = {o: n / n_with_organ for o, n in prim_count.items()} if n_with_organ else {}
            primary = [o for o, s in sorted(support.items(), key=lambda x: (-x[1], x[0])) if s >= args.min_support]
            # Secondary organs need the same support as primary ones, so a stray code from another
            # chapter (a shingles code inside a nerve-disorder endpoint) cannot create an organ.
            secondary = sorted(o for o, n in sec_count.items()
                               if n / n_with_organ >= args.min_support and o not in primary)
            lineage = "immune" if cats and n_immune / len(cats) >= 0.5 else ""
            row["notes"] = "; ".join(sorted(set(notes)))
            if not cats: reasons.append("no_icd10_codes")

        row.update(organs_primary=";".join(primary),
                   organ_support=";".join(f"{o}:{s:.2f}" for o, s in sorted(support.items(), key=lambda x: (-x[1], x[0]))),
                   organs_secondary=";".join(secondary), lineage=lineage)

        # --- status and agreement between the two levels ---------------------------------
        if not primary and not secondary:
            row["status"] = "not_scorable"
            reasons.append("no_organ_in_atlas_or_not_organ_based")
        else:
            row["status"] = "scorable" if primary else "secondary_only"
        if len(primary) >= BROAD_N_ORGANS:
            reasons.append(f"broad({len(primary)}_organs)")
        if c1["level1_type"] == "organ_system" and primary:
            outside = set(primary) - l1_organs
            if outside: reasons.append("level1_disagrees:" + ",".join(sorted(outside)))
            row["level1_agreement"] = "disagree" if outside else "agree"
        else:
            row["level1_agreement"] = "level1_uninformative"
        if row["status"] == "not_scorable": row["confidence"] = "n/a"
        elif reasons: row["confidence"] = "review"
        elif row["level1_agreement"] == "agree": row["confidence"] = "high"
        else: row["confidence"] = "medium"
        row["review_reasons"] = ";".join(reasons)
        out_rows.append(row)

    cols = ["Disease", "Disease_code", "Disease_category", "chapter", "level1_type", "level1_organs",
            "n_icd10_categories", "atlas_chapter_mismatch", "icd10_categories", "organs_primary", "organ_support", "organs_secondary",
            "lineage", "status", "confidence", "level1_agreement", "review_reasons", "source", "notes"]
    out = os.path.join(HERE, "disease_organ_map.tsv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader(); w.writerows(out_rows)

    # --- summary ---------------------------------------------------------------------------
    n = len(out_rows)
    print(f"{n} diseases -> {out}\n")
    print("status:     ", dict(Counter(r["status"] for r in out_rows)))
    print("confidence: ", dict(Counter(r["confidence"] for r in out_rows)))
    print("level-1 vs level-2:", dict(Counter(r["level1_agreement"] for r in out_rows)))
    by_ch = defaultdict(Counter)
    for r in out_rows: by_ch[r["chapter"]][r["confidence"]] += 1
    print("\nconfidence by chapter:")
    for ch in sorted(by_ch, key=lambda c: -sum(by_ch[c].values())):
        print(f"  {ch:>5} ({chapters[ch]['level1_type']:<12}) n={sum(by_ch[ch].values()):<3} {dict(by_ch[ch])}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-support", type=float, default=0.25,
                   help="minimum share of a disease's organ-bearing ICD-10 categories for an organ to be primary")
    main(p.parse_args())
