"""
Every organ a disease could plausibly involve, in evidence tiers, without letting the ICD-10 chapter's organ list
(level1_organs) through unchecked.

level1_organs is the organ list of the disease's whole ICD-10 chapter (circulatory -> heart, vasculature, brain,
lung, kidney, lymph node). Taken as is it adds ~700 disease-organ pairs to 142 diseases, most of them wrong
(varicose veins -> brain, kidney stones -> testis). A chapter organ is therefore kept only when an independent,
disease-specific source points to it as well:

  tier 1  primary      >= 25% of the disease's organ-bearing ICD-10 categories          (disease_organ_map.tsv)
  tier 2  secondary    < 25%, or listed as secondary by the ICD-10 block table          (disease_organ_map.tsv)
  tier 3  corroborated a level1 (chapter) organ that is ALSO supported by MONDO's disease location
                       (mondo_crosscheck.tsv) or named in the disease's own name (e.g. "Hypertensive RENAL
                       disease" -> kidney). Both conditions are required: a name match outside the chapter is
                       ignored (e.g. "NEUROmuscular dysfunction of bladder" does not add skeletal muscle).
  rejected             level1 organs with no such support; listed so the reader can see what was dropped.

Output: disease_possible_organs.tsv (one row per disease, all statuses).
Usage (repository root):
    python organ_mapping/build_possible_organs.py
"""
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# organ words in disease names. Anatomical / organ-defining terms only; no symptoms.
NAME_TERMS = {
    "brain": r"brain|cerebr\w*|stroke|isch[a]?emic attack|dementia|epilep\w*|mental|psych\w*",
    "eye": r"eye|ocular|retina\w*|visual|glaucoma|cataract|macul\w*|choroid",
    "esophagus": r"o?esophag\w*",
    "stomach": r"stomach|gastr\w*",
    "small.intestine": r"duoden\w*|intestin\w*|coeliac|celiac|enteritis|malabsorption",
    "colon": r"colon|colitis|intestin\w*|bowel|diverticul\w*",
    "rectum": r"rect\w*|anal|anus",
    "liver": r"liver|hepat\w*|biliary|cirrhosis",
    "pancreas": r"pancrea\w*|diabet\w*",
    "salivary.gland": r"saliva\w*",
    "tongue": r"tongue|oral cavity",
    "lung": r"lung|pulmonary|respiratory|pleura\w*|pneumon\w*|bronch\w*|asthma|copd",
    "skin": r"skin|cutaneous|derm\w*",
    "kidney": r"renal|kidney|nephr\w*",
    "urinary.bladder": r"bladder|urethr\w*|cystitis|urinary",
    "prostate": r"prostat\w*",
    "testis": r"test[ie]\w*",
    "ovary": r"ovar\w*",
    "endometrium": r"uter\w*|endometri\w*|menstru\w*|menorrh\w*",
    "placenta": r"pregnan\w*|labour|delivery|fetus|fetal|amniotic",
    "breast": r"breast",
    "adipose.tissue": r"obes\w*|adipos\w*",
    "blood": r"an[a]?emi\w*",
    "bone.marrow": r"an[a]?emi\w*|agranulocyt\w*",
    "lymph.node": r"lymph\w*",
    "spleen": r"spleen|splen\w*",
    "heart.muscle": r"heart|cardi\w*|myocard\w*|coronary|angina|arrhythmi\w*|atrial",
    "vasculature": r"vascular|arter\w*|vein\w*|venous|varicose|thromb\w*|embol\w*|hypertensi\w*|aneurysm|phlebitis",
}
NAME_RX = {o: re.compile(rf"\b(?:{p})\b", re.I) for o, p in NAME_TERMS.items()}


def split(x):
    return [o for o in re.split(r"[;,]", x or "") if o.strip()]


def main():
    m = pd.read_csv(os.path.join(HERE, "disease_organ_map.tsv"), sep="\t", dtype=str).fillna("")
    mondo = pd.read_csv(os.path.join(HERE, "mondo_crosscheck.tsv"), sep="\t", dtype=str).fillna("").set_index("Disease_code")
    rows = []
    for _, r in m.iterrows():
        prim = split(r.organs_primary)
        sec = [o for o in split(r.organs_secondary) if o not in prim]
        mondo_organs = {x.split(":")[0] for x in split(mondo.mondo_organs.get(r.Disease_code, ""))}
        name_organs = {o for o, rx in NAME_RX.items() if rx.search(r.Disease)}
        tier3, rejected, why = [], [], []
        for o in sorted(set(split(r.level1_organs)) - set(prim) - set(sec)):
            ev = (["MONDO"] if o in mondo_organs else []) + (["name"] if o in name_organs else [])
            if ev: tier3.append(o); why.append(f"{o}:{'+'.join(ev)}")
            else: rejected.append(o)
        # MONDO organs outside the chapter are reported, not added (MONDO's own extras are reviewed by hand)
        rows.append(dict(Disease=r.Disease, Disease_code=r.Disease_code, status=r.status, confidence=r.confidence,
                         tier1_primary=";".join(prim), tier2_secondary=";".join(sec), tier3_corroborated=";".join(tier3),
                         tier3_evidence=";".join(why), all_possible=";".join(prim + sec + tier3),
                         level1_rejected=";".join(rejected), mondo_outside_chapter=";".join(sorted(mondo_organs - set(prim) - set(sec) - set(tier3)))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "disease_possible_organs.tsv"), sep="\t", index=False)
    s = out[out.status == "scorable"]
    n = lambda col: s[col].map(lambda x: len(split(x))).sum()
    print(f"scorable diseases: {len(s)} | pairs: primary {n('tier1_primary')}, secondary {n('tier2_secondary')}, "
          f"corroborated {n('tier3_corroborated')} (in {(s.tier3_corroborated != '').sum()} diseases), level1 rejected {n('level1_rejected')}")
    print(s[s.tier3_corroborated != ""][["Disease", "tier1_primary", "tier3_evidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
