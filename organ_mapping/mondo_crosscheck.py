"""
Step 3 of the disease -> organ mapping: an independent cross-check against MONDO and UBERON.

The organ table in this folder was written from ICD-10 knowledge. MONDO (the community disease
ontology) records, for some diseases, where in the body they occur (`disease_has_location`), and
UBERON gives the anatomy hierarchy. This script derives organs from those two ontologies alone and
compares them with `disease_organ_map.tsv`.

For each disease and each of its ICD-10 categories:
  * the MONDO terms carrying that exact 3-character ICD-10 code as a cross-reference are found.
    Terms that carry only a sub-code (E78.71) are ignored by default: where MONDO has no term for the
    category itself, those are rare syndromes (Barth, Smith-Lemli-Opitz) that say nothing about the
    category (--allow-subcodes to include them);
  * the disease's name is also looked up among MONDO names and exact synonyms;
  * each term's locations are collected from the term and all its ancestors (a "liver disease"
    subtype inherits "liver");
  * each location is rolled up through UBERON (is_a / part_of) to the atlas organs in
    organ_to_uberon.tsv.
An organ is kept when at least --min-support of the disease's organ-relevant ICD-10 categories point to it,
the same rule as the main table.

Output: mondo_crosscheck.tsv and a printed summary. Run from the repository root; the ontology files
are downloaded once to data/ontologies/ (see README).

    python organ_mapping/mondo_crosscheck.py
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
NON_ORGAN_CHAPTER_LETTERS = set("RSTVWXYZ")


def parse_obo(path, keep_prefix):
    """Minimal OBO reader. Yields dicts with id, name, obsolete, exact synonyms, ICD10CM xrefs,
    is_a parents, part_of parents and disease_has_location targets."""
    cur = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("["):
                if cur and cur.get("id", "").startswith(keep_prefix): yield cur
                cur = {"is_a": [], "part_of": [], "loc": [], "icd10": [], "syn": [], "obsolete": False} if line.startswith("[Term]") else None
                continue
            if cur is None: continue
            if line.startswith("id: "): cur["id"] = line[4:].strip()
            elif line.startswith("name: "): cur["name"] = line[6:].strip()
            elif line.startswith("is_obsolete: true"): cur["obsolete"] = True
            elif line.startswith("is_a: "): cur["is_a"].append(line[6:].split()[0])
            elif line.startswith("relationship: part_of "): cur["part_of"].append(line.split()[2])
            elif line.startswith(("relationship: disease_has_location ", "intersection_of: disease_has_location ")):
                # a plain location statement, or one in a logical definition ("X is a disease located in Y")
                cur["loc"].append(line.split()[2])
            elif line.startswith("xref: ICD10CM:"): cur["icd10"].append(line[len("xref: ICD10CM:"):].split()[0])
            elif line.startswith("synonym: ") and " EXACT" in line:
                m = re.match(r'synonym: "(.*?)" EXACT', line)
                if m: cur["syn"].append(m.group(1))
    if cur and cur.get("id", "").startswith(keep_prefix): yield cur


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace(",", " ")).strip()


def closure(start, graph, cache):
    """All ancestors of `start` (including itself) following the given parent edges."""
    if start in cache: return cache[start]
    seen, stack = set(), [start]
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x)
        stack.extend(graph.get(x, ()))
    cache[start] = seen
    return seen


def main(args):
    organ_rows = list(csv.DictReader(open(os.path.join(HERE, "organ_to_uberon.tsv"), encoding="utf-8"), delimiter="\t"))
    anchors = {r["organ"]: set(r["uberon_anchors"].split(";")) for r in organ_rows}
    exact = {r["organ"]: set(filter(None, r["uberon_anchors_exact_only"].split(";"))) for r in organ_rows}

    # ---- UBERON: parents through is_a and part_of -----------------------------------------
    u_parents, u_name = {}, {}
    for t in parse_obo(os.path.join(args.ontology_dir, "uberon.obo"), "UBERON:"):
        u_parents[t["id"]] = [p for p in t["is_a"] + t["part_of"] if p.startswith("UBERON:")]
        u_name[t["id"]] = t.get("name", "")
    for organ, a in {**anchors, **exact}.items():
        for x in a:
            if x not in u_name: raise ValueError(f"organ_to_uberon.tsv: {organ}: {x} not in UBERON")
    u_cache, loc_cache = {}, {}

    def organs_of_location(loc):
        if not loc.startswith("UBERON:"): return set()      # cell-type locations (CL) are not organs
        if loc in loc_cache: return loc_cache[loc]
        anc = closure(loc, u_parents, u_cache)
        res = {o for o in anchors if anchors[o] & anc} | {o for o in exact if loc in exact[o]}
        loc_cache[loc] = res
        return res

    # ---- MONDO -----------------------------------------------------------------------------
    terms = {}
    for t in parse_obo(os.path.join(args.ontology_dir, "mondo.obo"), "MONDO:"):
        if not t["obsolete"]: terms[t["id"]] = t
    m_parents = {i: [p for p in t["is_a"] if p.startswith("MONDO:")] for i, t in terms.items()}
    m_cache = {}
    by_icd, by_name = defaultdict(set), defaultdict(set)
    for i, t in terms.items():
        for code in t["icd10"]: by_icd[code[:3]].add((i, code == code[:3]))
        for n in {t.get("name", "")} | set(t["syn"]): by_name[norm(n)].add(i)
    n_loc_terms = sum(1 for t in terms.values() if t["loc"])
    print(f"MONDO: {len(terms)} terms, {n_loc_terms} with an explicit location; ICD-10 categories covered: {len(by_icd)}")

    term_organs_cache = {}

    def organs_of_term(i):
        """Organs from the term's own and inherited disease_has_location (rolled up through UBERON)."""
        if i in term_organs_cache: return term_organs_cache[i]
        res = set()
        for a in closure(i, m_parents, m_cache):
            for loc in terms[a]["loc"] if a in terms else ():
                res |= organs_of_location(loc)
        term_organs_cache[i] = res
        return res

    # ---- compare ---------------------------------------------------------------------------
    mine = {r["Disease_code"]: r for r in csv.DictReader(open(os.path.join(HERE, "disease_organ_map.tsv"), encoding="utf-8"), delimiter="\t")}
    icd = {r["Disease_code"]: r for r in csv.DictReader(open(os.path.join(HERE, "disease_icd10.tsv"), encoding="utf-8"), delimiter="\t")}
    S = lambda x: set(filter(None, (x or "").split(";")))

    out, cls_count = [], Counter()
    for code, r in mine.items():
        cats = sorted({c[:3] for c in icd[code]["icd10_all"].split(",") if re.match(r"[A-Z]\d{2}", c)})
        cats = [c for c in cats if c[0] not in NON_ORGAN_CHAPTER_LETTERS]
        organ_cats, used_terms, basis = Counter(), [], set()
        for c in cats:
            cand = by_icd.get(c, set())
            ex = {i for i, is_exact in cand if is_exact}
            T = ex or ({i for i, _ in cand} if args.allow_subcodes else set())
            if T: basis.add("icd10")
            found = set()
            for i in T: found |= organs_of_term(i)
            for o in found: organ_cats[o] += 1
            used_terms += [i for i in T if organs_of_term(i)][:2]
        # the disease's own name, as a second, code-independent route to MONDO
        name_terms = by_name.get(norm(r["Disease"]), set()) | by_name.get(norm(r["Disease"].split(",")[0]), set())
        name_organs = set()
        for i in name_terms: name_organs |= organs_of_term(i)
        if name_terms: basis.add("name")

        n = max(len(cats), 1)
        support = {o: k / n for o, k in organ_cats.items()}
        m_organs = {o for o, s in support.items() if s >= args.min_support}
        if not m_organs and name_organs and len(cats) <= 1: m_organs = name_organs      # nothing via the code
        coverage = sum(1 for c in cats if any(organs_of_term(i) for i in ({i for i, e in by_icd.get(c, ()) if e} or ({i for i, _ in by_icd.get(c, ())} if args.allow_subcodes else set())))) / n

        P, Sec = S(r["organs_primary"]), S(r["organs_secondary"])
        if not m_organs:
            cls = "no_mondo_location" if basis else "no_mondo_match"
        elif not P and not Sec:
            cls = "mondo_has_organ_mine_none"
        elif m_organs & P:
            cls = "agree"
        elif m_organs & Sec:
            cls = "agree_secondary_only"
        else:
            cls = "disagree"
        cls_count[cls] += 1
        out.append({"Disease": r["Disease"], "Disease_code": code, "chapter": r["chapter"], "my_status": r["status"],
                    "my_primary": r["organs_primary"], "my_secondary": r["organs_secondary"],
                    "mondo_organs": ";".join(f"{o}:{s:.2f}" for o, s in sorted(support.items(), key=lambda x: -x[1]) if o in m_organs) or ";".join(sorted(m_organs)),
                    "mondo_extra": ";".join(sorted(m_organs - P - Sec)) if (P or Sec) else "",
                    "n_icd10_categories": len(cats), "mondo_coverage": f"{coverage:.2f}",
                    "mondo_basis": "+".join(sorted(basis)), "mondo_terms": ",".join(dict.fromkeys(used_terms))[:80],
                    "comparison": cls})

    cols = list(out[0].keys())
    path = os.path.join(HERE, "mondo_crosscheck.tsv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); w.writeheader(); w.writerows(out)
    print(f"\n{len(out)} diseases -> {path}")
    for k, v in cls_count.most_common(): print(f"  {k:<28} {v}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ontology-dir", default="data/ontologies")
    p.add_argument("--min-support", type=float, default=0.25)
    p.add_argument("--allow-subcodes", action="store_true", help="also use MONDO terms that carry only a sub-code of the category")
    main(p.parse_args())
