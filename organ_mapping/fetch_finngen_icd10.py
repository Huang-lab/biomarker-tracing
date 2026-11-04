"""
Step 1 of the disease -> organ mapping: get the ICD-10 definition of every disease.

The Proteome-Phenome Atlas names its diseases with FinnGen endpoint codes
(Disease_code, e.g. K11_ALCOLIV). FinnGen documents what each endpoint contains on Risteys
(https://risteys.finngen.fi/endpoints/<code>): the ICD-10 codes per registry, the parent ICD-10
block, and, for composite endpoints, the other endpoints they include.

For each disease in a download manifest this script
  * fetches (and caches) the Risteys page,
  * parses the ICD-10 codes of each registry, the parent block, the ICD hierarchy level and the
    included endpoints,
  * resolves included endpoints recursively, so a composite endpoint inherits their codes,
  * writes one row per disease to disease_icd10.tsv.

Diseases Risteys does not know (the Atlas includes a few non-FinnGen names such as SMOKING or
FALLS) get status "not_found"; they are left for the later curation step, not guessed.

Usage (from the repository root):
    python organ_mapping/fetch_finngen_icd10.py
    python organ_mapping/fetch_finngen_icd10.py --limit 5
"""
import argparse
import csv
import html
import os
import re
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://risteys.finngen.fi/endpoints/{code}"
MAX_DEPTH = 4                      # how deep included endpoints are followed

OUT_COLS = ["Disease", "Disease_code", "Disease_category", "status", "code_used", "finngen_name",
            "icd10_own", "icd10_via_included", "icd10_all", "parent_icd10_block",
            "icd_hierarchy_level", "included_endpoints", "registries_used"]


def page_text(raw_html: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw_html, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\s+", " ", t)


def parse_page(raw_html: str) -> dict:
    t = page_text(raw_html)
    out = {"name": "", "icd10_by_registry": {}, "included": [], "parent": "", "level": ""}

    # Endpoint name: the first <h1>/<h2> that is not site navigation.
    for h in re.findall(r"<h[12][^>]*>(.*?)</h[12]>", raw_html, re.S):
        h = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h))).strip()
        if h and "Risteys" not in h and h not in ("Endpoint definition", "All original rules") and len(h) < 200:
            out["name"] = h
            break

    # Registry filters: "<Registry>: [excluded ]<System> — <codes> <Registry>: ..."
    m = re.search(r"Registry filters:(.*?)\d+ out of \d+ registries used", t)
    if m:
        chunk = m.group(1)
        starts = list(re.finditer(r"(?P<reg>[A-Z][A-Za-z()/\- ]*?): (?P<exc>excluded )?(?P<sys>ICD-\d+|[A-Za-z]+[A-Za-z ]*?) — ", chunk))
        for i, s in enumerate(starts):
            end = starts[i + 1].start() if i + 1 < len(starts) else len(chunk)
            if s["sys"] == "ICD-10" and not s["exc"]:
                codes = re.findall(r"[A-Z]\d{2}[A-Za-z0-9.]*", chunk[s.end():end])
                out["icd10_by_registry"].setdefault(s["reg"].strip(), []).extend(codes)

    m = re.search(r"Include endpoints (.*?)(?: key figures| 6\. Filter| \d+ key figures)", t)
    if m:
        out["included"] = re.findall(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b", m.group(1))
    m = re.search(r"Parent code in ICD-10 ([A-Z0-9.\-]+)", t)
    if m: out["parent"] = m.group(1)
    m = re.search(r"Level in the ICD hierarchy (\S+)", t)
    if m: out["level"] = m.group(1)
    m = re.search(r"(\d+) out of (\d+) registries used", t)
    out["registries_used"] = f"{m.group(1)}/{m.group(2)}" if m else ""
    return out


class Fetcher:
    def __init__(self, cache_dir, delay):
        self.cache_dir, self.delay, self.last = cache_dir, delay, 0.0
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, code):
        """Return (status, parsed). status: ok | not_found | error:<why>. Pages are cached."""
        path = os.path.join(self.cache_dir, f"{code}.html")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f: raw = f.read()
            return ("not_found", None) if raw == "NOT_FOUND" else ("ok", parse_page(raw))
        for attempt in range(1, 4):
            wait = self.delay - (time.time() - self.last)
            if wait > 0: time.sleep(wait)
            self.last = time.time()
            try:
                r = requests.get(URL.format(code=code), timeout=60)
            except requests.exceptions.RequestException as e:
                status = f"error:{type(e).__name__}"
            else:
                if r.status_code == 404:
                    with open(path, "w", encoding="utf-8") as f: f.write("NOT_FOUND")
                    return "not_found", None
                if r.status_code == 200 and "Registry filters" in r.text:
                    with open(path, "w", encoding="utf-8") as f: f.write(r.text)
                    return "ok", parse_page(r.text)
                status = f"error:http_{r.status_code}"
            time.sleep(2 * attempt)
        return status, None

    def resolve(self, code, depth=0, seen=None):
        """ICD-10 codes of an endpoint: its own plus those of included endpoints (recursive)."""
        seen = seen if seen is not None else set()
        seen.add(code)
        status, p = self.get(code)
        if status != "ok":
            return status, None, set(), set()
        own = {c for codes in p["icd10_by_registry"].values() for c in codes}
        via = set()
        if depth < MAX_DEPTH:
            for child in p["included"]:
                if child in seen: continue
                st, _, c_own, c_via = self.resolve(child, depth + 1, seen)
                if st == "ok": via |= c_own | c_via
        return "ok", p, own, via - own


def main(args):
    with open(args.manifest, newline="", encoding="utf-8") as f:
        diseases = [r for r in csv.DictReader(f, delimiter="\t") if r["status"] == "ok"]
    if args.limit: diseases = diseases[: args.limit]

    fetcher = Fetcher(args.cache_dir, args.delay)
    rows = []
    for i, d in enumerate(diseases, 1):
        code = d["Disease_code"]
        status, p, own, via = fetcher.resolve(code)
        code_used = code
        # "<endpoint>_EXALLC" is the "controls excluding all cancers" variant of <endpoint>; it has no
        # page of its own, but the base endpoint defines the same cases.
        if status == "not_found" and code.endswith("_EXALLC"):
            code_used = code[: -len("_EXALLC")]
            status, p, own, via = fetcher.resolve(code_used)
        row = {"Disease": d["Disease"], "Disease_code": code, "Disease_category": d["Disease_category"],
               "status": status, "code_used": code_used if status == "ok" else "",
               "finngen_name": "", "icd10_own": "", "icd10_via_included": "",
               "icd10_all": "", "parent_icd10_block": "", "icd_hierarchy_level": "",
               "included_endpoints": "", "registries_used": ""}
        if status == "ok":
            row.update(finngen_name=p["name"], icd10_own=",".join(sorted(own)), icd10_via_included=",".join(sorted(via)),
                       icd10_all=",".join(sorted(own | via)), parent_icd10_block=p["parent"],
                       icd_hierarchy_level=p["level"], included_endpoints=",".join(p["included"]),
                       registries_used=p["registries_used"])
        rows.append(row)
        print(f"[{i}/{len(diseases)}] {code}: {status}" + (f" | {len(own | via)} ICD-10 codes" if status == "ok" else ""), flush=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS, delimiter="\t")
        w.writeheader(); w.writerows(rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    n_codes = sum(r["status"] == "ok" and r["icd10_all"] != "" for r in rows)
    print(f"\n{n_ok}/{len(rows)} found on Risteys, {n_codes} with at least one ICD-10 code; wrote {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/ppa_prevalent_sumstats/manifest.tsv")
    p.add_argument("--output", default=os.path.join(HERE, "disease_icd10.tsv"))
    p.add_argument("--cache-dir", default="data/finngen_cache")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
