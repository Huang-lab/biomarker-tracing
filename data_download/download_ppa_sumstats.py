"""
Download protein-disease summary statistics from the UK Biobank Proteome-Phenome Atlas
(https://proteome-phenome-atlas.com) for every disease on the site.

Adapted from the original 20-disease script. What changed:
  * the disease list is read from ppa_disease_list.tsv (all diseases on the site, with
    their ICD chapter) instead of a hand-picked list;
  * `--disease-type` selects "prevalent" (default; logistic regression, OR[95%CI]) or
    "incident" (Cox regression, HR[95%CI]);
  * files that already exist are skipped, so an interrupted run can simply be restarted;
  * the site answers HTTP 500 when it has no data for a disease/type (e.g. too few
    prevalent cases). That is recorded as "unavailable" in manifest.tsv, not retried
    forever, and not treated as a crash;
  * every response is checked to be a sumstats CSV before it is written, and written
    atomically, so a half-downloaded file never ends up in the output folder.

Output: <output-dir>/<Disease_name>.csv for each available disease, plus manifest.tsv
(one row per disease: status, number of proteins, file name). Files are named like the
original script (letters, digits, spaces->underscores, "-"); the pipeline uses the file
name as the disease name.

Usage:
    python data_download/download_ppa_sumstats.py                  # all prevalent diseases
    python data_download/download_ppa_sumstats.py --limit 5        # quick test
    python data_download/download_ppa_sumstats.py --disease-type incident --output-dir data/ppa_incident
"""
import argparse
import csv
import os
import time

import requests

BASE_URL = "https://proteome-phenome-atlas.com/api/getWideDownloadCsv"
HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED_HEADER_START = "Disease_category,Disease,Protein,"
MANIFEST_COLS = ["Disease", "Disease_code", "Disease_category", "status", "n_proteins", "file"]


def safe_filename(disease: str) -> str:
    """Same scheme as the original script: keep letters, digits, spaces, '_' and '-'."""
    name = "".join(c for c in disease if c.isalnum() or c in (" ", "_", "-")).strip()
    return name.replace(" ", "_")


def read_disease_list(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_manifest(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["Disease"]: r for r in csv.DictReader(f, delimiter="\t")}


def write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, delimiter="\t")
        w.writeheader()
        w.writerows(rows.values())


def fetch(disease, args):
    """Return (status, text). status: 'ok' | 'unavailable' | 'error:<why>'."""
    params = {
        "searchArg": disease,
        "diseaseType": args.disease_type,
        "population": args.population,
        "level": "1",
        "isDisease": "true",
        "isSensitive": "",
    }
    last = "error:unknown"
    for attempt in range(1, args.retries + 1):
        try:
            r = requests.get(BASE_URL, params=params, timeout=args.timeout)
        except requests.exceptions.RequestException as e:      # timeout, connection reset, ...
            last = f"error:{type(e).__name__}"
        else:
            if r.status_code == 200:
                text = r.content.decode("utf-8")
                if text.startswith(EXPECTED_HEADER_START):
                    return "ok", text
                last = "error:unexpected_content"
            elif r.status_code == 500:
                # The site's answer for "no such report". Confirm once, then give up.
                last = "unavailable"
                if attempt >= 2:
                    return "unavailable", ""
            else:
                last = f"error:http_{r.status_code}"
        time.sleep(args.delay * attempt)
    return last, ""


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.tsv")
    manifest = read_manifest(manifest_path)
    diseases = read_disease_list(args.disease_list)
    if args.limit: diseases = diseases[: args.limit]

    used_names = {}
    print(f"{len(diseases)} diseases, type={args.disease_type}; saving to {args.output_dir}\n")
    counts = {"ok": 0, "skipped": 0, "unavailable": 0, "error": 0}

    for i, d in enumerate(diseases, 1):
        name = d["Disease"]
        stem = safe_filename(name)
        if stem in used_names and used_names[stem] != name:       # two names collapsing to one file name
            stem = f"{stem}__{d['Disease_code']}"
        used_names[stem] = name
        filename = f"{stem}.csv"
        path = os.path.join(args.output_dir, filename)
        row = {"Disease": name, "Disease_code": d["Disease_code"], "Disease_category": d["Disease_category"],
               "status": "", "n_proteins": "", "file": ""}

        prev = manifest.get(name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            counts["skipped"] += 1
            manifest[name] = {**row, "status": "ok", "file": filename,
                              "n_proteins": (prev or {}).get("n_proteins", "")}
            continue
        if prev and prev["status"] == "unavailable" and not args.retry_unavailable:
            counts["skipped"] += 1
            continue

        print(f"[{i}/{len(diseases)}] {name}", end=" ... ", flush=True)
        status, text = fetch(name, args)
        if status == "ok":
            tmp = path + ".part"
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            os.replace(tmp, path)
            row.update(status="ok", n_proteins=text.count("\n") - 1, file=filename)
            print(f"saved ({row['n_proteins']} proteins)")
            counts["ok"] += 1
        elif status == "unavailable":
            row.update(status="unavailable")
            print(f"no {args.disease_type} data on the site")
            counts["unavailable"] += 1
        else:
            row.update(status=status)
            print(f"FAILED ({status})")
            counts["error"] += 1
        manifest[name] = row
        write_manifest(manifest_path, manifest)          # after every disease: safe to interrupt
        time.sleep(args.delay)                           # be polite to the server

    write_manifest(manifest_path, manifest)
    n_files = sum(1 for r in manifest.values() if r["status"] == "ok")
    print(f"\nDone. downloaded={counts['ok']} already-had={counts['skipped']} "
          f"unavailable={counts['unavailable']} failed={counts['error']} | {n_files} files in {args.output_dir}")
    if counts["error"]:
        print("Some downloads failed (see manifest.tsv, status starting with 'error'); rerun to retry them.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="data/ppa_prevalent_sumstats")
    p.add_argument("--disease-list", default=os.path.join(HERE, "ppa_disease_list.tsv"))
    p.add_argument("--disease-type", choices=["prevalent", "incident"], default="prevalent")
    p.add_argument("--population", default="all")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="only the first N diseases (testing)")
    p.add_argument("--retry-unavailable", action="store_true",
                   help="re-request diseases previously recorded as unavailable")
    main(p.parse_args())
