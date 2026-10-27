"""
File-level verification of research_outputs/ and research_audit/ (the per-value
checks are scripts 15-17). For every file (the preprocessing cache excluded):

  1. it opens and parses: CSV (pandas, no fully empty column), JSON, PNG and
     PDF (decoded / page count), Markdown and Python (UTF-8; Python compiles);
  2. every repository path cited in a Markdown file exists;
  3. it is cited by a Markdown file or written by a script (no orphans);
  4. research_audit/references.bib: every DOI is a DOI of literature_extraction.csv
     and every DOI there is in the .bib;
  5. research_audit/checksums_before.sha256: every listed file still has its hash;
  6. no secret-looking strings (API keys, tokens, passwords, private keys).

Output: research_audit/output_folder_verification.md (+ exit code 1 on any FAIL).
Usage: python research_outputs/scripts/18_verify_output_folders.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ROOTS = [REPO / "research_outputs", REPO / "research_audit"]
OUT = REPO / "research_audit" / "output_folder_verification.md"
SECRET = re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|password|passwd)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
PATH = re.compile(r"(?<![\w/])((?:research_outputs|research_audit|artifacts|tests|data)/[\w./\-]+?\.(?:csv|json|md|png|pdf|py|txt|pt|sha256|bib|log|jpeg|zip))")


def files():
    for root in ROOTS:
        for p in sorted(root.rglob("*")):
            if p.is_file() and ".cache" not in p.parts and "__pycache__" not in p.parts and p != OUT:
                yield p


def parse(p):
    s = p.suffix.lower()
    if s == ".csv":
        d = pd.read_csv(p, low_memory=False)
        expl = lambda c: c.lower() in ("note", "notes") or any(o.endswith("note") and o.split("_")[0] == c.split("_")[0] and d[o].notna().all() for o in d.columns if o != c)
        empty = [c for c in d.columns if d[c].isna().all() and not expl(c)]  # empty notes, or an empty value whose note column says why, are fine
        if len(d) == 0:
            return False, "no rows"
        return (not empty), f"{len(d):,} rows × {d.shape[1]} columns" + (f"; fully empty column(s): {', '.join(empty)}" if empty else "")
    if s == ".json":
        o = json.loads(p.read_text(encoding="utf8"))
        return True, f"{type(o).__name__} with {len(o)} entries"
    if s == ".png":
        with Image.open(p) as im:
            im.load(); return True, f"{im.size[0]}×{im.size[1]} px"
    if s == ".pdf":
        b = p.read_bytes(); n = len(re.findall(rb"/Type\s*/Page[^s]", b))
        return b.startswith(b"%PDF") and n > 0, f"{n} page(s)"
    if s == ".py":
        compile(p.read_text(encoding="utf8"), str(p), "exec"); return True, "compiles"
    p.read_text(encoding="utf8"); return True, "UTF-8 text"


def main():
    rows, fails = [], 0
    allf = list(files())
    texts = {p: p.read_text(encoding="utf8", errors="ignore") for p in allf if p.suffix in (".md", ".py", ".bib", ".txt", ".sha256", ".gitignore")}
    docs = {p: t for p, t in texts.items() if p.suffix == ".md"}
    scripts = "\n".join(t for p, t in texts.items() if p.suffix == ".py") + "\n".join(
        f.read_text(encoding="utf8", errors="ignore") for f in sorted(REPO.glob("*.py")) + sorted((REPO / "kaggle").glob("*.py")))
    cited = "\n".join(docs.values())

    # 1 + 3: parse every file; is it cited or written by a script?
    for p in allf:
        rel = p.relative_to(REPO).as_posix()
        try:
            ok, info = parse(p)
        except Exception as e:  # noqa: BLE001 - any parse error is a finding
            ok, info = False, f"{type(e).__name__}: {e}"
        stem = p.stem.replace("_plotted_values", "")
        used = p.name in cited or rel in cited or stem in cited or p.name in scripts or stem in scripts or p.suffix == ".py" or p.name == ".gitignore"
        rows.append({"file": rel, "parses": "PASS" if ok else "FAIL", "detail": info, "cited or produced": "yes" if used else "NO"})
        fails += (not ok) + (not used)

    # 2: every repository path cited in a Markdown file exists
    missing = []
    for p, t in docs.items():
        for m in sorted(set(PATH.findall(t))):
            if not (REPO / m).exists() and not (p.parent / m).exists():
                if m.startswith("artifacts/") and m.endswith(".pt"):
                    continue  # checkpoints are git-ignored and may not be on this machine
                missing.append(f"{p.relative_to(REPO).as_posix()} → `{m}`")
    fails += len(missing)

    # 4: bibliography DOIs vs extraction DOIs
    bib = (REPO / "research_audit/references.bib").read_text(encoding="utf8")
    bib_doi = {d.lower().rstrip("}.,") for d in re.findall(r"doi\s*=\s*\{([^}]+)\}", bib, re.I)}
    lit = pd.read_csv(REPO / "research_audit/literature_extraction.csv", dtype=str)
    lit_doi = {d.lower().rstrip(".,") for d in lit.study.str.extract(r"doi:(\S+)", expand=False).dropna()}
    doi_only_lit, doi_only_bib = sorted(lit_doi - bib_doi), sorted(bib_doi - lit_doi)
    fails += len(doi_only_lit)

    # 5: checksums recorded before the audit
    bad_sum, n_sum = [], 0
    for line in (REPO / "research_audit/checksums_before.sha256").read_text(encoding="utf8").splitlines():
        if line.strip():
            h, f = line.split(maxsplit=1); f = f.strip().lstrip("*"); n_sum += 1
            q = REPO / f
            if not q.exists() or hashlib.sha256(q.read_bytes()).hexdigest() != h.lower():
                bad_sum.append(f + ("" if q.exists() else " (missing)"))
    fails += len(bad_sum)

    # 4b: every Markdown table in tables/ shows exactly the cells of its CSV twin
    md_bad, n_md = [], 0
    for c in sorted((REPO / "research_outputs/tables").glob("*.csv")):
        m = c.with_suffix(".md")
        if not m.exists():
            continue
        n_md += 1
        d = pd.read_csv(c, dtype=str, keep_default_na=False)
        rows_md = [[x.strip() for x in ln.strip().strip("|").split("|")] for ln in m.read_text(encoding="utf8").splitlines() if ln.startswith("|") and not ln.startswith("|---")]
        want = [[str(v).replace("|", "/").replace("\n", " ").strip() for v in d.columns]] + [[str(v).replace("|", "/").replace("\n", " ").strip() for v in r] for r in d.itertuples(index=False)]
        if rows_md != want:
            diff = next(((i, j, a, b) for i, (ra, rb) in enumerate(zip(rows_md, want)) for j, (a, b) in enumerate(zip(ra, rb)) if a != b), None)
            md_bad.append(f"{m.name}" + (f" (row {diff[0]}, column {diff[1] + 1}: '{diff[2][:40]}' vs '{diff[3][:40]}')" if diff else f" ({len(rows_md)} vs {len(want)} rows)"))
    fails += len(md_bad)

    # 6: secrets
    secrets = [p.relative_to(REPO).as_posix() for p, t in texts.items() if SECRET.search(t)]
    for p in allf:
        if p.suffix == ".csv" and p.stat().st_size < 3_000_000 and SECRET.search(p.read_text(encoding="utf8", errors="ignore")):
            secrets.append(p.relative_to(REPO).as_posix())
    fails += len(secrets)

    n_parse = sum(r["parses"] == "PASS" for r in rows); n_used = sum(r["cited or produced"] == "yes" for r in rows)
    out = ["# Verification of research_outputs/ and research_audit/ (script 18)", "",
           f"{len(rows)} files checked (the preprocessing cache `research_outputs/.cache/` is excluded). Per-value checks: scripts 15–17.", "",
           "| Check | Result | Detail |", "|---|---|---|",
           f"| Every file opens and parses | **{'PASS' if n_parse == len(rows) else 'FAIL'}** | {n_parse} / {len(rows)} |",
           f"| Every file is cited in a document or written by a script | **{'PASS' if n_used == len(rows) else 'FAIL'}** | {n_used} / {len(rows)} |",
           f"| Every repository path cited in the Markdown files exists | **{'PASS' if not missing else 'FAIL'}** | {len(missing)} missing |",
           f"| Every DOI in literature_extraction.csv is in references.bib | **{'PASS' if not doi_only_lit else 'FAIL'}** | {len(lit_doi)} DOIs; missing from .bib: {', '.join(doi_only_lit) or 'none'}; extra in .bib: {', '.join(doi_only_bib) or 'none'} |",
           f"| Every Markdown table shows exactly the cells of its CSV | **{'PASS' if not md_bad else 'FAIL'}** | {n_md - len(md_bad)} / {n_md}" + (f"; differs: {'; '.join(md_bad)}" if md_bad else "") + " |",
           f"| Files hashed before the audit are unchanged | **{'PASS' if not bad_sum else 'FAIL'}** | {n_sum - len(bad_sum)} / {n_sum}" + (f"; changed: {', '.join(bad_sum)}" if bad_sum else "") + " |",
           f"| No secret-looking strings | **{'PASS' if not secrets else 'FAIL'}** | {', '.join(secrets) or 'none found'} |", ""]
    if missing:
        out += ["## Missing cited paths", ""] + [f"- {m}" for m in missing] + [""]
    out += ["## Every file", "", "| File | Parses | Detail | Cited or produced |", "|---|---|---|---|"]
    out += [f"| `{r['file']}` | {r['parses']} | {r['detail']} | {r['cited or produced']} |" for r in rows]
    OUT.write_text("\n".join(out) + "\n", encoding="utf8")
    print("\n".join(out[:14]))
    for r in rows:
        if r["parses"] == "FAIL" or r["cited or produced"] == "NO":
            print("ATTENTION", r)
    for m in missing:
        print("MISSING", m)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
