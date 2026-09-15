"""parts/*.csv 를 합쳐 brands.csv 를 만들고 검사한다."""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
COLS = ["id", "name_ko", "name_en", "aliases", "country", "sector", "category",
        "brand_type", "parent", "tier", "b2b"]


def load_taxonomy():
    tax = {}
    for line in (HERE / "SCHEMA.md").read_text().splitlines():
        m = re.match(r"- \*\*(.+?)\*\*: (.+)", line)
        if m:
            tax[m.group(1)] = {c.strip() for c in m.group(2).split(",")}
    return tax


def main():
    tax = load_taxonomy()
    rows, errors = {}, []
    for part in sorted((HERE / "parts").glob("*.csv")):
        with part.open(newline="") as f:
            for n, r in enumerate(csv.DictReader(f), start=2):
                where = f"{part.name}:{n}"
                r = {k: (r.get(k) or "").strip() for k in COLS}
                if r["id"] in rows:
                    errors.append(f"{where} 중복 id {r['id']} (먼저 {rows[r['id']]['_src']})")
                    continue
                if r["sector"] not in tax or r["category"] not in tax[r["sector"]]:
                    errors.append(f"{where} 분류 {r['sector']}/{r['category']}")
                if r["brand_type"] not in {"corporate", "product", "both"}:
                    errors.append(f"{where} brand_type {r['brand_type']}")
                if r["tier"] not in {"mega", "major", "niche"}:
                    errors.append(f"{where} tier {r['tier']}")
                if r["b2b"] not in {"0", "1"}:
                    errors.append(f"{where} b2b {r['b2b']}")
                if not re.fullmatch(r"[A-Z]{2}", r["country"]):
                    errors.append(f"{where} country {r['country']}")
                r["_src"] = where
                rows[r["id"]] = r

    for r in rows.values():
        if r["parent"] and r["parent"] not in rows:
            errors.append(f"{r['_src']} 없는 parent {r['parent']}")

    names = Counter(r["name_en"].lower() for r in rows.values())
    for name, c in names.items():
        if c > 1:
            errors.append(f"영문 이름 겹침: {name} ×{c}")

    with (HERE / "brands.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows.values(), key=lambda r: (r["sector"], r["category"], r["id"])):
            w.writerow(r)

    print(f"{len(rows)}개")
    for key in ["sector", "tier", "b2b", "brand_type"]:
        print(key, dict(Counter(r[key] for r in rows.values()).most_common()))
    print("country 상위", dict(Counter(r["country"] for r in rows.values()).most_common(10)))
    print(f"문제 {len(errors)}건")
    for e in errors:
        print(" ", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
