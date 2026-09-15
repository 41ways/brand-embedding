"""2차 보강(SPEC2) 검사. 사용: python3 check2.py [chunk_00 ...]"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
TICKER = re.compile(r"^[A-Z]{2,8}:[A-Z0-9.\-]{1,12}$")
VAGUE = {"제품", "상품", "서비스", "고급", "라이프스타일", "기타", "브랜드"}


def check(chunk):
    rows = list(csv.DictReader((HERE / "in" / f"{chunk}.csv").open(newline="")))
    want = {r["id"]: r for r in rows}
    path = HERE / "out2" / f"{chunk}.jsonl"
    if not path.exists():
        return [f"{chunk}: 출력 없음"]
    errs, seen = [], set()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            errs.append(f"{chunk}:{n} JSON 오류 {e}")
            continue
        i = d.get("id")
        if i not in want:
            errs.append(f"{chunk}:{n} 모르는 id {i}")
            continue
        if i in seen:
            errs.append(f"{i} 중복")
        seen.add(i)
        r = want[i]

        p = d.get("typical_price_krw")
        if p is not None and not (isinstance(p, int) and 100 <= p <= 5_000_000_000):
            errs.append(f"{i} typical_price_krw={p}")
        if (p is None) != (d.get("price_item") in (None, "")):
            errs.append(f"{i} price_item 과 가격이 짝이 안 맞음")

        prods = d.get("products")
        if not isinstance(prods, list) or not 3 <= len(prods) <= 8:
            errs.append(f"{i} products 개수")
        else:
            names = [x.strip().lower() for x in [r["name_ko"], r["name_en"], r.get("parent_name", "")] + r["aliases"].split("|") if len(x.strip()) >= 2]
            for x in prods:
                if not isinstance(x, str) or not x.strip() or x.strip() in VAGUE:
                    errs.append(f"{i} 모호한 상품 '{x}'")
                elif any(nm in x.lower() for nm in names):
                    errs.append(f"{i} 상품에 이름 '{x}'")

        if not isinstance(d.get("listed"), bool):
            errs.append(f"{i} listed")
        t = d.get("ticker")
        if d.get("listed") and not (isinstance(t, str) and TICKER.match(t)):
            errs.append(f"{i} ticker 형식 {t}")
        if not d.get("listed") and t:
            errs.append(f"{i} 비상장인데 ticker {t}")
        for k, hi in (("market_cap_usd_b", 6000), ("revenue_usd_b", 1000)):
            v = d.get(k)
            if v is not None and not (isinstance(v, (int, float)) and 0 < v <= hi):
                errs.append(f"{i} {k}={v}")
        if d.get("confidence") not in {"high", "mid", "low"}:
            errs.append(f"{i} confidence")
    for i in want:
        if i not in seen:
            errs.append(f"{chunk}: 빠진 id {i}")
    return errs


if __name__ == "__main__":
    chunks = sys.argv[1:] or sorted(p.stem for p in (HERE / "in").glob("chunk_*.csv"))
    total = []
    for c in chunks:
        e = check(c)
        total += e
        print(c, "OK" if not e else f"문제 {len(e)}건")
    for e in total[:200]:
        print(" ", e)
    sys.exit(1 if total else 0)
