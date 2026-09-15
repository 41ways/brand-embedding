"""보강 데이터 검사. 사용: python3 check.py [chunk_00 ...]  (없으면 전부)"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
ARCH = {"innocent", "explorer", "sage", "hero", "outlaw", "magician", "everyman", "lover", "jester", "caregiver", "creator", "ruler"}
OWN = {"family", "founder", "public", "state", "foundation", "coop", "pe", "chaebol", "subsidiary", "unknown"}


def names(r):
    out = [r["name_ko"], r["name_en"], r.get("parent_name", "")] + r["aliases"].split("|")
    return [n.strip() for n in out if len(n.strip()) >= 2]


def check(chunk):
    rows = list(csv.DictReader((HERE / "in" / f"{chunk}.csv").open(newline="")))
    want = {r["id"]: r for r in rows}
    path = HERE / "out" / f"{chunk}.jsonl"
    errs, seen = [], set()
    if not path.exists():
        return [f"{chunk}: 출력 없음"]
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
            errs.append(f"{chunk}:{n} 중복 {i}")
        seen.add(i)
        s = d.get("summary") or ""
        if not 80 <= len(s) <= 300:
            errs.append(f"{i} summary 길이 {len(s)}")
        for nm in names(want[i]):
            if re.search(re.escape(nm), s, re.I):
                errs.append(f"{i} summary 에 이름 '{nm}'")
        for k in ("price_pos", "exclusivity"):
            v = d.get(k)
            if v is not None and not (isinstance(v, (int, float)) and 0 <= v <= 1):
                errs.append(f"{i} {k}={v}")
        a = d.get("archetype") or {}
        if not a or set(a) - ARCH or not 1 <= len(a) <= 3 or abs(sum(a.values()) - 1) > 0.02:
            errs.append(f"{i} archetype {a}")
        f = d.get("founded")
        if f is not None and not (isinstance(f, int) and 1300 <= f <= 2026):
            errs.append(f"{i} founded {f}")
        if not isinstance(d.get("founder_named"), bool):
            errs.append(f"{i} founder_named")
        if d.get("ownership") not in OWN:
            errs.append(f"{i} ownership {d.get('ownership')}")
        if d.get("confidence") not in {"high", "mid", "low"}:
            errs.append(f"{i} confidence")
        if not d.get("why"):
            errs.append(f"{i} why 없음")
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
