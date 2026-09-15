"""유명 브랜드만 추린 부분 목록을 규칙별로 만든다 → data/subsets/<규칙>-<크기>.csv

기본 1,603개(brands.csv)는 건드리지 않는다. 부분 목록은 id 와 참고 열만 담는다.
공통: 소비자 브랜드만 (b2b=0, 기업집단 제외)

규칙 (뒤로 갈수록 v1 의 문제를 하나씩 고친 것)
  sum     한국어 위키 12개월 조회수 합 순              — 가장 단순한 기준
  med     월 중앙값×12 순, 모회사와 나눠 쓰는 문서는 ×0.3 — 화제성 튐·공유 문서 보정
  quota   med + 분야 쿼터(최소 max(2, N×1.5%), 최대 N×12%) + 같은 최상위 그룹 2개까지
  blend   quota 제약 + 점수만 한국 0.6 · 영어 0.25 · 언어 수 0.15 (z-점수, log)

v2 (플레이·목록 검토에서 나온 문제 반영) — 후보 풀을 좁히고 신호를 바꾼다
  풀 v2   소비자 브랜드 중 물류·농업·산업재·에너지 분야, 방송·신문, 기업 이름(corporate)을 뺀다
          (쿼터를 채우느라 야마토운수·미국우정청이 들어오고, MBC·조선일보가 쏠리고, BMW 와 BMW그룹이 같이 들어왔다)
  quota1  quota 와 같되 풀 v2 + 같은 최상위 그룹 1개 (애플·아이폰 중 하나만)
  panel   풀 v2 + 친숙도 패널(0~5) 점수 순, 같으면 조회수 — 쿼터·그룹 1개
  hybrid  풀 v2 + 패널 0.6 · 조회수 0.4 (z-점수) — 쿼터·그룹 1개

크기 150 · 200 · 300
"""
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent
sys.path.insert(0, str(DATA.parent / "embed"))
import preview as P  # noqa: E402

SIZES = (150, 200, 300)
RULES = ("sum", "med", "quota", "blend", "quota1", "panel", "hybrid")
DROP_SECTORS = {"물류", "농업·식품원료", "산업재·중공업", "에너지·화학·소재"}
DROP_CATEGORIES = {"방송·신문"}
SHARED_PENALTY = 0.3


def load():
    rows = list(csv.DictReader((DATA / "brands.csv").open(newline="")))
    fame = {r["id"]: r for r in csv.DictReader((DATA / "fame" / "fame_checked.csv").open(newline=""))}
    roots = dict(zip([r["id"] for r in rows], P.root_of(rows)))
    for r in rows:
        f = fame[r["id"]]
        pen = lambda lang: SHARED_PENALTY if f[f"{lang}_shared"] == "1" else 1
        r["ko_sum"] = int(f["ko_sum"])
        r["ko_med"] = int(f["ko_med12"]) * pen("ko")
        r["en_med"] = int(f["en_med12"]) * pen("en")
        r["sitelinks"] = int(f["sitelinks"] or 0)
        r["root"] = roots[r["id"]]
    panel = {}
    pdir = DATA / "fame" / "panel"
    for p in sorted(pdir.glob("out_[0-9]*.csv")):
        for x in csv.DictReader(p.open(newline="")):
            panel[x["id"]] = int(x["score"])
    for r in rows:
        r["panel"] = panel.get(r["id"])
    return rows


def pool_v2(rows):
    return [r for r in pool_of(rows) if r["sector"] not in DROP_SECTORS and r["category"] not in DROP_CATEGORIES
            and r["brand_type"] != "corporate"]


def zlog(vals):
    xs = [math.log1p(v) for v in vals]
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** .5 or 1
    return [(x - m) / sd for x in xs]


def pool_of(rows):
    return [r for r in rows if r["b2b"] == "0" and r["sector"] != "기업집단"]


def pick_top(pool, n, key):
    return sorted(pool, key=key)[:n]


def pick_quota(pool, n, key, fam_cap=2):
    """분야 최소를 먼저 채우고, 나머지는 점수 순. 분야 최대와 그룹 제한을 지킨다"""
    lo, hi = max(2, round(n * .015)), round(n * .12)
    order = sorted(pool, key=key)
    chosen, ids, per_sec, per_root = [], set(), Counter(), Counter()

    def ok(r):
        return per_sec[r["sector"]] < hi and per_root[r["root"]] < fam_cap and r["id"] not in ids

    def take(r):
        chosen.append(r); ids.add(r["id"]); per_sec[r["sector"]] += 1; per_root[r["root"]] += 1

    by_sec = defaultdict(list)
    for r in order:
        by_sec[r["sector"]].append(r)
    for sec, lst in by_sec.items():
        for r in lst:
            if per_sec[sec] >= lo:
                break
            if ok(r):
                take(r)
    for r in order:
        if len(chosen) >= n:
            break
        if ok(r):
            take(r)
    return chosen


def build(rows):
    pool = pool_of(rows)
    for r, a, b, c in zip(pool, zlog([r["ko_med"] for r in pool]), zlog([r["en_med"] for r in pool]), zlog([r["sitelinks"] for r in pool])):
        r["blend"] = .6 * a + .25 * b + .15 * c
    made = {}
    for n in SIZES:
        made[f"sum-{n}"] = pick_top(pool, n, key=lambda r: -r["ko_sum"])
        made[f"med-{n}"] = pick_top(pool, n, key=lambda r: -r["ko_med"])
        made[f"quota-{n}"] = pick_quota(pool, n, key=lambda r: -r["ko_med"])
        made[f"blend-{n}"] = pick_quota(pool, n, key=lambda r: -r["blend"])
    p2 = pool_v2(rows)
    for n in SIZES:
        made[f"quota1-{n}"] = pick_quota(p2, n, key=lambda r: -r["ko_med"], fam_cap=1)
    if all(r["panel"] is not None for r in p2):
        for r, a, b in zip(p2, [(r["panel"] - 2.5) / 1.3 for r in p2], zlog([r["ko_med"] for r in p2])):
            r["hybrid"] = .6 * a + .4 * b
        for n in SIZES:
            made[f"panel-{n}"] = pick_quota(p2, n, key=lambda r: (-r["panel"], -r["ko_med"]), fam_cap=1)
            made[f"hybrid-{n}"] = pick_quota(p2, n, key=lambda r: -r["hybrid"], fam_cap=1)
    else:
        print("패널 점수가 아직 다 없어서 panel·hybrid 는 건너뜀")
    return made


def main():
    rows = load()
    for old in HERE.glob("*-*.csv"):
        old.unlink()
    for name, lst in build(rows).items():
        with (HERE / f"{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "name_ko", "sector", "ko_sum", "ko_med"])
            for r in lst:
                w.writerow([r["id"], r["name_ko"], r["sector"], r["ko_sum"], int(r["ko_med"])])
        print(name, len(lst))


if __name__ == "__main__":
    main()
