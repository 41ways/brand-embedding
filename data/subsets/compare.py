"""부분 목록 비교표 → data/subsets/compare.md

지표
  한국%        한국 브랜드 비율
  분야/최대%   담긴 분야 수 / 가장 많은 분야 비율 (쏠림)
  그룹중복     같은 최상위 그룹이 2개 이상 들어간 그룹 수 (아이폰·맥북·에어팟 문제)
  낯섦%        한국어 위키 월 중앙값이 500 미만 — 한국 사람이 거의 안 찾아보는 브랜드
  패널≤2%      친숙도 패널 2점 이하 (관심 있는 사람만 아는 브랜드) · 패널평균
  튐%          12개월 합이 중앙값×12 의 3배 넘음 — 뉴스·주가로 한때 튄 문서
  다른분야%    기본 가중치로 부분 목록 안에서 계산한 상위 10 이웃 중 다른 분야
  힌트/이웃    상위 10 이웃이 받는 "닮은 점" 개수 평균 — 좁혀 갈 단서가 얼마나 되나
  문항         정답 세트 중 세 브랜드가 모두 들어간 문항 수 · 그 정답률
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "embed"))
import preview as P  # noqa: E402
import make_subsets as M  # noqa: E402


def main():
    rows = M.load()
    by = {r["id"]: r for r in rows}
    idx = {r["id"]: i for i, r in enumerate(rows)}
    _, extra = P.load()
    blocks = P.build_blocks(rows, extra)
    cos, mask = P.pairwise(blocks)
    w = P.MODES["all"]
    full = P.similarity(cos, mask, w)
    trip = list(csv.DictReader((ROOT / "tests" / "triplets.csv").open(newline="")))
    hint_blocks = [b for b in w if b != "scale"]

    variants = [("base", [r["id"] for r in rows])] + [(p.stem, [r["id"] for r in csv.DictReader(p.open())])
                                                      for p in sorted(HERE.glob("*-*.csv"), key=lambda p: (["sum", "med", "blend", "quota", "quota1", "panel", "hybrid"].index(p.stem.split("-")[0]), int(p.stem.split("-")[1])))]
    lines = ["| 목록 | 개수 | 한국% | 분야 / 최대% | 그룹중복 | 낯섦% | 패널≤2% · 평균 | 튐% | 다른분야% | 힌트/이웃 | 문항 · 정답률 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for name, ids in variants:
        sel = np.array([idx[i] for i in ids])
        n = len(sel)
        rs = [by[i] for i in ids]
        sec = Counter(r["sector"] for r in rs)
        fam = Counter(r["root"] for r in rs)
        pan = [r["panel"] for r in rs if r["panel"] is not None]
        pan_low = sum(p <= 2 for p in pan) / len(pan) if pan else 0
        pan_avg = sum(pan) / len(pan) if pan else 0
        kr = sum(r["country"] == "KR" for r in rs) / n
        unfamiliar = sum(r["ko_med"] < 6000 for r in rs) / n
        spiky = sum(r["ko_sum"] > 3 * max(r["ko_med"], 1) and r["ko_sum"] > 20000 for r in rs) / n
        sim = full[np.ix_(sel, sel)]
        nb = np.argsort(-sim, axis=1)[:, :10]
        secs = np.array([by[i]["sector"] for i in ids])
        cross = float((secs[nb] != secs[:, None]).mean())
        # 힌트: 부분 목록 안에서 묶음별 상위 5%
        hints = np.zeros((n, 10))
        for b in hint_blocks:
            cb = np.where(mask[b][np.ix_(sel, sel)] > 0, cos[b][np.ix_(sel, sel)], -np.inf)
            np.fill_diagonal(cb, -np.inf)
            cnt = np.isfinite(cb).sum(1)
            edge = -np.sort(-cb, axis=1)[np.arange(n), np.maximum(1, (cnt * .05).astype(int)) - 1]
            got = np.take_along_axis(cb, nb, 1)
            hints += (got >= edge[:, None]) & (got > 0) & np.isfinite(got)
        ok = tot = 0
        s_ids = set(ids)
        for t in trip:
            if t["mode"] != "all" or not {t["anchor"], t["near"], t["far"]} <= s_ids:
                continue
            tot += 1
            ok += full[idx[t["anchor"]], idx[t["near"]]] > full[idx[t["anchor"]], idx[t["far"]]]
        row = dict(name=name, n=n, kr=kr, nsec=len(sec), maxsec=max(sec.values()) / n, famdup=sum(c >= 2 for c in fam.values()), pan_low=pan_low, pan_avg=pan_avg,
                   unfamiliar=unfamiliar, spiky=spiky, cross=cross, hints=float(hints.mean()), trip=tot, acc=ok / tot if tot else 0)
        out.append(row)
        lines.append(f"| {name} | {n} | {kr:.0%} | {len(sec)} / {row['maxsec']:.0%} | {row['famdup']} | {unfamiliar:.0%} | {pan_low:.0%} · {pan_avg:.2f} | {spiky:.0%} | {cross:.0%} | {row['hints']:.2f} | {tot} · {row['acc']:.0%} |")
    (HERE / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    main()
