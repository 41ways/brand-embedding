"""부분 목록에 맞춰 '전부' 가중치를 다시 고른다 → embed/out/tuned_<목록>.json

    python3 embed/tune_subset.py panel-150 hybrid-200

목록을 줄이면 같은 업종 짝이 줄어 다른 분야 이웃이 11% → 48~54% 로 뛰었다.
플레이 후기: "업종이 딴판인데 순위만 가까운 이웃은 방해".
기준
  1. 목록 안에 세 브랜드가 다 들어간 정답 문항 틀림이 가장 적을 것
  2. 상위 10 이웃 중 다른 분야 15~30%
  3. 같으면 기본 가중치에 가까울 것
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import preview as P

ROOT = Path(__file__).resolve().parents[1]
GRID = {
    "summary": [.20, .30], "offering": [.10, .18, .26], "archetype": [.05, .10], "price": [.05, .10],
    "position": [.03, .06], "category": [.10, .20, .30], "heritage": [.02, .06], "size": [.02, .06],
    "origin": [.02, .05], "family": [.01], "scale": [.01, .03],
}
LO, HI = .15, .30


def main():
    rows, extra = P.load()
    idx = {r["id"]: i for i, r in enumerate(rows)}
    cos, mask = P.pairwise(P.build_blocks(rows, extra))
    trip = [t for t in csv.DictReader((ROOT / "tests" / "triplets.csv").open(newline="")) if t["mode"] == "all"]
    base = P.MODES["all"]
    for v in sys.argv[1:]:
        ids = [r["id"] for r in csv.DictReader((ROOT / "data" / "subsets" / f"{v}.csv").open())]
        sel = np.array([idx[i] for i in ids])
        loc = {i: k for k, i in enumerate(ids)}
        c = {b: cos[b][np.ix_(sel, sel)] for b in base}
        m = {b: mask[b][np.ix_(sel, sel)] for b in base}
        sec = np.array([rows[i]["sector"] for i in sel])
        ts = [t for t in trip if {t["anchor"], t["near"], t["far"]} <= set(ids)]
        keys = list(GRID)
        best = []
        for combo in itertools.product(*GRID.values()):
            w = {k: x / sum(combo) for k, x in zip(keys, combo)}
            s = P.similarity(c, m, w)
            fails = sum(s[loc[t["anchor"]], loc[t["near"]]] <= s[loc[t["anchor"]], loc[t["far"]]] for t in ts)
            nb = np.argpartition(-s, 10, axis=1)[:, :10]
            share = float((sec[nb] != sec[:, None]).mean())
            off = 0 if LO <= share <= HI else min(abs(share - LO), abs(share - HI))
            dist = sum(abs(w[k] - base[k]) for k in keys)
            best.append((fails, round(off, 3), dist, share, w))
        best.sort(key=lambda r: r[:3])
        fails, off, dist, share, w = best[0]
        s0 = P.similarity(c, m, base)
        nb0 = np.argpartition(-s0, 10, axis=1)[:, :10]
        share0 = float((sec[nb0] != sec[:, None]).mean())
        f0 = sum(s0[loc[t["anchor"]], loc[t["near"]]] <= s0[loc[t["anchor"]], loc[t["far"]]] for t in ts)
        print(f"{v}: 문항 {len(ts)}  기본 가중치 틀림 {f0}·다른분야 {share0:.0%} → 재조정 틀림 {fails}·다른분야 {share:.0%}")
        print("   ", " ".join(f"{k}={x:.3f}" for k, x in w.items()))
        (P.OUT / f"tuned_{v}.json").write_text(json.dumps({"weights": w, "fails": int(fails), "share": share,
                                                           "base_fails": int(f0), "base_share": share0, "n_trip": len(ts)}))


if __name__ == "__main__":
    main()
