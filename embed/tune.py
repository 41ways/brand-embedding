"""'전부' 모드 가중치를 정답 세트(tests/triplets.csv)로 고른다.

격자 탐색. 1순위 sanity·position 틀린 수가 적은 것, 2순위 다른 분야 이웃 비율이 10~35% 인 것,
3순위 현재 가중치와 가까운 것. 정답 세트가 작아 과적합되기 쉬우니 참고용 — 세트를 늘리면 다시 돌린다.
"""
import csv
import itertools
from pathlib import Path

import numpy as np

import preview as P

ROOT = Path(__file__).resolve().parents[1]
GRID = {
    "summary": [.24, .30, .36],
    "archetype": [.10, .15, .20],
    "position": [.10, .15, .20],
    "category": [.12, .18, .24, .30],
    "heritage": [.04, .08],
    "origin": [.03, .06],
    "family": [.02, .05],
    "scale": [.02, .05],
}


def sim_rows(cos, mask, weights, rows):
    """정답 세트에 나오는 행만 계산 (전체 행렬보다 훨씬 가볍다)."""
    out = {}
    for i in rows:
        num = sum(wt * mask[b][i] * cos[b][i] for b, wt in weights.items())
        den = sum(wt * mask[b][i] for b, wt in weights.items())
        out[i] = np.where(den > 0, num / np.maximum(den, 1e-9), 0) * np.minimum(1, den / P.FULL_SHARE)
    return out


def main():
    rows, extra = P.load()
    idx = {r["id"]: i for i, r in enumerate(rows)}
    sec = np.array([r["sector"] for r in rows])
    cos, mask = P.pairwise(P.build_blocks(rows, extra))
    trip = list(csv.DictReader((ROOT / "tests" / "triplets.csv").open(newline="")))
    hard = [t for t in trip if t["kind"] in ("sanity", "position") and t["mode"] == "all"]
    anchors = sorted({idx[t["anchor"]] for t in hard})
    current = P.MODES["all"]
    keys = list(GRID)

    results = []
    for combo in itertools.product(*GRID.values()):
        tot = sum(combo)
        w = {k: v / tot for k, v in zip(keys, combo)}
        s = sim_rows(cos, mask, w, anchors)
        fails = [t for t in hard if s[idx[t["anchor"]]][idx[t["near"]]] <= s[idx[t["anchor"]]][idx[t["far"]]]]
        results.append([len(fails), sum(abs(w[k] - current[k]) for k in keys), w, fails])
    results.sort(key=lambda r: r[:2])

    best = [r for r in results if r[0] == results[0][0]][:40]
    for r in best:
        s = P.similarity(cos, mask, r[2])
        nb = np.argpartition(-s, 10, axis=1)[:, :10]
        r.append(float((sec[nb] != sec[:, None]).mean()))
    best.sort(key=lambda r: (not 0.10 < r[4] < 0.35, r[1]))

    print(f"후보 {len(results)}개, 최소 틀림 {results[0][0]}/{len(hard)}")
    for nfail, _, w, fails, share in best[:5]:
        print(f"틀림 {nfail}  다른분야 {share:.3f}  " + " ".join(f"{k}={v:.3f}" for k, v in w.items()))
        for t in fails:
            print("   ", t["anchor"], "→", t["near"], "vs", t["far"])


if __name__ == "__main__":
    main()
