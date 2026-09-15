"""정답 세트: anchor 는 far 보다 near 에 가까워야 한다.

실제 임베딩(캐시된 bge-m3 결과)으로 돈다. 캐시가 없으면 건너뛴다 — 먼저
`python3 embed/preview.py` 를 한 번 돌릴 것.

- sanity·position: 전부 맞아야 함 (틀리면 무언가 망가진 것)
- cross: 업종을 넘는 가까움. 목표 비율 이상이면 통과, 틀린 줄은 출력
"""
import csv
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "embed"))
import preview as P  # noqa: E402

CROSS_TARGET = 0.8


def cached():
    return all((P.OUT / f).exists() for f in ("name_emb.npy", "summary_emb.npy"))


@unittest.skipUnless(cached(), "임베딩 캐시 없음: embed/preview.py 먼저 실행")
class Triplets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows, extra = P.load()
        cls.idx = {r["id"]: i for i, r in enumerate(rows)}
        cos, mask = P.pairwise(P.build_blocks(rows, extra))
        cls.sim = {m: P.similarity(cos, mask, w) for m, w in P.MODES.items()}
        with (ROOT / "tests" / "triplets.csv").open(newline="") as f:
            cls.triplets = list(csv.DictReader(f))

    def judge(self, kind):
        ok, bad = 0, []
        for t in self.triplets:
            if t["kind"] != kind:
                continue
            for k in ("anchor", "near", "far"):
                self.assertIn(t[k], self.idx, f"triplets.csv 에 없는 id {t[k]}")
            s = self.sim[t["mode"]]
            a, n, f = (self.idx[t[k]] for k in ("anchor", "near", "far"))
            if s[a, n] > s[a, f]:
                ok += 1
            else:
                bad.append(f"{t['anchor']}: {t['near']} {s[a, n]:.3f} ≤ {t['far']} {s[a, f]:.3f} ({t['note']})")
        return ok, bad

    def test_sanity(self):
        ok, bad = self.judge("sanity")
        self.assertEqual(bad, [], "\n".join(bad))

    def test_position(self):
        ok, bad = self.judge("position")
        self.assertEqual(bad, [], "\n".join(bad))

    def test_cross_industry(self):
        ok, bad = self.judge("cross")
        rate = ok / (ok + len(bad))
        print(f"\n[cross] {ok}/{ok + len(bad)} = {rate:.0%}" + "".join("\n  " + b for b in bad))
        self.assertGreaterEqual(rate, CROSS_TARGET, "\n".join(bad))

    def test_debatable_reported(self):
        """사람끼리도 갈리는 문항. 강제하지 않고 결과만 보여준다."""
        ok, bad = self.judge("debatable")
        if bad:
            print("\n[debatable] 모델 판단이 적어 둔 기대와 다름:\n  " + "\n  ".join(bad))

    def test_all_mode_mixes_industries(self):
        """전부 모드 상위 10 이웃 중 다른 분야가 10~35% — 업종 맞히기도, 뒤죽박죽도 아니게."""
        rows, _ = P.load()
        sec = np.array([r["sector"] for r in rows])
        nb = np.argsort(-self.sim["all"], axis=1)[:, :10]
        share = float((sec[nb] != sec[:, None]).mean())
        self.assertGreater(share, 0.10)
        self.assertLess(share, 0.35)


if __name__ == "__main__":
    unittest.main()
