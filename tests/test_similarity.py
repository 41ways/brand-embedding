"""유사도 계산 단위 테스트. 모델 없이 작은 가짜 묶음으로 돈다."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "embed"))
import preview as P  # noqa: E402


def block(vectors, has=None):
    v = P.l2(np.array(vectors, dtype=np.float32))
    return v, np.array(has if has is not None else [True] * len(v))


class Helpers(unittest.TestCase):
    def test_l2_keeps_zero_rows(self):
        m = P.l2(np.array([[3, 4], [0, 0]], dtype=np.float32))
        np.testing.assert_allclose(m[0], [.6, .8], rtol=1e-6)
        np.testing.assert_array_equal(m[1], [0, 0])

    def test_onehot_with_fixed_keys_ignores_unknown(self):
        m = P.onehot(["a", "z", "b"], keys=["a", "b"])
        np.testing.assert_array_equal(m, [[1, 0], [0, 0], [0, 1]])

    def test_soft_bins_peak_at_nearest_center(self):
        b = P.soft_bins(0.8, [0, .25, .5, .75, 1], .18)
        self.assertEqual(int(np.argmax(b)), 3)
        self.assertAlmostEqual(float(P.soft_bins(0.5, [.5], .18)[0]), 1.0)

    def test_root_of_follows_chain_and_survives_cycle(self):
        rows = [{"id": "a", "parent": "b"}, {"id": "b", "parent": "c"}, {"id": "c", "parent": ""},
                {"id": "x", "parent": "y"}, {"id": "y", "parent": "x"}]
        roots = P.root_of(rows)
        self.assertEqual(roots[:3], ["c", "c", "c"])
        self.assertIn(roots[3], {"x", "y"})


class Similarity(unittest.TestCase):
    def test_identical_brands_score_one(self):
        blocks = {"a": block([[1, 0], [1, 0], [0, 1]]), "b": block([[0, 1], [0, 1], [1, 0]])}
        cos, mask = P.pairwise(blocks)
        s = P.similarity(cos, mask, {"a": .5, "b": .5})
        self.assertAlmostEqual(float(s[0, 1]), 1.0, places=5)
        self.assertAlmostEqual(float(s[0, 2]), 0.0, places=5)

    def test_diagonal_excluded(self):
        blocks = {"a": block([[1, 0], [0, 1]])}
        cos, mask = P.pairwise(blocks)
        s = P.similarity(cos, mask, {"a": 1})
        self.assertTrue((np.diag(s) == -9).all())

    def test_missing_block_is_renormalized_not_zeroed(self):
        """한쪽에 없는 묶음은 0점으로 치지 않고 계산에서 뺀다."""
        a = block([[1, 0], [1, 0], [1, 0]])
        b = block([[1, 0], [0, 0], [0, 1]], has=[True, False, True])
        cos, mask = P.pairwise({"a": a, "b": b})
        s = P.similarity(cos, mask, {"a": .6, "b": .4})
        # 0-1: b 없음 → a 만으로 1.0 (공통 가중치 .6 ≥ .5 라 감점 없음)
        self.assertAlmostEqual(float(s[0, 1]), 1.0, places=5)
        # 0-2: 둘 다 있음 → .6*1 + .4*0
        self.assertAlmostEqual(float(s[0, 2]), 0.6, places=5)

    def test_low_overlap_is_penalized(self):
        a = block([[1, 0], [1, 0]])
        b = block([[1, 0], [1, 0]], has=[True, False])
        cos, mask = P.pairwise({"a": a, "b": b})
        s = P.similarity(cos, mask, {"a": .2, "b": .8})
        # 공통은 a(.2) 뿐 → 코사인 1 × (.2/.5)
        self.assertAlmostEqual(float(s[0, 1]), 0.4, places=5)

    def test_weights_sum_to_one(self):
        for mode, w in P.MODES.items():
            self.assertAlmostEqual(sum(w.values()), 1.0, places=6, msg=mode)


class CompanySize(unittest.TestCase):
    def test_own_then_parent_then_revenue(self):
        rows = [{"id": "group", "parent": ""}, {"id": "brand", "parent": "group"},
                {"id": "private", "parent": ""}, {"id": "unknown", "parent": ""}]
        ex = [{"market_cap_usd_b": 100}, {}, {"revenue_usd_b": 2}, {}]
        out = P.company_size(rows, ex)
        self.assertEqual(out[0], (2.0, "own"))
        self.assertEqual(out[1], (2.0, "parent"))
        self.assertAlmostEqual(out[2][0], np.log10(3))
        self.assertEqual(out[2][1], "revenue")
        self.assertEqual(out[3], (None, None))

    def test_inherited_size_is_softer(self):
        rows = [{"id": "g", "parent": ""}, {"id": "b", "parent": "g"}]
        m, has = P.size_block(rows, [{"market_cap_usd_b": 50}, {}])
        self.assertTrue(has.all())
        self.assertGreater(m[0, :len(P.SIZE_CENTERS)].max(), m[1, :len(P.SIZE_CENTERS)].max())


if __name__ == "__main__":
    unittest.main()
