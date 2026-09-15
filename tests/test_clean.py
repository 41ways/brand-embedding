"""설명글 나라 이름 제거."""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "embed"))
from clean import PLACES, strip_places  # noqa: E402


class StripPlaces(unittest.TestCase):
    def test_examples(self):
        cases = {
            "생명·손해보험과 자산운용을 아우르는 독일의 세계 최대급 보험 그룹이다.": "생명·손해보험과 자산운용을 아우르는 세계 최대급 보험 그룹이다.",
            "종신·건강·연금보험을 파는 한국 대형 생명보험사다.": "종신·건강·연금보험을 파는 대형 생명보험사다.",
            "국내에서는 온라인 자동차보험으로 친숙하다.": "온라인 자동차보험으로 친숙하다.",
            "지중해·북유럽·카리브해 등을 도는 크루즈": "지중해·카리브해 등을 도는 크루즈",
            "미국산 소고기와 일본식 카레": "소고기와 카레",
        }
        for src, want in cases.items():
            self.assertEqual(strip_places(src), want)

    def test_does_not_eat_words_that_contain_place_names(self):
        for text in ["부산물을 활용한다", "아로마 향", "인도어 운동", "파리바게뜨 같은 빵집", "영국적 절제"]:
            with self.subTest(text=text):
                self.assertIn(text.split()[0][:2], strip_places(text))

    def test_no_place_left_in_real_summaries(self):
        pat = re.compile(r"(?<![가-힣A-Za-z])(?:" + "|".join(map(re.escape, PLACES)) + r")(?![가-힣A-Za-z])")
        left = []
        for p in sorted((ROOT / "data" / "enrich" / "out").glob("chunk_*.jsonl")):
            for line in p.read_text().splitlines():
                d = json.loads(line)
                if pat.search(strip_places(d["summary"])):
                    left.append(d["id"])
        self.assertEqual(left, [])


if __name__ == "__main__":
    unittest.main()
