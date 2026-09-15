"""brands.csv 와 보강 데이터가 규칙을 지키는지."""
import csv
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def taxonomy():
    tax = {}
    for line in (DATA / "SCHEMA.md").read_text().splitlines():
        m = re.match(r"- \*\*(.+?)\*\*: (.+)", line)
        if m:
            tax[m.group(1)] = {c.strip() for c in m.group(2).split(",")}
    return tax


def read_brands():
    with (DATA / "brands.csv").open(newline="") as f:
        return list(csv.DictReader(f))


class BrandList(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (DATA / "brands.csv").open(newline="") as f:
            cls.rows = list(csv.DictReader(f))
        cls.by_id = {r["id"]: r for r in cls.rows}

    def test_ids_unique_and_slug(self):
        ids = [r["id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        bad = [i for i in ids if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", i)]
        self.assertEqual(bad, [])

    def test_taxonomy(self):
        tax = taxonomy()
        bad = [r["id"] for r in self.rows if r["category"] not in tax.get(r["sector"], ())]
        self.assertEqual(bad, [])

    def test_enums(self):
        for r in self.rows:
            self.assertIn(r["brand_type"], {"corporate", "product", "both"}, r["id"])
            self.assertIn(r["tier"], {"mega", "major", "niche"}, r["id"])
            self.assertIn(r["b2b"], {"0", "1"}, r["id"])
            self.assertRegex(r["country"], r"^[A-Z]{2}$", r["id"])

    def test_parents_exist_and_no_cycles(self):
        for r in self.rows:
            seen, cur = set(), r["id"]
            while self.by_id[cur]["parent"]:
                self.assertNotIn(cur, seen, f"소속 순환: {r['id']}")
                seen.add(cur)
                cur = self.by_id[cur]["parent"]
                self.assertIn(cur, self.by_id, f"{r['id']} 의 parent 없음")

    def test_brands_csv_matches_parts(self):
        """brands.csv 를 손으로 고치지 않았는지: merge.py 결과와 같아야 한다."""
        before = (DATA / "brands.csv").read_bytes()
        out = subprocess.run([sys.executable, str(DATA / "merge.py")], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertEqual(before, (DATA / "brands.csv").read_bytes())


class Enrichment(unittest.TestCase):
    def test_check_script_passes(self):
        out = subprocess.run([sys.executable, str(DATA / "enrich" / "check.py")], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stdout[-2000:])

    def test_check2_script_passes(self):
        out = subprocess.run([sys.executable, str(DATA / "enrich" / "check2.py")], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stdout[-2000:])

    def test_every_brand_enriched_once(self):
        ids = [r["id"] for r in read_brands()]
        for folder in ("out", "out2"):
            got = []
            for p in sorted((DATA / "enrich" / folder).glob("chunk_*.jsonl")):
                got += [json.loads(line)["id"] for line in p.read_text().splitlines() if line.strip()]
            self.assertEqual(len(got), len(set(got)), folder)
            self.assertEqual(set(got), set(ids), folder)

    def test_consumer_product_brands_have_price_position(self):
        """가격을 비교할 수 있는 소비자 제품 브랜드는 거의 다 price_pos 가 있어야 한다.
        무료 플랫폼·방송·제약·금융·기업 브랜드는 비어도 정상이라 뺀다."""
        no_price = {"인터넷·플랫폼·게임", "통신·미디어·엔터", "제약·헬스케어", "금융", "기업집단"}
        rows = {r["id"]: r for r in read_brands()}
        total = filled = 0
        for p in (DATA / "enrich" / "out").glob("chunk_*.jsonl"):
            for line in p.read_text().splitlines():
                d = json.loads(line)
                r = rows[d["id"]]
                if r["b2b"] == "0" and r["sector"] not in no_price and r["brand_type"] != "corporate":
                    total += 1
                    filled += d["price_pos"] is not None
        self.assertGreater(filled / total, 0.97)


if __name__ == "__main__":
    unittest.main()
