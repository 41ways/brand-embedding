"""브랜드 인지도 신호를 모은다 → data/fame/fame.csv

v1 (Wikidata 매칭 → 그 항목의 위키 문서, 결과는 fame_v1.csv) 에서 드러난 문제와 v2 의 대응
- 빠짐: 기업처럼 보이는 항목만 매칭해서 아이폰·인스타그램·자라 같은 제품 브랜드가 0
  → 한국어·영어 위키에서 **이름으로 문서를 직접 찾고** 넘겨주기를 따라간다. 동음이의 문서는 버린다
- 엉뚱한 문서: 루이비통 → LVMH, 페이스북 → 메타 플랫폼스
  → 넘겨주기 끝 제목에 브랜드 이름이 없으면 shared=1 (조회수를 모회사와 나눠 쓰는 문서)
- 화제성: 뉴스·주가로 한두 달 튄 문서(퓨리오사AI, 한화오션)가 위로
  → 12개월 합과 함께 **월 중앙값 × 12** 를 남긴다

열: id, ko_title, ko_shared, ko_sum, ko_med12, en_title, en_shared, en_sum, en_med12, sitelinks
"""
import csv
import hashlib
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent
CACHE = DATA / "verify" / "cache"
UA = "brand-embedding-fame/0.2 (https://github.com/41ways)"
START, END = "2025090100", "2026083100"


def get(url):
    path = CACHE / ("f2_" + hashlib.md5(url.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    err = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                data = json.load(r)
            path.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                path.write_text("{}")
                return {}
            err = e
            time.sleep(min(60, int(e.headers.get("Retry-After") or 0) or 5 * (attempt + 1)))
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(3 * (attempt + 1))
    raise err


def norm(s):
    return re.sub(r"[\s()·\-.,&'’!+]", "", str(s)).lower()


def resolve(lang, names):
    """이름 후보로 문서를 찾는다 → 제목 또는 None. 동음이의 문서는 건너뛴다"""
    suffix = ("기업", "브랜드") if lang == "ko" else ("company", "brand")
    cands = []
    for nm in names:
        cands += [nm, f"{nm} ({suffix[0]})", f"{nm} ({suffix[1]})"]
    cands = list(dict.fromkeys(cands))
    for k in range(0, len(cands), 50):
        q = urllib.parse.urlencode({"action": "query", "titles": "|".join(cands[k:k + 50]), "redirects": 1,
                                    "prop": "pageprops", "ppprop": "disambiguation", "format": "json"})
        d = get(f"https://{lang}.wikipedia.org/w/api.php?{q}")
        qq = d.get("query", {})
        redirect = {r["from"]: r["to"] for r in qq.get("redirects", [])}
        norm_map = {n["from"]: n["to"] for n in qq.get("normalized", [])}
        by_title = {p.get("title"): p for p in qq.get("pages", {}).values()}
        for c in cands[k:k + 50]:
            t = redirect.get(norm_map.get(c, c), norm_map.get(c, c))
            p = by_title.get(t)
            if not p or "missing" in p or "invalid" in p or "disambiguation" in p.get("pageprops", {}):
                continue
            return t
    return None


def views(lang, title):
    t = urllib.parse.quote(title.replace(" ", "_"), safe="")
    d = get(f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{t}/monthly/{START}/{END}")
    months = [it.get("views", 0) for it in d.get("items", [])]
    months += [0] * (12 - len(months))          # 조회가 없던 달은 0
    return sum(months), int(statistics.median(months) * 12)


def main():
    rows = list(csv.DictReader((DATA / "brands.csv").open(newline="")))
    old = {r["id"]: r for r in csv.DictReader((HERE / "fame_v1.csv").open(newline=""))}

    def one(r):
        al = [a for a in r["aliases"].split("|") if a]
        names = {"ko": [r["name_ko"]] + [a for a in al if re.search("[가-힣]", a)],
                 "en": [r["name_en"]] + [a for a in al if not re.search("[가-힣]", a)]}
        o = old.get(r["id"], {})
        rec = {"id": r["id"], "sitelinks": o.get("sitelinks", 0)}
        for lang in ("ko", "en"):
            title = resolve(lang, names[lang]) or o.get(f"{lang}_title") or None   # 이름으로 못 찾으면 v1 매칭 문서
            shared = int(bool(title) and not any(len(norm(n)) >= 2 and norm(n)[:3] in norm(title) for n in names["ko"] + names["en"]))
            s, m = views(lang, title) if title else (0, 0)
            rec.update({f"{lang}_title": title or "", f"{lang}_shared": shared, f"{lang}_sum": s, f"{lang}_med12": m})
        return rec

    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(one, rows))
    with (HERE / "fame.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "ko_title", "ko_shared", "ko_sum", "ko_med12", "en_title", "en_shared", "en_sum", "en_med12", "sitelinks"])
        w.writeheader()
        w.writerows(out)
    print("한국어 문서", sum(1 for o in out if o["ko_title"]), "(공유", sum(o["ko_shared"] for o in out), ") / 영어",
          sum(1 for o in out if o["en_title"]), "(공유", sum(o["en_shared"] for o in out), ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
