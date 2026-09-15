"""fame.csv 의 위키 문서가 정말 그 브랜드 문서인지 검증한다 → fame_checked.csv, fame_rejects.csv

v2(이름으로 문서 찾기)에서 생긴 오류: 짧거나 일반명사인 이름이 딴 문서로 간다.
  삿포로·칭따오 → 도시, 이솝 → 아이소포스(우화 작가), 츄이 → 츄바카, 카파 → 그리스 문자
또 표기만 다른 제목(바클레이즈 ↔ 바클리즈)을 "모회사와 나눠 쓰는 문서"로 잘못 판정했다.

검증: 문서의 Wikidata 항목(한·영 설명, P31 분류)을 받아
  - 도시·사람·문자·가상 인물 같은 분류면 버린다
  - 설명이 기업·브랜드·제품·서비스를 뜻하면 받는다
  - 둘 다 아니면 "모름" — 버리지는 않되 표시
공유 판정도 다시: 문서 항목이 브랜드 자신의 Wikidata 매칭과 같으면 공유 아님.
"""
import csv
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent
CACHE = DATA / "verify" / "cache"
UA = "brand-embedding-fame/0.2 (https://github.com/41ways)"

BAD_P31 = {
    "Q5": "사람", "Q515": "도시", "Q1637706": "대도시", "Q1549591": "대도시", "Q486972": "정착지", "Q3957": "소도시",
    "Q15284": "지방자치체", "Q6256": "나라", "Q9788": "문자", "Q9779": "문자", "Q15632617": "가상 인물",
    "Q95074": "가상 인물", "Q4167410": "동음이의", "Q23397": "호수", "Q8502": "산", "Q4022": "강", "Q13442814": "학술 논문",
    "Q11424": "영화", "Q7725634": "문학 작품", "Q482994": "음반", "Q134556": "싱글", "Q5398426": "TV 시리즈",
    "Q16521": "생물 분류군", "Q1549591": "대도시", "Q70208": "자치단체", "Q747074": "이탈리아 코무네",
}
GOOD_DESC = re.compile(
    r"기업|회사|브랜드|상표|제조|업체|그룹|음료|맥주|주류|소주|라면|과자|식품|항공|은행|보험|증권|카드|방송|채널|서비스|플랫폼|"
    r"게임|앱|애플리케이션|소프트웨어|자동차|의류|패션|화장품|체인|호텔|리조트|유통|백화점|마트|편의점|쇼핑|메신저|검색|웹사이트|"
    r"스마트폰|휴대전화|시계|보석|가구|완구|장난감|캐릭터|제약|의약품|커피|카페|레스토랑|음식점|치킨|피자|햄버거|엔터테인먼트|기획사|"
    r"company|corporation|brand|manufacturer|maker|beer|brewery|beverage|drink|airline|bank|insurer|broadcaster|channel|"
    r"service|platform|software|video game|app|retailer|restaurant|chain|product|smartphone|automobile|car|clothing|fashion|"
    r"cosmetic|website|social network|messaging|conglomerate|business|label|studio|franchise|toy|character|pharmaceutical|"
    r"hotel|resort|watch|jewel|furniture|coffee|energy drink|snack|noodle|search engine|streaming|marketplace|e-commerce",
    re.I)


def get(url):
    path = CACHE / ("f2_" + hashlib.md5(url.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                d = json.load(r)
            path.write_text(json.dumps(d))
            return d
        except Exception:  # noqa: BLE001
            time.sleep(3 * (attempt + 1))
    return {}


def items_for(lang, titles):
    out = {}
    for k in range(0, len(titles), 50):
        q = urllib.parse.urlencode({"action": "query", "titles": "|".join(titles[k:k + 50]), "prop": "pageprops",
                                    "ppprop": "wikibase_item", "format": "json"})
        d = get(f"https://{lang}.wikipedia.org/w/api.php?{q}")
        nm = {n["to"]: n["from"] for n in d.get("query", {}).get("normalized", [])}
        for p in d.get("query", {}).get("pages", {}).values():
            qid = p.get("pageprops", {}).get("wikibase_item")
            if qid:
                out[nm.get(p["title"], p["title"])] = qid
                out[p["title"]] = qid
    return out


def entities(qids):
    out = {}
    qids = list(dict.fromkeys(qids))
    for k in range(0, len(qids), 50):
        q = urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(qids[k:k + 50]),
                                    "props": "descriptions|claims", "languages": "ko|en", "format": "json"})
        out.update(get(f"https://www.wikidata.org/w/api.php?{q}").get("entities", {}))
    return out


def judge(ent):
    p31 = [c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") for c in ent.get("claims", {}).get("P31", [])]
    bad = [BAD_P31[p] for p in p31 if p in BAD_P31]
    desc = " / ".join((ent.get("descriptions", {}).get(l) or {}).get("value", "") for l in ("ko", "en"))
    if GOOD_DESC.search(desc):
        return "ok", desc
    if bad:
        return "reject:" + bad[0], desc
    return "unknown", desc


def second_chance(lang, name, skip):
    """버린 문서 대신 위키 검색으로 '이름 + 브랜드/기업' 을 찾아 검증을 통과하는 첫 문서"""
    word = "브랜드" if lang == "ko" else "brand"
    q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": f"{name} {word}", "srlimit": 5, "format": "json"})
    hits = [h["title"] for h in get(f"https://{lang}.wikipedia.org/w/api.php?{q}").get("query", {}).get("search", [])]
    hits = [h for h in hits if h not in skip and name[:2].lower() in h.lower()]
    if not hits:
        return None
    qm = items_for(lang, hits)
    ents = entities([qm[h] for h in hits if h in qm])
    for h in hits:
        if h in qm and judge(ents.get(qm[h], {}))[0] == "ok":
            return h
    return None


def main():
    fame = list(csv.DictReader((HERE / "fame.csv").open(newline="")))
    brands = {r["id"]: r for r in csv.DictReader((DATA / "brands.csv").open(newline=""))}
    match = {r["id"]: r["qid"] for r in csv.DictReader((DATA / "verify" / "report.csv").open(newline=""))}
    titles = {l: [f[f"{l}_title"] for f in fame if f[f"{l}_title"]] for l in ("ko", "en")}
    qmap = {l: items_for(l, titles[l]) for l in ("ko", "en")}
    ents = entities([q for l in qmap for q in qmap[l].values()])

    rejects = []
    for f in fame:
        for l in ("ko", "en"):
            t = f[f"{l}_title"]
            if not t:
                f[f"{l}_check"] = ""
                continue
            q = qmap[l].get(t)
            verdict, desc = judge(ents.get(q, {})) if q else ("unknown", "")
            f[f"{l}_check"] = verdict
            # 공유 다시 판정: 문서 항목이 브랜드 자신의 매칭 항목이면 공유가 아니다 (표기만 다른 제목)
            if q and match.get(f["id"]) == q:
                f[f"{l}_shared"] = "0"
            if verdict.startswith("reject"):
                b = brands[f["id"]]
                alt = second_chance(l, b["name_ko"] if l == "ko" else b["name_en"], {t})
                rejects.append({"id": f["id"], "name": b["name_ko"], "lang": l, "title": t,
                                "why": verdict, "desc": desc[:80], "views_med12": f[f"{l}_med12"], "retry": alt or ""})
                if alt:
                    from fetch_fame import views
                    s_, m_ = views(l, alt)
                    f[f"{l}_title"], f[f"{l}_check"], f[f"{l}_sum"], f[f"{l}_med12"] = alt, "ok:retry", s_, m_
                else:
                    f[f"{l}_sum"] = f[f"{l}_med12"] = "0"
    with (HERE / "fame_checked.csv").open("w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(fame[0]))
        w.writeheader(); w.writerows(fame)
    with (HERE / "fame_rejects.csv").open("w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["id", "name", "lang", "title", "why", "desc", "views_med12", "retry"])
        w.writeheader(); w.writerows(rejects)
    from collections import Counter
    print("버림", len(rejects), Counter(r["why"] for r in rejects).most_common(6))
    print("한국어 판정", Counter(f["ko_check"] for f in fame))
    return 0


if __name__ == "__main__":
    sys.exit(main())
