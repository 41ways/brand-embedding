"""Wikidata 로 brands.csv 를 대조한다.

1) 영문 이름으로 wbsearchentities → 후보 몇 개
2) wbgetentities 로 후보의 주장(claim)을 받아, 기업·브랜드처럼 보이는 후보를 고른다
3) 국가·설립 연도·모회사를 우리 값과 비교해 verify/report.csv 로 남긴다

결과 캐시는 verify/cache/ 에 두어 다시 돌리면 네트워크를 거의 안 쓴다.
"""
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
API = "https://www.wikidata.org/w/api.php"
UA = "brand-embedding-verify/0.1 (https://github.com/41ways)"

# 기업·브랜드로 볼 만한 P31 값 (일부)
ORG_TYPES = {
    "Q4830453", "Q783794", "Q891723", "Q431289", "Q167037", "Q6881511", "Q219577", "Q1616075",
    "Q43229", "Q18388277", "Q1589009", "Q2085381", "Q507619", "Q46970", "Q15265344", "Q1331793",
    "Q210167", "Q178706", "Q1058914", "Q524518", "Q17377208", "Q2005696", "Q1631129", "Q3918",
    "Q5621421", "Q1762059", "Q740752", "Q22687", "Q1664720", "Q936518", "Q11691", "Q27686",
    "Q170566", "Q1076968", "Q7275", "Q25379", "Q1520223", "Q4287745", "Q1497375", "Q2659904",
    "Q13235160", "Q1377711", "Q848014", "Q26836193", "Q1110684", "Q949149", "Q57655560",
}
DESC_HINT = re.compile(r"compan|brand|manufactur|corporat|conglomerat|retail|chain|airline|bank|maker|label|group|"
                       r"developer|publisher|studio|operator|provider|producer|enterprise|business|firm|house|"
                       r"agency|insur|restaurant|hotel|platform|service", re.I)


def get(params):
    key = CACHE / (re.sub(r"[^\w]+", "_", json.dumps(params, sort_keys=True))[:180] + ".json")
    if key.exists():
        return json.loads(key.read_text())
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    err = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            key.write_text(json.dumps(data))
            time.sleep(0.7)
            return data
        except urllib.error.HTTPError as e:
            err = e
            wait = int(e.headers.get("Retry-After") or 0) or 20 * (attempt + 1)
            time.sleep(min(wait, 120))
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(3 * (attempt + 1))
    raise err


def claim_ids(ent, prop):
    out = []
    for c in ent.get("claims", {}).get(prop, []):
        v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(v, dict) and "id" in v:
            out.append(v["id"])
    return out


def year(ent):
    for c in ent.get("claims", {}).get("P571", []):
        v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(v, dict) and "time" in v:
            m = re.match(r"[+-](\d{4})", v["time"])
            if m:
                return int(m.group(1))
    return None


def entities(ids, props="claims|labels|descriptions"):
    out = {}
    ids = [i for i in dict.fromkeys(ids) if i]
    for k in range(0, len(ids), 50):
        d = get({"action": "wbgetentities", "ids": "|".join(ids[k:k + 50]), "props": props, "languages": "en|ko"})
        out.update(d.get("entities", {}))
    return out


def label(ent):
    labs = ent.get("labels", {})
    return (labs.get("en") or labs.get("ko") or {}).get("value", "")


def main():
    rows = list(csv.DictReader((DATA / "brands.csv").open(newline="")))
    by_id = {r["id"]: r for r in rows}
    extra = {}
    for p in (DATA / "enrich" / "out").glob("chunk_*.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                extra[d["id"]] = d

    # 1) 검색
    cands = {}
    for n, r in enumerate(rows, 1):
        hits = get({"action": "wbsearchentities", "search": r["name_en"], "language": "en", "type": "item", "limit": 7})
        cands[r["id"]] = [h["id"] for h in hits.get("search", [])]
        if n % 200 == 0:
            print("검색", n)

    # 2) 후보 주장 받기
    ents = entities([q for qs in cands.values() for q in qs])

    def score(r, q):
        e = ents.get(q, {})
        s = 0
        if set(claim_ids(e, "P31")) & ORG_TYPES:
            s += 3
        desc = (e.get("descriptions", {}).get("en") or {}).get("value", "")
        if DESC_HINT.search(desc):
            s += 2
        if any(p in e.get("claims", {}) for p in ("P571", "P159", "P749", "P127", "P452", "P1454")):
            s += 1
        if label(e).lower() == r["name_en"].lower():
            s += 1
        return s

    match = {}
    for r in rows:
        ranked = sorted(cands[r["id"]], key=lambda q: -score(r, q))
        if ranked and score(r, ranked[0]) >= 4:
            match[r["id"]] = ranked[0]

    # 국가·모회사 이름을 위한 두 번째 조회
    refs = []
    for q in match.values():
        e = ents[q]
        refs += claim_ids(e, "P17") + claim_ids(e, "P495") + claim_ids(e, "P749") + claim_ids(e, "P127")
    ref_ents = entities(refs, props="claims|labels")

    def iso(q):
        for c in ref_ents.get(q, {}).get("claims", {}).get("P297", []):
            return c["mainsnak"].get("datavalue", {}).get("value")
        return None

    q_of = {i: q for i, q in match.items()}
    out_rows = []
    for r in rows:
        q = match.get(r["id"])
        rec = {"id": r["id"], "name_en": r["name_en"], "qid": q or "", "wd_label": "", "wd_desc": "",
               "country": r["country"], "wd_country": "", "country_ok": "",
               "founded": (extra.get(r["id"]) or {}).get("founded") or "", "wd_founded": "", "founded_ok": "",
               "parent": r["parent"], "wd_parents": "", "parent_ok": ""}
        if q:
            e = ents[q]
            rec["wd_label"] = label(e)
            rec["wd_desc"] = (e.get("descriptions", {}).get("en") or {}).get("value", "")
            cs = [iso(c) for c in claim_ids(e, "P17") + claim_ids(e, "P495")]
            cs = [c for c in cs if c]
            rec["wd_country"] = "|".join(dict.fromkeys(cs))
            if cs:
                rec["country_ok"] = "1" if r["country"] in cs else "0"
            y = year(e)
            rec["wd_founded"] = y or ""
            if y and rec["founded"]:
                rec["founded_ok"] = "1" if abs(int(rec["founded"]) - y) <= 2 else "0"
            ps = claim_ids(e, "P749") + claim_ids(e, "P127")
            rec["wd_parents"] = "|".join(label(ref_ents.get(p, {})) or p for p in ps)
            if ps:
                ours = r["parent"]
                ok = False
                seen = set()
                while ours and ours not in seen:  # 조상 누구든 맞으면 인정
                    seen.add(ours)
                    if q_of.get(ours) in ps or by_id[ours]["name_en"].lower() in rec["wd_parents"].lower():
                        ok = True
                        break
                    ours = by_id[ours]["parent"]
                rec["parent_ok"] = "1" if ok else ("0" if r["parent"] else "missing")
        out_rows.append(rec)

    with (HERE / "report.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    def cnt(k, v):
        return sum(1 for x in out_rows if x[k] == v)
    print(f"매칭 {len(match)}/{len(rows)}")
    print(f"국가 일치 {cnt('country_ok', '1')} 불일치 {cnt('country_ok', '0')}")
    print(f"설립 연도 일치 {cnt('founded_ok', '1')} 불일치 {cnt('founded_ok', '0')}")
    print(f"모회사 일치 {cnt('parent_ok', '1')} 불일치 {cnt('parent_ok', '0')} 우리 쪽 빈칸 {cnt('parent_ok', 'missing')}")


if __name__ == "__main__":
    main()
