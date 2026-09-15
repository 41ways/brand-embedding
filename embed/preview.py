"""브랜드 임베딩 미리보기. 결과는 embed/out/preview.json.

묶음(block)마다 벡터를 L2 정규화하고 브랜드별로 "이 묶음이 있음" 표시를 둔다.
두 브랜드의 유사도 = 둘 다 가진 묶음만으로 계산한 코사인의 가중 평균.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np

from clean import strip_places

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "embed" / "out"
OUT.mkdir(parents=True, exist_ok=True)

REGION = {
    "KR": "동아시아", "JP": "동아시아", "CN": "동아시아", "TW": "동아시아", "HK": "동아시아", "MO": "동아시아",
    "US": "북미", "CA": "북미", "MX": "중남미", "BR": "중남미", "AR": "중남미", "CL": "중남미", "CO": "중남미",
    "GB": "서유럽", "FR": "서유럽", "DE": "서유럽", "IT": "남유럽", "ES": "남유럽", "PT": "남유럽", "GR": "남유럽",
    "NL": "서유럽", "BE": "서유럽", "CH": "서유럽", "AT": "서유럽", "IE": "서유럽", "LU": "서유럽", "MC": "서유럽",
    "SE": "북유럽", "DK": "북유럽", "NO": "북유럽", "FI": "북유럽", "IS": "북유럽",
    "IN": "남·동남아", "SG": "남·동남아", "TH": "남·동남아", "VN": "남·동남아", "ID": "남·동남아", "MY": "남·동남아", "PH": "남·동남아",
    "AE": "중동", "SA": "중동", "QA": "중동", "IL": "중동", "TR": "중동",
    "AU": "오세아니아", "NZ": "오세아니아", "ZA": "아프리카",
}
ARCH = ["innocent", "explorer", "sage", "hero", "outlaw", "magician", "everyman", "lover", "jester", "caregiver", "creator", "ruler"]
OWN = ["family", "founder", "public", "state", "foundation", "coop", "pe", "chaebol", "subsidiary"]
ERAS = [1850, 1900, 1945, 1970, 1990, 2005, 2015]
PRICE_CENTERS = [3 + 0.75 * k for k in range(9)]   # log10(원): 1천원 ~ 10억원
SIZE_CENTERS = [-1 + 0.5 * k for k in range(10)]    # log10(십억달러): 1억 ~ 3조 달러

MODES = {
    "structure": {"category": 0.55, "family": 0.2, "origin": 0.1, "scale": 0.15},
    "meaning": {"summary": 0.5, "archetype": 0.3, "position": 0.2},
    # embed/tune.py 격자 탐색으로 고름 (train 틀림 최소 + 다른 분야 이웃 10% 이상)
    "all": {"summary": 0.32, "offering": 0.115, "archetype": 0.09, "price": 0.07, "position": 0.045,
            "category": 0.09, "heritage": 0.08, "size": 0.08, "origin": 0.06, "family": 0.015, "scale": 0.035},
}
FULL_SHARE = 0.5  # 공통 묶음 가중치가 이보다 적으면 유사도를 비례해 깎는다


def l2(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1
    return m / n


def onehot(values, keys=None):
    keys = keys or sorted(set(values))
    idx = {k: i for i, k in enumerate(keys)}
    m = np.zeros((len(values), len(keys)), dtype=np.float32)
    for r, v in enumerate(values):
        if v in idx:
            m[r, idx[v]] = 1
    return m


def soft_bins(x, centers, sigma):
    return np.exp(-((x - np.array(centers)) ** 2) / (2 * sigma ** 2))


def load():
    with (ROOT / "data" / "brands.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    extra = {}
    for folder in ("out", "out2"):  # 1차·2차 보강을 id 로 합친다 (confidence 는 1차 것을 유지)
        for p in sorted((ROOT / "data" / "enrich" / folder).glob("chunk_*.jsonl")):
            for line in p.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    cur = extra.setdefault(d["id"], {})
                    if folder == "out2":
                        d["confidence2"] = d.pop("confidence", None)
                    cur.update(d)
    apply_market_caps(extra)
    return rows, extra


def apply_market_caps(extra, path=ROOT / "data" / "market" / "caps.csv"):
    """data/market/update_caps.py 가 받은 실제 시총이 있으면 LLM 추정값 대신 쓴다."""
    if not path.exists():
        return
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "check") and r["id"] in extra:
                e = extra[r["id"]]
                e["market_cap_llm_usd_b"] = e.get("market_cap_usd_b")
                e["market_cap_usd_b"] = float(r["market_cap_usd_b"])
                e["market_cap_date"] = r["fetched"]


def bge(texts, cache, max_length):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if cache.exists():
        try:
            saved = np.load(cache, allow_pickle=True).item()
            if saved["texts"] == texts:
                return saved["emb"]
        except (ValueError, KeyError, AttributeError):
            pass
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3").eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(dev)
    torch.set_num_threads(4)
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            b = tok(texts[i:i + 16], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
            outs.append(model(**b).last_hidden_state[:, 0].float().cpu().numpy())
    e = l2(np.concatenate(outs))
    np.save(cache, {"texts": texts, "emb": e})
    return e


def root_of(rows):
    parent = {r["id"]: r["parent"] for r in rows}

    def up(i):
        seen = set()
        while parent.get(i) and i not in seen:
            seen.add(i)
            i = parent[i]
        return i
    return [up(r["id"]) for r in rows]


def build_blocks(rows, extra):
    n = len(rows)
    ones = np.ones(n, dtype=bool)
    tier = {"mega": [1, .5, 0], "major": [.5, 1, .5], "niche": [0, .5, 1]}
    ex = [extra.get(r["id"], {}) for r in rows]

    blocks = {}
    blocks["category"] = (l2(np.hstack([onehot([r["sector"] for r in rows]) * 0.6,
                                        onehot([r["sector"] + r["category"] for r in rows])])), ones)
    blocks["family"] = (l2(onehot(root_of(rows))), ones)
    blocks["origin"] = (l2(np.hstack([onehot([r["country"] for r in rows]),
                                      onehot([REGION.get(r["country"], "기타") for r in rows]) * 0.7])), ones)
    blocks["scale"] = (l2(np.hstack([np.array([tier[r["tier"]] for r in rows], dtype=np.float32),
                                     onehot([r["b2b"] for r in rows])])), ones)

    has = np.array([bool(e.get("summary")) for e in ex])
    summ = [strip_places(e.get("summary") or "") for e in ex]
    blocks["summary"] = (bge(summ, OUT / "summary_emb.npy", 256), has)

    arch = np.array([[e.get("archetype", {}).get(a, 0) for a in ARCH] for e in ex], dtype=np.float32)
    blocks["archetype"] = (l2(arch), arch.sum(1) > 0)

    pos = np.zeros((n, 10), dtype=np.float32)
    pos_has = np.zeros(n, dtype=bool)
    centers = [0, .25, .5, .75, 1]
    for i, e in enumerate(ex):
        for k, off in (("price_pos", 0), ("exclusivity", 5)):
            if e.get(k) is not None:
                pos[i, off:off + 5] = soft_bins(e[k], centers, .18)
                pos_has[i] = True
    blocks["position"] = (l2(pos), pos_has)

    her = np.zeros((n, len(OWN) + 2 + len(ERAS)), dtype=np.float32)
    her_has = np.zeros(n, dtype=bool)
    for i, e in enumerate(ex):
        if not e:
            continue
        her_has[i] = True
        if e.get("ownership") in OWN:
            her[i, OWN.index(e["ownership"])] = 1
        her[i, len(OWN) + int(bool(e.get("founder_named")))] = 0.8
        if e.get("founded"):
            her[i, len(OWN) + 2:] = soft_bins(e["founded"], ERAS, 25) * 0.8
    blocks["heritage"] = (l2(her), her_has)

    prods = [strip_places("; ".join(e.get("products") or [])) for e in ex]
    off_has = np.array([bool(p) for p in prods])
    blocks["offering"] = (bge(prods, OUT / "offering_emb.npy", 96), off_has)

    price = np.zeros((n, len(PRICE_CENTERS)), dtype=np.float32)
    price_has = np.zeros(n, dtype=bool)
    for i, e in enumerate(ex):
        if e.get("typical_price_krw"):
            price[i] = soft_bins(np.log10(e["typical_price_krw"]), PRICE_CENTERS, .45)
            price_has[i] = True
    blocks["price"] = (l2(price), price_has)

    size, size_has = size_block(rows, ex)
    blocks["size"] = (l2(size), size_has)
    return blocks


def company_size(rows, ex):
    """브랜드별 (log10 십억달러, 출처). 출처: own 자기 시총·기업가치 / parent 소속 시총 / revenue 매출."""
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    out = []
    for i, r in enumerate(rows):
        e = ex[i]
        if e.get("market_cap_usd_b"):
            out.append((np.log10(e["market_cap_usd_b"]), "own"))
            continue
        cur, seen, found = r["parent"], set(), None
        while cur and cur in by_id and cur not in seen:
            seen.add(cur)
            cap = ex[by_id[cur]].get("market_cap_usd_b")
            if cap:
                found = (np.log10(cap), "parent")
                break
            cur = rows[by_id[cur]]["parent"]
        if found:
            out.append(found)
        elif e.get("revenue_usd_b"):
            out.append((np.log10(e["revenue_usd_b"] * 1.5), "revenue"))  # 매출 1.5배를 대략의 가치로
        else:
            out.append((None, None))
    return out


def size_block(rows, ex):
    n = len(rows)
    m = np.zeros((n, len(SIZE_CENTERS) + 3), dtype=np.float32)
    has = np.zeros(n, dtype=bool)
    for i, (v, src) in enumerate(company_size(rows, ex)):
        if v is None:
            continue
        has[i] = True
        # 소속에서 물려받은 값은 조금 흐리게: 킷캣이 곧 네슬레 규모는 아니므로
        m[i, :len(SIZE_CENTERS)] = soft_bins(v, SIZE_CENTERS, .4) * (1 if src == "own" else .7)
        m[i, len(SIZE_CENTERS) + ["own", "parent", "revenue"].index(src)] = .3
    return m, has


def pairwise(blocks):
    cos = {k: v @ v.T for k, (v, _) in blocks.items()}
    mask = {k: np.outer(m, m).astype(np.float32) for k, (_, m) in blocks.items()}
    return cos, mask


def similarity(cos, mask, weights):
    """둘 다 가진 묶음만으로 코사인 가중 평균. 자기 자신 칸은 -9."""
    num = sum(wt * mask[b] * cos[b] for b, wt in weights.items())
    den = sum(wt * mask[b] for b, wt in weights.items())
    sim = np.where(den > 0, num / np.maximum(den, 1e-9), 0) * np.minimum(1, den / FULL_SHARE)
    np.fill_diagonal(sim, -9)
    return sim


def main():
    rows, extra = load()
    print("보강된 브랜드", len(extra), "/", len(rows))
    blocks = build_blocks(rows, extra)
    cos, mask = pairwise(blocks)

    from sklearn.manifold import TSNE

    keep = ["id", "name_ko", "name_en", "country", "sector", "category", "tier", "b2b", "parent"]
    brands = []
    for r in rows:
        b = {k: r[k] for k in keep}
        e = extra.get(r["id"])
        if e:
            b.update({k: e.get(k) for k in ["summary", "why", "archetype", "price_pos", "exclusivity", "founded", "founder_named", "ownership", "confidence",
                                        "typical_price_krw", "price_item", "products", "listed", "ticker", "market_cap_usd_b", "market_cap_date", "revenue_usd_b"]})
        brands.append(b)
    result = {"brands": brands, "modes": {}}

    for mode, w in MODES.items():
        sim = similarity(cos, mask, w)
        nb = np.argsort(-sim, axis=1)[:, :12]
        why = {b: np.round(np.where(np.take_along_axis(mask[b], nb, 1) > 0,
                                    np.take_along_axis(cos[b], nb, 1), -1), 2).tolist() for b in w}
        dist = np.clip(1 - sim, 0, 2)
        np.fill_diagonal(dist, 0)
        xy = TSNE(n_components=2, metric="precomputed", init="random", perplexity=30, random_state=7).fit_transform(dist)
        xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
        result["modes"][mode] = {
            "weights": w,
            "xy": np.round(xy, 4).tolist(),
            "nb": nb.tolist(),
            "sim": np.round(np.take_along_axis(sim, nb, axis=1), 3).tolist(),
            "why": why,
        }
        print(mode, "done")

    (OUT / "preview.json").write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    print("size", (OUT / "preview.json").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
