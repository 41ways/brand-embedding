"""게임(행선지4: Brand)용 압축 모델을 뽑는다.

    python3 embed/export_game.py ../wheretogo-brand

워커는 1,603×1,603 유사도표(5MB)를 들 수 없어서, 요청이 올 때 정답 한 줄만 계산한다.
그래서 묶음을 가볍게 바꾼다.
- summary·offering (bge-m3 1024차원) → 평균을 빼지 않는 SVD 로 128·96차원, 다시 L2 정규화
- category·origin·family (원-핫) → 번호만. 코사인은 번호가 같은지로 바로 계산된다
- 나머지 작은 묶음은 그대로
모든 실수 벡터는 int8(×127)로 담는다.

압축 전후 순위가 얼마나 같은지 재서 기준에 못 미치면 멈춘다.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

import preview as P

PCA_DIMS = {"summary": 128, "offering": 96}
DENSE = ["summary", "offering", "archetype", "position", "heritage", "price", "size", "scale"]
LABEL = {"summary": "설명", "offering": "상품", "archetype": "이미지", "position": "업계 위치", "heritage": "창업·소유",
         "price": "가격대", "size": "규모", "scale": "인지도", "category": "업종", "origin": "출신", "family": "소속"}


def pca(v, has, k):
    """평균을 빼지 않는 절단 SVD. bge 벡터는 한쪽으로 몰려 있어서, 평균을 빼면 원래 코사인 순서가
    크게 흐트러진다(128차원에서 순위 상관 0.93). 빼지 않으면 0.998"""
    _, _, vt = np.linalg.svd(v[has], full_matrices=False)
    out = v @ vt[:k].T
    out[~has] = 0
    return P.l2(out.astype(np.float32))


def quant(m):
    return np.clip(np.round(m * 127), -127, 127).astype(np.int8)


def compact_blocks(rows, blocks):
    """워커와 똑같은 계산을 파이썬으로 — 검증용"""
    sector = [r["sector"] for r in rows]
    cat = [r["sector"] + r["category"] for r in rows]
    region = [P.REGION.get(r["country"], "기타") for r in rows]
    roots = P.root_of(rows)
    dense = {}
    for b in DENSE:
        v, has = blocks[b]
        if b in PCA_DIMS:
            v = pca(v, has, PCA_DIMS[b])
        dense[b] = (quant(v), has)
    ids = {
        "sector": np.unique(sector, return_inverse=True)[1],
        "category": np.unique(cat, return_inverse=True)[1],
        "country": np.unique([r["country"] for r in rows], return_inverse=True)[1],
        "region": np.unique(region, return_inverse=True)[1],
        "root": np.unique(roots, return_inverse=True)[1],
    }
    return dense, ids


def compact_cos(dense, ids):
    cos, mask = {}, {}
    for b, (q, has) in dense.items():
        f = q.astype(np.float32) / 127
        f = P.l2(f)
        cos[b] = f @ f.T
        mask[b] = np.outer(has, has).astype(np.float32)
    eq = lambda k: (ids[k][:, None] == ids[k][None, :]).astype(np.float32)
    ones = np.ones_like(eq("root"))
    cos["category"] = (0.36 * eq("sector") + eq("category")) / 1.36
    cos["origin"] = (eq("country") + 0.49 * eq("region")) / 1.49
    cos["family"] = eq("root")
    for b in ("category", "origin", "family"):
        mask[b] = ones
    return cos, mask


def main():
    out_dir = Path(sys.argv[1])
    rows, extra = P.load()
    blocks = P.build_blocks(rows, extra)
    w = P.MODES["all"]

    full_cos, full_mask = P.pairwise(blocks)
    full = P.similarity(full_cos, full_mask, w)
    dense, ids = compact_blocks(rows, blocks)
    c_cos, c_mask = compact_cos(dense, ids)
    comp = P.similarity(c_cos, c_mask, w)

    # 검증: 정답 한 줄씩 순위 비교
    n = len(rows)
    rk = lambda s: np.argsort(np.argsort(-s, axis=1), axis=1)
    rf, rc = rk(full), rk(comp)
    spearman = np.array([np.corrcoef(rf[i], rc[i])[0, 1] for i in range(n)])
    top10 = np.mean([len(set(np.argsort(-full[i])[:10]) & set(np.argsort(-comp[i])[:10])) / 10 for i in range(n)])
    print(f"순위 상관(중앙) {np.median(spearman):.4f}  최저 {spearman.min():.4f}  상위10 겹침 {top10:.3f}")
    assert np.median(spearman) > 0.97 and top10 > 0.8, "압축으로 순위가 너무 달라짐"

    # 좌표: 미리보기에서 쓴 '전부' t-SNE
    pv = json.loads((P.OUT / "preview.json").read_text())
    assert [b["id"] for b in pv["brands"]] == [r["id"] for r in rows]
    xy = pv["modes"]["all"]["xy"]

    model = {
        "n": n,
        "weights": w,
        "full_share": P.FULL_SHARE,
        "labels": LABEL,
        "dense": {b: {"dim": int(q.shape[1]), "data": base64.b64encode(q.tobytes()).decode(),
                      "has": base64.b64encode(np.packbits(has.astype(np.uint8)).tobytes()).decode()}
                  for b, (q, has) in dense.items()},
        "ids": {k: v.astype(int).tolist() for k, v in ids.items()},
        # 검증용: 몇 개 정답에 대해 파이썬이 계산한 상위 20
        "golden": {rows[i]["id"]: [rows[j]["id"] for j in np.argsort(-comp[i])[:20]] for i in range(0, n, 97)},
    }
    brands = []
    for i, r in enumerate(rows):
        e = extra.get(r["id"], {})
        brands.append({
            "id": r["id"], "name": r["name_ko"], "en": r["name_en"],
            "alias": [a for a in r["aliases"].split("|") if a],
            "sector": r["sector"], "sub": r["category"], "country": r["country"],
            "x": round(xy[i][0], 4), "y": round(xy[i][1], 4),
            "products": (e.get("products") or [])[:4],
        })

    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "worker" / "src").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "brands.json").write_text(json.dumps({"brands": brands}, ensure_ascii=False, separators=(",", ":")))
    (out_dir / "worker" / "src" / "model.json").write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")))
    for f in ("data/brands.json", "worker/src/model.json"):
        print(f, (out_dir / f).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
