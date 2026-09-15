"""브랜드 맞히기를 터미널에서 한다 — 규칙·크기별 부분 목록을 사람(또는 에이전트)이 직접 해 보기 위한 도구.

    python3 embed/play.py prepare                     부분 목록마다 유사도표를 미리 계산 (한 번)
    python3 embed/play.py new quota-200 [--seed 3]    새 판 → 판 번호
    python3 embed/play.py list quota-200              고를 수 있는 브랜드 이름 (찾기 칸에 해당)
    python3 embed/play.py guess <판> <이름>           몇 번째로 닮았는지·점수·닮은 점
    python3 embed/play.py giveup <판>

게임과 같은 규칙: 순위와 점수, "닮은 점"(그 묶음에서 정답과 가장 닮은 5% 안)만 보여 준다.
판 기록은 embed/out/play/games/*.json, 부를 때마다 embed/out/play/log.jsonl 에 남는다.
base 는 기본 1,603개 전체.
"""
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "embed" / "out" / "play"
GAMES = OUT / "games"
SUBSETS = ROOT / "data" / "subsets"
MAX_GUESSES = 60
LABEL = {"summary": "설명", "offering": "상품", "archetype": "이미지", "position": "업계 위치", "heritage": "창업·소유",
         "price": "가격대", "size": "규모", "category": "업종", "origin": "출신", "family": "소속"}


def norm(s):
    return re.sub(r"[\s·\-.&'’!+()]", "", str(s)).lower()


def variants():
    return ["base"] + sorted(p.stem for p in SUBSETS.glob("*-*.csv"))


def prepare(weights=None, tag="", only=None, hint_pct=.05):
    import preview as P
    rows, extra = P.load()
    idx = {r["id"]: i for i, r in enumerate(rows)}
    cos, mask = P.pairwise(P.build_blocks(rows, extra))
    w = weights or P.MODES["all"]
    OUT.mkdir(parents=True, exist_ok=True)
    for v in (only or variants()):
        ids = [r["id"] for r in rows] if v == "base" else [r["id"] for r in csv.DictReader((SUBSETS / f"{v}.csv").open())]
        sel = np.array([idx[i] for i in ids])
        c = {b: cos[b][np.ix_(sel, sel)] for b in w}
        m = {b: mask[b][np.ix_(sel, sel)] for b in w}
        sim = P.similarity(c, m, w)
        n = len(sel)
        # 묶음별 5% 경계 (정답 행마다)
        near = np.zeros((n, n), dtype=np.uint16)
        for k, b in enumerate(b for b in w if b != "scale"):
            cb = np.where(m[b] > 0, c[b], -np.inf)
            np.fill_diagonal(cb, -np.inf)
            cnt = np.isfinite(cb).sum(1)
            kth = np.maximum(1, (cnt * hint_pct).astype(int))
            srt = -np.sort(-cb, axis=1)
            edge = srt[np.arange(n), kth - 1]
            near |= ((cb >= edge[:, None]) & (cb > 0) & np.isfinite(cb)).astype(np.uint16) << k
        meta = [{"id": rows[i]["id"], "name": rows[i]["name_ko"], "en": rows[i]["name_en"],
                 "alias": [a for a in rows[i]["aliases"].split("|") if a], "sub": rows[i]["category"],
                 "sector": rows[i]["sector"]} for i in sel]
        np.savez_compressed(OUT / f"{v}{tag}.npz", sim=sim.astype(np.float32), near=near)
        (OUT / f"{v}{tag}.json").write_text(json.dumps({"brands": meta, "blocks": [b for b in w if b != "scale"],
                                                         "weights": w}, ensure_ascii=False))
        print(v + tag, n)


def load_variant(v):
    meta = json.loads((OUT / f"{v}.json").read_text())
    z = np.load(OUT / f"{v}.npz")
    return meta, z["sim"], z["near"]


def ranks_for(sim, a):
    order = np.lexsort((np.arange(len(sim)), -sim[a]))
    order = order[order != a]
    r = np.zeros(len(sim), dtype=int)
    r[order] = np.arange(1, len(sim))
    return r


def find(meta, q):
    qn = norm(q)
    exact = [i for i, b in enumerate(meta["brands"]) if qn in [norm(b["name"]), norm(b["en"])] + [norm(a) for a in b["alias"]]]
    if exact:
        return exact[0], []
    part = [i for i, b in enumerate(meta["brands"]) if qn and (qn in norm(b["name"]) or qn in norm(b["en"]))]
    return None, part[:8]


def cmd_new(v, seed=None):
    meta, sim, _ = load_variant(v)
    n = len(meta["brands"])
    if seed is None:
        a = random.Random(time.time_ns()).randrange(n)
    else:
        # 시드마다 정답이 겹치지 않게: 목록마다 고정된 섞음 순서에서 seed 번째를 꺼낸다
        # (예전엔 Random(seed).randrange(n) — 시드 3·4 가 늘 같은 정답을 뽑았다)
        order = list(range(n))
        random.Random(f"{v}:order").shuffle(order)
        a = order[seed % n]
    GAMES.mkdir(parents=True, exist_ok=True)
    gid = f"{v}-{int(time.time() * 1000) % 10**9:09d}"
    (GAMES / f"{gid}.json").write_text(json.dumps({"variant": v, "answer": a, "seed": seed, "guesses": [], "done": False}))
    print(f"판 {gid} · {v} · {len(meta['brands'])}곳 중 하나가 정답")


def cmd_guess(gid, q):
    path = GAMES / f"{gid}.json"
    g = json.loads(path.read_text())
    if g["done"]:
        print("끝난 판입니다"); return
    meta, sim, near = load_variant(g["variant"])
    i, sug = find(meta, q)
    if i is None:
        print(f'"{q}" — 목록에 없음' + (" · 비슷한 이름: " + ", ".join(meta["brands"][k]["name"] for k in sug) if sug else ""))
        return
    b = meta["brands"][i]
    if any(x["i"] == i for x in g["guesses"]):
        print(f"{b['name']} 은(는) 이미 불렀습니다"); return
    a, n = g["answer"], len(meta["brands"])
    r = int(ranks_for(sim, a)[i])
    bits = int(near[a, i])
    hint = [LABEL[blk] for k, blk in enumerate(meta["blocks"]) if bits >> k & 1] if r else []
    score = 100 if r == 0 else round(100 - r * 100 / (n - 1) * .97, 2)
    g["guesses"].append({"i": i, "rank": r, "t": time.time()})
    with (OUT / "log.jsonl").open("a") as f:
        f.write(json.dumps({"game": gid, "variant": g["variant"], "no": len(g["guesses"]), "guess": b["id"], "rank": r, "n": n}) + "\n")
    if r == 0:
        g["done"] = True
        print(f"정답! {b['name']} ({b['en']}) · {len(g['guesses'])}번 만에")
    else:
        print(f"{b['name']} [{b['sub']}] · {n - 1}곳 중 {r}번째로 닮음 · {score:.2f}점 · 닮은 점: {' · '.join(hint[:3]) if hint else '없음'}")
        best = sorted(g["guesses"], key=lambda x: x["rank"])[:5]
        print("  지금까지 가까운 곳: " + ", ".join(f"{meta['brands'][x['i']]['name']}({x['rank']})" for x in best))
        if len(g["guesses"]) >= MAX_GUESSES:
            g["done"] = True
            print(f"{MAX_GUESSES}번을 넘겨 끝. 정답은 {meta['brands'][a]['name']}")
    path.write_text(json.dumps(g))


def cmd_giveup(gid):
    path = GAMES / f"{gid}.json"
    g = json.loads(path.read_text())
    meta, _, _ = load_variant(g["variant"])
    g["done"] = True; g["gaveup"] = True
    path.write_text(json.dumps(g))
    with (OUT / "log.jsonl").open("a") as f:
        f.write(json.dumps({"game": gid, "variant": g["variant"], "giveup": True, "no": len(g["guesses"])}) + "\n")
    b = meta["brands"][g["answer"]]
    print(f"포기 · 정답은 {b['name']} ({b['en']}, {b['sub']})")


def cmd_list(v):
    meta, _, _ = load_variant(v)
    by = {}
    for b in meta["brands"]:
        by.setdefault(b["sector"], []).append(b["name"])
    for sec in sorted(by):
        print(f"[{sec}] " + ", ".join(sorted(by[sec])))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__); sys.exit(0)
    if a[0] == "prepare":
        sys.path.insert(0, str(ROOT / "embed")); prepare()
    elif a[0] == "new":
        cmd_new(a[1], int(a[a.index("--seed") + 1]) if "--seed" in a else None)
    elif a[0] == "guess":
        cmd_guess(a[1], " ".join(a[2:]))
    elif a[0] == "giveup":
        cmd_giveup(a[1])
    elif a[0] == "list":
        cmd_list(a[1])
