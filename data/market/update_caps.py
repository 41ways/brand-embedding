"""티커로 실제 시가총액을 받아 data/market/caps.csv 에 쓴다.

실행: .venv/bin/python data/market/update_caps.py          (전체)
      .venv/bin/python data/market/update_caps.py --retry  (실패한 것만)
재갱신 주기는 정하지 않았다. 필요할 때 다시 돌리면 파일 전체를 새로 쓴다.

- 보강 데이터(out2)의 ticker "거래소:종목" 을 Yahoo Finance 기호로 바꿔 조회
- 현지 통화 시총을 그날 환율로 달러 환산
- LLM 이 적어 둔 값과 4배 넘게 차이 나면 status=check 로 표시(티커 오류 가능성)
- preview.py 는 status 가 ok·check 인 값을 LLM 값보다 우선한다
"""
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

HERE = Path(__file__).parent
DATA = HERE.parent

SUFFIX = {
    "NYSE": "", "NASDAQ": "", "OTC": "",
    "KRX": ".KS", "KOSDAQ": ".KQ",
    "TSE": ".T", "TYO": ".T",
    "ETR": ".DE", "FRA": ".F", "EPA": ".PA", "AMS": ".AS", "EBR": ".BR", "BIT": ".MI", "BME": ".MC",
    "LSE": ".L", "LON": ".L", "ISE": ".IR",
    "SWX": ".SW", "SIX": ".SW", "CPH": ".CO", "STO": ".ST", "OSL": ".OL", "HEL": ".HE", "WSE": ".WA", "IST": ".IS",
    "HKG": ".HK", "HKEX": ".HK", "TPE": ".TW", "TWSE": ".TW", "SSE": ".SS", "SHA": ".SS", "SZSE": ".SZ", "SHE": ".SZ",
    "SGX": ".SI", "NSE": ".NS", "PSE": ".PS", "HOSE": ".VN", "ASX": ".AX", "TSX": ".TO", "TADAWUL": ".SR",
}
# 보강 데이터의 티커가 옛 이름이거나 Yahoo 표기가 다른 곳
RENAMED = {"NYSE:ERJ": "EMBJ", "SWX:ROG": "RO.SW", "PSE:AC": "AC.PS", "PSE:JFC": "JFC.PS"}
# 통화 → 달러 환산 시 단위 보정 (영국 펜스 등)
MINOR = {"GBp": ("GBP", 100), "GBX": ("GBP", 100), "ZAc": ("ZAR", 100), "ILA": ("ILS", 100)}


def yahoo_symbols(ticker):
    ex, sym = ticker.split(":", 1)
    if ticker in RENAMED:
        return [RENAMED[ticker]]
    if ex not in SUFFIX:
        return []
    if ex in ("NYSE", "NASDAQ", "OTC"):
        return [sym.replace(".", "-")]
    if ex in ("HKG", "HKEX"):
        return [sym.lstrip("0").zfill(4) + ".HK"]
    if ex == "KRX":  # 코스닥 종목도 KRX 로 적힌 경우가 있다
        return [sym + ".KS", sym + ".KQ"]
    return [sym.replace(".", "-") + SUFFIX[ex]]  # MAERSK.B → MAERSK-B.CO


def fx_to_usd(currency, cache):
    if currency in MINOR:
        base, div = MINOR[currency]
        return fx_to_usd(base, cache) / div
    if currency == "USD":
        return 1.0
    if currency not in cache:
        cache[currency] = float(yf.Ticker(f"{currency}USD=X").fast_info["last_price"])
        time.sleep(0.3)
    return cache[currency]


def fetch(symbol):
    fi = yf.Ticker(symbol).fast_info  # FastInfo.get() 은 키 이름이 달라 None 을 준다 — [] 로 읽을 것
    cap, cur = fi["market_cap"], fi["currency"]
    if not cap or not cur:
        return None
    return float(cap), cur


def main():
    retry = "--retry" in sys.argv  # 이미 받은 줄은 두고 실패한 줄만 다시
    kept = {}
    if retry and (HERE / "caps.csv").exists():
        kept = {r["id"]: r for r in csv.DictReader((HERE / "caps.csv").open(newline="")) if r["status"] in ("ok", "check")}
    items = []
    for p in sorted((DATA / "enrich" / "out2").glob("chunk_*.jsonl")):
        for line in p.read_text().splitlines():
            d = json.loads(line)
            if d.get("ticker"):
                items.append(d)

    fx, out = {}, []
    today = datetime.now(timezone.utc).date().isoformat()
    for n, d in enumerate(items, 1):
        if d["id"] in kept:
            out.append(kept[d["id"]])
            continue
        rec = {"id": d["id"], "ticker": d["ticker"], "yahoo": "", "currency": "", "market_cap_local": "",
               "fx_usd": "", "market_cap_usd_b": "", "llm_usd_b": d.get("market_cap_usd_b") or "",
               "ratio": "", "status": "", "fetched": today}
        cands = yahoo_symbols(d["ticker"])
        if not cands:
            rec["status"] = "unknown_exchange"
        for sym in cands:
            try:
                got = fetch(sym)
            except Exception as e:  # noqa: BLE001 - 상장폐지·오타 등은 기록만
                got = None
                rec["status"] = f"error:{type(e).__name__}"
            time.sleep(0.25)
            if got:
                cap, cur = got
                usd = cap * fx_to_usd(cur, fx) / 1e9
                rec.update(yahoo=sym, currency=cur, market_cap_local=f"{cap:.0f}",
                           fx_usd=f"{fx_to_usd(cur, fx):.8g}", market_cap_usd_b=f"{usd:.3f}")
                llm = d.get("market_cap_usd_b")
                if llm:
                    r = usd / llm
                    rec["ratio"] = f"{r:.2f}"
                    rec["status"] = "ok" if 0.25 <= r <= 4 else "check"
                else:
                    rec["status"] = "ok"
                break
        else:
            rec["status"] = rec["status"] or "not_found"
        out.append(rec)
        if n % 50 == 0:
            print(n, "/", len(items), flush=True)

    with (HERE / "caps.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(Counter(r["status"].split(":")[0] for r in out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
