# brand-embedding

브랜드 1,603개(럭셔리부터 카길 같은 B2B, 작은 브랜드까지)를 여러 특징 묶음으로 임베딩해 "비슷한 브랜드"를 재는 실험. 행선지 게임을 브랜드 공간으로 옮기기 위한 준비 단계.

## 흐름

```
data/parts/*.csv ──merge.py──▶ data/brands.csv            목록 (분류·소속·국가·규모)
data/enrich/in ──(LLM 보강)──▶ data/enrich/out/*.jsonl    설명글·원형·가격 위치·창업/소유
                              data/verify/wikidata.py     Wikidata 대조 → report.csv
embed/preview.py ──▶ embed/out/preview.json               묶음별 벡터, 유사도, t-SNE 좌표
embed/build_viewer.py ──▶ embed/out/brand-space.html      비교 뷰어
embed/tune.py                                             정답 세트로 가중치 격자 탐색
```

## 유사도

묶음마다 L2 정규화한 벡터와 "이 브랜드에 이 묶음이 있음" 표시를 둔다. 두 브랜드의 유사도는 **둘 다 가진 묶음만으로** 계산한 코사인의 가중 평균이고, 공통 묶음 가중치가 0.5 미만이면 비례해 깎는다. 작은 브랜드나 B2B 처럼 빈 특징이 많은 브랜드도 불이익 없이 비교된다.

| 묶음 | 내용 |
|---|---|
| summary | 같은 틀로 쓴 한국어 요약(이름 가림) → bge-m3 |
| archetype | 브랜드 원형 12유형 비중 |
| position | 같은 업종 안 가격 위치·희소성 (구간 소프트 원-핫) |
| heritage | 소유 형태, 창업자 이름 여부, 설립 연대 |
| category / family / origin / scale | 분류 계층, 최상위 소속, 국가·권역, 규모·B2B |
| name | 이름 → bge-m3 (비교용, 철자에 끌려 단독으론 못 씀) |

## 실행

```bash
python3 data/merge.py
python3 data/enrich/check.py
python3 embed/preview.py && python3 embed/build_viewer.py
python3 -m unittest discover -s tests
```

`tests/triplets.csv` 는 "A 는 C 보다 B 에 가깝다" 정답 세트. sanity·position 은 전부 맞아야 하고, cross(업종을 넘는 가까움)는 80% 이상, debatable 은 기록만 한다.

## 알려진 한계

- 목록과 보강 데이터는 LLM 지식 기반 1차본. 2025~26 인수·합병이 걸린 소속·소유 형태는 Wikidata 대조 후 검수가 필요하다.
- 요약에 국적이 들어가 설명글 임베딩이 나라별로 뭉치는 경향이 있다(무인양품 → 일본 식품 브랜드).
- 정답 세트가 34문항이라 가중치가 과적합되기 쉽다.
