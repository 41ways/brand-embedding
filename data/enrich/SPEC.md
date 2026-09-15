# 1차 보강 데이터 규격

입력 `in/chunk_XX.csv` 한 줄마다 출력 `out/chunk_XX.jsonl` 에 JSON 한 줄. 필드:

```json
{"id":"hermes",
 "summary":"프랑스의 최고급 가죽·실크 하우스. 수작업 가방과 스카프가 대표 상품이며 …",
 "price_pos":0.98, "exclusivity":0.95,
 "archetype":{"ruler":0.5,"creator":0.3,"lover":0.2},
 "founded":1837, "founder_named":true, "ownership":"family",
 "confidence":"high",
 "why":"대기 명단으로 사야 하는 가방, 가족 지배"}
```

## summary (설명글)
- 한국어 2~3문장, 120~220자. 반드시 이 순서로: **무엇을 파나 → 주 고객 → 분위기·이미지 → 강점/차별점**.
- **브랜드 이름, 모회사 이름, 대표 하위 브랜드 이름, 창업자 이름을 쓰지 않는다.** "이 브랜드", "모회사" 식으로 돌려 쓴다. 이름이 새면 이름 철자로 가까워진다.
- 업종 단어는 써도 된다. 광고 문구 말고 사실과 일반 인식 위주.
- 기업집단·지주회사는 계열 전체의 성격을 요약한다.

## price_pos, exclusivity (업계 안 위치)
- `price_pos`: **같은 category 안에서** 가격이 어디쯤인지 0~1. 0=최저가, 0.5=중간, 1=최고가. 업종이 달라도 "그 업계의 최고급"이면 1 가까이.
- `exclusivity`: 0=누구나 쉽게 삼·어디서나 보임, 1=한정·희소·접근 어려움.
- B2B·기업집단·가격 개념이 안 맞으면 둘 다 `null`. 모르면 `null`.

## archetype (브랜드 원형)
12개 키 중 1~3개, 합이 1.0 (소수 둘째 자리):
`innocent`(순수) `explorer`(탐험가) `sage`(현자) `hero`(영웅) `outlaw`(반항아) `magician`(마법사) `everyman`(보통사람) `lover`(연인) `jester`(광대) `caregiver`(돌보는 자) `creator`(창조자) `ruler`(지배자)

업종 관성 말고 **브랜드가 풍기는 이미지**로 고른다(나이키=hero, 할리데이비슨=outlaw, 레고=creator, 디즈니=magician, 이케아=everyman, 존슨앤존슨=caregiver, 레드불=hero/explorer, 올드스파이스=jester).

## founded, founder_named, ownership (창업·소유)
- `founded`: 브랜드(또는 회사) 시작 연도 정수. 모르면 `null`.
- `founder_named`: 브랜드 이름이 창업자·가문 이름에서 왔으면 true (샤넬, 포르쉐, 혼다, 디즈니). 모르면 false.
- `ownership` 하나 (2026 기준 최상위 지배 형태):
  - `family` 가족·가문 지배(상장이어도 가문이 지배하면 family)
  - `founder` 창업자가 아직 지배
  - `public` 지배주주 없는 상장사
  - `state` 국영·공기업·국부펀드
  - `foundation` 재단·신탁 소유(롤렉스, 이케아 계열, 칼스버그, 노보)
  - `coop` 협동조합·상호회사
  - `pe` 사모펀드
  - `chaebol` 한국 기업집단 계열(오너 일가 지배 재벌 계열사·그룹)
  - `subsidiary` 위 어디에도 안 맞는 외국 대기업 자회사
  - `unknown`

## confidence, why
- `confidence`: 이 줄 전체에 대한 확신 `high` / `mid` / `low`. 작은 브랜드를 잘 모르면 `low` 로 두고 추측을 줄인다.
- `why`: 원형·위치를 그렇게 잡은 근거 한 줄(40자 안팎). 이름 써도 됨.
