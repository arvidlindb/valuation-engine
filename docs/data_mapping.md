# Datamappning – Yahoo Finance → standardiserade poster

Det här dokumentet beskriver hur datalagret (`src/valuation_engine/data/`) översätter Yahoo Finance
(via `yfinance`) till kontraktet i `core/types.py`. Yahoo är källan för bolagsdata; Sveriges
Riksbank är källan för växelkurser och riskfri ränta (avsnitt 5 och 8).

Alla siffror om täckning och datakvalitet nedan kommer från hela-universum-körningen
**2026-11-16** (yfinance 1.8.0) och är observationer, inte antaganden.

## 1. Flöde och filer

```
Yahoo ──fetch──> rå payload (JSON) ──> cache ──build──> CompanyData / PeerSnapshot
                                         │
                                         └── frysta kopior = tests/fixtures/
```

| Fil (i cache och i fixtures) | Innehåll |
|---|---|
| `universe.json` | Råa screener-rader för alla svenska symboler (kolumner + rader) |
| `companies/<TICKER>.json` | Rå payload per ticker: `info` (urval), tre årsrapporter, senaste två kvartalsbalansräkningarna, veckokurser 5 år, FX-kurser |
| `benchmark.json` | Veckokurser för jämförelseindex |
| `peer_snapshots.csv` | Härledd peer-tabell, en rad per bolag, belopp i SEK |
| `riksbank.json` | Riksbankens serier: senaste fixing och månadsvärden (snitt, ultimo, antal observationer) per valuta samt senaste 10-åriga statsobligationsränta |
| `fx.json`, `download_failures.csv`, `download_summary.json`, `download.log` | Endast cache: dagens FX-kurser, fellogg, täckningssammanfattning |

* Cache: `<repo>/.cache/valuation_engine/` (styrs med `VALUATION_ENGINE_CACHE_DIR`). Fixtures:
  `tests/fixtures/` (styrs med `VALUATION_ENGINE_FIXTURES_DIR`).
* Alla objekt byggs från den råa JSON-filen, aldrig direkt från ett nätverkssvar. Därför är en
  offline-laddning identisk med den online-laddning som skapade filen.
* Cachen har ingen automatisk utgångstid. Använd `refresh=True` eller
  `python -m valuation_engine.data.download --all --refresh`.

**Upplösningsordning**

| Läge | Ordning |
|---|---|
| Offline (`offline=True` eller `VALUATION_ENGINE_OFFLINE=1`) | fixtures → cache → `DataUnavailableError`. Aldrig nätverk (nätverkslagret vägrar dessutom själv när miljövariabeln är satt). |
| Online | cache (hoppas över med `refresh=True`) → nätverk (skrivs till cache) → vid nätverksfel: cache → fixtures |

## 2. Universum och aktieslag

Källa: `yf.screen(EquityQuery('eq', ['region', 'se']))`, paginerat, sorterat på ticker.

| Steg | Regel | Antal |
|---|---|---|
| 0 | Alla symboler från screenern (samtliga på börs `STO`) | 984 |
| 1 | `quoteType == 'EQUITY'` (63 ETF:er/ETP:er bort) | 921 |
| 2 | Har bolagsnamn (`longName`), pris > 0 och är inte teckningsoption/‑rätt/BTU/tracker enligt symbolmönster (`-TO*`, `-TR`, `-UR`, `-BTU`, `-BTA`, `-XBT*`, `BULL-`, `BEAR-`, `MINI-` …) | 737 |
| 3 | Ett aktieslag per bolag | **687** |

* Yahoo märker warranter, minifutures och bull/bear-certifikat som `EQUITY`, men ger dem inget
  `longName`. Det är det filtret som tar bort de flesta (184 st).
* **Gruppering:** symboler med samma rot (`ATCO-A.ST`, `ATCO-B.ST` → `ATCO`) eller samma `longName`
  är samma bolag. I dagens data ger de två nycklarna exakt samma grupper (45 bolag har fler än ett
  noterat slag).
* **Val av primärt slag:** preferensaktier (`-PREF`) och D-aktier (`-D`) väljs aldrig. Bland övriga
  väljs slaget med högst genomsnittlig omsättning i kronor (3 månaders snittvolym × kurs).
  Exempel: `ATCO-A`, `VOLV-B`, `INVE-B`, `ERIC-B`, `SHB-A`, `SEB-A`, `INDU-C`, `STE-R`.
* Bolag där endast preferensaktien är noterad utesluts (1 st: `HEIM-PREF.ST`, Heimstaden).

### Börsvärde för bolag med flera aktieslag

Kontrollerat för ATCO, VOLV, INVE, ERIC och SHB: Yahoos `marketCap` är
`impliedSharesOutstanding` (alla slag) × **kursen för det slag man frågar på**, medan
`sharesOutstanding` bara avser det egna slaget.

| Ticker | Kurs | `sharesOutstanding` | `impliedSharesOutstanding` | `marketCap` (mdr) |
|---|---|---|---|---|
| ATCO-A | 192,25 | 3,32 mdr | 4,88 mdr | 938 |
| ATCO-B | 168,00 | 1,56 mdr | 4,88 mdr | 820 |
| SHB-A | 165,15 | 1,94 mdr | 2,00 mdr | 331 |
| SHB-B | 259,20 | 0,04 mdr | 1,98 mdr | 513 |
| ERIC-A / ERIC-B | 108,60 / 106,30 | 0,26 / 2,98 mdr | 3,24 mdr | 352 / 344 |
| INVE-A / INVE-B | 427,60 / 436,25 | 1,25 / 1,82 mdr | 3,06 mdr | 1 310 / 1 337 |
| VOLV-A / VOLV-B | 324,20 / 323,40 | 0,44 / 1,59 mdr | 2,03 mdr | 659 / 658 |

Slutsats: `marketCap` är bolagsövergripande, men alla aktier värderas till det valda slagets kurs.
Därför gäller `shares_outstanding = market_cap / price` (antal aktier uttryckt i primärslagets
ekvivalenter), och därför är det viktigt att primärslaget är det likvida (SHB-B hade gett 55 %
för högt börsvärde).

## 3. Mappningstabell

Varje standardpost har en ordnad fallback-kedja. Första kandidaten som ger ett värde **för ett
visst räkenskapsår** vinner för det året; senare kandidater fyller bara luckor. Vilken kandidat
som användes sparas i `FinancialStatements.source_map` med nycklarna `income.<post>`,
`balance.<post>`, `cash_flow.<post>` och `market.<post>` (posten `depreciation_amortization` finns
i två rapporter, därav prefixet).

Syntax i `source_map`: `is:` resultaträkning, `bs:` balansräkning, `cf:` kassaflöde, `std:` redan
standardiserad post, `-` framför = Yahoos tecken vänds, `?` efter = valfri term (saknas → 0),
`abs(...)` = absolutbelopp, `[2022, 2023]` = kandidaten användes bara dessa år, `missing` = ingen
kandidat gav något värde.

### Resultaträkning (`income`)

| Post | Yahoo-taggar i fallback-ordning |
|---|---|
| `revenue` | `Total Revenue` → `Operating Revenue` |
| `cogs` | `Cost Of Revenue` → `Reconciled Cost Of Revenue` → `revenue − Gross Profit` |
| `gross_profit` | `Gross Profit` → `revenue − cogs` |
| `depreciation_amortization` | abs(`Reconciled Depreciation`) → abs(cf `Depreciation And Amortization`) → abs(cf `Depreciation Amortization Depletion`) → abs(`Depreciation And Amortization In Income Statement`) → abs(`Depreciation Amortization Depletion Income Statement`) → `EBITDA − EBIT` |
| `ebit` | `EBIT` → `Operating Income` → `Total Operating Income As Reported` (ingen beräkning från resultat före skatt, se 10.3) |
| `ebitda` | `EBITDA` → `Normalized EBITDA` → `ebit + depreciation_amortization` → `Operating Income + depreciation_amortization`. Reparation: år där `EBITDA` = `EBIT` trots D&A > 0 ersätts med `ebit + D&A` |
| `interest_expense` | abs(`Interest Expense`) → abs(`Interest Expense Non Operating`) |
| `pretax_income` | `Pretax Income` → `Net Income Including Noncontrolling Interests + Tax Provision` |
| `income_tax` | `Tax Provision` (kan vara negativ = skatteintäkt) |
| `net_income` | `Net Income Common Stockholders` → `Net Income` → `Net Income From Continuing Operation Net Minority Interest` → `Net Income Including Noncontrolling Interests + Minority Interests?` |
| `diluted_shares` | `Diluted Average Shares` → `Basic Average Shares` → bs `Ordinary Shares Number` → bs `Share Issued` |

### Balansräkning (`balance`)

| Post | Yahoo-taggar i fallback-ordning |
|---|---|
| `cash` | `Cash Cash Equivalents And Short Term Investments` → `Cash And Cash Equivalents + Other Short Term Investments?` → `Cash Financial + Cash Equivalents?` → `Cash Cash Equivalents And Federal Funds Sold` → `Cash And Due From Banks` |
| `accounts_receivable` | `Accounts Receivable` → `Gross Accounts Receivable + Allowance For Doubtful Accounts Receivable?` → `Receivables` |
| `inventory` | `Inventory` |
| `accounts_payable` | `Accounts Payable` → `Payables` → `Payables And Accrued Expenses` |
| `total_current_assets` | `Current Assets` |
| `total_current_liabilities` | `Current Liabilities` |
| `net_ppe` | `Net PPE` → `Gross PPE + Accumulated Depreciation` |
| `goodwill_intangibles` | `Goodwill And Other Intangible Assets` → `Goodwill + Other Intangible Assets?` → `Other Intangible Assets` |
| `total_assets` | `Total Assets` |
| `total_debt` | `Total Debt` → `Long Term Debt And Capital Lease Obligation + Current Debt And Capital Lease Obligation?` → `Current Debt And Capital Lease Obligation` → `Long Term Debt + Current Debt? + Capital Lease Obligations?` → `Current Debt + Capital Lease Obligations?` → `Capital Lease Obligations` |
| `total_liabilities` | `Total Liabilities Net Minority Interest` → `Total Assets − Total Equity Gross Minority Interest` |
| `minority_interest` | `Minority Interest` → `Total Equity Gross Minority Interest − Stockholders Equity` |
| `total_equity` | `Stockholders Equity` → `Common Stock Equity` → `Total Equity Gross Minority Interest − Minority Interest?` |
| `net_working_capital` | `accounts_receivable + inventory? − accounts_payable` (kräver både kundfordringar och leverantörsskulder; saknat lager räknas som 0) |

### Kassaflöde (`cash_flow`)

| Post | Yahoo-taggar i fallback-ordning |
|---|---|
| `cash_from_operations` | `Operating Cash Flow` → `Cash Flow From Continuing Operating Activities` |
| `depreciation_amortization` | abs(`Depreciation And Amortization`) → abs(`Depreciation Amortization Depletion`) → `income.depreciation_amortization` |
| `change_in_nwc` | `−Change In Working Capital` → `−(Change In Receivables + Change In Inventory? + Change In Payables And Accrued Expense? + Change In Other Working Capital?)` → årsförändring i `balance.net_working_capital` (första året NaN) |
| `capex` | abs(`Capital Expenditure`) → abs(`Purchase Of PPE + Purchase Of Intangibles?`) → abs(`Capital Expenditure Reported`) → abs(`Net PPE Purchase And Sale + Net Intangibles Purchase And Sale?`) → abs(`Purchase Of Intangibles`) |
| `dividends_paid` | abs(`Cash Dividends Paid`) → abs(`Common Stock Dividend Paid + Preferred Stock Dividend Paid?`) |
| `free_cash_flow` | `cash_from_operations − capex` → `Free Cash Flow` (endast om capex saknas) |

### Marknadsdata (`MarketData`)

| Fält | Källa i fallback-ordning |
|---|---|
| `price` | `info.currentPrice` → `info.regularMarketPrice` → `info.previousClose` → senaste veckokurs |
| `market_cap` | `info.marketCap` → `diluted_shares × price` (Yahoo saknar börsvärde för några depåbevis/småbolag) |
| `shares_outstanding` | `market_cap / price` |
| `total_debt` | senaste balansräkning (kvartal om nyare än årsbokslut), samma kedja som ovan; förkastas om kvartalet bara innehåller leasing fast årsbokslutet har lån (se 10.7) → `info.totalDebt` → senaste årsbokslut → 0 |
| `cash` | senaste balansräkning → `info.totalCash` → senaste årsbokslut → 0 |
| `minority_interest` | senaste balansräkning → senaste årsbokslut → 0 (negativa värden sätts till 0) |
| `preferred_equity` | senaste balansräkning: `Preferred Stock Equity` → `Preferred Stock` → 0 |
| `beta` | `info.beta` (endast information) |
| `as_of` | datum för `info.regularMarketTime`, annars hämtningsdatum |

## 4. Teckenkonventioner

| Post | Yahoo | Standardiserat |
|---|---|---|
| Kostnader, räntekostnad, D&A | positiva (i regel) | positiva; `abs()` på D&A och räntekostnad |
| `income_tax` | positiv kostnad | oförändrad, negativ = skatteintäkt |
| `capex` | negativ (utflöde) | **positiv** |
| `dividends_paid` | negativ (utflöde) | **positiv** |
| `change_in_nwc` | `Change In Working Capital` > 0 betyder kassainflöde (rörelsekapitalet minskar) | **omvänt tecken**: positivt när rörelsekapitalet växer (utflöde) |
| `free_cash_flow` | – | `cash_from_operations − capex` (identitet per konstruktion) |

Räkenskapsår = kalenderåret för periodens slutdatum (H&M med bokslut 30 november 2025 → 2025;
brutet år som slutar 31 mars 2026 → 2026). Slutar två perioder samma kalenderår (bytt
räkenskapsår) behålls den senare. Ett år behålls bara om resultaträkningen har minst två av
`revenue`, `pretax_income` och `net_income`; alla tre rapporter får samma årsindex.
Yahoo levererar ofta en femte, nästan tom kolumn – den faller bort här.

## 5. Valuta

Alla belopp för ett bolag uttrycks i **handelsvalutan** (`info.currency`). Om
`info.financialCurrency` avviker räknas rapporterna om med Riksbankens kurser
(`data/riksbank.py`, `data/fx.py`), på samma sätt som en koncernredovisning räknas om:

| Vad | Kurs | Riksbanksdata |
|---|---|---|
| Resultaträkning och kassaflöde | **Genomsnittskurs för räkenskapsåret** | observationsviktat snitt av månadssnitten för de 12 kalendermånader som slutar med bokslutsmånaden |
| Balansräkning (årsbokslut) | **Balansdagskurs** | sista fixing i bokslutsmånaden ("ultimo") |
| Marknadsögonblicket (`MarketData`: skuld, kassa, minoritet från senaste kvartal) | **Senaste avistakurs** | senaste fixing (`Observations/Latest`) |

* Serierna heter `SEK<VALUTA>PMI` (SEK per en enhet utländsk valuta), t.ex. `SEKUSDPMI`,
  `SEKEURPMI`. Alla sju valutor i universumet finns: EUR, USD, CAD, DKK, ISK, PLN, GBP.
  Korskurser (rapportvaluta → handelsvaluta när ingen av dem är SEK) räknas som kvoten mellan
  två serier.
* `diluted_shares` räknas inte om.
* `CompanyProfile.fx_rate` är **avistakursen som använts på marknadsögonblicket**, inte en
  kurs för rapporterna. Kurserna per år står i `source_map`:
  `fx.method`, `fx.spot`, `fx.average.<år>` och `fx.year_end.<år>`.
* **Brutna räkenskapsår** hanteras utan förenkling på årsnivå: snittet tas över de 12 månader
  som slutar med bokslutsmånaden (bokslut 31 mars 2026 → april 2025–mars 2026).
  **Förenklingar:** (1) räkenskapsåret antas alltid vara 12 månader (fel för förlängda eller
  förkortade år); (2) upplösningen är hela kalendermånader, så ett bokslut mitt i en månad får
  månadens sista fixing; (3) poster som är härledda mellan rapporter, som `free_cash_flow`, räknas
  om efter beräkningen i rapportvalutan, så identiteterna håller.
* **Avistakursens datum:** Riksbankens senaste fixing används om den ligger högst 10 dagar från
  den dag Yahoo-datat hämtades (kurs och börsvärde). Annars används Yahoos avistakurs som sparats
  i bolagets råfil, så att kurs och växelkurs hör till samma tidpunkt.
* **Fallback:** saknas valutan hos Riksbanken, går Riksbanken inte att nå (och varken cache eller
  fixtures har serien), eller saknas kurs för något räkenskapsår, används det gamla beteendet:
  Yahoos avistakurs (`USDSEK=X` o.s.v., sparad i råfilen) för alla år. Vägen som togs står i
  `source_map["fx.method"]`. I körningen 2026-11-16 fick alla 62 bolag som behöver omräkning
  Riksbankskurser per år; ingen föll tillbaka.
* Saknas `financialCurrency` antas den vara lika med handelsvalutan.
* `build_company(raw)` utan `fx_plan` använder alltid Yahoos avistakurs (bakåtkompatibelt);
  `load_company` använder Riksbanken.

**Peer-tabellen (SEK)** för bolag som handlas i annan valuta (`VSURE.ST`, EUR): marknadsvärden
(börsvärde, EV, skuld) räknas om med avistakursen och resultatposter med genomsnittskursen för
senaste räkenskapsår. Kolumnerna `fx_to_sek`, `fx_to_sek_average` och `fx_source` i CSV-filen
visar vad som använts.

**Effekt av bytet från en avistakurs till kurser per år** (fixturebolagen, mdr SEK):

| Bolag | Post | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| `ALIV-SDB.ST` (USD) | omsättning, gammal → ny | 89,1 → 89,5 | 105,5 → 111,2 | 104,7 → 109,7 | 108,9 → 106,2 |
| `ABB.ST` (USD) | omsättning, gammal → ny | 296,6 → 298,1 | 324,7 → 342,1 | 308,1 → 323,0 | 334,6 → 326,2 |
| `HEXA-B.ST` (EUR) | omsättning, gammal → ny | 57,3 → 54,9 | 60,4 → 62,4 | 60,0 → 61,8 | 60,2 → 60,0 |
| USD-bolag | förändring omsättning / skuld | +0,5 % / +3,6 % | +5,4 % / −0,3 % | +4,8 % / +9,2 % | −2,5 % / −8,7 % |

SEK-rapporterande bolag påverkas inte alls (bit-identiska före och efter).

### Riksbankens API

* Bas: `https://api.riksbank.se/swea/v1/`. Använda anrop: `Observations/Latest/{serie}` och
  `ObservationAggregates/{serie}/M/{från}/{till}` (endast avslutade månader returneras).
* **Hastighetsbegränsning:** utan API-nyckel svarar tjänsten HTTP 429 efter ungefär fem snabba
  anrop. Klienten väntar därför 13 sekunder mellan anrop
  (`VALUATION_ENGINE_RIKSBANK_THROTTLE_SECONDS`), backar exponentiellt vid 429 och gör högst
  fyra försök. Varje valuta kostar två anrop och hämtas en gång och sparas i `riksbank.json`.
* Färskhet online: en valutaserie i cachen återanvänds om den hämtades samma dag som bolagets
  Yahoo-data eller senare; räntan återanvänds om den hämtades i dag. `refresh=True` tvingar ny
  hämtning. Offline: fixtures → cache.
* Ett bolag som både rapporterar och handlas i SEK orsakar aldrig något Riksbanksanrop.

## 6. Kurshistorik och jämförelseindex

* 5 år, veckodata, justerade stängningskurser (`auto_adjust=True`) för aktien och index.
* `^OMXSPI` (OMX Stockholm PI) **finns** på Yahoo med 261 veckoobservationer och används. Fallback
  `^OMX` (OMXS30) är implementerad men behövdes inte. Faktiskt använt index står i
  `PriceHistory.benchmark`.
* Serierna nycklas på veckans måndag, inner-joinas och rader med saknade värden tas bort. Den
  vecka som pågick vid hämtningen tas bort. Färre än 52 gemensamma veckor → `price_history=None`.
* Båda serierna är prisindex/justerade kurser; `^OMXSPI` är ett prisindex (utan utdelningar) medan
  aktiekursen är utdelningsjusterad. Effekten på beta är försumbar.

## 7. Peer-tabellen (`PeerSnapshot`)

| Fält | Definition |
|---|---|
| `market_cap`, `enterprise_value`, `total_debt` | från `MarketData`, i SEK |
| `revenue`, `ebitda`, `ebit`, `net_income` | **senaste räkenskapsår** (inte rullande tolv månader), i SEK; saknas → `None` |
| `revenue_growth` | senaste år / föregående år − 1 (kräver två år i följd och positiv bas) |
| `ebitda_margin` | `ebitda / revenue` om omsättningen är positiv |
| `return_on_equity` | `net_income` / genomsnitt av ingående och utgående eget kapital (endast om > 0) |
| omräkning till SEK | marknadsvärden med avistakurs, resultatposter med årets genomsnittskurs (avsnitt 5) |
| `beta` | egen rå OLS-beta på veckoavkastning mot `^OMXSPI`; om kurshistorik saknas: Yahoos `beta` |

CSV-filen har dessutom informationskolumnerna `beta_source`, `fiscal_year`, `currency`,
`reporting_currency`, `fx_to_sek`, `as_of`, `fx_to_sek_average` och `fx_source`, som inte ingår
i `PeerSnapshot`.

Observera att EV bygger på senaste balansräkning (oftast Q3 2026) medan EBITDA avser senaste
helår (oftast 2025). Multiplarna är alltså "aktuellt EV / senaste helårs-EBITDA".

## 8. Marknadsparametrar

`load_market_parameters(*, offline=False, refresh=False) -> MarketParameters` och
`market_parameter_sources(...) -> dict[str, ParameterSource]` (värde, källa, URL, datum och
ursprung för varje parameter, avsett för Excel-exporten och rapporterna).

| Parameter | Värde 2026-11-16 | Källa | Fallback |
|---|---|---|---|
| `risk_free_rate` | 3,075 % (fixing 2026-11-13) | Riksbanken, 10-årig svensk statsobligation, serie `SEGVB10YC` | cache → nätverk → fixtures → standardvärdet i `MarketParameters` |
| `equity_risk_premium` | 4,23 % | Damodaran, Sverige (Aaa), total aktieriskpremie, 2026-01-01 | standardvärdet |
| `tax_rate` | 20,6 % | Damodaran, Sverige, bolagsskatt, 2026-01-01 | standardvärdet |
| `benchmark`, `long_term_growth` | `^OMXSPI`, 2 % | projektets standardvärden | – |
| `country_risk_premium` (endast i källistan) | 0 % | Damodaran; ingår redan i aktieriskpremien | – |

Damodarans siffror ligger i den incheckade filen
`src/valuation_engine/data/reference/market_reference.json` med källa
(`https://pages.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xlsx`), datum och en instruktion
för hur filen uppdateras. Excel-filen laddas aldrig ned eller tolkas vid körning. Siffrorna är
kontrollerade mot Damodarans HTML-version av tabellen 2026-11-16 (Sverige: Aaa, 0,00 %, 4,23 %,
20,60 %). Funktionerna kastar aldrig fel för saknad data.

## 9. Täckning (körning 2026-11-16)

Kommando: `python -m valuation_engine.data.download --all` (ca 18 minuter, 1,6 s per ticker,
ingen rate limiting från Yahoo under körningen).

| Mått | Antal |
|---|---|
| Symboler från screenern | 984 |
| `quoteType == EQUITY` | 921 |
| Stamaktier efter filter (namn, pris, symbolmönster) | 737 |
| Bolag efter deduplicering av aktieslag (= `list_universe()`) | 687 |
| Rådata nedladdad | 687 |
| **Laddade fullt ut (`CompanyData` + rad i peer-tabellen)** | **680** |
| Går inte att bygga: Yahoo saknar årsresultaträkning | 7 |
| Finans/fastighet (`is_financial`) | 103 (53 Financial Services, 50 Real Estate) |
| Saknar EBITDA senaste år | 26 (samtliga finans: banker, investmentbolag, kapitalförvaltare) |
| EBITDA ≤ 0 senaste år | 208 (190 icke-finansiella, 18 finans/fastighet) |
| Icke-finansiella med EBITDA > 0 (användbara för EV/EBITDA) | 387 av 577 |
| Rapporterar i annan valuta än SEK | 63 (EUR 38, USD 16, CAD 3, DKK 3, ISK 1, PLN 1, GBP 1) |
| … varav omräknade med Riksbankskurser per år | 62 (den 63:e, `VSURE.ST`, rapporterar i sin handelsvaluta) |
| … varav fallback till Yahoos avistakurs | 0 |
| Handlas i annan valuta än SEK | 1 (`VSURE.ST`, EUR) |
| Saknar kurshistorik ≥ 52 veckor (`price_history=None`) | 18 |
| Saknar beta i peer-tabellen | 17 |
| Färre än 4 räkenskapsår | 23 (1 år: 1, 2 år: 4, 3 år: 18) |
| Senaste räkenskapsår äldre än 2025 | 4 (`NOTEK`, `OCTV-SDB`, `SONE`: 2024; `PADEL`: 2022) |

De 7 som inte går att bygga (kontrollerat med två hämtningar): `ARENIT-SDB.ST`, `ATIN.ST`,
`IMP-A-SDB.ST`, `KDV-B.ST`, `NORDVI.ST`, `PJ.ST`, `UNIBAP.ST`. Yahoo returnerar tomma
årsrapporter för dem. Rådata ligger kvar i cachen och felen står i `download_failures.csv`.
De finns med i `list_universe()` men inte i peer-tabellen, och `load_company` ger
`DataUnavailableError`.

Sektorfördelning (680): Industrials 151, Healthcare 126, Technology 121, Consumer Cyclical 68,
Financial Services 53, Real Estate 50, Communication Services 42, Basic Materials 34,
Consumer Defensive 25, Utilities 6, Energy 4. Median EV/EBITDA bland icke-finansiella bolag med
positiv EBITDA är 10,5x (10:e percentilen 4,8x, 90:e 30,5x). 119 bolag har börsvärde under
100 Mkr.

### Fixtures

`tests/fixtures/` (ca 1 MB) är en fryst kopia av cachen: hela `universe.json`, hela
`peer_snapshots.csv`, `benchmark.json`, `riksbank.json` (alla sju valutor från 2019 samt
statsobligationsräntan) och rådata för 15 bolag: `ATCO-A.ST` (case), peers
`SAND.ST`, `ALFA.ST`, `SKF-B.ST`, `VOLV-B.ST`, `EPI-A.ST`, `TREL-B.ST`, `ASSA-B.ST`, `NIBE-B.ST`,
`INDT.ST`, EUR-rapporterande `HEXA-B.ST`, USD-rapporterande `ALIV-SDB.ST` och `ABB.ST`, banken
`SEB-A.ST` samt `VSURE.ST` (handlas i EUR, bara två räkenskapsår, 58 veckors kurshistorik).
Skapas med `python -m valuation_engine.data.download --freeze <tickers>`.

Verifierat: med `VALUATION_ENGINE_OFFLINE=1`, tom cachekatalog och blockerade sockets fungerar
`list_universe()`, `load_peer_snapshots()`, `load_market_parameters()` och `load_company()`
för alla 15 bolag (även med `python -W error`), och
`yfinance` importeras aldrig. En färsk online-laddning och efterföljande offline-laddning av
samma fil ger exakt lika `CompanyData` (kontrollerat med `check_exact=True`).

## 10. Observerade datakvalitetsproblem

Allt nedan är observerat i körningen, med antal bolag av de 680 laddade.

### Kontroll av Atlas Copco (`ATCO-A.ST`, räkenskapsår 2025)

| Identitet | Utfall |
|---|---|
| `gross_profit = revenue − cogs` | 72,21 = 168,34 − 96,13 mdr ✓ |
| `ebitda = ebit + D&A` | 44,08 = 34,55 + 9,53 mdr ✓ |
| `free_cash_flow = cfo − capex` | 26,38 = 32,57 − 6,19 mdr ✓ (lika med Yahoos `Free Cash Flow`) |
| `capex` = PPE + immateriella | 6,187 = 4,284 + 1,903 mdr ✓ (ingen dubbelräkning) |
| `total_debt` inkl. leasing | 34,83 = 22,75 + 4,70 + 7,38 mdr ✓ |
| EV mot Yahoos `enterpriseValue` | 950,9 mot 951,7 mdr (−0,1 %) ✓ |
| `change_in_nwc` mot förändring i balansens rörelsekapital | −1,6 mot −5,9 mdr ✗ – se punkt 6 |

### Problem och hur de hanteras

1. **Yahoos `EBIT` är inte rörelseresultatet.** `EBIT` = resultat före skatt + räntekostnad och
   innehåller därmed ränteintäkter, övriga finansiella poster och engångsposter. ATCO 2025:
   `EBIT` 34,55, `Operating Income` 34,81, rapporterat rörelseresultat 34,11 mdr. Vi använder
   `EBIT` först så att `ebitda = ebit + D&A` håller. För fastighetsbolag ingår
   värdeförändringar (Balder: `EBIT` 14,3 mot `Operating Income` 9,2 mdr), för Embracer
   nedskrivningar (−6,8 mot +0,06 mdr).
2. **Yahoos `EBITDA` är ibland fel.** I 14 bolag är `EBITDA` exakt lika med `EBIT` ett enstaka år
   trots redovisade avskrivningar (t.ex. `XANO-B` 2024, `TOBII` 2022). De åren ersätts med
   `ebit + D&A` och det syns i `source_map`. Därutöver avviker `EBITDA` mer än 1 % från
   `ebit + D&A` minst ett år i 53 bolag, oftast små bolag med nedskrivningar som Yahoo räknar
   bort (t.ex. `POLYG` 2023: `EBITDA` −13 Mkr, `EBIT` −313 Mkr, D&A 5 Mkr). Dessa lämnas orörda.
3. **Banker och investmentbolag saknar `EBIT`/`EBITDA`** (26 bolag) → NaN. Ingen
   ersättningsberäkning görs eftersom resultat före skatt + räntekostnad är meningslöst för en
   bank. Fem investmentbolag har negativ "omsättning" (`FIRST-B`, `KINV-B`, `NAXS`, `VEFAB`,
   `VNV`) eftersom Yahoo bokför värdeförändringar som intäkt.
4. **`Capital Expenditure` innehåller redan immateriella investeringar.** Av 444 bolag där både
   `Purchase Of PPE` och `Purchase Of Intangibles` finns stämmer
   `Capital Expenditure = PPE + immateriella` i 405. I resten är skillnaden oftast obetydlig eller
   så upprepar Yahoo samma belopp på båda raderna. Några få har väsentlig avvikelse där
   `Capital Expenditure` bara motsvarar PPE (`BILI-A` 2023, `CASE` 2023, `ABB` 2022). Vi lägger
   **aldrig** immateriella ovanpå `Capital Expenditure`: risken för dubbelräkning bedömdes som
   större än risken att missa dem. 32 bolag saknar capex helt alla år; då används Yahoos
   `Free Cash Flow` om det finns.
5. **`Total Debt` innehåller leasingskulder** (448 av 471 bolag med leasingrad stämmer exakt mot
   lån + leasing; resten är etikettfel hos Yahoo där summan ändå stämmer). 47 bolag saknar
   skuldrad helt (oftast skuldfria småbolag) → NaN i balansräkningen, 0 i `MarketData`.
6. **`Change In Working Capital` är bredare än operativt rörelsekapital.** Den innehåller övriga
   kortfristiga skulder m.m. och påverkas inte av valutaomräkning och förvärv, så den stämmer
   inte mot årsförändringen i `net_working_capital` (kundfordringar + lager − leverantörsskulder).
   En DCF bör därför inte blanda de två måtten.
7. **Glesa kvartalsbalansräkningar.** `MarketData` läser senaste balansräkning, normalt Q3 2026
   (339 bolag) eller Q2 2026 (162). För vissa bolag saknar Yahoos kvartalsdata lånerader, så
   att `Total Debt` bara är leasing (`ARJO-B`: 1,3 mot verkliga ca 5,4 mdr; `HOLM-B`: 0,2 mot
   ca 6,3 mdr). Det upptäcks (lån finns i årsbokslutet men kvartalets `Total Debt` ≤ leasing) och
   då används `info.totalDebt`. Totalt kommer skulden från `info.totalDebt` i 149 bolag och från
   årsbokslutet i 4. Kvarstående risk: i 70 bolag avviker vår skuld mer än 5 % från
   `info.totalDebt`, åt båda håll (`LOOMIS`: 17,6 mot 22,6 mdr), utan att det går att avgöra
   vilken som är rätt. Kassan avviker mot `info.totalCash` i ca 130 bolag; vi använder
   balansräkningens rad.
8. **Yahoos `enterpriseValue` är fel för bolag som rapporterar i annan valuta.** Yahoo adderar
   börsvärde i SEK och nettoskuld i rapportvalutan utan omräkning (`8TRA`: 190,5 + 29,3 − 3,0 =
   216,9 "mdr", där skulden är i EUR). Vårt EV räknar om först. Av 577 icke-finansiella bolag
   avviker vårt EV mer än 10 % från Yahoos i 99: 30 rapporterar i annan valuta, övriga 69 är
   främst småbolag där kassa eller skuld skiljer mellan `info` och balansräkningen. 6 bolag har
   negativt EV (kassa större än börsvärde).
9. **`diluted_shares` är opålitligt för småbolag.** Det är ett vägt genomsnitt och Yahoo justerar
   inte historiken konsekvent för splitar, omvända splitar och emissioner. I 83 bolag avviker
   `market_cap / price` mer än ca 30–40 % från senaste `diluted_shares`, i 31 med mer än en
   faktor 2,5 (`PADEL` 292x, `ATORX` 88x, `LXB` 0,02x). `DUNI.ST` har ett rent enhetsfel 2025
   (46 999 032 000 i stället för 46 999 032). Använd `MarketData.shares_outstanding` för värde
   per aktie, inte `diluted_shares`.
10. **Börsvärde saknas för 6 symboler** (`marketCap` är `None`). Tre av dem (`NTECH`, `OCTV-SDB`,
    `TANGEN-B`) får börsvärde = `diluted_shares × price`, vilket är osäkert för depåbevis; övriga
    tre saknar även rapporter.
11. **`financialCurrency` saknas för 10 bolag** → antas vara handelsvalutan (SEK).
12. **FX:** `ISKSEK=X` och `PLNSEK=X` finns inte på Yahoo (Yahoos avistakurs räknas då som
    korskurs via USD). Det spelar bara roll för fallbacken: Riksbanken har båda valutorna.
    Yahoos avistakurs är numera valfri vid nedladdning och fäller inte en ticker.
13. **Fyra hela år är det normala.** 657 bolag har exakt 4 räkenskapsår. Yahoos femte kolumn
    innehåller oftast ett enda strövärde och tas bort. Två bolag har lucka i årsserien
    (`GIG-SDB`, `SPRINT`); `revenue_growth` beräknas då inte.
14. **D&A:** resultaträkningens och kassaflödets D&A skiljer sig i 12 bolag. 5 bolag saknar D&A.
    `dividends_paid` saknas (NaN, inte 0) i 320 bolag som inte delat ut.
15. **Bruttoresultat:** 51 bolag (främst finans och tjänstebolag) saknar `cogs`/`gross_profit`.
    I 8 bolag stämmer inte `gross_profit = revenue − cogs` på grund av inkonsekvent Yahoo-data.
16. **Beta i peer-tabellen** är en rå regression och blir orimlig för illikvida aktier och aktier
    med ojusterade kurshopp: 21 bolag har beta under 0 eller över 3 (`FRAG.ST` 94,7, `WYLD.ST`
    19,4). Median 0,63. Comps-modellen bör filtrera eller winsorisera.
17. **Sektor** kommer från Yahoo (GICS-liknande). `is_financial` omfattar därför även t.ex.
    kapitalförvaltare och kreditbolag. Investmentbolag ligger under Financial Services.
18. **Tidpunkter blandas:** kurs och börsvärde är per 2026-11-13, balansposter i EV per senaste
    kvartal, resultatposter per senaste helår (2025 för 644 bolag, brutet 2026 för 32).
