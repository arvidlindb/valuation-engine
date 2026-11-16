# Design: valuation-engine

Det här dokumentet beskriver hur värderingsmotorn är uppbyggd, vilka val som gjorts och varför. Det är skrivet för den som vill förstå eller ändra koden. Vill du bara köra den räcker [README.md](README.md).

## Mål

Ge motorn en svensk ticker och få en värdering i den stil en investmentbank gör: flera metoder sida vid sida, ett intervall per metod och en ärlig redovisning av var metoderna är svaga. Det ska fungera för alla svenska bolag på Yahoo Finance, inte bara för ett utvalt bolag.

## Avgränsningar och grundval

| Val | Skäl |
|---|---|
| Bara svenska bolag, data från Yahoo Finance | Gratis och täcker hela Nasdaq Stockholm inklusive First North. |
| Riskfri ränta från Riksbanken, riskpremie och skatt från Damodaran | Källbelagda värden i stället för gissningar. Se "Marknadsparametrar". |
| Allt räknas om till handelsvalutan (normalt SEK) i datalagret | Modellerna ska aldrig behöva tänka på valuta. |
| Ett aktieslag per bolag | Annars räknas Atlas Copco A och B som två bolag. Det mest omsatta slaget väljs. |
| Antal aktier är börsvärde delat med kurs | Yahoos börsvärde gäller hela bolaget, medan antalet aktier bara gäller ett aktieslag. |
| Banker, försäkring, investmentbolag och fastighet får bara P/E | En kassaflödesvärdering på rörelsenivå och en LBO är inte meningsfulla för dem. |
| Case: Atlas Copco (`ATCO-A.ST`) | Stort, icke-finansiellt, stabil historik och tydliga svenska jämförelsebolag. |

## Arkitektur

```
Yahoo Finance, Riksbanken, referensfil, fixtures
        |
     data/            hämtning, cache, standardisering, valuta
        |
     core/types.py    gemensamma datatyper
        |
     pipeline.py      comps -> DCF -> LBO -> Monte Carlo
        |
  viz/  export/  app/  batch.py
```

Varje lager känner bara till lagret under sig. Modellerna importerar inte varandra, med ett undantag: Monte Carlo bygger på DCF-modellens kärna. Det är pipelinen som kopplar ihop dem.

## Gemensamt kontrakt

`src/valuation_engine/core/types.py` innehåller de typer som alla moduler delar: `CompanyData` (profil, rapporter, marknadsdata, kurshistorik), `PeerSnapshot`, `MarketParameters`, `DCFAssumptions` och `ValuationResult` med `low`, `base` och `high` per aktie.

Konventioner:
- Belopp är absoluta tal, inte tusental eller miljoner.
- Rapporterna är tabeller med räkenskapsår som index och standardiserade kolumnnamn. En post som saknas är `NaN`, aldrig en saknad kolumn.
- Kostnader, investeringar, avskrivningar, räntor och skatt lagras som positiva tal.
- Räntor och marginaler är decimaltal (0,08 betyder 8 %).

Publika funktioner:

```python
# data
load_company(ticker, *, offline=False, refresh=False) -> CompanyData
list_universe(*, offline=False, refresh=False) -> list[str]
load_peer_snapshots(*, offline=False, refresh=False) -> list[PeerSnapshot]
load_market_parameters(*, offline=False, refresh=False) -> MarketParameters

# modeller, var och en med .to_valuation_results()
dcf.run_dcf(company, assumptions=None, params=..., peers=None, ...) -> DCFResult
comps.run_comps(company, universe, *, config=None) -> CompsResult
lbo.run_lbo(company, assumptions=None, params=None) -> LBOResult
monte_carlo.run_monte_carlo(company, assumptions=None, config=None, ...) -> MonteCarloResult

# helheten
pipeline.run_valuation(ticker, ...) -> ValuationReport
```

Kommandoraden:

```
python -m valuation_engine ATCO-A.ST     en ticker
python -m valuation_engine --all         hela universumet
```

`offline=True` eller miljövariabeln `VALUATION_ENGINE_OFFLINE=1` gör att bara fixtures och lokal cache läses.

### MonteCarloResult

Fälten som diagram och export bygger på: `ticker`, `currency`, `price`, `base_value`, `n_simulations`, `values` och `values_exit` (värde per aktie per simulering), `inputs` (dragna drivare), `percentiles`, `mean`, `std`, `standard_error`, `prob_above_price`, `tornado`, `variance_contribution`, `convergence` och `config`.

## Data

Mappningen från Yahoos poster till de standardiserade står i [docs/data_mapping.md](docs/data_mapping.md). Det viktigaste:

- **Universum:** Yahoos sökning ger 984 svenska symboler. Efter att fonder, warranter och dubbla aktieslag rensats bort återstår 687 bolag. 680 går att ladda; de övriga 7 saknar årsrapporter hos Yahoo.
- **Reservkedjor:** varje post har en ordnad lista av källor. Vilken som användes sparas i `source_map`.
- **Valuta:** 63 bolag rapporterar i annan valuta än SEK. Resultat och kassaflöde räknas om till räkenskapsårets genomsnittskurs, balansräkningen till årets slutkurs och marknadsdata till dagskurs, allt från Riksbanken.
- **Cache och fixtures:** rådata sparas lokalt som JSON. Fixtures i `tests/fixtures/` är en fryst cache för 15 bolag plus hela universumlistan och jämförelsetabellen, så att tester och demo går utan nätverk.

Kända problem i Yahoos data som modellerna måste tåla:
- Antal aktier i resultaträkningen är opålitligt för småbolag. Därför används alltid börsvärde delat med kurs.
- Yahoos förändring i rörelsekapital är bredare än operativt rörelsekapital. DCF räknar därför rörelsekapitalet från balansräkningen.
- Skulden skiljer sig mer än 5 % mellan två av Yahoos fält för ett sjuttiotal bolag.
- Investeringar saknas för 32 bolag.
- Datumen är blandade: kurs från i dag, balansposter från senaste kvartal, resultat från senaste räkenskapsår.

## Marknadsparametrar

| Parameter | Värde | Källa |
|---|---|---|
| Riskfri ränta | 3,075 % | Riksbanken, 10-årig statsobligation, 2026-11-13 |
| Riskpremie | 4,23 % | Damodaran, Sverige, januari 2026 |
| Bolagsskatt | 20,6 % | Damodaran, januari 2026 |
| Index för beta | `^OMXSPI` | OMX Stockholm PI på Yahoo |
| Evig tillväxt | 2 % | Riksbankens inflationsmål, eget val |

Räntan hämtas på nytt vid varje körning mot nätet. Riskpremie och skatt ligger i en daterad referensfil som uppdateras för hand. Excel-exporten och terminalutskriften visar källa och datum för varje värde.

## Modellerna

### DCF

- Fem prognosår. Tillväxten startar på den historiska genomsnittliga årstakten (begränsad till mellan minus 5 och plus 15 %) och faller linjärt till den eviga tillväxten.
- Marginal, avskrivningar, investeringar och rörelsekapital är treårssnitt av historiken.
- Avkastningskravet på eget kapital är CAPM. Beta är snittet av bolagets eget (regression mot index, Blume-justerat) och jämförelsebolagens (avskuldsatt och återskuldsatt enligt Hamada).
- Kassaflödena diskonteras mitt i året. Gordon-slutvärdet får samma halvårsjustering, exit-slutvärdet får det inte, eftersom det är en försäljning vid årets slut.
- Två slutvärden räknas alltid, och varje metods underförstådda multipel respektive tillväxt redovisas så att de kan stämmas av mot varandra.

Två avsteg från läroboken:
- **Storlekspremie.** Små och illikvida aktier får för låga betavärden i regressionen. Utan korrigering blev medianen för WACC 6,3 % och småbolagen systematiskt övervärderade. Därför läggs en premie på avkastningskravet: 0 % över 20 mdkr, stigande till 3 % under 200 mkr. Nivåerna är ett eget antagande utan källa. Stängs av med `--no-size-premium`.
- **Golv för investeringar i slutåret.** I slutåret är investeringarna minst lika stora som avskrivningarna. Annars skulle bolaget krympa sin anläggningsbas i all evighet. För Atlas Copco kostar golvet ungefär 9 kr per aktie.

DCF räknas inte för finansiella bolag, bolag med negativ EBITDA eller bolag där kassaflödet i slutåret inte är positivt.

### Comps

- Kandidaterna kommer från samma sektor. De klustras med k-means på storlek, tillväxt, marginal, avkastning på eget kapital, beta och skuldsättning. Antalet kluster väljs med silhuettvärde.
- Inom bolagets kluster väljs de högst 12 närmaste bolagen, med ett avståndstillägg för annan bransch.
- Lågt, bas och högt värde är 25:e percentilen, medianen och 75:e percentilen av jämförelsebolagens multipel.

Ärligt om klustringen: silhuettvärdet väljer k = 2 för ungefär halva universumet, och då skiljer klustringen mest friska bolag från krisbolag. Det som i praktiken väljer jämförelsebolagen är rangordningen efter avstånd. Branschtillägget är inställt genom att titta på Atlas Copco och är bara stickprovat i övrigt.

### LBO

- Köp till kurs plus 30 % premie. Seniort lån på 4,0 gånger EBITDA och efterställt på 1,5 gånger, med tak för räntetäckning, skuldtjänst och lägsta andel eget kapital.
- Obligatorisk amortering, cash sweep och en checkkredit som tar tillfälliga underskott.
- Försäljning efter fem år. IRR och MOIC, samt en uppdelning av värdeskapandet på resultattillväxt, multipelförändring och amortering.
- Det som går in i football field är betalningsförmågan: vilket pris en köpare kan betala i dag och ändå nå 25, 20 respektive 15 % årlig avkastning.

### Monte Carlo

- Fyra slumpmässiga drivare: omsättningstillväxt, marginal, WACC och evig tillväxt. Varje dragning är en parallellförskjutning av basfallets bana.
- Normal, trunkerad normal eller triangulär fördelning per drivare. Korrelation mellan tillväxt och marginal (förval 0,3) via Cholesky och en gaussisk copula.
- Alla simuleringar går genom samma kärnfunktion som bas-DCF i ett enda vektoriserat anrop. Med noll varians ger varje simulering exakt basvärdet.
- Football field visar medianen och P10 till P90. Medelvärdet ligger några procent över basfallet, eftersom Gordons formel är konvex i WACC och tillväxt.

## Hur pipelinen kopplar ihop metoderna

Metoderna ska beskriva samma bolag. Därför:

1. Comps körs först och väljer jämförelsebolagen.
2. DCF får samma jämförelsebolag för sitt beta. Exit-multipeln är comps-medianen för EV/EBITDA, om den vilar på minst fem bolag, och kapas vid högst 20 gånger.
3. LBO får DCF-modellens rörelseprognos och samma exit-multipel.
4. Monte Carlo bygger på DCF-modellens basfall.

Priset för den här konsekvensen är att exit-DCF och LBO ärver marknadens värdering via comps. De är alltså inga oberoende metoder, och det syns tydligt för Atlas Copco, där Gordon-metoden och exit-metoden hamnar långt ifrån varandra.

En metod som inte går att tillämpa ger en förklaring i rapporten i stället för ett fel. Ett oväntat fel i en metod stoppar inte de andra.

## Tester

653 tester som går helt offline mot fixtures, med en spärr som fäller testet om något försöker nå nätverket. De täcker:

- Analytiska facit: DCF med konstant tillväxt ska ge Gordons formel, en LBO utan skuld ska ge en IRR som går att räkna för hand.
- Invarianter: sources är lika med uses, skulden blir aldrig negativ, Monte Carlo med noll varians ger bas-DCF.
- Hela kedjan: pipeline, kommandorad, Excel-export, diagram och app.

Täckningen för `models/` är 96 till 98 %.

## Resultat

Atlas Copco (kurs 192,25 kr, kr per aktie):

| Metod | Låg | Bas | Hög |
|---|---|---|---|
| DCF Gordon | 93,73 | 115,85 | 153,74 |
| DCF exit-multipel (19,43x) | 160,55 | 175,23 | 191,04 |
| Comps EV/EBITDA | 152,81 | 172,93 | 190,71 |
| Comps EV/Sales | 111,67 | 138,62 | 168,80 |
| Comps P/E | 162,82 | 178,33 | 196,54 |
| LBO | 104,38 | 117,93 | 135,34 |
| Monte Carlo P10/P50/P90 | 84,29 | 116,74 | 164,92 |

WACC är 7,65 % med beta 1,13. Hela resonemanget finns i [docs/case_report.md](docs/case_report.md).

Hela universumet: 294 bolag får alla fyra metoder, 386 får några av dem och 7 saknar data. 296 bolag får en DCF.

## Kända svagheter

- **Yahoos data:** fyra års historik, senaste bokslut mot dagens kurs, ingen stubbperiod.
- **Leasing:** leasingbetalningar syns varken i EBITDA eller i investeringarna, så kassaflödena överskattas för leasingtunga bolag i både DCF och LBO.
- **Storleksmönster:** storbolag värderas systematiskt lågt i DCF och småbolag högt. Bara ungefär vart fjärde DCF-värde hamnar inom 25 % från kursen.
- **Valuta i tillväxten:** för bolag som rapporterar i annan valuta innehåller den historiska tillväxten valutaeffekter.
- **Förvärv:** historisk tillväxt innehåller köpta bolag som DCF aldrig belastar kassaflödet med.
- **Tunna branscher:** jämförelsebolagen väljs inom Yahoos sektor, så Autoliv får till exempel H&M och Betsson.
- **P/E för investmentbolag och fastighet** styrs av värdeförändringar. P/B eller substansvärde vore rätt mått.
- **Monte Carlo:** osäkerheten i tillväxt bygger på högst tre observationer och slår ofta i taket på 5 procentenheter.

## Idéer framåt

- Årsredovisningar i ESEF-format (filings.xbrl.org) för huvudlistebolagen: upp till sex års historik och rapporterade siffror att stämma av Yahoo mot.
- Rullande tolv månader ur kvartalsdata, så att resultatet ligger närmare kursen i tid.
- P/B för finansiella bolag och fastighetsbolag.
- Tillväxt i rapportvalutan och justering för förvärv.
