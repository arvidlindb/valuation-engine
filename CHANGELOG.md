# Changelog

All notable changes to this project are written down here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (2026-11-16)

First version. It values every Swedish company on Yahoo Finance with four methods and shows the results side by side.

### Added

- Data layer for Swedish stocks on Yahoo Finance. It builds the company list from Yahoo's screener (984 symbols, 687 companies after filtering and keeping one share class each) and maps Yahoo's statements to one standard set of line items, with a fallback chain per item.
- Local cache for all downloaded data, and a download command that can be stopped and started again (`python -m valuation_engine.data.download --all`).
- Offline mode (`--offline` or `VALUATION_ENGINE_OFFLINE=1`) with frozen data for 15 companies in `tests/fixtures/`, so the tests and the demo run without a network.
- DCF on free cash flow to the firm: five year forecast from the company's own history, WACC from CAPM with a peer beta, Gordon growth and exit multiple terminal values, mid-year discounting, sensitivity tables and an equity bridge.
- Size premium on the cost of equity, by market cap. It is on by default and can be turned off with `--no-size-premium`.
- Trading comps: peers picked inside the sector with k-means clustering and a nearest neighbour ranking, then EV/EBITDA, EV/Sales and P/E with outlier filtering.
- LBO: sources and uses, senior and subordinated debt sized against coverage limits, cash sweep, revolver, IRR and MOIC, and the price a buyer could pay at 25%, 20% and 15% target IRR.
- Monte Carlo on the DCF: random growth, margin, WACC and terminal growth with correlation, percentiles, tornado, variance contribution and convergence. 100,000 simulations take about a tenth of a second.
- Football field chart and 15 other charts, as PNG and SVG.
- Excel export with seven sheets (Summary, Assumptions, DCF, Comps, LBO, Monte Carlo, Data). The Assumptions sheet shows where each market parameter comes from.
- Streamlit app where you pick a company, change the main assumptions and download the workbook.
- Pipeline that runs all four models on one company, and a command line: `python -m valuation_engine TICKER` for one or more companies and `--all` for the whole universe.
- Batch run that writes `universe_valuation.csv` and `universe_summary.md`. All 687 companies take about a minute.
- Case study on Atlas Copco: a notebook, a written report and figures.
- Tests: 653 of them, all offline, including independent recalculations of the DCF, WACC, comps, LBO and Monte Carlo.

### Changed

- The risk-free rate now comes from the Riksbank (10 year government bond, 3.075% on 2026-11-13) and the equity risk premium and tax rate from Damodaran's table for Sweden (4.23% and 20.6%, January 2026). Before this they were fixed assumptions of 2.5%, 5.5% and 20.6%.
- Companies that report in another currency are converted with Riksbank rates per fiscal year (average rate for the income statement and cash flow, year-end rate for the balance sheet). Before this one spot rate was used for all years.
- The DCF takes its peer beta and exit multiple from the peer group that comps selected, and no longer from everyone with the same industry label. For Atlas Copco this moved the exit multiple from 12.4x to 18.6x.
- The DCF's default exit multiple is the same EV/EBITDA median that comps reports, after its outlier filter (19.4x for Atlas Copco).
- The LBO uses the DCF's operating forecast and exits at the DCF's exit multiple, so the two models describe the same company. Before this the LBO made its own forecast and exited at its entry multiple.
- Banks and other financial companies count as "partial" in the batch status, also when the P/E comparison worked.

### Fixed

- The DCF only takes over the comps exit multiple when at least five peers have one. With fewer it falls back to its own peer median or the wider universe.
- The low and high end of the DCF range when WACC minus growth gets too small.
- The currency text for companies that report in another currency than they trade in.
- A mistyped or empty ticker now gives a clear message and exit code 1, not a traceback.
- IRR when the cash flows have more than one root.
- The app text when comps finds a single cluster.
- An import warning, and a matplotlib import that was not needed.
