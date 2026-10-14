# Course dataset — ML in Finance

This folder holds every data file the course notebooks use. All files are plain CSV (one is gzipped
CSV, which pandas reads the same way) so students can open them in Excel as well as Python.

The data come from free public sources and were assembled once by the instructor with
`build_dataset.ipynb`. Students never need to download anything from the internet; notebooks read
these files directly.

**Sample period:** January 2010 – December 2025 for analysis. Prices start in December 2008 so that
12-month features (momentum, volatility, beta) already exist in January 2010.

**Universe:** every stock that was a member of the S&P 500 at any time between 2010 and 2025, plus
the SPY ETF as the market benchmark.

---

## Files

| File | One row is… | Columns | Units / notes |
|---|---|---|---|
| `universe.csv` | a stock | `ticker` (Yahoo style, e.g. `BRK-B`), `index_ticker` (as in index lists, `BRK.B`), `name`, `sector` (GICS), `sub_industry`, `cik` (SEC id), `first_date` / `last_date` (first and last day in the S&P 500 within 2010–2025), `years_in_index`, `in_index_at_end`, `delisted`, `has_prices`, `price_start`, `has_fundamentals`, `duplicate_share_class`, `reused_ticker_suspect`, `info_source` | `delisted = True` means the stock **left the S&P 500** before 31 Dec 2025 (acquired, bankrupt, dropped to a smaller index, or renamed — see Known gaps). |
| `prices_monthly.csv` | a month-end | `date`, then one column per ticker, plus `SPY` | Adjusted close in USD (adjusted for splits **and** dividends, so `pct_change()` gives total returns). Dates are calendar month-ends. Blank = no price that month. |
| `prices_daily.csv.gz` | a trading day | `date`, one column per ticker, plus `SPY` | Same adjusted close, daily. Used for volatility, beta and the max-daily-return feature. |
| `splits.csv` | a stock split | `date`, `ticker`, `ratio` | 4.0 means 4-for-1. Kept for the Week 2 lesson on adjusted prices. |
| `fundamentals_quarterly.csv` | a company-quarter | `ticker`, `cik`, `source_cik` (SEC registrant the row came from; differs from `cik` for older quarters of companies that re-registered), `period_end`, `report_date`, `revenue`, `operating_income`, `d_and_a`, `ebitda`, `net_income`, `total_assets`, `total_debt`, `cash`, `book_equity`, `shares_outstanding`, `shares_date`, `shares_source`, `shares_adj` | USD. Income-statement items are **single-quarter** values (not year-to-date). Balance-sheet items are at `period_end`. `report_date` is the SEC filing date — **a number may only be used on or after its report date.** `shares_adj` is the share count restated to today's split basis (multiply by an adjusted price to get market cap). |
| `mktcap_monthly.csv` | a month-end | `date`, one column per ticker | Market capitalisation in **$ millions**, using the latest share count filed on or before that month-end (no look-ahead). |
| `factors_monthly.csv` | a month-end | `date`, `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `Mom`, `RF` | Fama–French five factors plus momentum, as **decimals** (0.01 = 1%). These are realised returns *during* that month. |
| `macro_monthly.csv` | a month-end | `date`, `yield_10y`, `yield_3m`, `term_spread`, `cpi_yoy`, `unemployment`, `vix` | Yields, CPI inflation and unemployment in percent; VIX in index points. Daily series use the last value of the month. |
| `headlines.csv` | a financial news sentence | `id`, `text`, `label` (positive / neutral / negative), `agreement` | `agreement` = share of the human annotators who agreed on the label (50, 66, 75 or 100 percent). No ticker or date. |
| `lm_dictionary.csv` | a word | `word`, `positive`, `negative`, `uncertainty`, `litigious` | 1 if the word belongs to that Loughran–McDonald list. Only words on at least one list are included. |

Instructor-only file produced alongside: `fundamentals_failures.csv` lists stocks for which the SEC
returned no usable data, with the reason.

**Folder layout:** the CSVs sit directly in `data/` (not in a sub-folder), and `prices_daily.csv.gz`
stays compressed — notebooks read it as is.

---

## Rules that keep the data leakage-safe

These matter for every model in the course; `mlfin.py` follows them and students should too.

1. **Fundamentals:** use a quarter's numbers only from its `report_date` onward, never from
   `period_end`. Typical gap: 25–45 days for quarterly reports, up to ~60 for annual reports.
2. **Market cap** is already point-in-time (prices at month-end × shares filed by then).
3. **Macro data** is labelled by the month it describes, but CPI and unemployment for month *m*
   are only published during month *m + 1*. Lag them by one month before using them as features.
   Yields and VIX are known at the month-end they are stamped with.
4. **Factors** are returns earned during the month. Using this month's factor return to predict this
   month's stock return is look-ahead.
5. **Membership:** `first_date`/`last_date` say when a stock was in the index. A strategy that only
   trades stocks that are *still* in the index today (`in_index_at_end`) is survivorship-biased —
   that is the point of the Week 2 demo.

---

## Sources, licences and citations

| Data | Source | Terms |
|---|---|---|
| Index membership history | `fja05680/sp500` on GitHub ("S&P 500 Historical Components & Changes"), built from Clenow's *Trading Evolved* list plus Wikipedia changes | Community-maintained; the author notes it is accurate on any given day but not guaranteed. |
| Current names, sectors, CIK | `datasets/s-and-p-500-companies` on GitHub (from Wikipedia) | Open data (PDDL). |
| Prices, splits, sectors for former members | Yahoo Finance via the `yfinance` package | Yahoo's terms allow personal use; **see "Hosting" below.** |
| Fundamentals | SEC EDGAR XBRL company facts (`data.sec.gov`) | US government data, public domain. |
| Factors | Kenneth R. French Data Library, Dartmouth | Free; cite the data library. |
| Macro | FRED, Federal Reserve Bank of St. Louis (DGS10, DGS3MO, CPIAUCSL, UNRATE, VIXCLS) | Free; VIX is © Cboe, redistributed by FRED. |
| Headlines | Financial PhraseBank v1.0 (Malo, Sinha, Korhonen, Wallenius & Takala, 2014, *JASIST* 65(4)) via Hugging Face `takala/financial_phrasebank` | CC BY-NC-SA 3.0 — non-commercial use with attribution; fine for teaching. |
| Sentiment dictionary | Loughran & McDonald (2011), *Journal of Finance* 66(1), word lists as bundled in the `pysentiment2` package (2018 master dictionary) | Free for academic use with citation. |

**Hosting.** The design spec planned a public GitHub repository. Everything above is fine to post
publicly *except* the Yahoo-derived prices, whose terms are for personal use. Many courses post such
data anyway, but the conservative choice is to keep the notebooks in a public repo and put the
`data/` folder somewhere only enrolled students can reach (a WVU Google Drive folder or the LMS).
This is an open decision for the instructor.

---

## Known gaps (documented, not fixed — several are teaching points)

- **Survivorship.** Yahoo keeps history only for securities that still exist. Most stocks that left
  the index because they were acquired or went bankrupt have no prices (`has_prices = False`). The
  dataset therefore over-represents survivors; the Week 2 lesson uses exactly this.
- **Renames look like removals.** The membership history records a ticker change (e.g. FB → META in
  2022, ANTM → ELV in 2022) as one stock leaving and another joining. The old ticker has no Yahoo
  data; the new ticker's price history covers the whole period, but its `first_date` is the rename
  date. Analyses that select "stocks in the index on date *t*" will miss such firms before the rename.
  Selecting the largest stocks by `mktcap_monthly.csv` on each date avoids this.
- **Re-used tickers.** Some tickers of former members now belong to a different company, and Yahoo
  returns the new company's history. Three safeguards remove these (flag `reused_ticker_suspect`):
  history that starts after the original firm's membership; a median market cap under $500m while
  supposedly in the index; and a hand-checked list found in the first build — CPWR (Compuware),
  EP (El Paso), BBT (BB&T), GENZ (Genzyme), KG (King Pharmaceuticals), PARA (Paramount Global).
- **Fundamentals are approximate.** Companies label the same line item with different XBRL tags; each
  field uses a priority list of tags. `total_debt` (long-term + current debt) and `ebitda`
  (operating income + depreciation & amortisation) are the least standardised; banks and insurers
  often lack `ebitda` and a clean `revenue`. Values are as **first reported**, not restated: when a
  company later re-files an old quarter under a different tag (common after the 2018 revenue-standard
  change), the earliest public figure and its date are kept.
- **Re-registered companies.** Some companies moved to a new SEC registrant (holding-company
  reorganisations, re-domiciling): Alphabet (from Google, 2015), ExxonMobil, Disney, Medtronic,
  Cigna, Apache/APA, DuPont, Eaton, ICE, Xerox, DXC (from CSC), TechnipFMC (from FMC Technologies).
  Their earlier quarters come from the old registrant (`source_cik`), checked by name before merging.
- **Gaps.** About 1.7% of company-quarters are missing, mostly fourth quarters in 2009–2011 where the
  SEC feed lacks the original annual report (only a restated copy filed years later, which is not
  used). ExxonMobil's quarterly revenue before 2018 is missing for the same kind of reason (its other
  fields are present). When building features, take the latest quarter available as of each date
  rather than requiring four consecutive quarters.
- **Berkshire Hathaway** stopped reporting a usable share count after 2015, so its market cap from
  2016 on uses the last good count (roughly 10–15% too high by 2025 because of buybacks).
- **Shell or mis-scaled quarters** (total assets under 5% of the neighbouring quarters — e.g. a new
  parent company filing before its merger closed, or a filing keyed in thousands instead of
  dollars) are dropped.
- **Share counts** are checked against the quarter's weighted-average share count and against the
  company's neighbouring quarters; figures off by a factor (filing typos such as ×1,000,000) are
  rejected (`shares_source = rejected_outlier`) and market cap carries the last good count forward.
- **Share classes.** Companies with two listed classes (GOOGL/GOOG, FOXA/FOX, NWSA/NWS) appear once per
  class with the same fundamentals and company-level market cap; `duplicate_share_class = True`
  marks the extra class so each company can be counted once. Berkshire Hathaway reports shares in
  Class A equivalents; the builder converts them to B shares (×1,500).
- **Membership dates** come from a community list and are approximate to within days.
- **Unemployment for October 2025** is blank: the BLS did not publish it (federal shutdown).
- **Headlines** have no tickers or dates, so they cannot be merged with returns; NB11 uses them to
  teach text classification, and the sentiment-as-a-feature extension is optional.
- **Planted errors** for the Week 2 data-problem hunt (a missing month, a bad split) are created inside
  NB02, not stored here. These files should stay clean.

---

## Differences from the design spec (Section 10)

- Added `splits.csv` and `mktcap_monthly.csv`. Market cap needs share counts and split-only prices to
  be lined up carefully; doing it once here keeps `mlfin.py` simple and leakage-safe.
- `fundamentals_quarterly.csv` has extra columns: `operating_income` and `d_and_a` (the parts of
  `ebitda`), `shares_date`, `shares_source`, `shares_adj`.
- `prices_daily.csv.gz` covers all tickers (gzipped to stay small) rather than 100.
- Prices start December 2008 rather than January 2010 (12-month look-back for features).
- `universe.csv` has extra columns explained above; `delisted` means "left the index".

---

## Rebuilding

Open `build_dataset.ipynb` in Google Colab, put your e-mail in `SEC_USER_AGENT`, and run all cells
(about 15–25 minutes). Two zips download: `course_data.zip` (these files) and `builder_cache.zip`
(instructor only — raw SEC facts and split-only prices, kept so problems can be fixed later without
re-downloading). Then run `dataset_checks.ipynb` on the result. Rebuilding later will give
slightly different numbers where sources have been corrected; record the build date below.

| Build date | Built by | Notes |
|---|---|---|
| 2026-10-14 | Ryan Chen (Colab) + offline fixes from `builder_cache.zip` and `sec_patch.zip` | 60/60 checks pass (`instructor/dataset_checks_results.ipynb`). 831 tickers in universe, 585 with prices, 584 with fundamentals. |
| 2026-10-14 (b) | offline rebuild during step 2 | 13 shell/mis-scaled company-quarters removed (34,366 rows). 60/60 checks pass. |
