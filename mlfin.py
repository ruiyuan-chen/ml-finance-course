"""
mlfin.py — the helper module for the course *Machine Learning in Finance*.

You do not need to read this file to do the labs. Each notebook tells you which helper it uses and
what it does in plain English. If you are curious how something works, open this file and read the
function, or paste it into an AI assistant and ask it to explain it line by line.

Everything here follows three rules that keep our results honest:
  1. A feature for month t only uses information that was public by the end of month t.
  2. Financial-statement numbers are used only after the date they were filed with the SEC.
  3. Models are always tested on months that come *after* the months they were trained on.
"""

import os
import re

import numpy as np
import pandas as pd

__version__ = "1.4"

# ---------------------------------------------------------------------------------------------
# Where the data lives
# ---------------------------------------------------------------------------------------------
# The course files can sit in a local folder (data/) or at a web address. Notebooks call
# set_data_location() once if needed; otherwise the first folder below that exists is used.
COURSE_URL = "https://raw.githubusercontent.com/ruiyuan-chen/ml-finance-course/main/"
DATA_URL = COURSE_URL + "data/"
_SEARCH = ["data", "../data", "../../data", "/content/data", "/content/drive/MyDrive/MLinFinance/data"]
_location = None
_cache = {}

BENCHMARK = "SPY"

FEATURES = ["mom_12_1", "rev_1", "vol_12", "size", "btm", "ep", "growth", "margin",
            "leverage", "beta_12", "max_1", "sector"]
NUMERIC_FEATURES = [f for f in FEATURES if f != "sector"]

FEATURE_DESCRIPTIONS = {
    "mom_12_1": "Momentum: return over the past 12 months, skipping the most recent month",
    "rev_1": "Reversal: last month's return",
    "vol_12": "Volatility: standard deviation of the last 12 monthly returns",
    "size": "Size: log of market capitalisation ($ millions)",
    "btm": "Value: book equity divided by market cap",
    "ep": "Value: last 12 months' earnings divided by market cap",
    "growth": "Growth: last 12 months' revenue vs the 12 months before",
    "margin": "Profitability: last 12 months' EBITDA divided by revenue",
    "leverage": "Debt: total debt divided by total assets",
    "beta_12": "Market risk: beta to the S&P 500 from the last 12 months of daily returns",
    "max_1": "Lottery-ness: the largest one-day return in the last month",
    "sector": "GICS sector (a category, not a number)",
}


def set_data_location(path_or_url):
    """Tell mlfin where the course CSV files are: a folder on this computer / Colab, or a web
    address ending in '/'. Call it once at the top of a notebook if the default doesn't work."""
    global _location
    _location = path_or_url if path_or_url.endswith("/") else path_or_url + "/"
    _cache.clear()


def setup(course_url=None, quiet=False):
    """Run once at the top of every notebook ("RUN ME FIRST"). Finds the course data (a local
    data/ folder if there is one, otherwise the course website), sets up tidy chart defaults,
    loads the monthly prices to check everything works, and prints a one-line confirmation."""
    import matplotlib.pyplot as plt
    global _location
    local = [d for d in _SEARCH if os.path.exists(os.path.join(d, "universe.csv"))]
    if local:
        set_data_location(local[0])
    elif course_url and "RYAN-GITHUB-USER" not in course_url:
        set_data_location(course_url.rstrip("/") + "/data/")
    plt.rcParams.update({"figure.figsize": (9, 4.5), "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    pm = load_prices("M")
    if not quiet:
        n = pm.shape[1] - 1
        print(f"Setup OK — {n} stocks, {len(pm)} months loaded "
              f"({pm.index[0]:%b %Y} to {pm.index[-1]:%b %Y}).")


def _where():
    global _location
    if _location:
        return _location
    env = os.environ.get("MLFIN_DATA")
    if env:
        return env if env.endswith("/") else env + "/"
    for d in _SEARCH:
        if os.path.exists(os.path.join(d, "universe.csv")):
            _location = d + "/"
            return _location
    if DATA_URL and "RYAN-GITHUB-USER" not in DATA_URL:
        _location = DATA_URL
        return _location
    raise FileNotFoundError(
        "mlfin can't find the course data. Put the data folder next to your notebook, "
        "or call mlfin.set_data_location('path/or/url/to/data/').")


def _read(name, **kw):
    if name not in _cache:
        df = pd.read_csv(_where() + name, **kw)
        # use one date resolution everywhere (pandas versions differ in their defaults)
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].astype("datetime64[ns]")
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.astype("datetime64[ns]")
        _cache[name] = df
    return _cache[name].copy()


# ---------------------------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------------------------
def load_prices(freq="M"):
    """Adjusted closing prices: one row per date, one column per stock ticker, plus SPY (the
    S&P 500 fund we use as 'the market'). freq="M" gives month-end prices; freq="D" gives daily.
    Prices are adjusted for splits and dividends, so percentage changes are total returns."""
    if freq.upper().startswith("M"):
        return _read("prices_monthly.csv", index_col="date", parse_dates=["date"])
    if freq.upper().startswith("D"):
        return _read("prices_daily.csv.gz", index_col="date", parse_dates=["date"])
    raise ValueError('freq must be "M" (monthly) or "D" (daily)')


def load_universe():
    """One row per stock that was in the S&P 500 at some point in 2010-2025: name, sector, when it
    was in the index, and whether we have prices and financial statements for it."""
    return _read("universe.csv", parse_dates=["first_date", "last_date", "price_start"])


def load_fundamentals():
    """Quarterly financial-statement numbers from SEC filings, one row per company per quarter.
    'report_date' is the day the numbers became public - never use a row before that date."""
    return _read("fundamentals_quarterly.csv", parse_dates=["period_end", "report_date", "shares_date"])


def load_mktcap():
    """Month-end market capitalisation in $ millions (rows = dates, columns = tickers)."""
    return _read("mktcap_monthly.csv", index_col="date", parse_dates=["date"])


def load_factors():
    """Monthly Fama-French factor returns (Mkt-RF, SMB, HML, RMW, CMA, Mom) and the risk-free rate
    RF, as decimals (0.01 = 1%)."""
    return _read("factors_monthly.csv", index_col="date", parse_dates=["date"])


def load_macro():
    """Monthly macro series: 10-year and 3-month Treasury yields, term spread, CPI inflation,
    unemployment and VIX. CPI and unemployment for a month are published the NEXT month."""
    return _read("macro_monthly.csv", index_col="date", parse_dates=["date"])


def load_headlines():
    """About 4,800 financial news sentences labelled positive / neutral / negative by people.
    'agreement' says what share of the labellers agreed (100 = everyone)."""
    return _read("headlines.csv")


def load_lm_dictionary():
    """The Loughran-McDonald finance word lists (positive, negative, uncertainty, litigious)."""
    return _read("lm_dictionary.csv")


def to_returns(prices):
    """Turn prices into simple returns: (price this period / price last period) - 1.
    The first row is empty because there is no earlier price to compare with."""
    return prices.pct_change(fill_method=None)


# ---------------------------------------------------------------------------------------------
# Building the feature panel
# ---------------------------------------------------------------------------------------------
def _fundamentals_as_known(fund):
    """For every company and every date on which it filed new numbers, what an investor knew on
    that date: the latest balance sheet, and trailing-12-month (TTM) revenue, earnings and EBITDA
    built only from quarters already filed. If one of the last four quarters is missing, the
    average of the available ones (at least 3) is scaled up to a year. Growth compares TTM revenue
    with the TTM revenue of the year before, again using only quarters already filed."""
    rows = []
    year, win = np.timedelta64(365, "D"), np.timedelta64(370, "D")
    for tic, g in fund.sort_values("period_end").groupby("ticker", sort=False):
        g = g.drop_duplicates("period_end")
        pe = g["period_end"].values.astype("datetime64[ns]")
        rd = g["report_date"].values.astype("datetime64[ns]")
        vals = {c: g[c].values.astype(float) for c in
                ["revenue", "net_income", "ebitda", "book_equity", "total_debt", "total_assets"]}
        for e in np.unique(rd):
            known = rd <= e
            latest = pe[known].max()
            i_last = np.where(known & (pe == latest))[0][0]
            cur = known & (pe > latest - win)
            prev = known & (pe > latest - year - win) & (pe <= latest - year + np.timedelta64(15, "D"))
            out = {"ticker": tic, "known_date": e, "period_end": latest}
            for c in ["revenue", "net_income", "ebitda"]:
                v = vals[c][cur]
                v = v[~np.isnan(v)]
                out[c + "_ttm"] = v.mean() * 4 if len(v) >= 3 else np.nan
            v = vals["revenue"][prev]
            v = v[~np.isnan(v)]
            out["revenue_ttm_prev"] = v.mean() * 4 if len(v) >= 3 else np.nan
            for c in ["book_equity", "total_debt", "total_assets"]:
                out[c] = vals[c][i_last]
            rows.append(out)
    return pd.DataFrame(rows)


def _cross_section_clip(df, cols, lo=0.01, hi=0.99):
    """Within each month, pull extreme values in to the 1st and 99th percentile."""
    q = df.groupby("date")[cols].transform(lambda s: s.quantile(lo)), \
        df.groupby("date")[cols].transform(lambda s: s.quantile(hi))
    return df[cols].clip(lower=q[0], upper=q[1])


def make_features(prices=None, fundamentals=None, universe="top500", start="2010-01-31",
                  end="2025-12-31", clip=True):
    """Build the course's stock-month table ("panel"): one row per stock per month-end, with the
    12 standard features (see mlfin.FEATURE_DESCRIPTIONS) and the things we try to predict:

      ret_next     the stock's return over the NEXT month
      beat_next    1 if that return beat SPY's next-month return, else 0
      ret_next_12  the stock's return over the next 12 months
      spy_next     SPY's return over the next month (for reference)

    Every feature on the row dated t uses only information public by the end of month t, so the
    labels are genuinely in the future. Financial-statement features use the latest quarter whose
    SEC filing date is on or before t (and are left empty if that filing is over 15 months old).

    universe="top500" keeps, each month, the 500 largest stocks by market cap (one share class per
    company). universe="members" keeps the stocks that were in the S&P 500 that month instead.
    clip=True pulls extreme feature values in to each month's 1st/99th percentile.
    Arguments prices/fundamentals are optional; by default the course files are loaded."""
    pm = load_prices("M") if prices is None else prices
    fund = load_fundamentals() if fundamentals is None else fundamentals
    pdly = load_prices("D")
    mcap = load_mktcap()
    uni = load_universe()

    stocks = [c for c in pm.columns if c != BENCHMARK]
    ret = to_returns(pm)
    spy = ret[BENCHMARK]

    feats = {}
    feats["mom_12_1"] = pm[stocks].shift(1) / pm[stocks].shift(12) - 1
    feats["rev_1"] = ret[stocks]
    feats["vol_12"] = ret[stocks].rolling(12, min_periods=10).std()
    feats["size"] = np.log(mcap.reindex(index=pm.index, columns=stocks))

    # daily-data features: beta over the last ~252 trading days, max daily return in the month
    dret = to_returns(pdly)
    m_spy = dret[BENCHMARK]
    cov = dret[stocks].rolling(252, min_periods=200).cov(m_spy)
    var = m_spy.rolling(252, min_periods=200).var()
    beta_d = cov.div(var, axis=0)
    feats["beta_12"] = beta_d.resample("ME").last().reindex(pm.index)
    feats["max_1"] = dret[stocks].resample("ME").max().reindex(pm.index)

    # labels
    lab = {
        "ret_next": ret[stocks].shift(-1),
        "ret_next_12": pm[stocks].shift(-12) / pm[stocks] - 1,
    }
    long = []
    for name, df in {**feats, **lab}.items():
        try:
            s = df.stack(future_stack=True)
        except TypeError:          # newer pandas: future behaviour is the default
            s = df.stack()
        s.name = name
        long.append(s)
    panel = pd.concat(long, axis=1)
    panel.index.names = ["date", "ticker"]
    panel = panel.reset_index()
    panel["mktcap"] = np.exp(panel["size"])
    panel["spy_next"] = panel["date"].map(spy.shift(-1))
    panel["beat_next"] = (panel["ret_next"] > panel["spy_next"]).astype(float)
    panel.loc[panel["ret_next"].isna(), "beat_next"] = np.nan

    # financial-statement features: what was known at each month-end (by SEC filing date)
    known = _fundamentals_as_known(fund).sort_values("known_date")
    known["known_date"] = known["known_date"].astype("datetime64[ns]")
    fcols = ["period_end", "book_equity", "net_income_ttm", "revenue_ttm", "revenue_ttm_prev",
             "ebitda_ttm", "total_debt", "total_assets"]
    panel = panel.sort_values("date")
    panel["date"] = panel["date"].astype("datetime64[ns]")
    panel = pd.merge_asof(panel, known[["ticker", "known_date"] + fcols], left_on="date",
                          right_on="known_date", by="ticker", direction="backward")
    stale = (panel["date"] - panel["period_end"]).dt.days > 456   # latest quarter over 15 months old
    panel.loc[stale, fcols] = np.nan
    mc_dollars = panel["mktcap"] * 1e6
    panel["btm"] = panel["book_equity"] / mc_dollars
    panel["ep"] = panel["net_income_ttm"] / mc_dollars
    panel["growth"] = panel["revenue_ttm"] / panel["revenue_ttm_prev"] - 1
    panel.loc[panel["revenue_ttm_prev"] <= 0, "growth"] = np.nan
    panel["margin"] = panel["ebitda_ttm"] / panel["revenue_ttm"]
    panel.loc[panel["revenue_ttm"] <= 0, "margin"] = np.nan
    panel["leverage"] = panel["total_debt"] / panel["total_assets"]

    # sector and universe
    info = uni.set_index("ticker")
    panel["sector"] = panel["ticker"].map(info["sector"]).fillna("Unknown")
    dup = set(info.index[info["duplicate_share_class"].fillna(False).astype(bool)])
    panel = panel[~panel["ticker"].isin(dup)]
    panel = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))]
    if universe == "top500":
        panel = panel[panel["mktcap"].notna()]
        panel["cap_rank"] = panel.groupby("date")["mktcap"].rank(ascending=False)
        panel = panel[panel["cap_rank"] <= 500].drop(columns="cap_rank")
    elif universe == "members":
        fd, ld = panel["ticker"].map(info["first_date"]), panel["ticker"].map(info["last_date"])
        panel = panel[(panel["date"] >= fd) & (panel["date"] <= ld)]
    elif universe != "all":
        raise ValueError('universe must be "top500", "members" or "all"')

    if clip:
        panel[NUMERIC_FEATURES] = _cross_section_clip(panel, NUMERIC_FEATURES)
    keep = ["date", "ticker", "sector"] + NUMERIC_FEATURES + ["mktcap", "ret_next", "beat_next",
                                                               "spy_next", "ret_next_12"]
    panel = panel[keep].sort_values(["date", "ticker"]).reset_index(drop=True)
    panel = panel[[c for c in ["date", "ticker"] + FEATURES + keep[14:] if c in panel.columns]]
    return panel


def prepare_xy(panel, features, target="ret_next"):
    """Pick the feature columns and the target from a panel, dropping rows where any of them is
    missing (we say how many). 'sector' is turned into one 0/1 column per sector.
    Returns X (features), y (target) and the rows that were kept."""
    cols = list(features)
    rows = panel.dropna(subset=[c for c in cols if c != "sector"] + [target])
    X = rows[[c for c in cols if c != "sector"]].copy()
    if "sector" in cols:
        X = X.join(pd.get_dummies(rows["sector"], prefix="sec", dtype=float))
    dropped = len(panel) - len(rows)
    if dropped:
        print(f"prepare_xy: dropped {dropped:,} of {len(panel):,} rows with missing values")
    return X, rows[target], rows


def time_split(panel, train_end="2017-12-31", test_end="2025-12-31", gap_months=1):
    """Split the panel by date: everything up to train_end is for training, everything after it
    up to test_end is for testing. Never shuffles - the test months always come later.
    The last month of training is dropped because its label (next month's return) falls in the
    test period. For labels that look further ahead (e.g. 12 months), set gap_months to match."""
    train_end, test_end = pd.Timestamp(train_end), pd.Timestamp(test_end)
    last_train = train_end - pd.offsets.MonthEnd(gap_months)
    train = panel[panel["date"] <= last_train]
    test = panel[(panel["date"] > train_end) & (panel["date"] <= test_end)]
    return train.copy(), test.copy()


# ---------------------------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------------------------
# Slow-moving features suit a strategy that holds stocks for a quarter. The fast ones (rev_1,
# max_1) change every month, so a model that leans on them trades much more (try adding them).
DEFAULT_BACKTEST_FEATURES = ["mom_12_1", "vol_12", "size", "btm", "ep", "growth", "leverage",
                             "beta_12"]


def default_model():
    """The course's standard model for backtests (Weeks 6-7 and the project template):
    gradient-boosted trees predicting the chance a stock beats SPY next month. Settings are kept
    modest (shallow trees, slow learning) so it doesn't simply memorise the past."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                          random_state=0)


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def walk_forward_backtest(panel, model, features=None, target=None, n_long=30, rebalance="Q",
                          cost_bps=20, weighting="equal", test_start="2015-01-01", max_weight=None,
                          max_per_cluster=None, cluster_k=8, verbose=True):
    """Simulate investing with a model the honest way ("walk-forward"):

      * At the start of each year the model is re-trained on ALL earlier months (only months whose
        outcome was already known), then used for that year without peeking ahead.
      * On each rebalance date ("M" monthly, "Q" quarterly, "A" yearly) it scores every stock in
        that month's universe and buys the n_long highest-scoring ones.
      * Between rebalances the holdings just ride the market (weights drift with prices).
      * Every time we trade we pay cost_bps basis points on the amount traded (20 bps = 0.20%).

    model     any scikit-learn model. A classifier is scored with its probability of class 1
              (default target "beat_next"); a regressor with its prediction ("ret_next").
    weighting "equal", "rank" (more weight to higher scores), "inverse_vol" (less weight to
              volatile stocks) or "cluster" (equal weights, but at most max_per_cluster stocks from
              any one cluster of stocks that move together; see cluster_stocks). max_weight caps
              any single stock's weight (e.g. 0.05).
    max_per_cluster  with any weighting: never hold more than this many stocks from one cluster.
              The clusters are re-estimated each year from the previous 5 years of monthly
              returns only (cluster_k groups), so they never use the future.

    Returns a dictionary with monthly returns of the strategy, SPY and an equal-weighted
    portfolio of the whole universe, cumulative growth of $1, turnover, the picks on every
    rebalance date and a table of statistics (see stats_table)."""
    from sklearn.base import clone, is_classifier
    features = DEFAULT_BACKTEST_FEATURES if features is None else list(features)
    if target is None:
        target = "beat_next" if is_classifier(model) else "ret_next"
    step = {"M": 1, "Q": 3, "A": 12, "Y": 12}[rebalance.upper()[0]]

    data = panel.dropna(subset=[c for c in features if c != "sector"]).copy()
    dates = sorted(d for d in data["date"].unique() if d >= pd.Timestamp(test_start))
    if not dates:
        raise ValueError("no months after test_start")
    rebal_dates = set(dates[::step])
    # next-month returns for ALL stocks (a holding that drops out of the universe still earns its return)
    nxt = to_returns(load_prices("M")).shift(-1)
    spy_next = panel.groupby("date")["spy_next"].first()
    ew_next = panel.groupby("date")["ret_next"].mean()

    if weighting == "cluster" and not max_per_cluster:
        max_per_cluster = int(np.ceil(n_long / cluster_k))
    monthly_returns = to_returns(load_prices("M")) if max_per_cluster else None
    clusters = None
    cluster_log = {}

    fitted_year, fitted, last_train = None, None, None
    w = pd.Series(dtype=float)
    rows, picks_log, turnover_log = [], {}, []
    for d in dates:
        if d in rebal_dates:
            if fitted_year != d.year:
                # train only on months whose outcome was known before this year began
                horizon = 12 if target == "ret_next_12" else 1
                cutoff = pd.Timestamp(year=d.year, month=1, day=1) - pd.offsets.MonthEnd(1 + horizon)
                train = data[(data["date"] <= cutoff)].dropna(subset=[target])
                X, y, _ = prepare_xy(train, features, target) if "sector" in features else \
                    (train[features], train[target], train)
                fitted = clone(model).fit(X, y)
                fitted._mlfin_columns = list(X.columns)
                sample = X.sample(min(5000, len(X)), random_state=0)
                last_train = (sample, y.loc[sample.index])
                fitted_year = d.year
                if max_per_cluster:
                    past = monthly_returns.loc[d - pd.DateOffset(years=5):d - pd.offsets.MonthEnd(1)]
                    clusters = cluster_stocks(past, k=cluster_k, plot=False)
                if verbose:
                    print(f"  {d.year}: trained on {len(X):,} stock-months up to {cutoff.date()}")
            now = data[data["date"] == d]
            if "sector" in features:
                Xn = pd.get_dummies(now[[c for c in features if c != "sector"]].join(
                    pd.get_dummies(now["sector"], prefix="sec", dtype=float)), dtype=float)
                Xn = Xn.reindex(columns=fitted._mlfin_columns, fill_value=0.0)
            else:
                Xn = now[features]
            sc = pd.Series(_scores(fitted, Xn), index=now["ticker"].values)
            ranked = sc.sort_values(ascending=False)
            if max_per_cluster:
                lab = clusters.reindex(ranked.index).fillna(-1)
                chosen, count = [], {}
                for t in ranked.index:
                    c = lab[t]
                    if count.get(c, 0) < max_per_cluster:
                        chosen.append(t)
                        count[c] = count.get(c, 0) + 1
                    if len(chosen) == n_long:
                        break
                top = ranked[chosen]
                cluster_log[d] = lab[chosen].value_counts().to_dict()
            else:
                top = ranked.head(n_long)
            if weighting in ("equal", "cluster"):
                new = pd.Series(1.0, index=top.index)
            elif weighting == "rank":
                new = pd.Series(np.arange(len(top), 0, -1, dtype=float), index=top.index)
            elif weighting == "inverse_vol":
                vol = now.set_index("ticker")["vol_12"].reindex(top.index)
                new = 1 / vol.clip(lower=vol.median() * 0.25)
            else:
                raise ValueError('weighting must be "equal", "rank", "inverse_vol" or "cluster"')
            new = new / new.sum()
            if max_weight:
                for _ in range(20):
                    over = new > max_weight
                    if not over.any():
                        break
                    excess = (new[over] - max_weight).sum()
                    new[over] = max_weight
                    new[~over] += excess * new[~over] / new[~over].sum()
            allt = new.index.union(w.index)
            turnover = (new.reindex(allt, fill_value=0) - w.reindex(allt, fill_value=0)).abs().sum()
            if w.empty:
                turnover = 1.0   # first purchase
            w = new
            picks_log[d] = list(top.index)
            turnover_log.append((d, turnover))
            cost = turnover * cost_bps / 1e4
        else:
            cost = 0.0
            turnover = 0.0
        r = nxt.loc[d].reindex(w.index).fillna(0.0) if d in nxt.index else pd.Series(0.0, index=w.index)
        gross = float((w * r).sum())
        net = gross - cost
        rows.append((d + pd.offsets.MonthEnd(1), net, gross, spy_next.get(d, np.nan), ew_next.get(d, np.nan)))
        # drift
        w = w * (1 + r)
        w = w / w.sum() if w.sum() > 0 else w

    res = pd.DataFrame(rows, columns=["date", "strategy", "strategy_gross", "SPY", "equal_weight_universe"]).set_index("date")
    res = res.dropna(subset=["SPY"])
    out = {
        "returns": res,
        "cumulative": (1 + res[["strategy", "SPY", "equal_weight_universe"]]).cumprod(),
        "turnover": pd.DataFrame(turnover_log, columns=["date", "turnover"]).set_index("date")["turnover"],
        "picks": picks_log,
        "settings": dict(n_long=n_long, rebalance=rebalance, cost_bps=cost_bps, weighting=weighting,
                         features=features, target=target, model=type(model).__name__,
                         test_start=str(test_start), max_weight=max_weight,
                         max_per_cluster=max_per_cluster),
        "clusters_held": cluster_log,
        "model": fitted,
        "train_sample": last_train,   # 5,000 rows of the final year's training data
    }
    out["stats"] = stats_table(out)
    return out


def stats_table(result):
    """The scorecard for a backtest, per year (annualised):
      Ann. return     average yearly growth rate
      Volatility      how much returns swing around (yearly standard deviation)
      Sharpe ratio    return above the risk-free rate per unit of volatility (higher is better)
      Max drawdown    the worst fall from a peak to a later low
      Beat SPY (months)  share of months the column beat SPY
      Turnover/yr     how much of the portfolio was replaced per year (1.0 = 100%)"""
    r = result["returns"]
    rf = load_factors()["RF"].reindex(r.index).fillna(0)
    out = {}
    for col in ["strategy", "SPY", "equal_weight_universe"]:
        x = r[col]
        years = len(x) / 12
        growth = (1 + x).prod()
        cum = (1 + x).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        ex = x - rf
        out[col] = {
            "Ann. return": growth ** (1 / years) - 1,
            "Volatility": x.std() * np.sqrt(12),
            "Sharpe ratio": ex.mean() / x.std() * np.sqrt(12),
            "Max drawdown": dd,
            "Beat SPY (months)": (x > r["SPY"]).mean() if col != "SPY" else np.nan,
        }
    t = pd.DataFrame(out)
    years = len(r) / 12
    t.loc["Turnover/yr", "strategy"] = result["turnover"].iloc[1:].sum() / years if len(result["turnover"]) > 1 else np.nan
    t.columns = ["Strategy (after costs)", "SPY", "Equal-weight universe"]
    return t.round(3)



# ---------------------------------------------------------------------------------------------
# Week 4: too many features, and regularization
# ---------------------------------------------------------------------------------------------
NAVY, GOLD, GREY = "#022e5c", "#eaaa00", "#b8bec8"


def add_noise_features(panel, n=25, seed=0):
    """Add n columns of pure random numbers called noise_1, noise_2, ... Each stock gets its own
    random level that wobbles a little month to month, so the columns look like real stock
    characteristics - but by construction they contain NO information about future returns.
    Returns the new panel and the list of new column names."""
    rng = np.random.default_rng(seed)
    names = [f"noise_{i + 1}" for i in range(n)]
    out = panel.reset_index(drop=True)
    if n == 0:
        return out.copy(), []
    tickers = np.sort(out["ticker"].unique())
    base = pd.DataFrame(rng.standard_normal((len(tickers), n)), index=tickers, columns=names)
    vals = base.reindex(out["ticker"]).values + 0.5 * rng.standard_normal((len(out), n))
    return pd.concat([out, pd.DataFrame(vals, columns=names)], axis=1), names


def decile_spread(y_pred, y_true, dates):
    """Each month, sort stocks into 10 groups by prediction; return the average monthly difference
    between the actual outcome of the top group and the bottom group (0.01 = 1% a month)."""
    df = pd.DataFrame({"p": np.asarray(y_pred), "y": np.asarray(y_true), "d": np.asarray(dates)})
    df["g"] = df.groupby("d")["p"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    m = df.groupby(["d", "g"])["y"].mean().unstack()
    return float((m[9] - m[0]).mean())


def linear_model(kind="ols", penalty=0.0):
    """A linear regression whose features are first put on the same scale (each feature rescaled
    to average 0 and spread 1, using the training data only).
      kind="ols"    plain line of best fit (NB03)            penalty is ignored
      kind="lasso"  penalized: pushes weak weights to exactly zero   try penalty 0.0005 - 0.005
      kind="ridge"  penalized: shrinks all weights toward zero       try penalty 1,000 - 100,000"""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    if kind == "ols" or penalty == 0:
        reg = LinearRegression()
    elif kind == "lasso":
        reg = Lasso(alpha=penalty, max_iter=5000)
    elif kind == "ridge":
        reg = Ridge(alpha=penalty)
    else:
        raise ValueError('kind must be "ols", "lasso" or "ridge"')
    return make_pipeline(StandardScaler(), reg)


def noise_experiment(panel, features, noise_counts=(0, 10, 25, 50, 100), train_start="2015-01-31",
                     train_end="2017-12-31", kind="ols", penalty=0.0, seed=0, plot=True):
    """Fit the same model again and again, each time with more useless noise features added, and
    record how well it does on the training data and on the test data. Returns a table."""
    rows = []
    for n in noise_counts:
        p, noise = add_noise_features(panel, n, seed=seed)
        train, test = time_split(p, train_end)
        train = train[train["date"] >= pd.Timestamp(train_start)]
        cols = list(features) + noise
        Xa, ya, ra = _quiet_xy(train, cols)
        Xb, yb, rb = _quiet_xy(test, cols)
        m = linear_model(kind, penalty).fit(Xa, ya)
        rows.append({"noise features": n, "train R2 (%)": m.score(Xa, ya) * 100,
                     "test R2 (%)": m.score(Xb, yb) * 100,
                     "test top-minus-bottom (% a month)": decile_spread(m.predict(Xb), yb, rb["date"]) * 100})
    res = pd.DataFrame(rows).set_index("noise features")
    if plot:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.plot(res.index, res["train R2 (%)"], "o-", color=NAVY, lw=2.5, label="Training data (the past it learned from)")
        ax.plot(res.index, res["test R2 (%)"], "o-", color=GOLD, lw=2.5, label="Test data (the future)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Number of useless random features added")
        ax.set_ylabel("R-squared (%)")
        ax.set_title("More features: the past looks better, the future gets worse")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    return res.round(3)


def _quiet_xy(panel, cols, target="ret_next"):
    rows = panel.dropna(subset=[c for c in cols if c != "sector"] + [target])
    X = rows[[c for c in cols if c != "sector"]].copy()
    if "sector" in cols:
        X = X.join(pd.get_dummies(rows["sector"], prefix="sec", dtype=float))
    return X, rows[target], rows


def plot_coefficients(model, feature_names=None, title=None, top=40):
    """Bar chart of a linear model's weights (on the common scale, so sizes are comparable).
    Real features are navy, noise features grey; weights that are exactly zero are listed below."""
    plt = _plt()
    reg = model[-1] if hasattr(model, "steps") else model
    names = list(feature_names) if feature_names is not None else list(getattr(model, "feature_names_in_", []))
    w = pd.Series(np.ravel(reg.coef_), index=names)
    nz = w[w != 0]
    show = nz.reindex(nz.abs().sort_values(ascending=False).index[:top]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.28 * max(len(show), 4) + 1.4))
    colors = [GREY if n.startswith("noise_") else NAVY for n in show.index]
    ax.barh(show.index, show.values * 100, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Weight (% next-month return per 1 standard deviation of the feature)")
    ax.set_title(title or f"What the {type(reg).__name__} model kept (navy = real, grey = noise)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()
    real = [n for n in names if not n.startswith("noise_")]
    noise = [n for n in names if n.startswith("noise_")]
    kept_r, kept_n = int((w[real] != 0).sum()), int((w[noise] != 0).sum()) if noise else 0
    msg = f"Kept {kept_r} of {len(real)} real features"
    if noise:
        msg += f" and {kept_n} of {len(noise)} noise features"
    zero = [n for n in real if w[n] == 0]
    print(msg + "." + (f"  Real features set to zero: {', '.join(zero)}" if zero else ""))
    return w


def plot_time_folds(train, n_splits=5):
    """Draw how time-series cross-validation slices the TRAINING period: each round learns from
    the earlier months (navy) and is checked on the next block of months (gold)."""
    from sklearn.model_selection import TimeSeriesSplit
    plt = _plt()
    dates = pd.DatetimeIndex(np.sort(train["date"].unique()))
    fig, ax = plt.subplots(figsize=(9, 0.45 * n_splits + 1.2))
    for k, (a, b) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(dates)):
        y = n_splits - k
        start_a, end_a = dates[a[0]] - pd.offsets.MonthEnd(1), dates[a[-1]]
        start_b, end_b = dates[b[0]] - pd.offsets.MonthEnd(1), dates[b[-1]]
        ax.barh(y, end_a - start_a, left=start_a, color=NAVY, height=0.6)
        ax.barh(y, end_b - start_b, left=start_b, color=GOLD, height=0.6)
    ax.set_yticks(range(1, n_splits + 1), [f"round {n_splits - i + 1}" for i in range(1, n_splits + 1)])
    ax.set_xlabel("Training period")
    ax.set_title("Time-series cross-validation inside the training period (navy = learn, gold = check)")
    plt.tight_layout()
    plt.show()


def tune_penalty(train, features, penalties, kind="lasso", n_splits=5, target="ret_next", plot=True):
    """Pick the penalty using ONLY the training period. The training months are cut into blocks
    (time-series cross-validation): learn on the early blocks, check on the next block, repeat.
    For each penalty we record the average top-minus-bottom spread on the check blocks.
    Returns a table; the best penalty is the one with the highest average check-block spread."""
    from sklearn.model_selection import TimeSeriesSplit
    X, y, rows = _quiet_xy(train, list(features), target)
    d = rows["date"].values
    dates = np.sort(np.unique(d))
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(dates))
    out = []
    for pen in penalties:
        sp, r2 = [], []
        for a, b in folds:
            ia, ib = np.isin(d, dates[a]), np.isin(d, dates[b])
            m = linear_model(kind, pen).fit(X[ia], y[ia])
            sp.append(decile_spread(m.predict(X[ib]), y[ib], d[ib]) * 100)
            r2.append(m.score(X[ib], y[ib]) * 100)
        full = linear_model(kind, pen).fit(X, y)
        kept = int((np.ravel(full[-1].coef_) != 0).sum())
        out.append({"penalty": pen, "check-block spread (% a month)": np.mean(sp),
                    "worst block": np.min(sp), "best block": np.max(sp),
                    "check-block R2 (%)": np.mean(r2), "features kept": kept})
    res = pd.DataFrame(out).set_index("penalty")
    best = res["check-block spread (% a month)"].idxmax()
    if plot:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 3.8))
        labels = [f"{p:g}" for p in res.index]
        ax.bar(labels, res["check-block spread (% a month)"],
               color=[GOLD if p == best else NAVY for p in res.index])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel(f"{kind.capitalize()} penalty")
        ax.set_ylabel("Average spread on check blocks (% a month)")
        ax.set_title(f"Choosing the penalty with cross-validation (best = gold: {best:g})")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()
    print(f"Best penalty by cross-validation: {best:g}")
    return res.round(3)


# ---------------------------------------------------------------------------------------------
# Weeks 5-6: classification and committees of trees
# ---------------------------------------------------------------------------------------------
def add_beat_label(panel, months=1):
    """Add a 0/1 label: did the stock beat SPY over the next `months` months?
    months=1 is the same as the built-in beat_next. Returns the new panel and the label's name."""
    if months == 1:
        return panel.copy(), "beat_next"
    pm = load_prices("M")
    fwd = pm.shift(-months) / pm - 1
    try:
        s = fwd.drop(columns=BENCHMARK).stack(future_stack=True)
    except TypeError:
        s = fwd.drop(columns=BENCHMARK).stack()
    s.index.names = ["date", "ticker"]
    name = f"beat_next_{months}m"
    out = panel.merge(s.rename("_r").reset_index(), on=["date", "ticker"], how="left")
    spy = out["date"].map(fwd[BENCHMARK])
    out[name] = (out["_r"] > spy).astype(float)
    out.loc[out["_r"].isna() | spy.isna(), name] = np.nan
    return out.drop(columns="_r"), name


def add_crash_label(panel, fall=0.5):
    """Add a 0/1 label 'crash_next_12': 1 if the stock lost more than `fall` (0.5 = 50%) of its
    value over the next 12 months. Rare - which is exactly what makes it interesting."""
    out = panel.copy()
    out["crash_next_12"] = (out["ret_next_12"] < -fall).astype(float)
    out.loc[out["ret_next_12"].isna(), "crash_next_12"] = np.nan
    return out


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan


def classify_scores(y_true, proba, threshold=0.5):
    """How a probability model does once you pick a threshold ("say yes if probability > threshold").
      share flagged : how often the model says yes
      accuracy      : share of all calls that were right
      precision     : of the times it said yes, how often it was right
      recall        : of all the actual yeses, how many it caught
      base rate     : how often the answer is actually yes (what a know-nothing model would see)
      AUC           : ranking score that ignores the threshold: 0.5 = coin flip, 1.0 = perfect"""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba)
    yes = p > threshold
    tp = int((yes & (y == 1)).sum())
    out = pd.Series({
        "threshold": threshold,
        "share flagged": yes.mean(),
        "accuracy": (yes == (y == 1)).mean(),
        "precision": tp / yes.sum() if yes.sum() else np.nan,
        "recall": tp / (y == 1).sum() if (y == 1).sum() else np.nan,
        "base rate": y.mean(),
        "AUC": _auc(y, p),
    })
    return out.round(3)


def plot_probabilities(proba, y_true, labels=("Lagged SPY", "Beat SPY"), threshold=None, title=None):
    """Histogram of the model's predicted probabilities, split by what actually happened.
    If the two colors sit on top of each other, the model can't tell the groups apart."""
    plt = _plt()
    p, y = np.asarray(proba), np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(p.min(), p.max(), 41)
    ax.hist(p[y == 0], bins=bins, density=True, alpha=0.45, color=NAVY, label=f"Actually: {labels[0]}")
    ax.hist(p[y == 1], bins=bins, density=True, histtype="step", lw=2.5, color=GOLD, label=f"Actually: {labels[1]}")
    if threshold is not None:
        ax.axvline(threshold, color="black", ls="--", lw=1.2, label=f"threshold = {threshold}")
    ax.set_xlabel(f"Predicted probability of '{labels[1]}'")
    ax.set_ylabel("How often (density)")
    ax.set_title(title or "What probabilities does the model give?")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_threshold_tradeoff(y_true, proba, threshold=None, title=None):
    """Precision, recall and share flagged as the threshold moves. Being pickier (higher threshold)
    flags fewer stocks; if the model has real signal, the ones it flags are right more often."""
    plt = _plt()
    p, y = np.asarray(proba), np.asarray(y_true).astype(int)
    ts = np.quantile(p, np.linspace(0.0, 0.995, 120))
    prec, rec, share = [], [], []
    for t in ts:
        yes = p > t
        prec.append((y[yes] == 1).mean() * 100 if yes.sum() >= 20 else np.nan)
        rec.append(((yes) & (y == 1)).sum() / max((y == 1).sum(), 1) * 100)
        share.append(yes.mean() * 100)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.plot(ts, rec, color=NAVY, lw=2.5, label="Recall: % of actual yeses caught")
    ax.plot(ts, share, color=GREY, lw=2, ls="--", label="Share of stocks flagged")
    ax.set_title("Pickier = fewer flags, fewer caught")
    ax.set_ylabel("%")
    ax.set_ylim(0, 102)
    ax2.plot(ts, prec, color=GOLD, lw=2.5, label="Precision: % of flagged that were right")
    ax2.axhline(y.mean() * 100, color="black", lw=1, ls=":", label=f"Base rate ({y.mean():.1%})")
    ax2.set_title("...but is each flag more likely right?")
    top = np.nanmax(prec) if np.isfinite(np.nanmax(prec)) else 100
    ax2.set_ylim(0, max(top * 1.25, y.mean() * 200))
    for a in (ax, ax2):
        if threshold is not None:
            a.axvline(threshold, color="C3", lw=1.2, label=f"your threshold = {threshold}")
        a.set_xlabel("Threshold (say yes if probability is above this)")
        a.legend(fontsize=8, loc="best")
        a.grid(alpha=0.3)
    fig.suptitle(title or "The threshold trade-off: pickier vs. more generous", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_tree(model, feature_names, class_names=("Lagged SPY", "Beat SPY"), title=None):
    """Draw a (small) decision tree as a flowchart. Each box shows the question asked, how many
    training rows reached it, and the share of each outcome. Readable up to depth 3 or 4."""
    plt = _plt()
    from sklearn.tree import plot_tree as _pt
    depth = model.get_depth()
    fig, ax = plt.subplots(figsize=(min(4 + 3.2 * 2 ** min(depth, 4) / 2, 22), 2.2 * depth + 1.5))
    _pt(model, feature_names=list(feature_names), class_names=list(class_names), filled=True,
        impurity=False, proportion=True, rounded=True, fontsize=8, max_depth=4, ax=ax)
    ax.set_title(title or f"A decision tree of depth {depth}: the flowchart the machine wrote")
    plt.tight_layout()
    plt.show()


def compare_models(models, X_train, y_train, X_test, y_test, dates_test=None, top=0.10, plot=True):
    """Fit each model in the dict {name: model} on the training data and score it on BOTH periods.
    Columns: train AUC and test AUC (0.5 = coin flip), test accuracy at threshold 0.5, and the hit
    rate among the `top` share of stocks each model rates most likely (formed month by month when
    dates are given) - the stocks you would actually act on. Also shows the time to train."""
    import time
    rows = []
    for name, m in models.items():
        t0 = time.time()
        m.fit(X_train, y_train)
        secs = time.time() - t0
        pa, pb = _scores(m, X_train), _scores(m, X_test)
        d = pd.DataFrame({"p": pb, "y": np.asarray(y_test)})
        if dates_test is not None:
            d["d"] = np.asarray(dates_test)
            d["rk"] = d.groupby("d")["p"].rank(pct=True, ascending=False)
        else:
            d["rk"] = d["p"].rank(pct=True, ascending=False)
        rows.append({"model": name, "train AUC": _auc(y_train, pa), "test AUC": _auc(y_test, pb),
                     "test accuracy": ((pb > 0.5) == (np.asarray(y_test) == 1)).mean(),
                     f"hit rate in top {top:.0%}": d.loc[d["rk"] <= top, "y"].mean(),
                     "base rate": np.asarray(y_test).mean(), "seconds to train": secs})
    res = pd.DataFrame(rows).set_index("model")
    if plot:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(res) + 1.6))
        yy = np.arange(len(res))
        ax.barh(yy + 0.2, res["train AUC"], height=0.4, color=NAVY, label="Training data (the past)")
        ax.barh(yy - 0.2, res["test AUC"], height=0.4, color=GOLD, label="Test data (the future)")
        ax.set_yticks(yy, res.index)
        ax.invert_yaxis()
        ax.axvline(0.5, color="black", lw=1, ls=":")
        ax.set_xlim(0.45, 1.0)
        ax.set_xlabel("AUC (0.5 = coin flip, 1.0 = perfect ranking)")
        ax.set_title("Same data, different models: the past vs. the future")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.show()
    return res.round(3)


# ---------------------------------------------------------------------------------------------
# Week 7: honest backtesting
# ---------------------------------------------------------------------------------------------
def plot_backtest_loop(test_start=2015, end=2025, data_start=2010):
    """The walk-forward loop as a picture: each year the model is re-trained on everything before
    that year (navy) and then trades that year only (gold). It never sees its own trading year."""
    plt = _plt()
    years = list(range(test_start, end + 1))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(years) + 1.4))
    for i, y in enumerate(years):
        row = len(years) - i
        ax.barh(row, y - data_start, left=data_start, color=NAVY, height=0.65)
        ax.barh(row, 1, left=y, color=GOLD, height=0.65)
    ax.set_yticks(range(1, len(years) + 1), [str(y) for y in reversed(years)])
    ax.set_xlim(data_start, end + 1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Trading year")
    ax.set_title("Walk-forward: re-train on the past (navy), trade the next year (gold), repeat")
    plt.tight_layout()
    plt.show()


def _ann(x):
    x = pd.Series(x).dropna()
    return (1 + x).prod() ** (12 / len(x)) - 1 if len(x) else np.nan


def cost_experiment(result, costs=(0, 10, 20, 50), plot=True):
    """Re-price ONE backtest at several trading costs (basis points per dollar traded; 20 bps =
    0.20%). The picks don't change, only what you pay to make them. Returns a table."""
    r = result["returns"]
    gross = r["strategy_gross"]
    tc = pd.Series(0.0, index=r.index)
    for d, t in result["turnover"].items():
        key = d + pd.offsets.MonthEnd(1)
        if key in tc.index:
            tc[key] = t
    rows, curves = [], {}
    for c in costs:
        net = gross - tc * c / 1e4
        curves[c] = (1 + net).cumprod()
        rows.append({"cost (bps)": c, "Ann. return": _ann(net), "SPY": _ann(r["SPY"]),
                     "Beats SPY?": "yes" if _ann(net) > _ann(r["SPY"]) else "no"})
    if plot:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        shades = ["#022e5c", "#3f6aa0", "#8c6100", "#eaaa00", "#b8bec8"]
        for (c, cur), col in zip(curves.items(), shades):
            ax.plot(cur.index, cur.values, color=col, lw=2, label=f"Strategy, {c} bps cost")
        spy = (1 + r["SPY"]).cumprod()
        ax.plot(spy.index, spy.values, color="black", lw=1.5, ls="--", label="SPY")
        ax.set_title("Same picks, different trading costs")
        ax.set_ylabel("Value of $1 invested")
        ax.set_xlabel("Date")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    out = pd.DataFrame(rows).set_index("cost (bps)")
    out[["Ann. return", "SPY"]] = out[["Ann. return", "SPY"]].round(3)
    return out


def rebalance_experiment(panel, model, freqs=("M", "Q", "A"), cost_bps=20, **kwargs):
    """Run the same backtest with monthly, quarterly and yearly rebalancing and compare: trading
    more often uses newer predictions but pays more costs. Extra arguments go to
    walk_forward_backtest (e.g. features=..., n_long=...)."""
    names = {"M": "monthly", "Q": "quarterly", "A": "yearly"}
    rows = []
    for f in freqs:
        res = walk_forward_backtest(panel, model, rebalance=f, cost_bps=cost_bps, verbose=False, **kwargs)
        st = res["stats"]
        rows.append({"rebalance": names.get(f, f),
                     "Ann. return before costs": _ann(res["returns"]["strategy_gross"]),
                     "Ann. return after costs": st.loc["Ann. return", "Strategy (after costs)"],
                     "SPY": st.loc["Ann. return", "SPY"],
                     "Turnover/yr": st.loc["Turnover/yr", "Strategy (after costs)"],
                     "Sharpe ratio": st.loc["Sharpe ratio", "Strategy (after costs)"],
                     "Max drawdown": st.loc["Max drawdown", "Strategy (after costs)"]})
    return pd.DataFrame(rows).set_index("rebalance").round(3)


def pick_hit_rate(result, panel, label="beat_next"):
    """How often the stocks the strategy bought actually beat SPY the next month, compared with
    all stocks in the universe in the same months. This is the strategy's 'accuracy'."""
    lab = panel.set_index(["date", "ticker"])[label]
    hits, base = [], []
    for d, picks in result["picks"].items():
        hits.append(lab.reindex([(d, t) for t in picks]).mean())
        base.append(panel.loc[panel["date"] == d, label].mean())
    out = pd.Series({"picks that beat SPY next month": np.nanmean(hits),
                     "all stocks that beat SPY (same months)": np.nanmean(base)})
    return out.round(3)


def excess_return_tstat(result):
    """Is the strategy's edge over SPY more than luck? Average monthly return above SPY, and its
    t-statistic (roughly: how many standard errors away from zero). Rule of thumb: below 2 means
    the edge could easily be luck."""
    ex = (result["returns"]["strategy"] - result["returns"]["SPY"]).dropna()
    t = ex.mean() / ex.std() * np.sqrt(len(ex))
    out = pd.Series({"months": len(ex), "avg monthly return above SPY (%)": round(ex.mean() * 100, 3),
                     "t-statistic": round(t, 2), "could be luck?": "yes" if abs(t) < 2 else "unlikely"})
    print(f"Average edge over SPY: {ex.mean() * 100:.2f}% a month over {len(ex)} months; t = {t:.2f}. "
          + ("That could easily be luck." if abs(t) < 2 else "That is unlikely to be pure luck."))
    return out


# ---------------------------------------------------------------------------------------------
# Week 8: valuation with ML
# ---------------------------------------------------------------------------------------------
VALUATION_PREDICTORS = ["roe", "growth", "margin", "leverage", "size", "beta_12", "vol_12", "mom_12_1"]


def make_valuation_panel(month=12, start="2010-01-31", end="2024-12-31"):
    """One row per stock at each year-end (month=12) with what an investor knew then:
      ev         enterprise value = market cap + debt - cash ($)
      ev_ebitda  EV / trailing-12-month EBITDA (only when both are positive)
      pb         price-to-book = market cap / book equity (only when book equity is positive)
      roe        return on equity = trailing-12-month net income / book equity
    plus the usual features, the company name and sector, and ret_next_12 (the NEXT 12 months'
    return, for testing). Financial-statement numbers are used only after they were filed."""
    panel = make_features(clip=False)
    snap = panel[(panel["date"].dt.month == month) & (panel["date"] >= pd.Timestamp(start))
                 & (panel["date"] <= pd.Timestamp(end))].sort_values("date").copy()
    fund = load_fundamentals()
    known = _fundamentals_as_known(fund).sort_values("known_date")
    known["known_date"] = known["known_date"].astype("datetime64[ns]")
    snap["date"] = snap["date"].astype("datetime64[ns]")
    cols = ["ebitda_ttm", "net_income_ttm", "total_debt", "book_equity", "period_end"]
    snap = pd.merge_asof(snap, known[["ticker", "known_date"] + cols], left_on="date",
                         right_on="known_date", by="ticker", direction="backward")
    cash = fund[["ticker", "report_date", "period_end", "cash"]].dropna(subset=["report_date"]).copy()
    cash["report_date"] = pd.to_datetime(cash["report_date"]).astype("datetime64[ns]")
    cash = cash.rename(columns={"period_end": "_pe_cash"}).sort_values("report_date")
    snap = pd.merge_asof(snap.sort_values("date"), cash, left_on="date", right_on="report_date",
                         by="ticker", direction="backward")
    stale = (snap["date"] - snap["period_end"]).dt.days > 456
    snap.loc[stale, ["ebitda_ttm", "net_income_ttm", "total_debt", "book_equity"]] = np.nan
    snap["cash"] = snap["cash"].where(snap["_pe_cash"] == snap["period_end"])
    mc = snap["mktcap"] * 1e6
    snap["ev"] = mc + snap["total_debt"].fillna(0) - snap["cash"].fillna(0)
    snap["ev_ebitda"] = np.where((snap["ebitda_ttm"] > 0) & (snap["ev"] > 0), snap["ev"] / snap["ebitda_ttm"], np.nan)
    snap["pb"] = np.where(snap["book_equity"] > 0, mc / snap["book_equity"], np.nan)
    snap["roe"] = np.where(snap["book_equity"] > 0, snap["net_income_ttm"] / snap["book_equity"], np.nan)
    snap["year"] = snap["date"].dt.year
    names = load_universe().set_index("ticker")["name"]
    snap["name"] = snap["ticker"].map(names)
    keep = ["date", "year", "ticker", "name", "sector", "mktcap", "ev", "ev_ebitda", "pb", "roe", "growth",
            "margin", "leverage", "size", "beta_12", "vol_12", "mom_12_1", "ret_next_12"]
    return snap[keep].sort_values(["year", "ticker"]).reset_index(drop=True)


def fit_valuation(vp, multiple="pb", predictors=None, sector=None, model=None, folds=5):
    """'Comparables' done by machine. Each year, predict every firm's (log) multiple from its
    fundamentals and sector, using a model trained on the OTHER firms that year (cross-validation),
    so no firm is its own comparable. Adds columns:
      predicted      the multiple the model says the firm 'should' trade at
      gap            log(actual / predicted): below 0 = cheap vs. its comparables, above 0 = expensive
      comps_gap      the same gap against the plain sector median (classic comps)
    Prints how much of the differences in multiples each method explains (R-squared)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.base import clone
    predictors = list(predictors or VALUATION_PREDICTORS)
    model = model or HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0)
    d = vp.dropna(subset=[multiple] + predictors).copy()
    if multiple == "ev_ebitda":
        d = d[d["sector"] != "Financials"]
    if sector is not None:
        d = d[d["sector"] == sector]
    out, r2 = [], []
    for y, g in d.groupby("year"):
        lo, hi = g[multiple].quantile([0.01, 0.99])
        g = g[(g[multiple] >= lo) & (g[multiple] <= hi)].copy()
        if len(g) < 5 * folds:
            continue
        X = g[predictors]
        if sector is None:
            X = X.join(pd.get_dummies(g["sector"], prefix="sec", dtype=float))
        yv = np.log(g[multiple])
        pred = cross_val_predict(clone(model), X, yv, cv=KFold(folds, shuffle=True, random_state=0))
        comps = yv.groupby(g["sector"]).transform("median")
        g["predicted"] = np.exp(pred)
        g["gap"] = yv - pred
        g["comps_gap"] = yv - comps
        ss = ((yv - yv.mean()) ** 2).sum()
        r2.append({"year": y, "firms": len(g), "R2 ML": 1 - ((yv - pred) ** 2).sum() / ss,
                   "R2 sector median": 1 - ((yv - comps) ** 2).sum() / ss})
        out.append(g)
    res = pd.concat(out)
    r2 = pd.DataFrame(r2).set_index("year")
    print(f"{multiple}: {len(res):,} firm-years. Share of differences in the multiple explained (R-squared), "
          f"average over years: ML {r2['R2 ML'].mean():.2f}, sector median {r2['R2 sector median'].mean():.2f}")
    res.attrs["r2_ml"] = round(float(r2["R2 ML"].mean()), 3)
    res.attrs["r2_sector_median"] = round(float(r2["R2 sector median"].mean()), 3)
    res.attrs["multiple"] = multiple
    return res


def plot_valuation(vals, year, n_label=6, title=None):
    """Actual vs. predicted multiple for one year (log scales). Firms below the diagonal trade
    below what their fundamentals 'justify' (cheap); above it, expensive. Labels the extremes."""
    plt = _plt()
    mult = vals.attrs.get("multiple", "multiple")
    g = vals[vals["year"] == year]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(g["predicted"], g[mult], s=14, color=GREY, alpha=0.8)
    lo = min(g["predicted"].min(), g[mult].min()) * 0.9
    hi = max(g["predicted"].max(), g[mult].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="actual = predicted")
    for _, r in pd.concat([g.nsmallest(n_label, "gap"), g.nlargest(n_label, "gap")]).iterrows():
        col = GOLD if r["gap"] < 0 else NAVY
        ax.scatter(r["predicted"], r[mult], s=40, color=col)
        ax.annotate(r["ticker"], (r["predicted"], r[mult]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Predicted {mult} (what the fundamentals 'justify')")
    ax.set_ylabel(f"Actual {mult}")
    ax.set_title(title or f"{year}: cheap (gold, below the line) and expensive (navy, above)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def valuation_returns(vals, col="gap", groups=5, plot=True):
    """Do 'cheap' stocks beat 'expensive' ones over the NEXT 12 months? Each year, sort firms into
    groups by the gap (group 1 = cheapest); show each group's average next-12-month return, and the
    cheapest-minus-most-expensive difference year by year."""
    d = vals.dropna(subset=["ret_next_12"]).copy()
    d["group"] = d.groupby("year")[col].transform(
        lambda s: pd.qcut(s.rank(method="first"), groups, labels=False) + 1)
    by = d.groupby(["year", "group"])["ret_next_12"].mean().unstack()
    avg = by.mean() * 100
    spread = (by[1] - by[groups]) * 100
    if plot:
        plt = _plt()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.bar(avg.index, avg.values, color=[GOLD] + [GREY] * (groups - 2) + [NAVY])
        a1.set_xticks(avg.index)
        a1.set_xlabel(f"Group (1 = cheapest by {col}, {groups} = most expensive)")
        a1.set_ylabel("Average next-12-month return (%)")
        a1.set_title("Next-year return by valuation group")
        a2.bar(spread.index, spread.values, color=[GOLD if v > 0 else NAVY for v in spread.values])
        a2.axhline(0, color="black", lw=0.8)
        a2.set_xlabel("Year the groups were formed")
        a2.set_ylabel("Cheapest minus most expensive (%)")
        a2.set_title(f"Cheap beat expensive in {(spread > 0).sum()} of {len(spread)} years")
        for a in (a1, a2):
            a.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()
    return pd.DataFrame({"avg next-12m return (%)": avg.round(1)}), spread.round(1)


# ---------------------------------------------------------------------------------------------
# Week 9: structure without a target
# ---------------------------------------------------------------------------------------------
def plot_correlation(returns, tickers=None, n=40, seed=0, title=None):
    """Heatmap of how monthly returns move together (correlation: 1 = in lockstep, 0 = unrelated).
    Stocks are sorted by sector so related companies sit next to each other."""
    plt = _plt()
    r = returns.drop(columns=[BENCHMARK], errors="ignore")
    if tickers is None:
        ok = r.columns[r.notna().mean() >= 0.95]
        tickers = list(pd.Series(ok).sample(min(n, len(ok)), random_state=seed))
    sec = load_universe().set_index("ticker")["sector"].reindex(tickers).fillna("?")
    order = list(sec.sort_values(kind="stable").index)
    c = r[order].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), [f"{t} ({sec[t][:12]})" for t in order], fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlation")
    ax.set_title(title or "Which stocks move together? (sorted by sector)")
    plt.tight_layout()
    plt.show()
    return c


def pca_bets(returns, tickers=None, n_stocks=30, seed=0, plot=True):
    """Principal component analysis: how many independent 'bets' hide in a portfolio? Finds the
    few combined movements that explain most of the ups and downs of the stocks.
    Returns the share of the movement explained by each component."""
    from sklearn.decomposition import PCA
    r = returns.drop(columns=[BENCHMARK], errors="ignore")
    ok = r.columns[r.notna().mean() >= 0.95]
    if tickers is None:
        tickers = list(pd.Series(ok).sample(min(n_stocks, len(ok)), random_state=seed))
    x = r[tickers].dropna()
    z = (x - x.mean()) / x.std()
    p = PCA().fit(z.values)
    ev = pd.Series(p.explained_variance_ratio_, index=range(1, len(tickers) + 1), name="share explained")
    if plot:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 4))
        top = ev.head(15)
        ax.bar(top.index, top.values * 100, color=[GOLD] + [NAVY] * (len(top) - 1))
        ax.plot(top.index, top.cumsum().values * 100, "o-", color="black", lw=1.2, label="cumulative")
        ax.set_xticks(top.index)
        ax.set_xlabel("Component (1 = the biggest common movement)")
        ax.set_ylabel("% of all the stocks' ups and downs")
        ax.set_title(f"{len(tickers)} stocks, but how many real bets?")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()
    spy = returns[BENCHMARK].reindex(x.index) if BENCHMARK in returns else None
    if spy is not None:
        pc1 = p.transform(z.values)[:, 0]
        print(f"Component 1 explains {ev.iloc[0]:.0%} on its own; its correlation with the S&P 500 is "
              f"{abs(np.corrcoef(pc1, spy.values)[0, 1]):.2f} (it is 'the market'). "
              f"{int((ev.cumsum() < 0.5).sum()) + 1} components explain half of all the movement.")
    return ev.round(3)


def diversification_experiment(returns, labels, n=10, draws=300, seed=0, plot=True):
    """Build many equal-weighted n-stock portfolios three ways: all n stocks from ONE cluster,
    n stocks picked at random, and one stock from each of n different clusters. Compare how much
    the portfolios swing (annualised volatility) and how closely their stocks move together
    (average correlation). Lower = better diversified for the same number of stocks."""
    rng = np.random.default_rng(seed)
    r = returns[labels.index]
    groups = {c: list(t) for c, t in labels.groupby(labels).groups.items()}
    big = [c for c in groups if len(groups[c]) >= n]
    k = len(groups)
    vols = {"all from one cluster": [], "random picks": [], "one per cluster": []}
    cors = {key: [] for key in vols}
    iu = np.triu_indices(n, 1)
    for _ in range(draws):
        picks = {
            "all from one cluster": list(rng.choice(groups[rng.choice(big)], n, replace=False)),
            "random picks": list(rng.choice(labels.index, n, replace=False)),
            "one per cluster": [rng.choice(groups[c]) for c in rng.choice(list(groups), n, replace=n > k)],
        }
        for key, pk in picks.items():
            vols[key].append(r[pk].mean(axis=1).std() * np.sqrt(12) * 100)
            cors[key].append(np.nanmean(r[pk].corr().values[iu]))
    if plot:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 4))
        data = [vols[key] for key in vols]
        bp = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False, widths=0.55)
        for patch, col in zip(bp["boxes"], [NAVY, GREY, GOLD]):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_yticks([1, 2, 3], list(vols))
        ax.set_xlabel(f"Volatility of an equal-weighted {n}-stock portfolio (% a year)")
        ax.set_title(f"Same number of stocks, different diversification ({draws} portfolios each)")
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.show()
    return pd.DataFrame({"median volatility (% a year)": {key: np.median(v) for key, v in vols.items()},
                         "avg correlation between the stocks": {key: np.median(v) for key, v in cors.items()}}).round(2)


# ---------------------------------------------------------------------------------------------
# Week 10: portfolio construction
# ---------------------------------------------------------------------------------------------
PORTFOLIO_STOCKS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "WMT", "HD", "CAT"]

WEIGHTING_NAMES = {"equal": "Equal weight", "rank": "Rank-weighted", "inverse_vol": "Inverse volatility",
                   "cluster": "Cluster-balanced"}


def compare_weightings(panel, model, schemes=("equal", "rank", "inverse_vol", "cluster"), n_long=30,
                       rebalance="Q", cost_bps=20, plot=True, **kwargs):
    """Same model, same ranked list, different ways of turning the list into a portfolio.
    Runs one walk-forward backtest per weighting scheme and returns (table, results).
    Extra arguments go to walk_forward_backtest (e.g. features=..., max_weight=0.05)."""
    results, rows = {}, []
    for sc in schemes:
        res = walk_forward_backtest(panel, model, n_long=n_long, rebalance=rebalance, cost_bps=cost_bps,
                                    weighting=sc, verbose=False, **kwargs)
        results[sc] = res
        st = res["stats"]["Strategy (after costs)"]
        rows.append({"scheme": WEIGHTING_NAMES.get(sc, sc), "Ann. return": st["Ann. return"],
                     "Volatility": st["Volatility"], "Sharpe ratio": st["Sharpe ratio"],
                     "Max drawdown": st["Max drawdown"], "Turnover/yr": st["Turnover/yr"]})
    spy = results[schemes[0]]["stats"]["SPY"]
    rows.append({"scheme": "SPY", "Ann. return": spy["Ann. return"], "Volatility": spy["Volatility"],
                 "Sharpe ratio": spy["Sharpe ratio"], "Max drawdown": spy["Max drawdown"], "Turnover/yr": np.nan})
    if plot:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        cols = [NAVY, "#3f6aa0", "#8c6100", GOLD, GREY]
        for (sc, res), col in zip(results.items(), cols):
            cur = res["cumulative"]["strategy"]
            ax.plot(cur.index, cur.values, color=col, lw=2, label=WEIGHTING_NAMES.get(sc, sc))
        spyc = results[schemes[0]]["cumulative"]["SPY"]
        ax.plot(spyc.index, spyc.values, color="black", lw=1.5, ls="--", label="SPY")
        ax.set_title(f"Same {n_long} picks, different weights (after {cost_bps} bps costs)")
        ax.set_ylabel("Value of $1 invested")
        ax.set_xlabel("Date")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    return pd.DataFrame(rows).set_index("scheme").round(3), results


def _mv_inputs(tickers, start, end, drop_year=None):
    r = to_returns(load_prices("M"))[list(tickers)].loc[str(start):str(end)].dropna()
    if drop_year is not None:
        r = r[r.index.year != int(drop_year)]
    return r.mean() * 12, r.cov() * 12, r


def mean_variance_weights(tickers=None, start="2015", end="2019", drop_year=None, long_only=False):
    """'Optimal' weights from mean-variance optimization (the maximum-Sharpe portfolio) using
    average past returns and past covariances as if they were the truth. With long_only=False
    the optimizer may short (negative weights). Returns weights that sum to 1."""
    tickers = PORTFOLIO_STOCKS if tickers is None else list(tickers)
    mu, cov, _ = _mv_inputs(tickers, start, end, drop_year)
    if not long_only:
        w = np.linalg.solve(cov.values, mu.values)
        w = w / w.sum()
    else:
        from scipy.optimize import minimize
        n = len(tickers)
        neg_sharpe = lambda x: -(x @ mu.values) / np.sqrt(x @ cov.values @ x)
        res = minimize(neg_sharpe, np.ones(n) / n, bounds=[(0, 1)] * n,
                       constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], method="SLSQP")
        w = res.x
    return pd.Series(w, index=tickers).round(3)


def fragility_experiment(tickers=None, start="2015", end="2019", drop_year=2018, plot=True):
    """The 'optimal' portfolio is fragile. Estimate maximum-Sharpe weights on start-end, then
    again after dropping ONE year of data, and compare. Shorting is allowed, as in the textbook
    version. Returns a table of weights (as % of the portfolio)."""
    tickers = PORTFOLIO_STOCKS if tickers is None else list(tickers)
    a = mean_variance_weights(tickers, start, end) * 100
    b = mean_variance_weights(tickers, start, end, drop_year=drop_year) * 100
    lab_b = f"{start}–{end} without {drop_year}"
    out = pd.DataFrame({f"{start}–{end}": a, lab_b: b, "Equal weight": 100 / len(tickers)})
    if plot:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(tickers))
        ax.bar(x - 0.2, out.iloc[:, 0], 0.4, color=NAVY, label=out.columns[0])
        ax.bar(x + 0.2, out.iloc[:, 1], 0.4, color=GOLD, label=out.columns[1])
        ax.axhline(100 / len(tickers), color="black", ls="--", lw=1, label="Equal weight")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, tickers)
        ax.set_ylabel("Weight (% of the portfolio; below 0 = short)")
        ax.set_title("'Optimal' mean-variance weights, estimated twice")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()
    return out.round(1)


def out_of_sample_check(tickers=None, est_start="2015", est_end="2019", test_start="2020", test_end="2025"):
    """Hold the portfolios estimated on est_start-est_end through test_start-test_end (rebalanced
    monthly to the same weights) and compare with equal weight: return, volatility, Sharpe."""
    tickers = PORTFOLIO_STOCKS if tickers is None else list(tickers)
    r = to_returns(load_prices("M"))[tickers].loc[str(test_start):str(test_end)].dropna()
    rf = load_factors()["RF"].reindex(r.index).fillna(0)
    ports = {"Mean-variance (shorting allowed)": mean_variance_weights(tickers, est_start, est_end),
             "Mean-variance (long only)": mean_variance_weights(tickers, est_start, est_end, long_only=True),
             "Equal weight": pd.Series(1 / len(tickers), index=tickers)}
    rows = {}
    for k, w in ports.items():
        x = r @ w.reindex(tickers).values
        cum = (1 + x).cumprod()
        rows[k] = {"Ann. return": _ann(x), "Volatility": x.std() * np.sqrt(12),
                   "Sharpe ratio": (x - rf).mean() / x.std() * np.sqrt(12),
                   "Max drawdown": (cum / cum.cummax() - 1).min(),
                   "Largest position": w.abs().max(), "Short positions": int((w < 0).sum())}
    return pd.DataFrame(rows).T.round(3)


def plot_frontier(tickers=None, start="2015", end="2019", n_portfolios=4000, seed=0, title=None):
    """The efficient frontier as a picture: thousands of random long-only portfolios of the same
    stocks, each a dot at its risk (volatility) and return. The upper-left edge of the cloud is the
    'efficient frontier'; the stocks on their own are marked in gold."""
    tickers = PORTFOLIO_STOCKS if tickers is None else list(tickers)
    mu, cov, _ = _mv_inputs(tickers, start, end)
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(tickers)) * 0.4, n_portfolios)
    ret = W @ mu.values
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov.values, W))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(vol * 100, ret * 100, s=4, color=NAVY, alpha=0.25, label="random portfolios")
    sv = np.sqrt(np.diag(cov.values))
    ax.scatter(sv * 100, mu.values * 100, s=60, color=GOLD, edgecolor="black", zorder=3, label="single stocks")
    for t, x, y in zip(tickers, sv, mu.values):
        ax.annotate(t, (x * 100, y * 100), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ew = np.ones(len(tickers)) / len(tickers)
    ax.scatter([np.sqrt(ew @ cov.values @ ew) * 100], [ew @ mu.values * 100], marker="*", s=250,
               color="white", edgecolor="black", zorder=4, label="equal weight")
    ax.set_xlabel(f"Risk: volatility (% a year), {start}–{end}")
    ax.set_ylabel(f"Average return (% a year), {start}–{end}")
    ax.set_title(title or "The efficient frontier: the upper-left edge of the cloud")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# ---------------------------------------------------------------------------------------------
# Week 12: the full pipeline (project template)
# ---------------------------------------------------------------------------------------------
MODEL_MENU = {
    "boosting": "gradient-boosted trees (the course default)",
    "forest": "random forest",
    "logistic": "logistic regression (a straight-line model)",
    "tree": "a single small decision tree",
}


def make_model(name="boosting", seed=0):
    """The project's model menu. Every model predicts the chance a stock beats SPY next month.
    name: "boosting", "forest", "logistic" or "tree" (see MODEL_MENU)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    name = name.lower()
    if name == "boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=seed)
    if name == "forest":
        return RandomForestClassifier(n_estimators=100, min_samples_leaf=200, max_features=0.5, max_samples=0.25,
                                      n_jobs=-1, random_state=seed)
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "tree":
        return DecisionTreeClassifier(max_depth=4, min_samples_leaf=500, random_state=seed)
    raise ValueError(f"model must be one of {list(MODEL_MENU)}")


def monkey_portfolios(panel, n_funds=200, n_long=30, rebalance="Q", cost_bps=20, test_start="2015-01-01",
                      seed=0, result=None, plot=True):
    """'Could a monkey have done it?' Build n_funds funds that pick n_long stocks AT RANDOM from the
    same universe on the same dates, pay the same trading costs, and compare. If many random funds
    beat your fund, your edge is not convincing. Also shows how many random funds look 'significant'
    (t-statistic above 2 against SPY) purely by chance. Pass result= to mark your own fund."""
    rng = np.random.default_rng(seed)
    step = {"M": 1, "Q": 3, "A": 12, "Y": 12}[rebalance.upper()[0]]
    dates = sorted(d for d in panel["date"].unique() if d >= pd.Timestamp(test_start))
    spy_next = panel.groupby("date")["spy_next"].first().reindex(dates)
    keep = spy_next.notna().values
    dates = [d for d, k in zip(dates, keep) if k]
    spy_next = spy_next.reindex(dates)
    nxt = to_returns(load_prices("M")).shift(-1)
    universe = panel.groupby("date")["ticker"].apply(list)
    rf = load_factors()["RF"]
    rows, curves = [], []
    for f in range(n_funds):
        rets, held = [], []
        for i, d in enumerate(dates):
            cost = 0.0
            if i % step == 0:
                new = list(rng.choice(universe[d], min(n_long, len(universe[d])), replace=False))
                overlap = len(set(new) & set(held)) / n_long if held else 0.0
                cost = (2 * (1 - overlap) if held else 1.0) * cost_bps / 1e4
                held = new
            r = nxt.loc[d].reindex(held).fillna(0.0).mean() if d in nxt.index else 0.0
            rets.append(r - cost)
        x = pd.Series(rets, index=[d + pd.offsets.MonthEnd(1) for d in dates])
        ex = x - spy_next.values
        rows.append({"Ann. return": _ann(x), "Sharpe ratio": (x - rf.reindex(x.index).fillna(0)).mean() / x.std() * np.sqrt(12),
                     "t-stat vs SPY": ex.mean() / ex.std() * np.sqrt(len(ex))})
        curves.append(x)
    t = pd.DataFrame(rows)
    spy_ann = _ann(pd.Series(spy_next.values))
    summary = {"random funds": n_funds,
               "median random fund, ann. return": round(t["Ann. return"].median(), 3),
               "SPY, ann. return": round(spy_ann, 3),
               "random funds that beat SPY": f"{(t['Ann. return'] > spy_ann).mean():.0%}",
               "random funds with t > 2 vs SPY ('significant' by luck)": f"{(t['t-stat vs SPY'] > 2).mean():.0%}"}
    mine = None
    if result is not None:
        mine = result["stats"].loc["Ann. return", "Strategy (after costs)"]
        summary["your fund, ann. return"] = round(mine, 3)
        summary["random funds your fund beat"] = f"{(t['Ann. return'] < mine).mean():.0%}"
    if plot:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(t["Ann. return"] * 100, bins=30, color=GREY, edgecolor="white", label=f"{n_funds} random funds")
        ax.axvline(spy_ann * 100, color="black", ls="--", lw=1.5, label=f"SPY {spy_ann:.1%}")
        if mine is not None:
            ax.axvline(mine * 100, color=GOLD, lw=3, label=f"your fund {mine:.1%}")
        ax.set_xlabel(f"Annual return after {cost_bps} bps costs (%)")
        ax.set_ylabel("Number of random funds")
        ax.set_title(f"Could a monkey have done it? {n_long} random stocks, {dict(M='monthly', Q='quarterly', A='yearly').get(rebalance.upper()[0])}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()
    return pd.Series(summary, name="monkey test"), t.round(3)


def portfolio_concentration(result, k=8):
    """How many clusters (groups of stocks that move together, Week 9) the fund's most recent
    picks fall into. Clusters are estimated on the 5 years before January of that year (the same
    clusters a "cluster" or max_per_cluster backtest used)."""
    last = max(result["picks"])
    picks = result["picks"][last]
    # same window the backtest uses: the 5 years before the first rebalance of that year
    d0 = min(d for d in result["picks"] if d.year == last.year)
    past = to_returns(load_prices("M")).loc[d0 - pd.DateOffset(years=5):d0 - pd.offsets.MonthEnd(1)]
    labels = cluster_stocks(past, k=k, plot=False)
    info = load_universe().drop_duplicates("ticker").set_index("ticker")
    sec = info["sector"].reindex(labels.index)
    names = {c: sec[labels.index[labels == c]].value_counts().index[0] for c in labels.unique()}
    lab = labels.reindex(picks)
    out = (lab.map(lambda c: f"cluster {int(c)}: mostly {names[c]}" if pd.notna(c) else "not clustered (new stock)")
           .value_counts().rename("picks").to_frame())
    print(f"Picks on {last:%b %Y}: {len(picks)} stocks in {lab.nunique()} of {k} clusters; "
          f"the largest cluster holds {out['picks'].max()} of them.")
    return out


def fund_summary(result, panel, monkey=None, name="Our fund"):
    """The numbers for the pitch, in one table."""
    st = result["stats"]
    ex = (result["returns"]["strategy"] - result["returns"]["SPY"]).dropna()
    hit = pick_hit_rate(result, panel)
    rows = {
        "Annual return after costs": f"{st.loc['Ann. return', 'Strategy (after costs)']:.1%}",
        "SPY annual return": f"{st.loc['Ann. return', 'SPY']:.1%}",
        "Sharpe ratio (SPY)": f"{st.loc['Sharpe ratio', 'Strategy (after costs)']:.2f} ({st.loc['Sharpe ratio', 'SPY']:.2f})",
        "Worst drawdown (SPY)": f"{st.loc['Max drawdown', 'Strategy (after costs)']:.0%} ({st.loc['Max drawdown', 'SPY']:.0%})",
        "Turnover per year": f"{st.loc['Turnover/yr', 'Strategy (after costs)']:.1f}x",
        "Picks that beat SPY next month (all stocks)": f"{hit.iloc[0]:.1%} ({hit.iloc[1]:.1%})",
        "Edge over SPY, t-statistic": f"{ex.mean() / ex.std() * np.sqrt(len(ex)):.2f}",
        "Test period": f"{result['returns'].index.min():%b %Y} – {result['returns'].index.max():%b %Y}",
    }
    if monkey is not None and "random funds your fund beat" in monkey:
        rows["Random funds beaten"] = monkey["random funds your fund beat"]
    return pd.Series(rows, name=name).to_frame()

# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------
def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_cumulative(result, title=None, log=False):
    """Growth of $1 for the strategy, SPY and the equal-weighted universe."""
    plt = _plt()
    c = result["cumulative"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(c.index, c["strategy"], label="Strategy (after costs)", lw=2)
    ax.plot(c.index, c["SPY"], label="SPY (S&P 500)", lw=1.5)
    ax.plot(c.index, c["equal_weight_universe"], label="Equal-weight universe", lw=1, ls="--")
    s = result["settings"]
    ax.set_title(title or f"Growth of $1 - top {s['n_long']}, rebalance {s['rebalance']}, cost {s['cost_bps']} bps")
    ax.set_ylabel("Value of $1 invested")
    ax.set_xlabel("Date")
    if log:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_drawdown(result, title=None):
    """How far each portfolio was below its previous peak, month by month."""
    plt = _plt()
    c = result["cumulative"]
    dd = c / c.cummax() - 1
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.fill_between(dd.index, dd["strategy"] * 100, 0, alpha=0.35, color="C0", label="Strategy")
    ax.plot(dd.index, dd["SPY"] * 100, color="C1", label="SPY", lw=1.4)
    ax.set_title(title or "Drawdown: % below the previous peak")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_importance(model, feature_names=None, X=None, y=None, top=15, title=None):
    """Bar chart of which features the model relies on most.
    Random forests and single trees use their built-in importances; linear models use the size of
    their coefficients (meaningful only if features are on similar scales). Pass X and y to use
    'permutation importance' instead: how much worse the model gets when one feature is shuffled.
    You can also pass a backtest result: mlfin.plot_importance(result) explains the final model.
    Gradient boosting (HistGradientBoosting...) has no built-in importances, so it always needs
    X and y (or a backtest result)."""
    plt = _plt()
    if isinstance(model, dict) and "model" in model:
        X, y = model["train_sample"]
        model = model["model"]
    names = list(feature_names) if feature_names is not None else getattr(model, "_mlfin_columns", None)
    if X is not None and y is not None:
        from sklearn.inspection import permutation_importance
        scoring = "roc_auc" if hasattr(model, "predict_proba") else "r2"
        pi = permutation_importance(model, X, y, n_repeats=5, random_state=0, scoring=scoring)
        vals, names = pi.importances_mean, list(X.columns)
        label = f"Drop in {'AUC' if scoring == 'roc_auc' else 'R-squared'} when shuffled (below 0 = shuffling helped)"
    elif hasattr(model, "feature_importances_"):
        vals, label = model.feature_importances_, "Importance"
    elif hasattr(model, "coef_"):
        vals, label = np.abs(np.ravel(model.coef_)), "|Coefficient|"
    else:
        raise ValueError("This model has no built-in importances. Pass some data to measure them: "
                         "mlfin.plot_importance(model, X=X_test, y=y_test)")
    if names is None:
        names = [f"x{i}" for i in range(len(vals))]
    s = pd.Series(vals, index=names).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(s) + 1.2))
    ax.barh(s.index, s.values)
    ax.set_title(title or f"What the model relies on ({type(model).__name__})")
    ax.set_xlabel(label)
    ax.set_ylabel("Feature")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()
    return s.sort_values(ascending=False)


def plot_confusion(y_true, y_pred, labels=("Lagged SPY", "Beat SPY"), title=None):
    """2x2 table of predictions vs what actually happened, with counts and percentages."""
    plt = _plt()
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.6)   # keep every cell light enough to read
    tot = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / tot:.0%})", ha="center", va="center",
                    color="black", fontsize=11)
    ax.set_xticks([0, 1], [f"Predicted\n{labels[0]}", f"Predicted\n{labels[1]}"])
    ax.set_yticks([0, 1], [f"Actually\n{labels[0]}", f"Actually\n{labels[1]}"])
    ax.set_title(title or "Confusion matrix")
    plt.tight_layout()
    plt.show()
    return cm


def plot_deciles(y_pred, y_true, title=None, dates=None):
    """Sort stocks into 10 groups by the model's prediction (1 = lowest, 10 = highest) and show
    the average ACTUAL outcome in each group. A useful model gives bars that rise left to right.
    If dates are given, groups are formed within each month."""
    plt = _plt()
    df = pd.DataFrame({"pred": np.asarray(y_pred), "true": np.asarray(y_true)})
    if dates is not None:
        df["date"] = np.asarray(dates)
        df["decile"] = df.groupby("date")["pred"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    else:
        df["decile"] = pd.qcut(df["pred"].rank(method="first"), 10, labels=False) + 1
    m = df.groupby("decile")["true"].mean()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(m.index, m.values * 100)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Prediction group (1 = lowest prediction, 10 = highest)")
    ax.set_ylabel("Average actual outcome (%)")
    ax.set_title(title or "Do higher predictions come true?")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()
    return m


# ---------------------------------------------------------------------------------------------
# Unsupervised learning
# ---------------------------------------------------------------------------------------------
def cluster_stocks(returns, k=8, random_state=0, plot=True, min_months=None):
    """Group stocks whose returns move together, using k-means.
    Each stock's monthly returns are standardised (so only the up/down pattern matters, not the
    size of the moves); stocks with too much missing data are left out.
    Returns a Series: ticker -> cluster number (0..k-1). With plot=True it also shows the stocks
    on a 2-D map (the first two principal components) coloured by cluster."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    r = returns.drop(columns=[BENCHMARK], errors="ignore")
    min_months = min_months or int(0.9 * len(r))
    r = r.loc[:, r.notna().sum() >= min_months]
    r = r.fillna(r.mean())
    z = (r - r.mean()) / r.std()
    Z = z.T.values
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Z)
    labels = pd.Series(km.labels_, index=z.columns, name="cluster")
    if plot:
        plt = _plt()
        xy = PCA(n_components=2, random_state=random_state).fit_transform(Z)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels.values, cmap="tab10", s=18)
        ax.set_title(f"{len(labels)} stocks in {k} clusters (map of return patterns)")
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        ax.legend(*sc.legend_elements(), title="Cluster", fontsize=8, loc="best")
        plt.tight_layout()
        plt.show()
    return labels


# ---------------------------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------------------------
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def sentiment_dictionary(text):
    """Score text with the Loughran-McDonald finance dictionary:
        (positive words - negative words) / (positive words + negative words)
    giving a number from -1 (all negative) to +1 (all positive); 0 if no listed words appear.
    Works on a single string or a pandas Series / list of strings."""
    if "_lm" not in _cache:
        lm = load_lm_dictionary()
        _cache["_lm"] = (set(lm.loc[lm["positive"] == 1, "word"]), set(lm.loc[lm["negative"] == 1, "word"]))
    pos, neg = _cache["_lm"]

    def one(t):
        words = _WORD.findall(str(t).lower())
        p = sum(w in pos for w in words)
        n = sum(w in neg for w in words)
        return 0.0 if p + n == 0 else (p - n) / (p + n)

    if isinstance(text, str):
        return one(text)
    return pd.Series([one(t) for t in text], index=getattr(text, "index", None), name="lm_sentiment")



def dictionary_label(text, neutral_band=0.0):
    """Turn the dictionary score into a label: positive (> band), negative (< -band), else neutral."""
    sc = sentiment_dictionary(text)
    if isinstance(sc, float):
        return "positive" if sc > neutral_band else "negative" if sc < -neutral_band else "neutral"
    return pd.Series(np.where(sc > neutral_band, "positive", np.where(sc < -neutral_band, "negative", "neutral")),
                     index=sc.index, name="dictionary_label")


def dictionary_words(text):
    """Which Loughran-McDonald positive and negative words appear in a piece of text."""
    sentiment_dictionary("warm up")
    pos, neg = _cache["_lm"]
    words = _WORD.findall(str(text).lower())
    return {"positive": [w for w in words if w in pos], "negative": [w for w in words if w in neg]}


def text_classifier(vocab_size=2000):
    """A simple trainable text model: TF-IDF word (and two-word) counts + logistic regression.
    vocab_size = how many of the most common words/phrases it may use (None = all)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(max_features=vocab_size, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, C=5))


def headline_split(headlines=None, test_size=0.25, seed=0):
    """Split the labelled headlines into training and test sets (same label mix in both)."""
    from sklearn.model_selection import train_test_split
    h = load_headlines() if headlines is None else headlines
    return train_test_split(h, test_size=test_size, random_state=seed, stratify=h["label"])


def text_experiment(vocab_size=2000, test_size=0.25, seed=0):
    """Three ways to label financial headlines, graded on the same unseen test headlines:
    always say 'neutral' (the most common label), the Loughran-McDonald dictionary, and a
    classifier trained on the other headlines. Returns (accuracy table, fitted classifier, test set)."""
    train, test = headline_split(test_size=test_size, seed=seed)
    clf = text_classifier(vocab_size).fit(train["text"], train["label"])
    pred = clf.predict(test["text"])
    acc = pd.Series({
        "always 'neutral'": (test["label"] == "neutral").mean(),
        "finance dictionary (Loughran-McDonald)": (dictionary_label(test["text"]).values == test["label"].values).mean(),
        f"trained classifier ({vocab_size or 'all'} words)": (pred == test["label"].values).mean(),
    }, name="accuracy on unseen headlines").round(3)
    print(f"Trained on {len(train):,} labelled headlines, tested on {len(test):,} it never saw.")
    return acc.to_frame(), clf, test


def vocab_experiment(sizes=(50, 100, 500, 2000, None), seed=0):
    """Accuracy on the training and the test headlines as the classifier is allowed more words."""
    train, test = headline_split(seed=seed)
    rows = []
    for v in sizes:
        clf = text_classifier(v).fit(train["text"], train["label"])
        rows.append({"words allowed": v or "all", "train accuracy": (clf.predict(train["text"]) == train["label"]).mean(),
                     "test accuracy": (clf.predict(test["text"]) == test["label"]).mean()})
    return pd.DataFrame(rows).set_index("words allowed").round(3)


def text_confusion(test, clf=None):
    """Rows = what the human labellers said; columns = what the method said."""
    pred = clf.predict(test["text"]) if clf is not None else dictionary_label(test["text"]).values
    order = ["negative", "neutral", "positive"]
    return pd.crosstab(pd.Categorical(test["label"], order), pd.Categorical(pred, order),
                       rownames=["human label"], colnames=["predicted"], dropna=False)


def dictionary_mistakes(n=10, seed=1, agreement=100):
    """Headlines where every human labeller agreed, but the dictionary disagreed with them."""
    h = load_headlines()
    h = h[h["agreement"] >= agreement]
    d = dictionary_label(h["text"])
    wrong = h[d != h["label"]].assign(dictionary=d[d != h["label"]])
    out = wrong.sample(min(n, len(wrong)), random_state=seed)
    out = out.assign(words_found=[dictionary_words(t) for t in out["text"]])
    print(f"The dictionary disagrees with unanimous human labels on {len(wrong):,} of {len(h):,} headlines "
          f"({len(wrong) / len(h):.0%}).")
    return out[["text", "label", "dictionary", "words_found"]].rename(columns={"label": "human"})


def plot_top_words(clf, n=12, title=None):
    """The words and phrases that push the classifier most toward 'positive' and toward 'negative'."""
    vec, lr = clf.steps[0][1], clf.steps[-1][1]
    names = np.array(vec.get_feature_names_out())
    classes = list(lr.classes_)
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * n + 1.5))
    for ax, cl, col in zip(axes, ["positive", "negative"], [GOLD, NAVY]):
        coef = lr.coef_[classes.index(cl)]
        idx = np.argsort(coef)[-n:]
        ax.barh(names[idx], coef[idx], color=col)
        ax.set_title(f"Words that push toward '{cl}'")
        ax.set_xlabel("Weight in the model")
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle(title or "What the text classifier learned", fontsize=12)
    plt.tight_layout()
    plt.show()


def score_headlines(texts, clf):
    """Score your own headlines with both methods, side by side."""
    texts = list(texts)
    proba = clf.predict_proba(texts)
    out = pd.DataFrame({"headline": texts,
                        "dictionary score": sentiment_dictionary(pd.Series(texts)).round(2).values,
                        "dictionary label": dictionary_label(pd.Series(texts)).values,
                        "classifier label": clf.predict(texts),
                        "classifier confidence": proba.max(axis=1).round(2),
                        "dictionary words found": [dictionary_words(t) for t in texts]})
    return out

# ---------------------------------------------------------------------------------------------
# Week 1: the overfitting game
# ---------------------------------------------------------------------------------------------
RULE_FEATURES = ["mom_12_1", "rev_1", "vol_12", "size", "btm", "ep", "growth", "leverage",
                 "beta_12", "max_1"]


def make_random_rules(n_rules=200, conditions_per_rule=3, features=None, seed=0):
    """Invent n_rules random stock-picking rules. Each rule combines a few random conditions such as
    "momentum in the top 30%" or "volatility in the bottom 40%" (percentages are each month's
    ranking of all stocks). Nothing about these rules is clever - they are pure chance."""
    rng = np.random.default_rng(seed)
    features = features or RULE_FEATURES
    rules = []
    for i in range(n_rules):
        chosen = rng.choice(features, size=conditions_per_rule, replace=False)
        conds = []
        for f in chosen:
            pct = int(rng.choice([20, 30, 40, 50]))
            side = str(rng.choice(["top", "bottom"]))
            conds.append((str(f), side, pct))
        text = " AND ".join(f"{f} in {side} {pct}%" for f, side, pct in conds)
        rules.append({"rule_id": i + 1, "conditions": conds, "rule": text})
    return rules


def test_rules(panel, rules, start, end, min_stocks=5):
    """Score every rule over the months from start to end: each month, buy the stocks that pass
    the rule, and measure how much better (or worse) they did than SPY the following month.
    Returns one row per rule with the average monthly result in percent."""
    p = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))]
    p = p.dropna(subset=["ret_next", "spy_next"])
    feats = sorted({f for r in rules for f, _, _ in r["conditions"]})
    ranks = p.groupby("date")[feats].rank(pct=True)
    excess = p["ret_next"] - p["spy_next"]
    out = []
    for r in rules:
        mask = pd.Series(True, index=p.index)
        for f, side, pct in r["conditions"]:
            mask &= (ranks[f] >= 1 - pct / 100) if side == "top" else (ranks[f] <= pct / 100)
        sel = excess[mask].groupby(p.loc[mask, "date"])
        monthly = sel.mean()[sel.size() >= min_stocks]
        out.append({"rule_id": r["rule_id"], "rule": r["rule"],
                    "avg_monthly_vs_SPY_%": monthly.mean() * 100 if len(monthly) else np.nan,
                    "stocks_per_month": sel.size().mean() if len(monthly) else 0})
    return pd.DataFrame(out).set_index("rule_id")


def plot_rules_in_vs_out(in_sample, out_of_sample, highlight=10, title=None):
    """Each dot is one rule: how it did in the period we searched (x) against how the same rule
    did afterwards (y). The highlighted dots are the rules that looked best in the search period."""
    plt = _plt()
    df = pd.DataFrame({"in": in_sample["avg_monthly_vs_SPY_%"],
                       "out": out_of_sample["avg_monthly_vs_SPY_%"]}).dropna()
    best = df["in"].nlargest(highlight).index
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(df["in"], df["out"], s=14, alpha=0.5, label="all rules")
    ax.scatter(df.loc[best, "in"], df.loc[best, "out"], s=40, color="C3",
               label=f"the {highlight} best-looking rules")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Search period: average monthly return vs SPY (%)")
    ax.set_ylabel("Later period: average monthly return vs SPY (%)")
    ax.set_title(title or "Do rules that worked before keep working?")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return df["in"].corr(df["out"])


# ---------------------------------------------------------------------------------------------
# Week 2: data problems
# ---------------------------------------------------------------------------------------------
def survivorship_portfolios(start="2010-01-31", end="2025-12-31"):
    """Two equal-weighted portfolios, rebalanced monthly:
      "Today's S&P 500 members" - the stocks in the index at the END of 2025, held all along.
          Choosing them in 2010 would have required knowing which companies would survive.
      "Members at the time" - each month, the stocks that were actually in the index then
          (as far as we have prices for them).
    Returns their monthly returns. The gap between them is survivorship bias."""
    pm = load_prices("M")
    uni = load_universe()
    ret = to_returns(pm).loc[start:end]
    survivors = [t for t in uni.loc[uni["in_index_at_end"], "ticker"] if t in ret.columns]
    nxt = ret.shift(-1)
    rows = []
    for d in ret.index[:-1]:
        members = uni[(uni["first_date"] <= d) & (uni["last_date"] >= d)]["ticker"]
        members = [t for t in members if t in ret.columns]
        rows.append((nxt.index[ret.index.get_loc(d) + 1],
                     nxt.loc[d, survivors].mean(), nxt.loc[d, members].mean()))
    return pd.DataFrame(rows, columns=["date", "Today's S&P 500 members",
                                       "Members at the time"]).set_index("date")


def messy_prices():
    """A small 'raw' price file for five stocks, 2018-2021, of the kind you might download
    yourself. It contains two deliberate data problems for you to find (Week 2 exercise)."""
    pm = load_prices("M").loc["2018-01-31":"2021-12-31", ["AAPL", "KO", "JPM", "XOM", "NKE"]].copy()
    # problem 1: a stock split that was NOT adjusted for (AAPL 4-for-1, 31 Aug 2020)
    pm.loc[:"2020-07-31", "AAPL"] *= 4
    # problem 2: a month that is simply missing for one stock
    pm.loc["2019-06-30", "JPM"] = np.nan
    return pm.round(2)


# ---------------------------------------------------------------------------------------------
# Week 3: pictures
# ---------------------------------------------------------------------------------------------
def plot_scatter(panel, x, y="ret_next", sample=3000, seed=0, title=None):
    """Scatter one feature against next month's return for a random sample of stock-months, with
    the straight 'line of best guess' through the cloud."""
    plt = _plt()
    d = panel[[x, y]].dropna()
    d = d.sample(min(sample, len(d)), random_state=seed)
    slope, intercept = np.polyfit(d[x], d[y], 1)
    xs = np.linspace(d[x].quantile(0.01), d[x].quantile(0.99), 50)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(d[x], d[y] * 100, s=6, alpha=0.3)
    ax.plot(xs, (intercept + slope * xs) * 100, color="C3", lw=2.5, label="line of best guess")
    ax.set_xlabel(f"{x}: {FEATURE_DESCRIPTIONS.get(x, x)}")
    ax.set_ylabel("Next month's return (%)")
    ax.set_ylim(np.percentile(d[y] * 100, [0.5, 99.5]))
    ax.set_title(title or f"{x} vs next month's return ({len(d):,} random stock-months)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return slope


def plot_timeline(train_end="2017-12-31", start="2010", end="2025"):
    """The train / test split drawn as a timeline: the model learns only from the left part and is
    graded only on the right part."""
    plt = _plt()
    s, e, t = pd.Timestamp(f"{start}-01-01"), pd.Timestamp(f"{end}-12-31"), pd.Timestamp(train_end)
    fig, ax = plt.subplots(figsize=(9, 1.6))
    ax.barh(0, (t - s).days, left=s, color="C0", height=0.5)
    ax.barh(0, (e - t).days, left=t, color="C1", height=0.5)
    ax.text(s + (t - s) / 2, 0, "TRAIN: the model learns here", ha="center", va="center", color="white", fontsize=11)
    ax.text(t + (e - t) / 2, 0, "TEST: graded here", ha="center", va="center", color="white", fontsize=11)
    ax.set_yticks([])
    ax.set_title(f"Time-ordered split at {t:%d %b %Y}")
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_histogram(returns, title=None, bins=40, period="Monthly"):
    """Histogram of returns (in %) with the bell curve a 'normal' distribution with the same average
    and volatility would have. Bars far out beyond the curve are 'fat tails': extreme months happen
    more often than the bell curve says."""
    plt = _plt()
    r = pd.Series(returns).dropna() * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(r, bins=bins, density=True, alpha=0.7, label=f"actual {period.lower()} returns")
    xs = np.linspace(r.min(), r.max(), 200)
    mu, sd = r.mean(), r.std()
    ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color="C3", lw=2,
            label="bell curve (normal) with same average & volatility")
    ax.set_xlabel(f"{period} return (%)")
    ax.set_ylabel("How often (density)")
    ax.set_title(title or f"{period} returns of {getattr(returns, 'name', '')}")
    ax.legend()
    plt.tight_layout()
    plt.show()
    from math import erfc, sqrt
    z = (r - mu) / sd
    actual = int((z.abs() > 4).sum())
    expected = len(r) * erfc(4 / sqrt(2))
    print(f"{len(r):,} {period.lower()} returns. Moves bigger than 4 standard deviations: "
          f"{actual} actually happened; a bell curve expects {expected:.1f}.")
