"""Core logic for the PPC vs Organic Overlap tool.

Parses a Google Search Console queries export and a Google Ads search terms
report, normalises both to a common schema, and builds the overlap views
(both / paid only / organic only / savings opportunities).

Kept free of Streamlit imports so it can be unit tested on its own.
"""

import csv
import io
import re

import pandas as pd

GSC_QUERY_COLUMNS = ("top queries", "query", "search query", "queries")
ADS_QUERY_COLUMN = "search term"

GSC_RENAME = {
    "top queries": "query",
    "search query": "query",
    "queries": "query",
    "query": "query",
    "clicks": "clicks",
    "impressions": "impressions",
    "ctr": "ctr",
    "position": "position",
}

ADS_RENAME = {
    "search term": "query",
    "impr.": "impressions",
    "impressions": "impressions",
    "clicks": "clicks",
    "cost": "cost",
    "conversions": "conversions",
    "all conv. value": "conv_value",
    "conv. value": "conv_value",
    "campaign": "campaign",
    "ad group": "ad_group",
    "match type": "match_type",
    "added/excluded": "added_excluded",
    "currency code": "currency_code",
}


def _decode_upload(raw: bytes) -> tuple[str, str]:
    """Decode an uploaded CSV to text, returning (text, separator).

    Google Ads offers both plain CSV (UTF-8) and Excel CSV (UTF-16 with
    tabs); GSC exports are UTF-8. BOM sniffing covers all three.
    """
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16"), "\t"
    return raw.decode("utf-8-sig", errors="replace"), ","


def normalise_query(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip formatting (thousands separators, %, currency symbols, ' --')
    from a numeric column and cast to float."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace({"": "0", "-": "0", "--": "0"})
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def parse_gsc_csv(raw: bytes) -> pd.DataFrame:
    """Parse a GSC queries export (Performance report > Export > Queries tab,
    or an API pull with a 'query' column). Returns one row per query with
    clicks, impressions, ctr and best position."""
    text, sep = _decode_upload(raw)
    df = pd.read_csv(io.StringIO(text), sep=sep, on_bad_lines="skip", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    if not any(c in df.columns for c in GSC_QUERY_COLUMNS):
        raise ValueError(
            "No query column found. Expected a GSC queries export with a "
            "'Top queries' or 'Query' column."
        )

    df = df.rename(columns={k: v for k, v in GSC_RENAME.items() if k in df.columns})
    for col in ("clicks", "impressions", "position"):
        if col not in df.columns:
            raise ValueError(f"GSC export is missing the '{col}' column.")
        df[col] = _clean_numeric(df[col])

    df["query"] = df["query"].map(normalise_query)
    df = df[df["query"] != ""]

    grouped = (
        df.groupby("query")
        .agg(
            gsc_clicks=("clicks", "sum"),
            gsc_impressions=("impressions", "sum"),
            gsc_position=("position", "min"),
        )
        .reset_index()
    )
    grouped["gsc_ctr"] = (
        (grouped["gsc_clicks"] / grouped["gsc_impressions"].replace(0, pd.NA)) * 100
    ).astype(float).round(2).fillna(0.0)
    return grouped


def parse_ads_csv(raw: bytes) -> pd.DataFrame:
    """Parse a Google Ads search terms report (Campaigns > Insights and
    reports > Search terms > Download). Handles the two-line preamble,
    'Total:' footer rows and formatted numbers."""
    text, sep = _decode_upload(raw)

    # The report preamble line "Search terms report" also contains the words
    # "search term", so require an exact column match, not a substring hit.
    header_idx = None
    for idx, line in enumerate(text.splitlines()[:20]):
        fields = next(csv.reader([line], delimiter=sep), [])
        if any(f.strip().lower() == ADS_QUERY_COLUMN for f in fields):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(
            "No 'Search term' column found. Make sure you are uploading the "
            "search terms report (not the keywords report)."
        )

    df = pd.read_csv(
        io.StringIO(text), sep=sep, skiprows=header_idx, on_bad_lines="skip", dtype=str
    )
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in ADS_RENAME.items() if k in df.columns})

    df = df[df["query"].notna()]
    df = df[~df["query"].str.startswith("Total:", na=False)]

    if "added_excluded" in df.columns:
        df = df[~df["added_excluded"].str.strip().eq("Excluded")]

    for col in ("impressions", "clicks", "cost", "conversions", "conv_value"):
        if col in df.columns:
            df[col] = _clean_numeric(df[col])
        else:
            df[col] = 0.0

    for col in ("campaign", "ad_group", "match_type"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["query"] = df["query"].map(normalise_query)
    df = df[df["query"] != ""]

    def _join_unique(values: pd.Series) -> str:
        return ", ".join(sorted({v for v in values if v}))

    grouped = (
        df.groupby("query")
        .agg(
            ads_impressions=("impressions", "sum"),
            ads_clicks=("clicks", "sum"),
            ads_cost=("cost", "sum"),
            conversions=("conversions", "sum"),
            conv_value=("conv_value", "sum"),
            campaigns=("campaign", _join_unique),
            ad_groups=("ad_group", _join_unique),
            match_types=("match_type", _join_unique),
        )
        .reset_index()
    )
    grouped["ads_ctr"] = (
        (grouped["ads_clicks"] / grouped["ads_impressions"].replace(0, pd.NA)) * 100
    ).astype(float).round(2).fillna(0.0)
    return grouped


def tag_brand(df: pd.DataFrame, brand_terms: list[str]) -> pd.DataFrame:
    """Add an is_brand column: True when the query contains any brand term."""
    df = df.copy()
    terms = [t.strip().lower() for t in brand_terms if t.strip()]
    if terms:
        pattern = "|".join(re.escape(t) for t in terms)
        df["is_brand"] = df["query"].str.contains(pattern, regex=True)
    else:
        df["is_brand"] = False
    return df


def build_overlap(df_gsc: pd.DataFrame, df_ads: pd.DataFrame) -> dict:
    """Return the three keyword views keyed 'both', 'paid_only', 'organic_only'."""
    both = df_gsc.merge(df_ads, how="inner", on="query")
    both = both.sort_values(by="ads_cost", ascending=False).reset_index(drop=True)

    paid_only = (
        df_ads[~df_ads["query"].isin(df_gsc["query"])]
        .sort_values(by="ads_cost", ascending=False)
        .reset_index(drop=True)
    )
    organic_only = (
        df_gsc[~df_gsc["query"].isin(df_ads["query"])]
        .sort_values(by="gsc_clicks", ascending=False)
        .reset_index(drop=True)
    )
    return {"both": both, "paid_only": paid_only, "organic_only": organic_only}


def savings_view(both: pd.DataFrame, max_position: float) -> pd.DataFrame:
    """Overlap keywords where the site already ranks at or above max_position
    organically, sorted by ad spend. These are the review candidates for
    reducing paid spend."""
    view = both[both["gsc_position"] <= max_position].copy()
    view = view.sort_values(by="ads_cost", ascending=False).reset_index(drop=True)
    cols = [
        "query",
        "gsc_position",
        "ads_cost",
        "ads_clicks",
        "gsc_clicks",
        "ads_impressions",
        "gsc_impressions",
        "conversions",
        "conv_value",
        "campaigns",
        "ad_groups",
    ]
    if "is_brand" in view.columns:
        cols.insert(1, "is_brand")
    return view[[c for c in cols if c in view.columns]]
