"""PPC vs Organic Overlap - compare Google Ads search terms with GSC queries.

Upload a GSC queries export and a Google Ads search terms report. The app
shows a Venn diagram of the keyword overlap, side-by-side paid and organic
stats, and a savings opportunities view: ad spend on search terms where the
site already ranks strongly in organic search.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib_venn import venn2

from overlap_logic import (
    build_overlap,
    parse_ads_csv,
    parse_gsc_csv,
    savings_view,
    tag_brand,
)

SAMPLE_DIR = Path(__file__).parent / "sample_data"

st.set_page_config(page_title="PPC vs Organic Overlap", page_icon="⚡", layout="wide")

st.write("# PPC vs Organic Overlap")
st.markdown(
    "Compare your **Google Ads search terms** with your **Google Search Console "
    "queries** to see where paid and organic overlap, and where you are paying "
    "for clicks on terms you already rank well for organically."
)

# ------------------------------- sidebar ------------------------------------

st.sidebar.header("Data")
use_sample = st.sidebar.toggle(
    "Use sample data", value=False, help="Explore the app with a synthetic demo dataset."
)

gsc_file = st.sidebar.file_uploader(
    "GSC queries export (CSV)",
    type=["csv"],
    help="Search Console > Performance > Export > Queries tab. API pulls with a 'query' column also work.",
)
ads_file = st.sidebar.file_uploader(
    "Google Ads search terms report (CSV)",
    type=["csv"],
    help="Google Ads > Insights and reports > Search terms > Download > CSV.",
)

st.sidebar.header("Settings")
max_position = st.sidebar.slider(
    "Organic strength threshold (average position)",
    min_value=1.0,
    max_value=10.0,
    value=3.0,
    step=0.5,
    help="The savings view lists overlap terms ranking at or above this organic position.",
)
brand_input = st.sidebar.text_input(
    "Brand terms (comma separated)",
    value="",
    help="Optional. Queries containing any of these terms are tagged as brand.",
)
brand_filter = st.sidebar.radio(
    "Show", options=["All terms", "Brand only", "Non-brand only"], index=0
)

# ------------------------------- load data ----------------------------------

gsc_bytes = ads_bytes = None
if use_sample:
    gsc_bytes = (SAMPLE_DIR / "gsc_queries_sample.csv").read_bytes()
    ads_bytes = (SAMPLE_DIR / "ads_search_terms_sample.csv").read_bytes()
    if not brand_input:
        brand_input = "grandview"
    st.info(
        "Showing synthetic sample data for a fictional hotel (Grandview). "
        "Upload your own exports in the sidebar to analyse real data."
    )
elif gsc_file is not None and ads_file is not None:
    gsc_bytes = gsc_file.getvalue()
    ads_bytes = ads_file.getvalue()

if gsc_bytes is None or ads_bytes is None:
    st.info(
        "Upload both files in the sidebar (or switch on sample data) to get started."
    )
    st.stop()

try:
    df_gsc = parse_gsc_csv(gsc_bytes)
except ValueError as exc:
    st.error(f"Could not read the GSC export: {exc}")
    st.stop()

try:
    df_ads = parse_ads_csv(ads_bytes)
except ValueError as exc:
    st.error(f"Could not read the Google Ads report: {exc}")
    st.stop()

brand_terms = [t for t in brand_input.split(",") if t.strip()]
df_gsc = tag_brand(df_gsc, brand_terms)
df_ads = tag_brand(df_ads, brand_terms)

if brand_filter == "Brand only":
    df_gsc = df_gsc[df_gsc["is_brand"]]
    df_ads = df_ads[df_ads["is_brand"]]
elif brand_filter == "Non-brand only":
    df_gsc = df_gsc[~df_gsc["is_brand"]]
    df_ads = df_ads[~df_ads["is_brand"]]

if df_gsc.empty or df_ads.empty:
    st.warning("No queries left after filtering. Adjust the brand filter or terms.")
    st.stop()

views = build_overlap(df_gsc, df_ads)
both, paid_only, organic_only = views["both"], views["paid_only"], views["organic_only"]
savings = savings_view(both, max_position)

total_spend = float(df_ads["ads_cost"].sum())
overlap_spend = float(both["ads_cost"].sum())
savings_spend = float(savings["ads_cost"].sum())

# ------------------------------- headline -----------------------------------

col_venn, col_stats = st.columns([1, 1])

with col_venn:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    venn2(
        subsets=(len(organic_only), len(paid_only), len(both)),
        set_labels=("Organic (GSC)", "Paid (Google Ads)"),
        set_colors=("#2A9D8F", "#E76F51"),
        alpha=0.75,
        ax=ax,
    )
    ax.set_title("Keyword overlap")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_stats:
    st.metric("Organic queries", f"{len(df_gsc):,}")
    st.metric("Paid search terms", f"{len(df_ads):,}")
    st.metric("Overlapping terms", f"{len(both):,}")
    st.metric(
        "Ad spend on overlapping terms",
        f"{overlap_spend:,.2f}",
        help="Total cost of paid search terms that also appear in your GSC queries. Account currency.",
    )
    st.metric(
        f"Ad spend where organic position is {max_position:g} or better",
        f"{savings_spend:,.2f}",
        delta=(f"{savings_spend / total_spend:.1%} of uploaded spend" if total_spend else None),
        delta_color="off",
        help="Spend on terms you already rank strongly for. Review these for reduction tests, starting with brand terms.",
    )

st.caption(
    "Spend figures are sums from the uploaded Google Ads report, in the "
    "account currency. Reducing spend on a term does not guarantee organic "
    "recovers every paid click; validate with an incrementality or geo holdout test."
)

# ------------------------------- tables -------------------------------------


def _download(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=filename,
    )


tab_savings, tab_both, tab_paid, tab_organic = st.tabs(
    [
        f"Savings opportunities ({len(savings):,})",
        f"Overlap ({len(both):,})",
        f"Paid only ({len(paid_only):,})",
        f"Organic only ({len(organic_only):,})",
    ]
)

with tab_savings:
    st.markdown(
        f"Terms where you pay for ads **and** rank at position **{max_position:g} or "
        "better** organically, sorted by spend. Brand terms with a number one "
        "organic ranking are usually the first candidates for a spend reduction test."
    )
    st.dataframe(savings, use_container_width=True, hide_index=True)
    _download(savings, "Download savings opportunities CSV", "savings_opportunities.csv")

with tab_both:
    st.markdown("All terms appearing in both channels, with paid and organic stats side by side.")
    st.dataframe(both, use_container_width=True, hide_index=True)
    _download(both, "Download overlap CSV", "overlap_keywords.csv")

with tab_paid:
    st.markdown(
        "Terms you pay for with no organic impressions in the GSC export. "
        "Content or landing page gaps if they convert."
    )
    st.dataframe(paid_only, use_container_width=True, hide_index=True)
    _download(paid_only, "Download paid-only CSV", "paid_only_keywords.csv")

with tab_organic:
    st.markdown(
        "Terms with organic visibility and no recorded ad traffic. "
        "Expansion candidates if paid coverage is wanted."
    )
    st.dataframe(organic_only, use_container_width=True, hide_index=True)
    _download(organic_only, "Download organic-only CSV", "organic_only_keywords.csv")

st.markdown("---")
st.markdown("*Created by [Lee Foot](https://leefoot.com)*")
