"""
Delta Audit Tool - Detect Significant Traffic Changes
Automatically identifies weeks with the biggest traffic shifts in GSC data.
Useful for Google update impact analysis.

Author: Lee Foot
Date: January 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(
    page_title="Delta Audit Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Delta Audit Tool")
st.markdown("""
Detect weeks with significant traffic changes in Google Search Console data.
Automatically identifies the week with the biggest traffic shift - useful for
analyzing Google update impacts.
""")

# Sidebar configuration
st.sidebar.header("Configuration")
window_size = st.sidebar.slider(
    "Rolling Window Size (days)",
    min_value=3,
    max_value=14,
    value=7,
    help="Number of days for the rolling average calculation"
)

date_column = st.sidebar.text_input(
    "Date Column Name",
    value="date",
    help="Name of the date column in your CSV"
)

clicks_column = st.sidebar.text_input(
    "Clicks Column Name",
    value="clicks",
    help="Name of the clicks column in your CSV"
)

impressions_column = st.sidebar.text_input(
    "Impressions Column Name",
    value="impressions",
    help="Name of the impressions column in your CSV"
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload your GSC data CSV",
    type=['csv'],
    help="Upload a CSV export from Google Search Console with date, clicks, and impressions columns"
)

if uploaded_file is not None:
    try:
        # Load the data
        data = pd.read_csv(uploaded_file, low_memory=False)

        st.subheader("Data Preview")
        st.dataframe(data.head(10))

        # Validate columns exist
        required_cols = [date_column, clicks_column, impressions_column]
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
            st.info(f"Available columns: {', '.join(data.columns.tolist())}")
            st.stop()

        # Process the data
        with st.spinner("Analyzing traffic patterns..."):
            # Ensure the date column is correctly named and clean any non-date values
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

            # Drop rows where the date conversion failed
            data = data.dropna(subset=[date_column])

            # Convert numeric columns
            data[clicks_column] = pd.to_numeric(data[clicks_column], errors='coerce')
            data[impressions_column] = pd.to_numeric(data[impressions_column], errors='coerce')

            # Drop rows where numeric conversion failed
            data = data.dropna(subset=[clicks_column, impressions_column])

            # Set the date column as the index
            data.set_index(date_column, inplace=True)

            # Ensure the index is unique by aggregating data on the same date
            data = data.groupby(data.index).sum(numeric_only=True)

            # Calculate daily total clicks
            daily_clicks = data[clicks_column].resample('D').sum()

            # Use a rolling window to calculate the moving average
            rolling_clicks = daily_clicks.rolling(window=window_size).mean()

            # Identify the date of significant change in the rolling average
            rolling_clicks_diff = rolling_clicks.diff().abs()
            significant_change_date = rolling_clicks_diff.idxmax()

            # Display the significant change date
            st.success(f"**Most Significant Traffic Change:** {significant_change_date.strftime('%Y-%m-%d')}")

            # Calculate weekly data from Monday to Sunday
            weekly_data = data.resample('W-MON').sum()

            # Add a column to indicate if the week data is partial
            weekly_data['partial'] = np.where(data.resample('W-MON').size() < 7, True, False)

            # Filter out partial weeks
            weekly_data_complete = weekly_data[weekly_data['partial'] == False]

            # Snap the significant change date to the corresponding week
            significant_week_start = significant_change_date - pd.Timedelta(days=significant_change_date.weekday())
            significant_week_end = significant_week_start + pd.Timedelta(days=6)

            st.info(f"**Significant Week:** {significant_week_start.strftime('%Y-%m-%d')} to {significant_week_end.strftime('%Y-%m-%d')}")

            # Visualizations
            st.subheader("Traffic Visualization")

            # Daily clicks line chart
            fig_daily = px.line(
                x=daily_clicks.index,
                y=daily_clicks.values,
                title="Daily Clicks Over Time",
                labels={'x': 'Date', 'y': 'Clicks'}
            )

            # Add vertical line for significant change
            fig_daily.add_vline(
                x=significant_change_date,
                line_dash="dash",
                line_color="red",
                annotation_text="Significant Change"
            )

            st.plotly_chart(fig_daily, use_container_width=True)

            # Rolling average chart
            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(
                x=rolling_clicks.index,
                y=rolling_clicks.values,
                name=f'{window_size}-Day Rolling Average',
                line=dict(color='blue')
            ))
            fig_rolling.add_vline(
                x=significant_change_date,
                line_dash="dash",
                line_color="red",
                annotation_text="Significant Change"
            )
            fig_rolling.update_layout(
                title=f'{window_size}-Day Rolling Average of Clicks',
                xaxis_title='Date',
                yaxis_title='Clicks (Rolling Avg)'
            )
            st.plotly_chart(fig_rolling, use_container_width=True)

            # Week-over-week comparison
            st.subheader("Week-over-Week Comparison")

            # Ensure there are enough weeks to compare
            pre_week_start = significant_week_start - pd.Timedelta(days=7)
            post_week_start = significant_week_start + pd.Timedelta(days=7)

            if (pre_week_start in weekly_data.index) and (post_week_start in weekly_data.index):
                # Define the periods before and after the significant week
                pre_week = pre_week_start
                post_week = post_week_start

                # Extract the weekly data for comparison
                pre_week_data = weekly_data.loc[pre_week]
                significant_week_data = weekly_data.loc[significant_week_start]
                post_week_data = weekly_data.loc[post_week]

                # Calculate the absolute and relative changes
                absolute_change_clicks = post_week_data[clicks_column] - pre_week_data[clicks_column]
                relative_change_clicks = (absolute_change_clicks / pre_week_data[clicks_column]) * 100
                absolute_change_impressions = post_week_data[impressions_column] - pre_week_data[impressions_column]
                relative_change_impressions = (absolute_change_impressions / pre_week_data[impressions_column]) * 100

                # Create comparison summary
                comparison_summary = pd.DataFrame({
                    'Metric': ['Clicks', 'Impressions'],
                    f'{pre_week_start.strftime("%Y-%m-%d")} - {(pre_week_start + pd.Timedelta(days=6)).strftime("%Y-%m-%d")}': [
                        pre_week_data[clicks_column], pre_week_data[impressions_column]
                    ],
                    f'{significant_week_start.strftime("%Y-%m-%d")} - {significant_week_end.strftime("%Y-%m-%d")} (Significant)': [
                        significant_week_data[clicks_column], significant_week_data[impressions_column]
                    ],
                    f'{post_week_start.strftime("%Y-%m-%d")} - {(post_week_start + pd.Timedelta(days=6)).strftime("%Y-%m-%d")}': [
                        post_week_data[clicks_column], post_week_data[impressions_column]
                    ],
                    'Absolute Change (WoW)': [absolute_change_clicks, absolute_change_impressions],
                    'Relative Change (WoW %)': [f"{relative_change_clicks:.2f}%", f"{relative_change_impressions:.2f}%"]
                })

                st.dataframe(comparison_summary, use_container_width=True)

                # Color-coded metrics
                col1, col2 = st.columns(2)

                with col1:
                    delta_color = "normal" if absolute_change_clicks >= 0 else "inverse"
                    st.metric(
                        label="Clicks Change (WoW)",
                        value=f"{int(absolute_change_clicks):,}",
                        delta=f"{relative_change_clicks:.1f}%",
                        delta_color=delta_color
                    )

                with col2:
                    delta_color = "normal" if absolute_change_impressions >= 0 else "inverse"
                    st.metric(
                        label="Impressions Change (WoW)",
                        value=f"{int(absolute_change_impressions):,}",
                        delta=f"{relative_change_impressions:.1f}%",
                        delta_color=delta_color
                    )

                # Download comparison report
                st.subheader("Download Report")

                buffer = BytesIO()
                comparison_summary.to_csv(buffer, index=False)
                buffer.seek(0)

                st.download_button(
                    label="📥 Download Comparison Report (CSV)",
                    data=buffer,
                    file_name="delta_audit_comparison.csv",
                    mime="text/csv"
                )

            else:
                st.warning("Not enough data to compare the weeks before and after the significant change.")

            # Weekly trend table
            st.subheader("Weekly Traffic Summary")
            weekly_display = weekly_data_complete[[clicks_column, impressions_column]].copy()
            weekly_display.index = weekly_display.index.strftime('%Y-%m-%d')
            weekly_display = weekly_display.sort_index(ascending=False)
            st.dataframe(weekly_display, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Upload a GSC data CSV file to get started.")

    st.markdown("""
    ### Expected CSV Format
    Your CSV should contain at minimum:
    - **date** - Date column (YYYY-MM-DD format)
    - **clicks** - Number of clicks
    - **impressions** - Number of impressions

    You can customize column names in the sidebar.

    ### How it Works
    1. Upload your GSC data export
    2. The tool calculates a rolling average of daily clicks
    3. It identifies the date with the largest change in the rolling average
    4. Compares the weeks before and after the significant change
    5. Outputs a comparison report showing the impact

    ### Use Cases
    - Analyzing Google algorithm update impacts
    - Identifying sudden traffic changes
    - Comparing performance before/after site changes
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by [Lee Foot](https://leefoot.co.uk) · [Bluesky](https://bsky.app/profile/leefootseo.bsky.social) · [LinkedIn](https://www.linkedin.com/in/lee-foot/)")
