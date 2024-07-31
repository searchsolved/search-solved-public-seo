# Display results after form submission
if st.session_state.show_results:
    # Create tabs
    tab_names = ["Folder Visualization", "Status Code Visualization", "Frequently Changed Pages", "robots.txt Changes",
                 "Download URLs"]

    # Use a single selectbox to choose the active tab
    selected_tab = st.selectbox("Select a tab:", tab_names, key="tab_selector", on_change=on_tab_change,
                                index=tab_names.index(st.session_state.active_tab))

    # Update active_tab based on selection
    if selected_tab != st.session_state.active_tab:
        st.session_state.active_tab = selected_tab

    # Display content based on the selected tab
    if st.session_state.active_tab == "Folder Visualization":
        st.header("Folder Visualization")
        st.plotly_chart(visualize_folder_types_over_time(st.session_state.unique_urls, st.session_state.vis_type),
                        use_container_width=True)

    elif st.session_state.active_tab == "Status Code Visualization":
        st.header("Status Code Visualization")
        st.plotly_chart(visualize_status_codes_over_time(st.session_state.unique_urls, st.session_state.vis_type),
                        use_container_width=True)

    elif st.session_state.active_tab == "Frequently Changed Pages":
        # ... (code for Frequently Changed Pages tab)

    elif st.session_state.active_tab == "robots.txt Changes":
        # ... (code for robots.txt Changes tab)

    elif st.session_state.active_tab == "Download URLs":
        st.header("Download URLs")

        # Add filter options here
        filter_option = st.radio(
            "Select URL filter option:",
            ["All (HTML, Images, CSS, JS, etc.)", "HTML only", "HTML + Images"],
            index=0,
            help="Choose which types of URLs to include in the download. 'HTML only' includes HTML files and robots.txt, but excludes .json, .xml, and other .txt files."
        )

        # Add option for unique URLs
        unique_only = st.checkbox("Export only unique URLs", value=False,
                                  help="If checked, only one instance of each URL will be exported, regardless of how many times it was captured.")

        # Add option to validate against robots.txt
        validate_robots = st.checkbox("Validate URLs against latest robots.txt", value=False,
                                      help="If checked, URLs will be checked against the latest version of robots.txt.")

        # Function to apply filter
        def apply_filter(url, option):
            if option == "All (HTML, Images, CSS, JS, etc.)":
                return True
            elif option == "HTML only":
                return url.endswith(('.html', '.htm', '/')) or url.endswith('robots.txt')
            elif option == "HTML + Images":
                return url.endswith(('.html', '.htm', '/', '.jpg', '.jpeg', '.png', '.gif', '.svg')) or url.endswith('robots.txt')

        # Fetch and parse robots.txt if validation is requested
        rp = None
        if validate_robots:
            rp = get_latest_robots_txt(st.session_state.domain)

        # Filter and deduplicate URLs
        filtered_urls = {}
        for url, timestamp, statuscode, digest in st.session_state.unique_urls:
            if apply_filter(url, filter_option):
                full_url = f"https://{st.session_state.domain}{url}"
                if unique_only:
                    if url not in filtered_urls or timestamp > filtered_urls[url][0]:
                        filtered_urls[url] = (timestamp, statuscode, digest)
                else:
                    if url not in filtered_urls:
                        filtered_urls[url] = []
                    filtered_urls[url].append((timestamp, statuscode, digest))

        # Convert back to list format and check against robots.txt
        final_urls = []
        for url, data in filtered_urls.items():
            full_url = f"https://{st.session_state.domain}{url}"
            if unique_only:
                timestamp, statuscode, digest = data
                allowed = "N/A"
                if rp:
                    allowed = "Allowed" if rp.can_fetch("*", full_url) else "Blocked"
                final_urls.append((url, timestamp, statuscode, digest, allowed))
            else:
                for timestamp, statuscode, digest in data:
                    allowed = "N/A"
                    if rp:
                        allowed = "Allowed" if rp.can_fetch("*", full_url) else "Blocked"
                    final_urls.append((url, timestamp, statuscode, digest, allowed))

        # Prepare CSV content with headers
        csv_content = "URL,Timestamp,Status Code,Digest,Robots.txt Status\n"
        csv_content += "\n".join([f"{url},{timestamp},{statuscode},{digest},{allowed}" for url, timestamp, statuscode, digest, allowed in final_urls])

        # Convert to bytes with UTF-8-SIG encoding (UTF-8 with BOM)
        csv_bytes = csv_content.encode('utf-8-sig')

        st.download_button(
            label="Download Filtered URLs",
            data=csv_bytes,
            file_name=f"{st.session_state.domain}_filtered_urls.csv",
            mime="text/csv",
            key="download"
        )

        st.write(f"Total URLs after filtering: {len(final_urls)}")

        # Display summary of robots.txt validation
        if validate_robots and rp:
            allowed_count = sum(1 for url in final_urls if url[4] == "Allowed")
            blocked_count = sum(1 for url in final_urls if url[4] == "Blocked")
            st.write(f"URLs allowed by robots.txt: {allowed_count}")
            st.write(f"URLs blocked by robots.txt: {blocked_count}")
