#!/usr/bin/env python3
"""
OnCrawl Data Extractor - CLI Version
Extract data from OnCrawl crawls using the API.

Author: Lee Foot
Website: https://leefoot.co.uk

Usage:
    python oncrawl_extractor_cli.py --token YOUR_TOKEN --list-workspaces
    python oncrawl_extractor_cli.py --token YOUR_TOKEN --workspace WS_ID --list-projects
    python oncrawl_extractor_cli.py --token YOUR_TOKEN --crawl CRAWL_ID --preset "404 Pages"
    python oncrawl_extractor_cli.py --token YOUR_TOKEN --crawl CRAWL_ID --preset "301 Redirects" -o redirects.csv
"""

import argparse
import sys
import json
from datetime import datetime
from io import StringIO

import requests
import pandas as pd

# Constants
BASE_URL = "https://app.oncrawl.com/api/v2"

# Preset queries
PRESET_QUERIES = {
    "404-pages": {
        "name": "404 Pages",
        "description": "All pages returning 404 status code",
        "fields": ["url", "status_code", "depth", "inrank", "nb_inlinks"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 404]}
            ]
        }
    },
    "301-redirects": {
        "name": "301 Redirects",
        "description": "All pages with 301 redirects and their destinations",
        "fields": ["url", "status_code", "redirect_location", "final_redirect_location", "redirect_count"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 301]}
            ]
        }
    },
    "302-redirects": {
        "name": "302 Redirects",
        "description": "All pages with 302 temporary redirects",
        "fields": ["url", "status_code", "redirect_location", "final_redirect_location", "redirect_count"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 302]}
            ]
        }
    },
    "redirect-chains": {
        "name": "Redirect Chains",
        "description": "Pages with multiple redirects (redirect count > 1)",
        "fields": ["url", "status_code", "redirect_count", "redirect_location", "final_redirect_location", "is_redirect_loop"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["redirect_count", "gt", 1]}
            ]
        }
    },
    "redirect-loops": {
        "name": "Redirect Loops",
        "description": "Pages caught in redirect loops",
        "fields": ["url", "status_code", "redirect_count", "redirect_location", "final_redirect_location"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["is_redirect_loop", "equals", True]}
            ]
        }
    },
    "stale-links": {
        "name": "Stale Links",
        "description": "Redirecting pages that still receive internal links",
        "fields": ["url", "status_code", "redirect_location", "nb_inlinks", "inrank"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "between", [300, 399]]},
                {"field": ["nb_inlinks", "gt", 0]}
            ]
        }
    },
    "indexable": {
        "name": "Indexable Pages",
        "description": "All indexable pages",
        "fields": ["url", "status_code", "meta_robots_index", "canonical_evaluation", "title", "h1"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["meta_robots_index", "equals", True]},
                {"field": ["robots_txt_denied", "equals", False]},
                {"field": ["canonical_evaluation", "equals", "matching"]}
            ]
        }
    },
    "non-indexable": {
        "name": "Non-Indexable Pages",
        "description": "Pages blocked from indexing",
        "fields": ["url", "status_code", "meta_robots_index", "robots_txt_denied", "canonical_evaluation"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"or": [
                    {"field": ["meta_robots_index", "equals", False]},
                    {"field": ["robots_txt_denied", "equals", True]},
                    {"field": ["canonical_evaluation", "equals", "not_matching"]}
                ]}
            ]
        }
    },
    "no-title": {
        "name": "Pages Without Title",
        "description": "Pages missing title tags",
        "fields": ["url", "status_code", "title", "h1"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["title", "is_empty", True]}
            ]
        }
    },
    "no-h1": {
        "name": "Pages Without H1",
        "description": "Pages missing H1 tags",
        "fields": ["url", "status_code", "title", "h1"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["h1", "is_empty", True]}
            ]
        }
    },
    "slow-pages": {
        "name": "Slow Pages",
        "description": "Pages with load time over 3 seconds",
        "fields": ["url", "status_code", "delay_total", "delay_first_byte"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["delay_total", "gt", 3000]}
            ]
        }
    },
    "orphan-pages": {
        "name": "Orphan Pages",
        "description": "Pages with no internal links",
        "fields": ["url", "status_code", "nb_inlinks", "depth"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["nb_inlinks", "equals", 0]}
            ]
        }
    },
    "deep-pages": {
        "name": "Deep Pages",
        "description": "Pages more than 5 clicks deep",
        "fields": ["url", "status_code", "depth", "nb_inlinks"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]},
                {"field": ["status_code", "equals", 200]},
                {"field": ["depth", "gt", 5]}
            ]
        }
    },
    "all-pages": {
        "name": "All Fetched Pages",
        "description": "Complete list of all fetched pages",
        "fields": ["url", "status_code", "depth", "nb_inlinks", "inrank"],
        "oql": {
            "and": [
                {"field": ["fetched", "equals", True]}
            ]
        }
    }
}


def get_headers(api_token):
    """Return headers for API authentication."""
    return {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }


def get_workspaces(api_token):
    """Fetch all workspaces."""
    try:
        response = requests.get(
            f"{BASE_URL}/account",
            headers=get_headers(api_token),
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            workspace_ids = data.get('account', {}).get('workspace_ids', [])

            workspaces = []
            for ws_id in workspace_ids:
                ws_response = requests.get(
                    f"{BASE_URL}/workspaces/{ws_id}",
                    headers=get_headers(api_token),
                    timeout=30
                )
                if ws_response.status_code == 200:
                    ws_data = ws_response.json().get('workspace', {})
                    workspaces.append({
                        'id': ws_id,
                        'name': ws_data.get('name', ws_id)
                    })
            return workspaces
        else:
            print(f"Error: {response.status_code} - {response.text}", file=sys.stderr)
            return []
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return []


def get_projects(api_token, workspace_id):
    """Fetch all projects in a workspace."""
    try:
        all_projects = []
        offset = 0
        limit = 100

        while True:
            response = requests.get(
                f"{BASE_URL}/workspaces/{workspace_id}/projects",
                headers=get_headers(api_token),
                params={'offset': offset, 'limit': limit},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                projects = data.get('projects', [])
                all_projects.extend(projects)

                total = data.get('meta', {}).get('total', 0)
                if offset + limit >= total or len(projects) == 0:
                    break
                offset += limit
            else:
                break

        return all_projects
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return []


def get_crawls(api_token, workspace_id, project_id):
    """Fetch all completed crawls for a project."""
    try:
        response = requests.get(
            f"{BASE_URL}/workspaces/{workspace_id}/crawls",
            headers=get_headers(api_token),
            params={
                'filters[project_id]': project_id,
                'filters[status]': 'done'
            },
            timeout=30
        )

        if response.status_code == 200:
            crawls = response.json().get('crawls', [])
            crawls = sorted(crawls, key=lambda x: x.get('created_at', 0), reverse=True)
            return crawls
        return []
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return []


def export_crawl_data(api_token, crawl_id, fields, oql_query, url_filter=None):
    """Export data from a crawl."""
    try:
        query = {
            "fields": fields,
            "oql": oql_query
        }

        if url_filter:
            query["oql"] = {
                "and": [
                    oql_query,
                    {"field": ["url", "contains", url_filter]}
                ]
            }

        response = requests.post(
            f"{BASE_URL}/data/crawl/{crawl_id}/pages?export=true",
            headers=get_headers(api_token),
            json=query,
            timeout=120
        )

        if response.status_code == 200:
            csv_content = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(csv_content), sep=';', quotechar='"')
            return df, None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract data from OnCrawl crawls using the API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available presets:
{chr(10).join(f'  {k}: {v["description"]}' for k, v in PRESET_QUERIES.items())}

Examples:
    %(prog)s --token YOUR_TOKEN --list-workspaces
    %(prog)s --token YOUR_TOKEN --workspace WS_ID --list-projects
    %(prog)s --token YOUR_TOKEN --workspace WS_ID --project PROJ_ID --list-crawls
    %(prog)s --token YOUR_TOKEN --crawl CRAWL_ID --preset 404-pages
    %(prog)s --token YOUR_TOKEN --crawl CRAWL_ID --preset 301-redirects -o redirects.csv
    %(prog)s --token YOUR_TOKEN --crawl CRAWL_ID --preset stale-links --url-filter "/products/"

Author: Lee Foot (https://leefoot.co.uk)
        """
    )

    # Authentication
    parser.add_argument("--token", required=True, help="OnCrawl API access token")

    # Listing commands
    parser.add_argument("--list-workspaces", action="store_true", help="List all workspaces")
    parser.add_argument("--list-projects", action="store_true", help="List projects in workspace")
    parser.add_argument("--list-crawls", action="store_true", help="List crawls in project")
    parser.add_argument("--list-presets", action="store_true", help="List available preset queries")

    # Selection
    parser.add_argument("--workspace", help="Workspace ID")
    parser.add_argument("--project", help="Project ID")
    parser.add_argument("--crawl", help="Crawl ID")

    # Query
    parser.add_argument("--preset", help="Preset query name (e.g., 404-pages, 301-redirects)")
    parser.add_argument("--url-filter", help="Filter results to URLs containing this text")

    # Output
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--format", choices=["csv", "xlsx", "json"], default="csv",
                        help="Output format (default: csv)")

    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("Available preset queries:\n")
        for key, config in PRESET_QUERIES.items():
            print(f"  {key}")
            print(f"    {config['description']}")
            print(f"    Fields: {', '.join(config['fields'])}")
            print()
        return

    # List workspaces
    if args.list_workspaces:
        workspaces = get_workspaces(args.token)
        if workspaces:
            print("Workspaces:\n")
            for ws in workspaces:
                print(f"  ID: {ws['id']}")
                print(f"  Name: {ws['name']}")
                print()
        else:
            print("No workspaces found or invalid token.")
        return

    # List projects
    if args.list_projects:
        if not args.workspace:
            print("Error: --workspace required for --list-projects", file=sys.stderr)
            sys.exit(1)

        projects = get_projects(args.token, args.workspace)
        if projects:
            print(f"Projects in workspace {args.workspace}:\n")
            for p in projects:
                print(f"  ID: {p['id']}")
                print(f"  Name: {p['name']}")
                print()
        else:
            print("No projects found.")
        return

    # List crawls
    if args.list_crawls:
        if not args.workspace or not args.project:
            print("Error: --workspace and --project required for --list-crawls", file=sys.stderr)
            sys.exit(1)

        crawls = get_crawls(args.token, args.workspace, args.project)
        if crawls:
            print(f"Crawls in project {args.project}:\n")
            for c in crawls:
                crawl_date = datetime.fromtimestamp(c.get('created_at', 0) / 1000).strftime('%Y-%m-%d %H:%M')
                print(f"  ID: {c['id']}")
                print(f"  Date: {crawl_date}")
                print(f"  URLs: {c.get('fetched_urls', 'N/A')}")
                print()
        else:
            print("No crawls found.")
        return

    # Extract data
    if args.preset and args.crawl:
        preset_key = args.preset.lower().replace(" ", "-")

        if preset_key not in PRESET_QUERIES:
            print(f"Error: Unknown preset '{args.preset}'", file=sys.stderr)
            print(f"Available presets: {', '.join(PRESET_QUERIES.keys())}", file=sys.stderr)
            sys.exit(1)

        preset_config = PRESET_QUERIES[preset_key]

        if not args.quiet:
            print(f"Extracting: {preset_config['name']}")
            print(f"Crawl ID: {args.crawl}")
            if args.url_filter:
                print(f"URL filter: {args.url_filter}")

        df, error = export_crawl_data(
            args.token,
            args.crawl,
            preset_config['fields'],
            preset_config['oql'],
            args.url_filter
        )

        if error:
            print(error, file=sys.stderr)
            sys.exit(1)

        if df is None or df.empty:
            print("No data found matching the query.")
            return

        if not args.quiet:
            print(f"Extracted {len(df):,} rows")

        # Generate output filename
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"oncrawl_{preset_key}_{timestamp}.{args.format}"

        # Save output
        if args.format == "csv":
            df.to_csv(output_file, index=False)
        elif args.format == "xlsx":
            df.to_excel(output_file, index=False, sheet_name='Data')
        elif args.format == "json":
            df.to_json(output_file, orient="records", indent=2)

        if not args.quiet:
            print(f"Output saved to: {output_file}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
