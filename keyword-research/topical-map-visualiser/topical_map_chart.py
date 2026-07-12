"""
Topical Map Visualiser - Core chart builder

Builds a hierarchical JSON structure from a tagged keyword CSV
(parent topic > subtopic > keyword) and renders it as an interactive,
zoomable D3.js circle packing chart via Jinja2.

Author: Lee Foot
Website: https://www.leefoot.com
"""

import json

from jinja2 import Template

# Chart titles for each supported metric
CHART_TITLES = {
    'count': 'Keyword Count by Topic',
    'impressions': 'Keyword Impressions by Topic',
    'first_page_count': 'First Page Keywords by Topic',
    'top_3_count': 'Top 3 Position Keywords by Topic',
    'clicks': 'Total Clicks by Topic',
}

METRIC_CHOICES = list(CHART_TITLES.keys())


def create_hierarchy(df, metric='count', parent_col='Parent', child_col='Child',
                     keyword_col='query', position_col='position',
                     impressions_col='impressions', clicks_col='clicks'):
    """Build a root > parent > child > keyword hierarchy with a value per keyword."""
    if metric not in METRIC_CHOICES:
        raise ValueError(f"Invalid metric '{metric}'. Choose from: {', '.join(METRIC_CHOICES)}")

    hierarchy = {'name': 'root', 'children': []}
    for _, row in df.iterrows():
        parent = str(row[parent_col])
        child = str(row[child_col])
        query = str(row[keyword_col])

        if metric == 'count':
            value = 1
        elif metric == 'impressions':
            value = row[impressions_col]
        elif metric == 'first_page_count':
            value = 1 if 1 <= row[position_col] <= 10 else 0
        elif metric == 'top_3_count':
            value = 1 if 1 <= row[position_col] <= 3 else 0
        elif metric == 'clicks':
            value = row[clicks_col]

        parent_node = next((item for item in hierarchy['children'] if item['name'] == parent), None)
        if parent_node is None:
            parent_node = {'name': parent, 'children': []}
            hierarchy['children'].append(parent_node)

        child_node = next((item for item in parent_node['children'] if item['name'] == child), None)
        if child_node is None:
            child_node = {'name': child, 'children': []}
            parent_node['children'].append(child_node)

        child_node['children'].append({'name': query, 'value': value})

    # Aggregate values for parent and child nodes
    def aggregate_values(node):
        if 'children' in node:
            node['value'] = sum(aggregate_values(child) for child in node['children'])
        return node['value']

    aggregate_values(hierarchy)
    return hierarchy


# HTML template with D3.js zoomable circle packing chart
TEMPLATE_STRING = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ chart_title }}</title>
    <script src="https://d3js.org/d3.v6.min.js"></script>
    <style>
        body { margin: 0; overflow: hidden; font-family: Arial, sans-serif; }
        #chart { width: 100vw; height: 100vh; }
        .label { pointer-events: none; }
        #chart-title {
            position: absolute;
            top: 20px;
            left: 20px;
            font-size: 24px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="chart-title">{{ chart_title }}</div>
    <div id="chart"></div>
    <script>
        const data = {{ data | safe }};

        const width = window.innerWidth;
        const height = window.innerHeight;

        // Define a pastel colour palette
        const pastelColors = [
            "#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#FFDFBA",
            "#E0BBE4", "#D4F0F0", "#FFC6FF", "#DAEAF6", "#B5EAD7"
        ];

        // Create a colour scale that assigns colours to top-level nodes
        const colorScale = d3.scaleOrdinal()
            .domain(data.children.map(d => d.name))
            .range(pastelColors);

        const pack = data => d3.pack()
            .size([width * 0.9, height * 0.9]) // Reduce size to 90% to ensure full visibility
            .padding(3)
            (d3.hierarchy(data)
            .sum(d => d.value)
            .sort((a, b) => b.value - a.value));

        const root = pack(data);
        let focus = root;
        let view;

        const svg = d3.select("#chart")
            .append("svg")
            .attr("viewBox", `0 0 ${width} ${height}`)
            .style("display", "block")
            .style("cursor", "pointer")
            .on("click", (event) => zoom(event, root));

        const g = svg.append("g")
            .attr("transform", `translate(${width / 2},${height / 2})`);

        const node = g.selectAll("circle")
            .data(root.descendants().slice(1))
            .join("circle")
            .attr("fill", d => {
                while (d.depth > 1) d = d.parent;
                return d.data.name === data.name ? "#fff" : colorScale(d.data.name);
            })
            .attr("fill-opacity", d => d.children ? 0.6 : 1)
            .attr("pointer-events", d => !d.children ? "none" : null)
            .on("mouseover", function() { d3.select(this).attr("stroke", "#000"); })
            .on("mouseout", function() { d3.select(this).attr("stroke", null); })
            .on("click", (event, d) => focus !== d && (zoom(event, d), event.stopPropagation()));

        const label = g.append("g")
            .attr("class", "labels")
            .selectAll("text")
            .data(root.descendants())
            .join("text")
            .style("fill-opacity", d => d.parent === root ? 1 : 0)
            .style("display", d => d.parent === root ? "inline" : "none")
            .style("font", d => Math.min(d.r / 3, 16) + "px sans-serif")
            .style("pointer-events", "none")
            .attr("class", "label")
            .attr("text-anchor", "middle")
            .text(d => d.data.name);

        // Set initial view to show the entire chart
        zoomTo([root.x, root.y, root.r * 2.05]); // Slightly increase zoom out to ensure full visibility

        function zoomTo(v) {
            const k = Math.min(width, height) / v[2];

            view = v;

            label.attr("transform", d => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`);
            node.attr("transform", d => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`);
            node.attr("r", d => d.r * k);
        }

        function zoom(event, d) {
            const focus0 = focus;

            focus = d;

            const transition = svg.transition()
                .duration(event.altKey ? 7500 : 750)
                .tween("zoom", d => {
                    const i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2]);
                    return t => zoomTo(i(t));
                });

            label
                .filter(function(d) { return d.parent === focus || this.style.display === "inline"; })
                .transition(transition)
                .style("fill-opacity", d => d.parent === focus ? 1 : 0)
                .on("start", function(d) { if (d.parent === focus) this.style.display = "inline"; })
                .on("end", function(d) { if (d.parent !== focus) this.style.display = "none"; });
        }
    </script>
</body>
</html>
"""


def render_chart(df, metric='count', chart_title=None, **column_kwargs):
    """Return the rendered HTML for the chart as a string."""
    hierarchy = create_hierarchy(df, metric=metric, **column_kwargs)
    hierarchy_json = json.dumps(hierarchy)

    if chart_title is None:
        chart_title = CHART_TITLES.get(metric, 'Keyword Analysis by Topic')

    template = Template(TEMPLATE_STRING)
    return template.render(data=hierarchy_json, chart_title=chart_title)
