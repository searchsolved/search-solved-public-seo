import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
from jinja2 import Template
from openai import OpenAI
import tiktoken
import ast
import re
from stqdm import stqdm


# App title and introduction
st.title('Entity Extraction & Visualisation App by Lee Foot.')
st.write("This app extracts named entities from text using OpenAI's API and visualizes them using a D3 circle packing chart.")

# Dropdown for detailed instructions
with st.expander("User Guide"):
    st.write("""
    **Operation Steps:**
    1. **Text Input**: Enter the desired text into the provided area. The text is used for entity extraction.
    2. **Entity Extraction**: Click 'Process Text'. The application employs OpenAI's API to perform entity extraction. The text is divided into batches if necessary, improving the precision of entity detection.
    3. **Results Display**: Post-processing, the application visualises the entities in a D3 circle packing chart, sorted with SpaCy-style NER labels and contextual information.
    4. **Entity Details**: Entities are linked to Wikipedia for additional information when available.
    5. **Downloads**: Two download options are available: a CSV file detailing the entities and an HTML file of the D3 chart. The CSV includes Wikipedia URLs where applicable.

    **Note**: The token count influences the extraction detail. Lowering the token count may result in multiple batch processing, potentially increasing the output's fidelity for longer texts.
    """)

# Constants for OpenAI model and token limits
api_key = st.text_input("Enter your OpenAI API key:", type="password")
model = st.selectbox("Choose the model:", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"])
max_tokens = st.slider("Maximum Tokens:", min_value=500, max_value=2000, value=1000, step=100)
input_text = st.text_area("Enter your text here:", height=150)


# Function to extract context around specified entities in a text
def extract_context(text, entity, context_size=5):
    """
    Extracts context around each occurrence of a specified entity in a given text.

    Parameters:
    text (str): The text from which context is to be extracted.
    entity (str): The entity around which context is needed.
    context_size (int, optional): The number of words around the entity to include in the context. Defaults to 5.

    Returns:
    list of str: A list where each element is a string containing the context around an occurrence of the entity.
    """
    pattern = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)
    matches = list(pattern.finditer(text))

    if not matches:
        return ['No context found']

    contexts = []
    for match in matches:
        start_index = match.start()
        words = text.split()
        word_start_index = len(text[:start_index].split())
        context_before = ' '.join(words[max(0, word_start_index - context_size):word_start_index])
        context_after = ' '.join(
            words[word_start_index + len(entity.split()):min(len(words), word_start_index + context_size)])
        contexts.append(f"{context_before} ... {context_after}")

    return contexts

# Function to count occurrences of each entity in the text
def count_entities(input_text, entities_sorted):
    """
    Counts the occurrences of each entity in a given text.

    Parameters:
    input_text (str): The text in which entities are to be counted.
    entities_sorted (list): A list of entities to be counted in the text.

    Returns:
    dict: A dictionary where keys are entities and values are the count of their occurrences in the text.
    """
    counts = {entity: 0 for entity in entities_sorted}
    working_text = input_text.lower()

    for entity in entities_sorted:
        entity_pattern = re.escape(entity.lower())
        # Updated pattern to handle cases where the entity may have a period followed by more characters
        pattern = re.compile(r'\b' + entity_pattern + r'(?!\w)', re.IGNORECASE)
        matches = pattern.findall(working_text)

        if matches:
            counts[entity] = len(matches)
            # Replace the entire match, including potential punctuation
            working_text = pattern.sub('', working_text)

    return counts

# Function to process text and create hierarchy
def process_text(api_key, input_text):
    client = initialize_openai_client(api_key)

    # Process text to create DataFrame
    df = batch_process_and_create_dataframe(input_text, client)

    # Apply necessary transformations to the DataFrame
    df['context'] = df['context'].apply(safe_literal_eval)
    df = df.explode('context')

    # Create hierarchy for D3.js visualization
    hierarchy = create_hierarchy(df)

    return df, hierarchy

# Function to process text in batches and create a DataFrame with named entity recognition results
def batch_process_and_create_dataframe(input_text, client):
    """
    Processes the input text in batches for named entity recognition and creates a DataFrame with the results.

    Parameters:
    input_text (str): The text to be processed for named entity recognition.

    Returns:
    pandas.DataFrame: A DataFrame containing the results of the entity recognition process.
    """
    # Initialize the tokenizer
    enc = tiktoken.encoding_for_model(model)

    # Tokenize the input text
    tokens = enc.encode(input_text)

    # Split the tokens into batches
    batches = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]

    # Master dictionary for aggregated results
    aggregated_results = {}

    # Processing with stqdm
    for batch in stqdm(batches, desc="Processing batches"):
        # Display the disclaimer only when processing starts
        if batch == batches[0]:
            st.warning(
                "**Disclaimer:** The OpenAI API is in a preview state as of 23rd November 2023. Processing may be slow or you may experience brief freezing due to API responsiveness during this phase.")
        batch_text = enc.decode(batch)
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": json.dumps({
                        "task": "Named Entity Recognition",
                        "instructions": "Identify all named entities. Add Spacy_label, descriptive_tag (3 words max) & Wikipedia URL for each entity. Output JSON",
                        "expected_output_format": {
                            "entities": [
                                {
                                    "entity": "<ENTITY_NAME>",
                                    "spacy_label": "<SPACY_LABEL>",
                                    "descriptive_tag": "<DESCRIPTIVE_TAG>",
                                    "wikipedia_url": "<WIKIPEDIA_URL>"
                                }
                            ]
                        }
                    })
                },
                {
                    "role": "user",
                    "content": batch_text
                }
            ]
        )
        # Extract and store the response
        batch_response = json.loads(response.choices[0].message.content)

        # Update the master dictionary with batch response
        if isinstance(batch_response, dict) and 'entities' in batch_response:
            for entity in batch_response['entities']:
                entity_name = entity.get('entity')
                if entity_name and entity_name not in aggregated_results:
                    aggregated_results[entity_name] = {
                        'entity': entity_name,
                        'spacy_label': entity.get('spacy_label'),
                        'descriptive_tag': entity.get('descriptive_tag'),
                        'wikipedia_url': entity.get('wikipedia_url'),
                    }
        else:
            # Handle unexpected response format
            print(f"Unexpected response format: {batch_response}")

    # Extract entities from aggregated_results to create a sorted list of unique entity names
    entities = list(aggregated_results.keys())
    entities_sorted = sorted(entities, key=len, reverse=True)

    # Use entities_sorted for context and count extraction
    entity_counts = count_entities(input_text, entities_sorted)
    for entity in entities_sorted:
        context = extract_context(input_text, entity)
        aggregated_results[entity]['context'] = context
        aggregated_results[entity]['count'] = entity_counts[entity]  # Make sure this line correctly adds the count

    # Convert to DataFrame
    df = pd.DataFrame(list(aggregated_results.values()))

    # convert case of tags
    df['spacy_label'] = df['spacy_label'].str.upper()  # convert primary tag to UPPER care
    df['descriptive_tag'] = df['descriptive_tag'].str.upper()  # convert secondary tag to Title case

    return df


# Function to create a hierarchical data structure from the DataFrame for visualization
def create_hierarchy(df):
    """
    Creates a hierarchical structure from a DataFrame for visualization purposes.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing entities with their respective categories and counts.

    Returns:
    dict: A dictionary representing the hierarchical structure of the data.
    """
    hierarchy = {'name': 'root', 'children': []}
    for _, row in df.iterrows():
        primary = next((item for item in hierarchy['children'] if item['name'] == row['spacy_label']), None)
        if primary is None:
            primary = {'name': row['spacy_label'], 'children': []}
            hierarchy['children'].append(primary)
        secondary = next((item for item in primary['children'] if item['name'] == row['descriptive_tag']), None)
        if secondary is None:
            secondary = {'name': row['descriptive_tag'], 'children': []}
            primary['children'].append(secondary)
        entity = next((item for item in secondary['children'] if item['name'] == row['entity']), None)
        if entity is None:
            entity = {'name': row['entity'], 'value': row['count']}
            secondary['children'].append(entity)

    return hierarchy


# Function to safely evaluate a string as a Python expression
def safe_literal_eval(s):
    """
    Safely evaluates a string as a Python expression. If the string is not a valid Python expression, it is returned as is.

    Parameters:
    s (str): The string to be evaluated.

    Returns:
    any: The result of the evaluated expression or the original string if it's not a valid Python expression.
    """
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # Return the original string if it's not a list representation
        return [s]


# Function to convert context string to a list of items, if applicable
def convert_to_list_if_string(context):
    """
    Converts a string context to a list of context items, if applicable.

    Parameters:
    context (str): The context string to be converted.

    Returns:
    list or str: A list of context items if the context can be split, otherwise the original context string.
    """
    if isinstance(context, str):
        try:
            # Split the context string into a list of context items
            context_list = [c.strip() for c in context.split('...')]
            return context_list
        except ValueError:
            # Handle the case where the context is not a valid Python expression
            return [context]
    return context


# Initialize OpenAI client
def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)

# Function to render D3.js circle packing chart
def render_d3_circle_packing(hierarchy):
    hierarchy_json = json.dumps(hierarchy, indent=2)
    template = Template(template_string)
    rendered_html = template.render(hierarchy_json=hierarchy_json)
    return rendered_html


# HTML and JavaScript template for the D3.js circle packing chart
template_string = """
<!DOCTYPE html>
<meta charset="utf-8">
<style>
  /* CSS styling */
  circle {
    fill-opacity: 0.7;
    stroke: #fff;
    stroke-width: 1px;
    transition: stroke 0.2s; /* Add transition for smooth highlighting */
  }

  circle:hover {
    stroke: #000; /* Change stroke color to black on hover */
  }

  .leaf circle {
    fill: #ff7f0e;
  }

  .node text {
    font: 12px sans-serif;
  }

  .node--leaf text {
    text-shadow: 0 1px 0 #fff;
  }

  .label {
    pointer-events: none;
    text-anchor: middle;
    font-size: 11px;
    font-family: 'Gill Sans', sans-serif;
  }
</style>

<body>
<script src="https://d3js.org/d3.v6.min.js"></script>
<script>

  var data = {{ hierarchy_json | safe }};


  var width = 932;
  var height = width;
  var format = d3.format(",d");

  var pack = data => d3.pack()
      .size([width - 2, height - 2])
      .padding(3)
      (d3.hierarchy(data)
      .sum(d => d.value)
      .sort((a, b) => b.value - a.value));

  var color = d3.scaleLinear()
      .domain([0, 5])
      .range(["hsl(152,80%,80%)", "hsl(228,30%,40%)"])
      .interpolate(d3.interpolateHcl);

  const root = pack(data);
  let focus = root;
  let view;

  const svg = d3.create("svg")
      .attr("viewBox", `-${width / 2} -${height / 2} ${width} ${height}`)
      .attr("width", width)
      .attr("height", height)
      .style("max-width", "100%")
      .style("height", "auto")
      .style("margin", "0 -14px")
      .style("background", "rgb(240, 240, 240)") // Set the background color here
      .style("cursor", "pointer")
      .on("click", (event) => zoom(event, root));

  const node = svg.append("g")
      .selectAll("circle")
      .data(root.descendants().slice(1))
      .join("circle")
      .attr("fill", d => d.children ? color(d.depth) : "white")
      .attr("pointer-events", d => !d.children ? "none" : null)
      .on("mouseover", function() { d3.select(this).attr("stroke", "#000"); })
      .on("mouseout", function() { d3.select(this).attr("stroke", null); })
      .on("click", (event, d) => focus !== d && (zoom(event, d), event.stopPropagation()));

  const label = svg.append("g")
      .attr("pointer-events", "none")
      .attr("text-anchor", "top")
      .selectAll("text")
      .data(root.descendants())
      .join("text")
      .style("fill-opacity", d => d.parent === root ? 1 : 0)
      .style("display", d => d.parent === root ? "inline" : "none")
      .text(d => d.data.name);

  zoomTo([root.x, root.y, root.r * 2]);

  function zoomTo(v) {
    const k = width / v[2];
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

  document.body.appendChild(svg.node());

</script>
</body>
"""

# Initialize session state variables
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
    st.session_state['hierarchy'] = None


# Button to process the text
if st.button('Process Text'):
    if api_key and input_text:
        with st.spinner('Processing...'):
            df, hierarchy = process_text(api_key, input_text)
            st.session_state['processed_data'] = df
            st.session_state['hierarchy'] = hierarchy

            # Generate HTML for D3 visualization
            rendered_html = render_d3_circle_packing(hierarchy)
            components.html(rendered_html, height=800)

# Check if data is processed and available in the session state
if st.session_state['processed_data'] is not None:
    df = st.session_state['processed_data']
    st.write(df)

    # Provide download links
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV", data=csv, file_name='extracted_entities.csv', mime='text/csv')

if st.session_state['hierarchy'] is not None:
    hierarchy = st.session_state['hierarchy']
    rendered_html = render_d3_circle_packing(hierarchy)

    # Provide download link for D3 visualization
    st.download_button(label="Download D3 Visualization", data=rendered_html, file_name='circle_packing_chart.html', mime='text/html')

# Run the app with `streamlit run your_script_name.py`

# Add a credit with an email link at the bottom of your app
st.markdown("""
---
*Developed by Lee Foot - [Website](https://LeeFoot.com)*
*Hire me for bespoke work: [hello@LeeFoot.com](mailto:hello@LeeFoot.com)*
""", unsafe_allow_html=True)
